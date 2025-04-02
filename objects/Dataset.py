import uuid
import os
from config import *
import json
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import shutil

BATCH_CAN_CONTAINS = ["cdens","vdens","cospectra","density"]

def _open_batch(batch_name):
    assert os.path.exists(TRAINING_BATCH_FOLDER), LOGGER.error(f"Can't open batch {batch_name}, no folder exists.")
    batch_path = os.path.join(TRAINING_BATCH_FOLDER,batch_name)

    files = glob.glob(batch_path+"/*.npy")
    files = [f.split("/")[-1] for f in files]

    imgs = [[] for _ in range(len(np.unique([int(f.split("_")[0]) for f in files])))]
    order = []
    for bc in BATCH_CAN_CONTAINS:
        pot_files = [f for f in files if bc in f]

        if len(pot_files) <= 0 :
            continue

        ids = [int(f.split("_")[0]) for f in pot_files]
        indexes = np.argsort(ids)
        for j,i in enumerate(indexes):
            imgs[j].append(os.path.join(batch_path,pot_files[i]))
        order.append(bc)
    return imgs, order

def getDataset(name):
    ds = Dataset()
    try:
        ds.load_from_name(name, changeName=True)
    except AssertionError:
        LOGGER.error(f"Can't load dataset: {name}")
        return None
    return ds

class Dataset():
    """Dataset object which contains just the imgs paths for reduce the memory usage"""
    def __init__(self):
        self.batch = []
        self.settings = {}
        """settings contains:
            'order', eg: 'order':['cdens','vdens','cospectra']
        """
        self.name = str(uuid.uuid4())

        self.active_batch = []

    def get_element_index(self, name):
        assert "order" in self.settings, LOGGER.error("No order list in dataset settings")
        for i,o in enumerate(self.settings["order"]):
            if o == name:
                return i

    def load_from_name(self, name, changeName=False):
        LOGGER.log(f"Loading dataset {name}")
        if changeName:
            self.name = name
        batch, order = _open_batch(name)
        self.batch.extend(batch)
        self.settings["order"] = order

    def add(self,imgs_path):
        self.batch.append(imgs_path)
    
    def get(self, indexes = None):
        if not(indexes is None):
            if not(type(indexes) is list) or len(indexes) < 2:
                return self.load(np.array(self.batch)[indexes])
            else:
                return self.load(np.array(self.batch)[np.array(indexes)])
        else:
            return self.load(np.array(self.batch))

    def load(self, paths):
        result = []
        for pair in paths:
            if not(type(pair) is list or type(pair) is np.ndarray):
                result.append(np.load(pair))
                continue
            temp = []
            for p in pair:
                temp.append(np.load(p))
            result.append(temp)
        del self.active_batch
        self.active_batch = result
        return result

    def split(self, cutoff=0.7):
        LOGGER.log(f"Splitting dataset {self.name} with cutoff at {cutoff}")
        batch = np.array(self.batch)
        cut_index = int(cutoff * len(batch))

        b1 = Dataset()
        b1.batch = batch[:cut_index]
        b1.settings = self.settings #TODO
        b1.name = self.name + "_b1"
        b2 = Dataset()
        b2.batch = batch[cut_index:]
        b2.settings = self.settings #TODO
        b2.name = self.name + "_b2"

        return (b1, b2)
    
    def clone(self, new_name):
        ds = Dataset()
        ds.batch = self.batch
        ds.settings = self.settings
        ds.name = new_name
        return ds

    def downsample(self, channel_names, target_depths, method="mean"):
        LOGGER.log(f"Downsampling ({method}) channels: {channel_names} to depths {target_depths}")
        ds = self.clone(self.name+"_downsampled")
        ds.save(force=True)
        channel_indexes = [ds.get_element_index(c) for c in channel_names] if type(channel_names) is list else [ds.get_element_index(channel_names)]
        target_depths = target_depths if type(target_depths) is list else [target_depths]
        for bi in range(len(ds.batch)):
            batch = ds.get(bi)
            for ci,i in enumerate(channel_indexes):
                img = batch[i]
                original_depth = img.shape[-1]
                target_depth = target_depths[ci]
                factor = original_depth // target_depth

                if original_depth % target_depth != 0:
                    LOGGER.warn(f"Warning: {original_depth} is not perfectly divisible by {target_depth}, possible data loss.")

                if method == "mean":
                    batch[i] = img.reshape(128, 128, target_depth, factor).mean(axis=-1)
                elif method == "max":
                    batch[i] = img.reshape(128, 128, target_depth, factor).max(axis=-1)
                else:
                    batch[i] = img[:, :, ::factor]
            ds.save_batch(batch,bi)
            del batch
        return ds

    def save_batch(self, batch, i):
        if not(os.path.exists(TRAINING_BATCH_FOLDER)):
            os.mkdir(TRAINING_BATCH_FOLDER)
        batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(self.name).split("batch_")[-1])
        if not(os.path.exists(batch_path)):
            os.mkdir(batch_path)
        order = self.settings["order"]
        for j,o in enumerate(order):
            np.save(os.path.join(batch_path,str(i)+"_"+o+".npy"), batch[j])

    def save_settings(self):
        if not(os.path.exists(TRAINING_BATCH_FOLDER)):
            os.mkdir(TRAINING_BATCH_FOLDER)
        batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(self.name).split("batch_")[-1])
        if not(os.path.exists(batch_path)):
            os.mkdir(batch_path)
        with open(os.path.join(batch_path,'settings.json'), 'w') as file:
            json.dump(self.settings, file, indent=4)
    
    def save(self,batch=None, name=None, force=False):

        batch = self.get() if batch is None else batch

        if not(os.path.exists(TRAINING_BATCH_FOLDER)):
            os.mkdir(TRAINING_BATCH_FOLDER)

        batch_uuid = self.name if name is None else name

        if os.path.exists(os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid).split("batch_")[-1])) and force:
            LOGGER.warn(f"Dataset {batch_uuid} already exists, but force save enabled so previous batch was removed.")
            shutil.rmtree(os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid).split("batch_")[-1]))

        while os.path.exists(os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid).split("batch_")[-1])):
            self.name = str(uuid.uuid4())
            LOGGER.warn(f"Dataset {batch_uuid} already exists, change to: {str(self.name)}")
            batch_uuid = self.name

        batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid).split("batch_")[-1])
        os.mkdir(batch_path)

        with open(os.path.join(batch_path,'settings.json'), 'w') as file:
            json.dump(self.settings, file, indent=4)

        order = self.settings["order"]
        for i,img in enumerate(batch):
            for j,o in enumerate(order):
                np.save(os.path.join(batch_path,str(i)+"_"+o+".npy"), img[j])

        LOGGER.log(f"Dataset with {len(batch)} images saved.")

        return True
    
    def plot_correlation(self, X_i=0, Y_i=1, ax=None, bins_number=256, show_yx = False):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        batch = self.get()
        #from scripts.COSpectrum import getIntegratedIntensity
        c1 = np.array([np.log(b[X_i])/np.log(10) for b in batch]).flatten()
        c2 = np.array([np.log(b[Y_i])/np.log(10) for b in batch]).flatten()

        nan_indices = np.isnan(c1) | np.isnan(c2)
        good_indices = ~nan_indices
        c1= c1[good_indices]
        c2 = c2[good_indices]

        _, _, _,hist = ax.hist2d(c1, c2, bins=(bins_number,bins_number), norm=LogNorm())
        if show_yx:
            yx = np.linspace(np.min(c1), np.max(c1), 10)
            plt.plot(yx,yx,linestyle="--",color="red",label=r"$y=x$")
            plt.legend()

        plt.colorbar(hist, ax=ax, label="counts")
        plt.legend()
        fig.tight_layout()

        return fig, ax