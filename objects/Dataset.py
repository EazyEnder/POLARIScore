import uuid
import os
from config import *
import json
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

BATCH_CAN_CONTAINS = ["cdens","vdens","cospectra"]

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

    def load_from_name(self, name):
        batch, order = _open_batch(name)
        self.batch.extend(batch)
        self.settings["order"] = order

    def add(self,imgs_path):
        self.batch.append(imgs_path)
    
    def get(self, indexes = None):
        b_min = 0 
        b_max = -1
        if not(indexes is None):
            if not(type(indexes) is list):
                return self.load(np.array(self.batch)[indexes])
            elif len(indexes) < 2:
                return self.load(np.array(self.batch)[indexes])
            else:
                b_min = indexes[0]
                b_max = indexes[1]
        else:
            return self.load(np.array(self.batch))
        paths = np.array(self.batch)[b_min:b_max]
        batch = self.load(paths)
        return batch

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

    def split(self, batch, cutoff=0.7):
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
    
    def save(self,batch, name=None):      
        if not(os.path.exists(TRAINING_BATCH_FOLDER)):
            os.mkdir(TRAINING_BATCH_FOLDER)

        batch_uuid = self.name if name is None else name
        while os.path.exists(os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid))):
            self.name = str(uuid.uuid4())
            LOGGER.warn(f"Batch {batch_uuid} already exists, change to: {str(self.name)}")
            batch_uuid = self.name

        batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid))
        os.mkdir(batch_path)

        with open(os.path.join(batch_path,'settings.json'), 'w') as file:
            json.dump(self.settings, file, indent=4)

        order = self.settings["order"]
        for i,img in enumerate(batch):
            for j,o in enumerate(order):
                np.save(os.path.join(batch_path,str(i)+"_"+o+".npy"), img[j])

        LOGGER.log(f"batch with {len(batch)} images saved.")

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