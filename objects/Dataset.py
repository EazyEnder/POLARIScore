import uuid
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config import *
from utils import plot_lines, listDictToString
import json
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import shutil
from matplotlib.widgets import Slider

BATCH_CAN_CONTAINS = ["cdens","vdens","cospectra","density","vdensdiffuse","vdensdense"]
AUGMENTS = {
    "sum": lambda x: np.sum(x, dim=-1)
}

def _formate_name(name):
    return name.split("_")[-1]

def _open_batch(batch_name):
    assert os.path.exists(TRAINING_BATCH_FOLDER), LOGGER.error(f"Can't open batch {batch_name}, no folder exists.")
    batch_path = os.path.join(TRAINING_BATCH_FOLDER,batch_name)

    files = glob.glob(batch_path+"/*.npy")
    files = [f.split("/")[-1] for f in files]

    imgs = [[] for _ in range(len(np.unique([int(f.split("_")[0]) for f in files])))]
    order = []
    for bc in BATCH_CAN_CONTAINS:
        pot_files = [f for f in files if bc == f.split(".")[0].split("_")[-1]]

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

    def get_element_index(self, names):
        assert "order" in self.settings, LOGGER.error("No order list in dataset settings")

        names = names if type(names) is list else [names]

        indexes = []
        for n in names:
            found = False
            for i,o in enumerate(self.settings["order"]):
                if o == _formate_name(n):
                    indexes.append(i)
                    found = True
                    break
            assert found, LOGGER.error(f"Index not found for {n}")

        if len(indexes) == 1:
            return indexes[0]
        if len(indexes) == 0:
            return None
        
        return indexes

    def load_from_name(self, name, changeName=False):
        LOGGER.log(f"Loading dataset {name}")
        if changeName:
            self.name = name
        batch, order = _open_batch(name)
        self.batch.extend(batch)
        self.settings["order"] = order

        settings = {}
        with open(os.path.join( os.path.join(TRAINING_BATCH_FOLDER,name),'settings.json')) as file:
            settings = json.load(file)

        if "areas_explored" in settings:
            self.settings["areas_explored"] = eval(settings["areas_explored"].replace('array', 'np.array'))
        if "img_size" in settings:
            self.settings["img_size"] = settings["img_size"]

    def add(self,imgs_path):
        self.batch.append(imgs_path)
    
    def get(self, indexes = None):
        if len(self.batch) == 0:
            LOGGER.error("Can't load images in dataset because it's empty.")
            return
        if not(indexes is None):
            if not(type(indexes) is list):
                return self.load(np.array(self.batch)[indexes])
            elif len(indexes) < 2:
                return self.load(np.array(self.batch)[indexes[0]])
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

    def downsample(self, channel_names, target_depths, methods="mean"):
        LOGGER.log(f"Downsampling ({methods}) channels: {channel_names} to depths {target_depths}")
        ds = self.clone(self.name+"_downsampled")
        ds.save(force=True)
        channel_indexes = [ds.get_element_index(c) for c in channel_names] if type(channel_names) is list else [ds.get_element_index(channel_names)]
        target_depths = target_depths if type(target_depths) is list else [target_depths]
        methods = methods if type(methods) is list else [methods]
        for bi in range(len(ds.batch)):
            batch = ds.get(bi)
            for ci,i in enumerate(channel_indexes):
                img = batch[i]
                original_depth = img.shape[-1]
                target_depth = target_depths[ci]
                factor = original_depth // target_depth

                if original_depth % target_depth != 0:
                    LOGGER.warn(f"Warning: {original_depth} is not perfectly divisible by {target_depth}, possible data loss.")

                method = methods[ci]
                if method == "mean":
                    batch[i] = img.reshape(128, 128, target_depth, factor).mean(axis=-1)
                elif method == "max":
                    batch[i] = img.reshape(128, 128, target_depth, factor).max(axis=-1)
                elif method == "crop":
                    assert target_depth % 2 == 0, LOGGER.error(f"Target depth {target_depth} need to be even if method 'crop' is used for downsampling.")
                    batch[i] = img[:, :, img.shape[-1]//2-target_depth//2:img.shape[-1]//2+target_depth//2]
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
    
    def save_diagnostic(self):
        batch = self.get()
        map_index = self.get_element_index("cdens")
        result_dicts = []
        for i,b in enumerate(batch):
            data = np.array(b[map_index]).flatten()
        
            result_dicts.append({
                "index": i,
                "mean": np.mean(data),
                "std_log10": np.std(np.log10(data)),
                "min": np.min(data),
                "max": np.max(data),
                "median": np.median(data)
            })

        string = listDictToString(result_dicts)
        if not(os.path.exists(TRAINING_BATCH_FOLDER)):
            os.mkdir(TRAINING_BATCH_FOLDER)
        batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(self.name).split("batch_")[-1])
        if not(os.path.exists(batch_path)):
            os.mkdir(batch_path)
        path = os.path.join(batch_path, "diagnostic.txt")
        if os.path.exists(path):
            LOGGER.warn(f"Previous diagnostic file was removed for dataset {self.name}.")
            os.remove(path)
        with open(path, "w") as file:
            file.write(string)
        LOGGER.log(f"Diagnostic of {self.name} saved.")

        return result_dicts

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

    def plot(self, enable_slider=True, element_index=0):

        element_index = element_index
        
        fig = plt.figure(figsize=(8,8))
        fig.suptitle("Dataset "+self.name+" "+str(element_index+1))
        
        ax1 = plt.subplot(2,2,1)
        ax2 = plt.subplot(2,2,2)
        ax3 = plt.subplot(2,2,3)
        ax4 = plt.subplot(2,2,4)

        axes_histo = None
        axes_map = None
        axes_map2 = None

        def update_element_index(val):
            nonlocal axes_histo
            nonlocal axes_map
            nonlocal axes_map2
            element_index = int(val)
            ax1.clear()
            ax1.set_visible(False)
            if axes_map is not None:
                for a in axes_map:
                    a.remove()
            axes_map = None
            if element_index > -1:
                _, axes_map = self.plot_map(ax=ax1, element_index=element_index, enable_slider=False)
            ax2.clear()
            self.plot_correlation(ax=ax2, element_index=element_index, PDF=True)#, contour_levels=[0.38,0.69, 0.95]
            ax3.clear()
            ax3.set_visible(False)
            if axes_map2 is not None:
                for a in axes_map2:
                    a.remove()
            axes_map2 = None
            if element_index > -1:
                _, axes_map2 = self.plot_map(ax=ax3, element_index=element_index, map_index= 1, enable_slider=False)
            if axes_histo is not None:
                for a in axes_histo:
                    a.remove()
            _, axes_histo = self.plot_histo(ax=ax4, element_index=element_index, enable_slider=False)
            fig.canvas.draw_idle()

        update_element_index(element_index)

        if enable_slider:
            ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
            slider = Slider(ax_slider, 'Element index', -1, len(self.batch) - 1, valinit=element_index, valfmt='%0i')
            slider.on_changed(update_element_index)  
            plt.show()

        return fig

    def plot_histo(self, ax=None, element_index=-1, map_index=1, method=np.log10, enable_slider=True, lims=[1,8]):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        bbox = ax.get_position()
        width = bbox.width
        height = bbox.height
        left = bbox.x0
        bottom = bbox.y0
        ax.set_visible(False)

        ax_histo = fig.add_axes([left, bottom+0.1, width, height-0.1])
        histo_bins = 20

        batch = self.get(indexes=element_index if element_index > -1 else None)
        if element_index > -1:
            batch = [batch]

        def update_map_index(val):
            map_index = int(val)
            ax_histo.clear()
            data = np.array([method(b[map_index]) for b in batch]).flatten()
            counts, bin_edges = np.histogram(data, bins=histo_bins, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            ax_histo.hist(data, bins=histo_bins, alpha=1.0, label=self.settings["order"][map_index], density=True)
            ax_histo.plot(bin_centers, counts, color='black', linestyle='-')

            if lims is not None:
                ax_histo.set_xlim((lims[0],lims[1]))

            fig.canvas.draw_idle()
            ax_histo.set_yscale('log')
            fig.canvas.draw_idle()
            ax_histo.legend()

        update_map_index(map_index)

        if enable_slider:
            ax_slider = fig.add_axes([left, bottom, width, 0.03], zorder=10)
            slider = Slider(ax_slider, 'i', 0, len(self.settings['order']) - 1, valinit=map_index, valfmt='%0i')
            slider.on_changed(update_map_index) 

            return fig, [ax_histo, ax_slider] 

        return fig, [ax_histo]

    def plot_map(self, ax=None, element_index=0, map_index=0, enable_slider=True, show_title=True):        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if element_index < 0:
            element_index = 0

        bbox = ax.get_position()
        width = bbox.width
        height = bbox.height
        left = bbox.x0
        bottom = bbox.y0
        ax.set_visible(False)
        ax_map = fig.add_axes([left, bottom+0.1, width, height-0.1])

        batch = self.get(indexes=element_index if element_index > -1 else None)
        im = ax_map.imshow(batch[map_index] if len(batch[map_index].shape) <= 2 else np.sum(batch[map_index], axis=-1), norm=LogNorm(), cmap="jet",
                           extent=[val for ae in self.settings["areas_explored"][0][map_index] for val in (ae - self.settings["img_size"], ae + self.settings["img_size"])] if "areas_explored" in self.settings else None)

        if show_title:
            ax_map.set_title(self.settings['order'][map_index])

        if("areas_explored" in self.settings):
            ax_map.set_xlabel("[pc]")
            ax_map.set_ylabel("[pc]")
        plt.colorbar(im, label=self.settings['order'][map_index])

        if enable_slider:
            ax_slider = fig.add_axes([left, bottom-0.03, width, 0.03])
            slider = Slider(ax_slider, 'i', 0, len(self.settings['order']) - 1, valinit=map_index, valfmt='%0i')

            def update_map_index(val):
                map_index = int(val)
                im.set_data(batch[map_index] if len(batch[map_index].shape) <= 2 else np.sum(batch[map_index], axis=-1))
                im.set_norm(LogNorm())
                im.set_extent([val for ae in self.settings["areas_explored"][0][map_index] for val in (ae - self.settings["img_size"], ae + self.settings["img_size"])] if "areas_explored" in self.settings else None)
                if show_title:
                    ax_map.set_title(self.settings['order'][map_index])
                plt.colorbar(im, label=self.settings['order'][map_index])
                fig.canvas.draw_idle()

            slider.on_changed(update_map_index)    

            return fig, [ax_map, ax_slider] 

        return fig, [ax_map]

    def plot_correlation(self, X_i=0, Y_i=1, ax=None, bins_number=256, show_yx = False, method=np.log10, contour_levels=0, PDF=False, lines=[0,1,2], element_index=-1):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        batch = self.get(indexes=element_index if element_index > -1 else None)
        if element_index > -1:
            batch = [batch]
        c1 = np.array([method(b[X_i]) for b in batch]).flatten()
        c2 = np.array([method(b[Y_i]) for b in batch]).flatten()

        ax.set_xlabel(self.settings["order"][X_i])
        ax.set_ylabel(self.settings["order"][Y_i])


        nan_indices = np.isnan(c1) | np.isnan(c2)
        good_indices = ~nan_indices
        c1= c1[good_indices]
        c2 = c2[good_indices]

        if type(contour_levels) is list or contour_levels > 1:
            hist, xedges, yedges = np.histogram2d(c1, c2, bins=(bins_number, bins_number), density=PDF)
            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])
            X, Y = np.meshgrid(xcenters, ycenters)

            if PDF and type(contour_levels) is list:
                hist_flat = hist.flatten()
                idx = np.argsort(hist_flat)[::-1]
                hist_sorted = hist_flat[idx]
                cumsum = np.cumsum(hist_sorted)
                cumsum /= cumsum[-1]

                level_values = []
                for cl in contour_levels:
                    try:
                        i = np.where(cumsum >= cl)[0][0]
                        level_values.append(hist_sorted[i])
                    except IndexError:
                        level_values.append(hist_sorted[-1])
                level_prob_map = dict(zip(level_values, contour_levels))

                contour = ax.contour(X, Y, hist.T, levels=sorted(level_values), colors="black")
                ax.clabel(contour, fmt=lambda x: f"{level_prob_map.get(x, x):.2f}", inline=True, fontsize=8)
            else:
                contour = ax.contour(X, Y, hist.T, levels=contour_levels if type(contour_levels) is list else int(contour_levels), norm=LogNorm(), colors="black")
                ax.clabel(contour, fmt=lambda x: r"$10^{{{:.0f}}}$".format(np.log10(x)) if not(PDF) else r"${:.2f}$".format(x), inline=True, fontsize=8)

        _, _, _,hist = ax.hist2d(c1, c2, bins=(bins_number,bins_number), norm=LogNorm(), density=PDF)
        
        x_min, x_max = np.min(c1), np.max(c1)
        y_min, y_max = np.min(c2), np.max(c2)
        if type(contour_levels) is list or contour_levels > 1:
            if contour.collections:
                outer_contour = contour.collections[0]
                all_paths = outer_contour.get_paths()

                if all_paths:
                    all_vertices = np.concatenate([p.vertices for p in all_paths])
                    x_min, x_max = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
                    y_min, y_max = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])

                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)

        if type(lines) is list and len(lines) > 0:
            plot_lines(None, None, ax, lines=lines, x_max=x_max, x_min=x_min, y_max=y_max, y_min=y_min)

        
        #plt.colorbar(hist, ax=ax, label="PDF" if PDF else "counts")
        #ax.grid(True)
        ax.set_axisbelow(True)
        
        
        if show_yx:
            yx = np.linspace(np.min([c1.min(), c2.min()]), np.max([c1.max(), c2.max()]), 10)
            plt.plot(yx,yx,linestyle="--",color="red",label=r"$y=x$")

        return fig, ax

if __name__ == "__main__":

    #from objects.Simulation_DC import Simulation_DC
    #sim = Simulation_DC(name="orionMHD_lowB_0.39_512", global_size=66.0948, init=False)
    #sim.init(loadTemp=True, loadVel=True)
    #sim.plot(axis=1)

    ds = getDataset("batch_orionMHD_lowB_0.39_512_13CO_mass")
    ds.plot_map(map_index=0, element_index=4, enable_slider=0, show_title=False)
    #fig, ax = ds.plot_correlation(PDF=True, contour_levels=[0.38,0.69,0.95])
    #ds.plot_correlation(PDF=True, contour_levels=[0.38,0.69,0.95])
    plt.show()