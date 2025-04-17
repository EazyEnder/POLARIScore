import os
import sys
if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)
from config import CACHES_FOLDER, LOGGER
import numpy as np
from physics_utils import *
import uuid
import matplotlib.pyplot as plt

class Spectrum():
    """
    Object for spectra, this can also contains map of spectrum but a lot of functions will not work, use SpectrumMap instead. 
    """
    def __init__(self,spectrum, name=None):
        self.name = "spectrum_"+str(uuid.uuid4()) if name is None else name
        if not("spectrum" in name):
            name = "spectrum_"+name
        self.spectrum = spectrum

    def save(self,folder=None, replace=False, log=True):
        folder = CACHES_FOLDER if folder is None else folder
        #TODO maybe bad idea
        if not(os.path.exists(folder)):
            os.mkdir(folder)
        path = os.path.join(folder,self.name+".npy")
        if os.path.exists(path):
            if not(replace):
                LOGGER.error(f"Can't save spectrum {self.name} because there is already a spectrum called this way in the folder and replace is set to False")
                return
            os.remove(path)
        if log:
            LOGGER.log(f"Spectrum {self.name} saved")
        np.save(path,self.spectrum)

    def plot(self, ax=None, channels=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        channels = np.arange(len(self.spectrum)) if channels is None else channels
        ax.plot(channels,self.spectrum)
        return fig, ax

def loadSpectrum(name, folder=None, absolute_path=None):
    folder = CACHES_FOLDER if folder is None else folder
    path = os.path.join(folder,name.split(".npy")[0]+".npy")
    if not(absolute_path is None):
        path = absolute_path
    if not(os.path.exists(path)):
        LOGGER.error(f"Can't load spectrum because the file is not found: {path}")
        return 
    return np.load(path)