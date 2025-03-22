import sys
import os
if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)
import yt
import h5py
from utils import *
from config import *
import json
from astropy import units as u
import matplotlib.pyplot as plt
# Create a slice plot of density
#slc = yt.SlicePlot(ds, "z", "density")
#slc.show()

class Simulation_ARM():
    def __init__(self, name, global_size, init=True):
        self.name = name
        """Simulatio name, name of the folder where the sim is in"""
        self.folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../sims/"+name+"/")
        """Path to the folder where the simulation is stored"""
        self.file = os.path.join(self.folder,"cellsToPoints_cgs.h5")
        """Path to the simulation data"""
        self.data = None
        """Raw simulation data"""
        self.global_size = global_size

        if init:
            self.init()

    def init(self):
        """
        Load files and data in self variables
        """

        with h5py.File(self.file, "r") as f:
            points = f["points"][:] 
            densities = f["scalars/density"][:] 
            sizes = f["scalars/size"][:]

        #hist, xedges, yedges = np.histogram2d(x, y, bins=1000, weights=density)
        #plt.imshow(hist.T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap="viridis", aspect="auto")


if __name__ == "__main__":
    sim = Simulation_ARM("orionMHD_lowB_0.4_AMR", 66.0948)
    print(sim.data)
    plt.show()