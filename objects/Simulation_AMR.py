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
from astropy.constants import m_p
from astropy import units as u
import matplotlib.pyplot as plt
# Create a slice plot of density
#slc = yt.SlicePlot(ds, "z", "density")
#slc.show()

class Simulation_ARM():
    def __init__(self, name, global_size, init=True):
        self.name = name
        """Simulatio name, name of the folder where the sim is in"""
        self.folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../data/sims/"+name+"/")
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
            density = f["scalars/density"][:] 
            size = f["scalars/size"][:]

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        parsec_to_cm = 3.0857e18*self.global_size*0.4
        mean_molecular_weight = 1.4
        number_density = density / (mean_molecular_weight * m_p.value)  # cm⁻³

        x, y, z = points[:, 0] * parsec_to_cm, points[:, 1] * parsec_to_cm, points[:, 2] * parsec_to_cm
        data = {
            "particle_position_x": (x, "cm"),
            "particle_position_y": (y, "cm"),
            "particle_position_z": (z, "cm"),
            "particle_density": (number_density, "cm**-3"),
            "particle_size": (size, "cm"),
            "particle_volume": (size**3, "cm**3"),
        }

        # Define domain bounding box (min/max values of coordinates)
        bbox = np.array([[x.min(), x.max()], [y.min(), y.max()], [z.min(), z.max()]])

        # Load as a particle dataset in yt
        self.data = yt.load_particles(data, bbox=bbox)
        self.data.add_deposited_particle_field(("deposit", "number_density"), "sum")
        

if __name__ == "__main__":
    sim = Simulation_ARM("orionMHD_lowB_0.4_AMR", 66.0948)
    prj = yt.ProjectionPlot(sim.data, "z", ("deposit", "number_density"), weight_field=None)
    prj.save(EXPORT_FOLDER+"particle_density_projection.png")