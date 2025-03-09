import os
from astropy.io import fits
import numpy as np
from astropy import units as u

"""
This file contains all utils variables, like for simulation, plots...
"""

SIM_NAME = "orionMHD_lowB_0.39_512"
"""Simulatio name, name of the folder where the sim is in"""

SIM_SIZE = 66.0948
"""Real spatial size of the global simulation in parsec"""
SIM_DATA_NAME = "datacube.fits"
"""Name of the file where the simulation data is stored"""
SIM_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),"sims/"+SIM_NAME+"/")
"""Path to the folder where the simulation is stored"""
SIM_FILE = os.path.join(SIM_FOLDER,SIM_DATA_NAME)
"""Path to the simulation data"""

EXPORT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),"export/")
"""Where all the objects saves are stored (ex: models, training_batchs)"""

TRAINING_BATCH_FOLDER = os.path.join(EXPORT_FOLDER,"training_batchs")
"""Path to the training batchs"""
MODEL_FOLDER = os.path.join(EXPORT_FOLDER,"models")
"""Path to the models folder"""

#Load the sim data
simfile = fits.open(SIM_FILE)
DATA = simfile[0].data
"""Raw simulation data"""
simfile.close()

#Load the sim settings
import json
with open(os.path.join(SIM_FOLDER,"processing_config.json"), "r") as file:
    SIM_header = json.load(file)
    """Dict of sim settings"""

SIM_nres = SIM_header["run_parameters"]["nres"]
"""Resolution of the simulation (pixels*pixels), i.e shape of the matrix"""
SIM_relative_size = SIM_header["run_parameters"]["size"]
"""Relative size of the simulation to the global simulation"""
SIM_center = np.array([SIM_header["run_parameters"]["xcenter"],SIM_header["run_parameters"]["ycenter"],SIM_header["run_parameters"]["zcenter"]])
"""Center of the simulation to the global simulation"""

SIM_cell_size = (SIM_SIZE*SIM_relative_size/SIM_nres) * u.parsec
"""Simulation cell size in cm"""
SIM_cell_size = SIM_cell_size.to(u.cm)

SIM_size = SIM_SIZE*SIM_relative_size
"""Real spatial size of the simulation in parsec"""
SIM_axis = ([SIM_center[0]*SIM_SIZE-SIM_size/2,SIM_center[0]*SIM_SIZE+SIM_size/2],[SIM_center[1]*SIM_SIZE-SIM_size/2,SIM_center[1]*SIM_SIZE+SIM_size/2],[SIM_center[2]*SIM_SIZE-SIM_size/2,SIM_center[2]*SIM_SIZE+SIM_size/2])
"""Simulation area in parsec"""

RANDOM_BATCH_SCORE_offset = 1.
RANDOM_BATCH_SCORE_fct = lambda x: 1./(1+np.exp(-2.*(x-RANDOM_BATCH_SCORE_offset)))
"""To generate batch, we use a score that'll go through this function. If a random number between 0. and 1. is lower that this function, then the generate training image is keeped. By default this is a sigmoid"""

import matplotlib.cm as cm
FIGURE_CMAP = cm.Dark2
FIGURE_CMAP_MIN = 0.
FIGURE_CMAP_MAX = 1.0