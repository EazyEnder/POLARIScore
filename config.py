import os
from astropy.io import fits
import numpy as np
from astropy import units as u

"""
This file contains all sim variables: header and data
"""

SIM_NAME = "orionMHD_lowB_0.39_512"
#SIM size in parsec
SIM_SIZE = 66.0948
SIM_DATA_NAME = "datacube.fits"
SIM_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),"sims/"+SIM_NAME+"/")
SIM_FILE = os.path.join(SIM_FOLDER,SIM_DATA_NAME)

EXPORT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),"export/")

TRAINING_BATCH_FOLDER = os.path.join(EXPORT_FOLDER,"training_batchs")

file = fits.open(SIM_FILE)
DATA = file[0].data
file.close()

import json
with open(os.path.join(SIM_FOLDER,"processing_config.json"), "r") as file:
    SIM_header = json.load(file)

SIM_nres = SIM_header["run_parameters"]["nres"]
SIM_relative_size = SIM_header["run_parameters"]["size"]
SIM_center = np.array([SIM_header["run_parameters"]["xcenter"],SIM_header["run_parameters"]["ycenter"],SIM_header["run_parameters"]["zcenter"]])

SIM_cell_size = (SIM_SIZE*SIM_relative_size/SIM_nres) * u.parsec
SIM_cell_size = SIM_cell_size.to(u.cm)

SIM_size = SIM_SIZE*SIM_relative_size
SIM_axis = ([SIM_center[0]*SIM_SIZE-SIM_size/2,SIM_center[0]*SIM_SIZE+SIM_size/2],[SIM_center[1]*SIM_SIZE-SIM_size/2,SIM_center[1]*SIM_SIZE+SIM_size/2],[SIM_center[2]*SIM_SIZE-SIM_size/2,SIM_center[2]*SIM_SIZE+SIM_size/2])

RANDOM_BATCH_SCORE_offset = 1.
RANDOM_BATCH_SCORE_fct = lambda x: 1./(1+np.exp(-2.*(x-RANDOM_BATCH_SCORE_offset)))