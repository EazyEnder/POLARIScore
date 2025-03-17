import os
import numpy as np

"""
This file contains all utils variables, like for simulation, plots...
"""
SIM_DATA_NAME = "datacube.fits"
"""Name of the file where the simulation data is stored"""

EXPORT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),"export/")
"""Where all the objects saves are stored (ex: models, training_batchs)"""

TRAINING_BATCH_FOLDER = os.path.join(EXPORT_FOLDER,"training_batchs")
"""Path to the training batchs"""
MODEL_FOLDER = os.path.join(EXPORT_FOLDER,"models")
"""Path to the models folder"""

RANDOM_BATCH_SCORE_offset = 1.
RANDOM_BATCH_SCORE_fct = lambda x: 1./(1+np.exp(-2.*(x-RANDOM_BATCH_SCORE_offset)))
"""To generate batch, we use a score that'll go through this function. If a random number between 0. and 1. is lower that this function, then the generate training image is keeped. By default this is a sigmoid"""

import matplotlib.cm as cm
FIGURE_CMAP = cm.Dark2
FIGURE_CMAP_MIN = 0.
FIGURE_CMAP_MAX = 1.0

from Logger import Logger
LOGGER = Logger(level=2, auto_save=0)

FIGURE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),"paper/figure/")