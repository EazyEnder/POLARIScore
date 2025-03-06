import numpy as np
from config import *

from scipy.ndimage import rotate
def convert_pc_to_index(pc):
    return(int(np.floor((pc)/SIM_size*SIM_nres)))
def compute_column_density(data_cube, axis=0, cell_size=SIM_cell_size):
    return np.sum(data_cube, axis=axis) * cell_size.value
def compute_volume_weighted_density(data_cube, axis=0):
    return np.sum(data_cube, axis=axis) / data_cube.shape[0]
def compute_mass_weighted_density(data_cube, axis=0):
    return np.sum(np.power(data_cube,2), axis=axis) / np.sum(data_cube, axis= axis)
def compute_mean_density(data_cube, axis=0):
    return np.mean(data_cube, axis=axis)
def compute_max_density(data_cube, axis=0):
    return np.max(data_cube, axis=axis)
def rotate_cube(data_cube, angle, axis):
    """Rotates the cube around a given axis (0=X, 1=Y, 2=Z) by a given angle in degrees."""
    return rotate(data_cube, angle, axes=axis, reshape=False, mode="nearest")
def compute_pdf(data_slice, bins=100, func=lambda x: np.log(x)/np.log(10), center=False):
    hist = np.histogram(func(data_slice.flatten()),bins=bins, density=True)
    probabilities = hist[0]
    edges = hist[1]
    if(center):
        center_i = np.argmax(probabilities)
        edges = edges - (edges[center_i+1]+edges[center_i])/2
    return [probabilities, edges]