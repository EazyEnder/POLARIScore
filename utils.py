import numpy as np
from config import *

from scipy.ndimage import rotate
def convert_pc_to_index(pc,nres):
    return(int(np.floor((pc)/nres*nres)))
def compute_column_density(data_cube,cell_size, axis=0):
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
    """
    Compute the probability density function of a matrix

    Args:
        data_slice(np array): the matrix
        bins(int): How many bins
        func: function applied to the flatten data, like convert to log10 by default
        center(bool, default:False): Center the bins and pdf to 0
    
    Returns:
        probabilities, edges
    """
    hist = np.histogram(func(data_slice.flatten()),bins=bins, density=True)
    probabilities = hist[0]
    edges = hist[1]
    if(center):
        center_i = np.argmax(probabilities)
        edges = edges - (edges[center_i+1]+edges[center_i])/2
    return [probabilities, edges]

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """Print a progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

def divide_matrix_to_sub(matrix,final_dim=128):
    final_dim = int(2**round(np.log2(final_dim)))
    img_number = int(matrix.shape[0]/final_dim)
    imgs = []
    for i in range(img_number):
        for j in range(img_number):
            imgs.append(matrix[i*final_dim:(i+1)*final_dim,j*final_dim:(j+1)*final_dim])
    return imgs

def group_matrix(mats):
    grid_size = int(np.sqrt(len(mats)))
    final_dim = len(mats[0])
    new_mat_shape = final_dim * grid_size
    result = np.zeros((new_mat_shape, new_mat_shape))
    for idx, mat in enumerate(mats):
        row_idx = idx // grid_size
        col_idx = idx % grid_size
        result[row_idx * final_dim: (row_idx + 1) * final_dim,
               col_idx * final_dim: (col_idx + 1) * final_dim] = np.array(mat)
    return result