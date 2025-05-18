import numpy as np

from scipy.ndimage import rotate
def convert_pc_to_index(pc,nres,size,start=0):
    return(int(np.floor((pc-start)/(size)*nres)))


def compute_column_density(data_cube,cell_size, axis=0):
    return np.sum(data_cube, axis=axis) * cell_size.value
def compute_volume_weighted_density(data_cube, axis=0, divide=False):
    if divide:
        data_cube = data_cube.astype(np.float32, copy=False)
        threshold = 300
        dense_mask = data_cube >= threshold
        diffuse_mask = ~dense_mask

        dense_count = np.sum(dense_mask, axis=axis)
        diffuse_count = np.sum(diffuse_mask, axis=axis)

        dense_mean = np.max(data_cube * dense_mask, axis=axis)
        diffuse_mean = np.max(data_cube * diffuse_mask, axis=axis)

        #dense_mean = np.divide(dense_sum, dense_count, out=np.zeros_like(dense_sum), where=dense_count != 0)
        dense_mean += 0.01
        #diffuse_mean = np.divide(diffuse_sum, diffuse_count, out=np.zeros_like(diffuse_sum), where=diffuse_count != 0)
        diffuse_mean += 0.01

        return [diffuse_mean, dense_mean]
    return np.sum(data_cube, axis=axis) / data_cube.shape[0]
def compute_mass_weighted_density(data_cube, axis=0):
    return np.sum(np.power(data_cube,2), axis=axis) / np.sum(data_cube, axis= axis)
def compute_squared_weighted_density(data_cube, axis=0):
    return np.sum(np.power(data_cube,2), axis=axis) / data_cube.shape[0]
def compute_max_density(data_cube, axis=0):
    return np.max(data_cube, axis=axis)
def compute_derivative(data_slice, order=1, axis=0):
    d = data_slice
    for o in range(order):
        d = np.gradient(d)[axis]
    return d

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
    hist = np.histogram(func(data_slice.flatten()[~np.isnan(data_slice.flatten())]),bins=bins, density=True)
    probabilities = hist[0]
    edges = hist[1]
    if(center):
        center_i = np.argmax(probabilities)
        edges = edges - (edges[center_i+1]+edges[center_i])/2
    return [probabilities, edges]

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """Print a progress bar"""
    if total == 0:
        total = 1
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

def movingAverage(l, n=5, return_std=False):
    l = np.asarray(l, dtype=float)
    cs = np.cumsum(l)
    cs[n:] = cs[n:] - cs[:-n]
    moving_avg = cs[n-1:] / n
    
    if return_std:
        stds = np.array([np.std(l[i:i+n], ddof=0) for i in range(len(l) - n + 1)])
        return moving_avg, stds
    
    return moving_avg

def movingMin(l, n=5, exclude_zeros=False):
    result = []
    for i in range(len(l) - n + 1):
        window = l[i:i+n]
        if exclude_zeros:
            flag = [False for _ in range(n)]
            for j,w in enumerate(window):
                if w < 1e-5:
                    flag[j] = True
            if True in flag and False in flag:
                result.extend(window)
                continue
        result.append(min(window))
    return np.array(result, dtype=object)

def applyBaseline(t,y,T,Y):

    last_t = 0
    last_y = [y[0]]

    coefs = []
    for i in range(len(y)):
        y1 = y[i]
        t1 = t[i]
        coefs.append((y1 - last_y[-1]) / (t1 - last_t))
        last_t = t1
        last_y.append(y1)
    coefs.append(0.)

    int_time = []
    for j in range(len(t)+1):
        t_left = 0
        if j > 0:
            t_left = t[j-1]
        t_right = T[-1]
        if j < len(t):
            t_right = t[j]
        int_time.append((t_left,t_right))
    t_edges = np.array([0] + list(t))
    indices = np.searchsorted(t_edges[1:], T, side='right')
    for i in range(len(T)):
        j = indices[i]
        tl = t_edges[j]
        Y[i] = Y[i] - (coefs[j] * (T[i] - tl) + last_y[j])

    return Y

import matplotlib.pyplot as plt
def plot_function(function, ax=None, res=100, lims=[0,1], **args):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    X = np.linspace(lims[0],lims[1],res)
    Y = function(X)

    ax.plot(X,Y,**args)
    ax.grid(True)
    
    return fig, ax


def plot_lines(x,y, ax, lines=[0,1,2], x_max=None, x_min=None, y_max=None, y_min=None):
    x_max = np.max(x) if x_max is None else x_max
    x_min = np.min(x) if x_min is None else x_min
    y_max = np.max(y) if y_max is None else y_max
    y_min = np.min(y) if y_min is None else y_min
    if not(lines is None):
        axisx_length = (x_max-x_min)
        axisy_length = (y_max-y_min)
        x_corner = axisx_length*0.7+x_min
        y_corner = axisy_length*0.1+y_min
        length = axisx_length*0.2  
        for l in lines:
            ax.plot([x_corner, x_corner + length],
                    [y_corner, y_corner + length*l],
                    '--', lw=1, color="black")
            if l != 0:
                ax.text(x_corner + length + length*0.1, y_corner + l*length, f'$x^{l}$', color='black')
    return ax

import platform
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False

try:
    import GPUtil
    gputil_available = True
except ImportError:
    gputil_available = False
from config import LOGGER
def get_system_info():
    system_info = {}
    if not(psutil_available) or not(psutil_available):
        LOGGER.warn("Can't get system config informations because GPUtil or psutil are not installed.")
        return system_info

    # CPU
    system_info['CPU'] = {
        'Processor': platform.processor(),
        'Physical Cores': psutil.cpu_count(logical=False),
        'Total Cores': psutil.cpu_count(logical=True),
        'Max Frequency (MHz)': psutil.cpu_freq().max,
        'Current Frequency (MHz)': psutil.cpu_freq().current,
    }

    # RAM
    svmem = psutil.virtual_memory()
    system_info['RAM'] = {
        'Total (GB)': round(svmem.total / (1024 ** 3), 2),
    }

    # GPU
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            'Name': gpu.name,
            'Memory Total (MB)': gpu.memoryTotal,
            'Driver Version': gpu.driver,
        })
    system_info['GPU'] = gpu_info if gpu_info else 'No GPU Found'

    # System
    system_info['System'] = {
        'System': platform.system(),
        'Node Name': platform.node(),
        'Release': platform.release(),
        'Version': platform.version(),
        'Machine': platform.machine(),
    }

    return system_info