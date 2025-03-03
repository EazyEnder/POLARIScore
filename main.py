from astropy import units as u
import os
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
    return np.sum(np.pow(data_cube,2), axis=axis) / np.sum(data_cube, axis= axis)
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

column_density_xy = compute_volume_weighted_density(DATA, axis=0)  # Top-down
column_density_xz = compute_volume_weighted_density(DATA, axis=1)  # Side view
column_density_yz = compute_volume_weighted_density(DATA, axis=2)  # Front view

import scipy
def compute_smoothness(matrix):
    log_matrix = np.log1p(matrix)  # log1p(x) = log(1 + x), prevents log(0)

    laplacian = scipy.ndimage.laplace(log_matrix)
    raw_score = np.var(laplacian)

    normalized_score = raw_score

    return normalized_score

def generate_random_batch(data_cube,number=8,size=5,random_rotate=True,limit_area=([27,40,26,39],[26.4,40,22.5,44.3],[26.4,39,21,44.5])):
    column_density_xy = compute_column_density(data_cube, axis=0)
    column_density_xz = compute_column_density(data_cube, axis=1)
    column_density_yz = compute_column_density(data_cube, axis=2)
    volume_density_xy = compute_mass_weighted_density(data_cube, axis=0)
    volume_density_xz = compute_mass_weighted_density(data_cube, axis=1)
    volume_density_yz = compute_mass_weighted_density(data_cube, axis=2)

    imgs = []
    img_generated = 0
    areas_explored = [[],[],[]]
    iteration = 0
    while img_generated < number and iteration < number*10:
        iteration += 1
        if iteration >= number*10:
            print("Error: failed to generated random batches, nbr of img generated:"+str(len(imgs)))
            return imgs

        random = np.random.random()
        c_dens = column_density_xy
        v_dens = volume_density_xy
        face = 0
        if random < 1/3:
            c_dens = column_density_xz
            v_dens = volume_density_xz
            face = 1
        elif random < 2/3:
            c_dens = column_density_yz
            v_dens = volume_density_yz
            face = 2
        
        limits = limit_area[face]
        center = np.array([limits[0]+(limits[1]-limits[0])*np.random.random(),limits[2]+(limits[3]-limits[2])*np.random.random()])
        
        #Verify if the region is already covered by a previous generated image
        flag = False
        for point in areas_explored[face]:
            if np.linalg.norm(center-point) < 1. * size:
                flag = True
                break
        if flag:
            continue
        areas_explored[face].append(center)

        c_x, c_y = center
        c_x = convert_pc_to_index(c_x)-int(np.floor(SIM_axis[0][0]/SIM_size*SIM_nres))
        c_y = convert_pc_to_index(c_y)-int(np.floor(SIM_axis[0][0]/SIM_size*SIM_nres))
        s = convert_pc_to_index(size)
        start_x = c_x - s // 2
        start_y = c_y - s // 2
        end_x = c_x + s // 2
        end_y = c_y + s // 2

        cropped_cdens = c_dens[start_x:end_x, start_y:end_y]
        cropped_vdens = v_dens[start_x:end_x, start_y:end_y]

        # Randomly choose a rotation (0, 90, 180, or 270 degrees)
        rotated_cdens = cropped_cdens
        rotated_vdens = cropped_vdens
        if random_rotate:
            k = np.random.choice([0, 1, 2, 3])
            rotated_cdens = np.rot90(cropped_cdens, k)
            rotated_vdens = np.rot90(cropped_vdens, k)

        b = (rotated_cdens, rotated_vdens)
        imgs.append(b)
        img_generated += 1
    return imgs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_batch(batch):
    batch_nbr = len(batch)
    fig, axes = plt.subplots(2,batch_nbr)
    for i in range(batch_nbr):
        data1 = batch[i][0]
        data2 = batch[i][1]
        axes[0][i].set_title(str(np.round(compute_smoothness(data1),3)))
        d1 = axes[0][i].imshow(data1, cmap="jet", norm=LogNorm(vmin=np.min(data1), vmax=np.max(data1)))
        d2 = axes[1][i].imshow(data2, cmap="jet", norm=LogNorm(vmin=np.min(data2), vmax=np.max(data2)))
    fig.subplots_adjust( left=None, bottom=None,  right=None, top=None, wspace=None, hspace=None)
batch = generate_random_batch(DATA)
plot_batch(batch)

fig, axes = plt.subplots(2, 3, figsize=(9, 6))

def plot(column, data):
    cd = axes[0][column].imshow(data, extent=[SIM_axis[0][0], SIM_axis[0][1], SIM_axis[1][0],SIM_axis[1][1]], cmap="jet", norm=LogNorm(vmin=np.min(data), vmax=np.max(data)))
    plt.colorbar(cd,ax=axes[0][column], label="Column Density")
    pdf = compute_pdf(data)
    axes[1][column].plot([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0])
    axes[1][column].scatter([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0])
    axes[1][column].set_xlabel("s")
    axes[1][column].set_ylabel("p")
    axes[1][column].set_title("PDF")

# XY Projection (Top-down)
plot(0,column_density_xy)
axes[0][0].set_title("Top-Down View (XY Projection)")
axes[0][0].set_xlabel("X [pc]")
axes[0][0].set_ylabel("Y [pc]")

# XZ Projection (Side view)
plot(1,column_density_xz)

axes[0][1].set_title("Side View (XZ Projection)")
axes[0][1].set_xlabel("X [pc]")
axes[0][1].set_ylabel("Z [pc]")

# YZ Projection (Front view)
plot(2,column_density_yz)
axes[0][2].set_title("Front View (YZ Projection)")
axes[0][2].set_xlabel("Y [pc]")
axes[0][2].set_ylabel("Z [pc]")

plt.show()