import os

from sympy import limit_seq
from config import LOGGER, TRAINING_BATCH_FOLDER, FIGURE_FOLDER
import inspect
import uuid
import json
import numpy as np
import glob
from utils import *

import scipy
def compute_smoothness(matrix):
    log_matrix = np.log1p(matrix)  # log1p(x) = log(1 + x), prevents log(0)
    laplacian = scipy.ndimage.laplace(log_matrix-np.min(log_matrix))
    raw_score = np.var(laplacian)
    return raw_score

def compute_img_score(cdens,vdens):
    score = 0
    sm1 = compute_smoothness(cdens)
    sm2 = compute_smoothness(vdens)*0.5
    score = sm1+sm2
    diff_matrix = (cdens-np.min(cdens))/(np.max(cdens)-np.min(cdens))-(vdens-np.min(vdens))/(np.max(vdens)-np.min(vdens))
    sr1 = np.var(diff_matrix.flatten())*5
    score += sr1
    return (score,(sm1,sm2,sr1))

def rebuild_batch(cdens, vdens):
    batch = []
    for i in range(len(cdens)):
        batch.append((cdens[i], vdens[i]))
    return batch

def mix_batch(batch1, batch2, settings1=None, settings2=None,randomized=True):
    batch = batch1
    batch.extend(batch2)
    r_imgs = []
    for r_id in np.random.permutation(len(batch)):
        r_imgs.append(batch[r_id])
    batch = r_imgs
    if settings1 is None or settings2 is None:
        return batch, settings1
    #mix the 2 settings dict
    settings = None
    return batch, settings

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_batch(batch, b_name="",same_limits=True, number_per_row = 8):
    batch_nbr = len(batch)
    fig, axes = plt.subplots(int(2*np.ceil(batch_nbr/number_per_row)),number_per_row)
    if number_per_row==1:
        axes = [[axes[0]],[axes[1]]]
    fig.suptitle(b_name)
    for i in range(batch_nbr):
        data1 = batch[i][0]
        data2 = batch[i][1]
        #score = 0.
        #axes[2*(i//8)][i%8].set_title(str(np.round(score[0],3)))
        min_dat1 = np.min(data1)
        max_dat1 = np.max(data1) 
        d1 = axes[2*(i//number_per_row)][i%number_per_row].imshow(data1, cmap="jet", norm=LogNorm(vmin=np.min(data1), vmax=np.max(data1)))
        d2 = axes[2*(i//number_per_row)+1][i%number_per_row].imshow(data2, cmap="jet", norm=(LogNorm(vmin=np.min(data2), vmax=np.max(data2)) if not(same_limits) else LogNorm(min_dat1, max_dat1)))
    fig.subplots_adjust( left=None, bottom=None,  right=None, top=None, wspace=None, hspace=None)

    return fig, axes

def plot_batch_correlation(batch, ax=None, bins_number=256, show_yx = True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    column_density = np.array([np.log(b[0])/np.log(10) for b in batch]).flatten()
    volume_density = np.array([np.log(b[1])/np.log(10) for b in batch]).flatten()

    nan_indices = np.isnan(column_density) | np.isnan(volume_density)
    good_indices = ~nan_indices
    column_density= column_density[good_indices]
    volume_density = volume_density[good_indices]

    _, _, _,hist = ax.hist2d(column_density, volume_density, bins=(bins_number,bins_number), norm=LogNorm())
    if show_yx:
        yx = np.linspace(np.min(column_density), np.max(column_density), 10)
        plt.plot(yx,yx,linestyle="--",color="red",label=r"$y=x$")
        plt.legend()

    plt.colorbar(hist, ax=ax, label="counts")
    plt.legend()
    fig.tight_layout()

    return fig, ax

def split_batch(batch, cutoff=0.7):
    batch = np.array(batch)
    cut_index = int(cutoff * len(batch))
    return (batch[:cut_index],batch[cut_index:])

def open_batch(batch_name, return_path=False):
    if not(os.path.exists(TRAINING_BATCH_FOLDER)):
        return
    batch_path = os.path.join(TRAINING_BATCH_FOLDER,batch_name)

    imgs = []

    files = glob.glob(batch_path+"/*.npy")
    files = [f.split("/")[-1] for f in files]
    ids = [int(f.split("_")[0]) for f in files]
    indexes = np.argsort(ids)
    
    check_ids = []
    for j,i in enumerate(indexes):
        if ids[i] in check_ids:
            continue
        file1 = files[i]
        file2 = files[indexes[j+1]]
        if("cdens" in file1):
            file_c = file1
            file_v = file2
        else:
            file_c = file2
            file_v = file1
        check_ids.append(ids[i])
        if return_path:
            imgs.append(((os.path.join(batch_path,file_c)),(os.path.join(batch_path,file_v))))
        else:
            imgs.append((np.load(os.path.join(batch_path,file_c)),np.load(os.path.join(batch_path,file_v))))
    return imgs
    

if __name__ == "__main__":
    from objects.Simulation_DC import *

    sim_MHD = Simulation_DC(name="orionMHD_lowB_0.39_512", global_size=66.0948, init=False)
    sim_MHD.init(loadTemp=True, loadVel=True)
    sim_MHD.generate_batch(number=64, force_size=128, what_to_compute={"cospectra": True, "density":True})

    from objects.Dataset import getDataset
    ds = getDataset("batch_orionMHD_lowB_0.39_512")
    ds = ds.downsample(channel_names=["cospectra","density"], target_depths=[128,128], methods=["crop","mean"])
    #ds.load_from_name("batch_orionMHD_lowB_0.39_512")
    
    """
    print(ds.settings["order"])
    imgs = ds.get(12)
    density = imgs[-1]
    print(density.shape)

    import plotly.figure_factory as ff
    from skimage import measure

    log_data = np.log(sim_MHD.data) / np.log(10)
    binary_mask = log_data > 3

    # Extract isosurface at a chosen density threshold
    verts, faces, _, _ = measure.marching_cubes(binary_mask.astype(float), level=0.5)

    # Create 3D mesh
    fig = ff.create_trisurf(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            simplices=faces, colormap="Viridis")

    fig.show(renderer="iframe")"
    """
    

    plt.show()