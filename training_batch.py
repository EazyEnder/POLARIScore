import os

from sympy import limit_seq
from config import LOGGER, TRAINING_BATCH_FOLDER, FIGURE_FOLDER
import inspect
import uuid
import json
import numpy as np
import glob
from networks.nn_UNet import UNet
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

def split_batch(batch, cutoff=0.7):
    batch = np.array(batch)
    cut_index = int(cutoff * len(batch))
    return (batch[:cut_index],batch[cut_index:])

def check_if_batch_exists(settings):
    """todo"""
    return False

def save_batch(batch, settings, name=None):
    if check_if_batch_exists(settings):
        return False
    
    if not(os.path.exists(TRAINING_BATCH_FOLDER)):
        os.mkdir(TRAINING_BATCH_FOLDER)

    batch_uuid = uuid.uuid4() if name is None else name
    while os.path.exists(os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid))):
        batch_uuid = uuid.uuid4()

    batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid))
    os.mkdir(batch_path)

    with open(os.path.join(batch_path,'settings.json'), 'w') as file:
        json.dump(settings, file, indent=4)

    for i,img in enumerate(batch):
        cdens = img[0]
        vdens = img[1]
        np.save(os.path.join(batch_path,str(i)+"_cdens.npy"), cdens)
        np.save(os.path.join(batch_path,str(i)+"_vdens.npy"), vdens)

    LOGGER.log(f"batch with {len(batch)} images saved.")

    return True

def rebuild_batch(cdens, vdens):
    batch = []
    for i in range(len(cdens)):
        batch.append((cdens[i], vdens[i]))
    return batch

def open_batch(batch_name):
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
        imgs.append((np.load(os.path.join(batch_path,file_c)),np.load(os.path.join(batch_path,file_v))))
    return imgs

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
    _, _, _,hist = ax.hist2d(column_density, volume_density, bins=(bins_number,bins_number), norm=LogNorm())
    if show_yx:
        yx = np.linspace(np.min(column_density), np.max(column_density), 10)
        plt.plot(yx,yx,linestyle="--",color="red",label=r"$y=x$")
        plt.legend()

    plt.colorbar(hist, ax=ax)
    fig.tight_layout()

    return fig, ax
    

if __name__ == "__main__":
    from objects.Simulation_DC import Simulation_DC

    sim_MHD = Simulation_DC(name="orionMHD_lowB_0.39_512", global_size=66.0948)
    sim_HD = Simulation_DC(name="orionHD_all_512", global_size=66.0948)
    
    bHD, settingsHD = sim_HD.generate_batch(number=64, force_size=128, limit_area=[None,None,None])
    bMHD, settingsMHD = sim_MHD.generate_batch(number=64, force_size=128)
    final_b, _ = mix_batch(bHD,bMHD)
    save_batch(final_b, settingsMHD, name="mixt")
    plot_batch(final_b, same_limits=False)
    plot_batch_correlation(final_b)

    plt.show()