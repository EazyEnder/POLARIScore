import os
from config import *
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

def generate_batch(data_cube,method=compute_mass_weighted_density,number=8,size=5,random_rotate=True,limit_area=([27,40,26,39],[26.4,40,22.5,44.3],[26.4,39,21,44.5])):
    column_density_xy = compute_column_density(data_cube, axis=0)
    column_density_xz = compute_column_density(data_cube, axis=1)
    column_density_yz = compute_column_density(data_cube, axis=2)
    volume_density_xy = method(data_cube, axis=0)
    volume_density_xz = method(data_cube, axis=1)
    volume_density_yz = method(data_cube, axis=2)

    imgs = []
    scores = []
    img_generated = 0
    areas_explored = [[],[],[]]
    iteration = 0
    while img_generated < number and iteration < number*100 :
        iteration += 1
        if iteration >= number*100:
            print("Error: failed to generated all the random batches, nbr of img generated:"+str(len(imgs)))
            break

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
            if np.linalg.norm(center-point) < 0.75 * size:
                flag = True
                break
        if flag:
            continue
        c_x, c_y = center
        c_x = convert_pc_to_index(c_x)-int(np.floor(SIM_axis[0][0]/SIM_size*SIM_nres))
        c_y = convert_pc_to_index(c_y)-int(np.floor(SIM_axis[0][0]/SIM_size*SIM_nres))

        #todo find the neareast 2^x
        #s = int(np.floor(convert_pc_to_index(size)/2)*2)
        s = 128
        start_x = c_x - s // 2
        start_y = c_y - s // 2
        end_x = c_x + s // 2 + s%2
        end_y = c_y + s // 2 + s%2

        #If the random area is outside the sim
        if(start_x < 0 or start_y < 0 or end_x >= SIM_nres or end_y >= SIM_nres):
            continue
        cropped_cdens = c_dens[start_x:end_x, start_y:end_y]
        cropped_vdens = v_dens[start_x:end_x, start_y:end_y]

        #Verify if there is no low density region (outside cloud) inside the area
        if(((cropped_vdens < 10).sum()) > s*s*0.01):
            continue

        # Randomly choose a rotation (0, 90, 180, or 270 degrees)
        rotated_cdens = cropped_cdens
        rotated_vdens = cropped_vdens
        if random_rotate:
            k = np.random.choice([0, 1, 2, 3])
            rotated_cdens = np.rot90(cropped_cdens, k)
            rotated_vdens = np.rot90(cropped_vdens, k)


        b = (rotated_cdens, rotated_vdens)

        score = compute_img_score(b[0],b[1])
        if(np.random.random() > RANDOM_BATCH_SCORE_fct(score[0])):
            continue

        imgs.append(b)
        scores.append(score)
        areas_explored[face].append(center)
        img_generated += 1
    
    settings = {
        "SIM_name":SIM_NAME,
        "method": method.__name__,
        "img_number": len(imgs),
        "img_size": size,
        "areas_explored":str(areas_explored),
        "scores": str(scores),
        "scores_fct": inspect.getsourcelines(RANDOM_BATCH_SCORE_fct)[0][0],
        "scores_offset": str(RANDOM_BATCH_SCORE_offset),
        "number_goal": number,
        "iteration": iteration,
        "random_rotate": random_rotate,
    }
    return (imgs,settings)

def check_if_batch_exists(settings):
    return False

def save_batch(batch, settings):
    if check_if_batch_exists(settings):
        return False
    
    if not(os.path.exists(TRAINING_BATCH_FOLDER)):
        os.mkdir(TRAINING_BATCH_FOLDER)

    batch_uuid = uuid.uuid4()
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
    
    

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_batch(batch, b_name="",same_limits=False):
    batch_nbr = len(batch)
    fig, axes = plt.subplots(int(2*np.ceil(batch_nbr/8)),8)
    fig.suptitle(b_name)
    for i in range(batch_nbr):
        data1 = batch[i][0]
        data2 = batch[i][1]
        #score = 0.
        #axes[2*(i//8)][i%8].set_title(str(np.round(score[0],3)))
        min_dat1 = np.min(data1)
        max_dat1 = np.max(data1) 
        d1 = axes[2*(i//8)][i%8].imshow(data1, cmap="jet", norm=LogNorm(vmin=np.min(data1), vmax=np.max(data1)))
        d2 = axes[2*(i//8)+1][i%8].imshow(data2, cmap="jet", norm=(LogNorm(vmin=np.min(data2), vmax=np.max(data2)) if not(same_limits) else LogNorm(min_dat1, max_dat1)))
    fig.subplots_adjust( left=None, bottom=None,  right=None, top=None, wspace=None, hspace=None)

def plot_batch_correlation(batch):
    fig, ax = plt.subplots(1,1)
    column_density = np.array([np.log(b[0])/np.log(10) for b in batch]).flatten()
    volume_density = np.array([np.log(b[1])/np.log(10) for b in batch]).flatten()
    _, _,_,hist = ax.hist2d(column_density, volume_density, bins=(256,256), norm=LogNorm())
    plt.colorbar(hist, ax=ax)
    fig.tight_layout()

if __name__ == "__main__":
    b_name = "batch_37392b55-be04-4e8c-aa49-dca42fa684fc"
    b = open_batch(b_name)
    #b, settings = generate_batch(DATA, method=compute_mass_weighted_density, number=64)
    #save_batch(b, settings)
    plot_batch(b)
    plt.show()