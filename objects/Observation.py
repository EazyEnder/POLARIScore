import os
import sys
from astropy.io import fits
from astropy.wcs import WCS
if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)
from config import *
import matplotlib.pyplot as plt 
import numpy as np
from utils import *
from matplotlib.colors import LogNorm
import torch
import torch.nn.functional as F

class Observation():
    def __init__(self,name,file_name):

        self.name = name
        self.folder = os.path.join(OBSERVATIONS_FOLDER, name)
        """Path to the folder where the observation is stored"""

        file_name = file_name.split(".fits")[0]+".fits"
        self.file = os.path.join(self.folder,file_name)
        """Path to the observation data"""
        self.data = None
        self.wcs = None

        self.init()
    
    def init(self):
        file = fits.open(self.file)
        f = file[0]
        self.data = f.data
        self.wcs = WCS(f.header)
        file.close()

    def predict(self, model_trainer, patch_size=(128, 128), nan_value=0.0, overlap=0.5):

        input_matrix = self.data
        input_tensor = torch.tensor(input_matrix.astype(np.float32))
        nan_mask = np.isnan(input_matrix)
        input_tensor[nan_mask] = nan_value
        height, width = input_matrix.shape
        patch_height, patch_width = patch_size
        stride_height = int(patch_height * (1 - overlap))
        stride_width = int(patch_width * (1 - overlap))

        output_tensor = torch.zeros_like(input_tensor)
        count_tensor = torch.zeros_like(input_tensor)

        i_range = range(0, height - patch_height + 1, stride_height)
        j_range = range(0, width - patch_width + 1, stride_width)

        for i0,i in enumerate(i_range):
            for j0,j in enumerate(j_range):
                printProgressBar(i0*len(j_range)+j0,len(i_range)*len(j_range),prefix="Obs Pred")
                patch = input_tensor[i:i+patch_height, j:j+patch_width]
                patch = patch.unsqueeze(0).unsqueeze(0)
                
                output_patch = model_trainer.predict_image(patch)
                output_patch = output_patch.squeeze(0).squeeze(0) 
                
                output_tensor[i:i+patch_height, j:j+patch_width] += output_patch
                count_tensor[i:i+patch_height, j:j+patch_width] += 1

        output_tensor = output_tensor / count_tensor
        output_matrix = output_tensor.numpy()
        output_matrix[nan_mask] = np.nan

        return output_matrix

    def plot(self, data=None, norm=LogNorm(vmin=1e20)):
        plt.figure()
        ax = plt.subplot(projection=self.wcs)
        data = self.data if data is None else data
        im = ax.imshow(data, cmap="rainbow", norm=norm if not(norm is None) else LogNorm())
        overlay = ax.get_coords_overlay('fk5')
        overlay.grid(color='black', ls='dotted')
        overlay[0].set_axislabel('Right Ascension (J2000)')
        overlay[1].set_axislabel('Declination (J2000)')
        plt.colorbar(im, label=r"$N_H(cm^{-2})$")

        return ax

if __name__ == "__main__":
    obs = Observation("OrionA", "column_density_map")
    obs.plot()

    from networks.Trainer import load_trainer
    trainer = load_trainer("UNet_At")
    prediction = obs.predict(trainer,patch_size=(512,512), overlap=0.5)
    obs.plot(prediction,norm=LogNorm(vmin=1e1,vmax=1e5))

    plt.show()