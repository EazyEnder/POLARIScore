import os
import sys
from astropy.io import fits
from astropy.wcs import WCS
if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)
from config import *
from utils import compute_pdf
import matplotlib.pyplot as plt 
import numpy as np
from utils import *
from matplotlib.colors import LogNorm
import torch
import torch.nn.functional as F
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
import astropy.units as u
import re

def _crop(wcs, lims):
    ra_min, ra_max, dec_min, dec_max = lims
    corner_coords = SkyCoord([ra_min, ra_max, ra_max, ra_min], 
                            [dec_min, dec_min, dec_max, dec_max], 
                            unit="deg", frame="fk5")
    x_pix, y_pix = skycoord_to_pixel(corner_coords, wcs)
    x_min, x_max = int(np.min(x_pix)), int(np.max(x_pix))
    y_min, y_max = int(np.min(y_pix)), int(np.max(y_pix))
    return (x_min,x_max,y_min,y_max)

class Observation():
    def __init__(self,name,file_name):

        self.name = name
        self.folder = os.path.join(OBSERVATIONS_FOLDER, name)
        """Path to the folder where the observation is stored"""

        file_name = file_name.split(".fits")[0]+".fits"
        self.file = os.path.join(self.folder,file_name)
        """Path to the observation data"""
        self.data = None
        self.prediction = None
        self.wcs = None
        self.cores = None
        """Cores [{**core1_properties}]"""

        self.init()
    
    def init(self):
        file = fits.open(self.file)
        f = file[0]
        self.data = f.data
        self.wcs = WCS(f.header)
        file.close()

    def predict(self, model_trainer, patch_size=(128, 128), nan_value=-1.0, overlap=0.5, downsample_factor=1, apply_baseline=False):

        input_matrix = self.data
        input_tensor = torch.tensor(input_matrix.astype(np.float32))
        nan_mask = np.isnan(input_matrix)
        if nan_value < 0:
            nan_value = float(np.nanmin(self.data))
        input_tensor[nan_mask] = nan_value

        downsampled_tensor = F.interpolate(input_tensor.unsqueeze(0).unsqueeze(0), 
                                       scale_factor=1.0/downsample_factor, 
                                       mode='bilinear', align_corners=True).squeeze(0).squeeze(0)

        height, width = downsampled_tensor.shape
        patch_height, patch_width = patch_size
        stride_height = int(patch_height * (1 - overlap))
        stride_width = int(patch_width * (1 - overlap))

        output_tensor = torch.zeros_like(downsampled_tensor)
        count_tensor = torch.zeros_like(downsampled_tensor)

        i_range = range(0, height - patch_height + 1, stride_height)
        j_range = range(0, width - patch_width + 1, stride_width)

        for i0,i in enumerate(i_range):
            for j0,j in enumerate(j_range):
                printProgressBar(i0*len(j_range)+j0,len(i_range)*len(j_range),prefix="Obs Pred")
                patch = downsampled_tensor[i:i+patch_height, j:j+patch_width]
                patch = patch.unsqueeze(0).unsqueeze(0)
                
                output_patch = model_trainer.predict_image(patch)
                output_patch = output_patch.squeeze(0).squeeze(0) 

                if apply_baseline:
                    output_patch = model_trainer.apply_baseline(output_patch, log=False)
                
                output_tensor[i:i+patch_height, j:j+patch_width] += output_patch
                count_tensor[i:i+patch_height, j:j+patch_width] += 1

        print("")
        output_tensor = output_tensor / count_tensor

        upsampled_output = F.interpolate(output_tensor.unsqueeze(0).unsqueeze(0), 
                                     size=(input_matrix.shape[0], input_matrix.shape[1]), 
                                     mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        output_matrix = upsampled_output.numpy()
        output_matrix[nan_mask] = np.nan

        self.prediction = output_matrix

        return output_matrix

    def getCores(self, force_compute=False):

        if not(self.cores is None) and not(force_compute):
            return self.cores

        observed_cores_path = os.path.join(self.folder, "observed_core_catalog.txt")
        if(not(os.path.exists(observed_cores_path))):
            return
        with open(observed_cores_path, "r", encoding="utf-8") as file:
            observed_lines = file.readlines()
        derived_cores_path = os.path.join(self.folder, "derived_core_catalog.txt")
        if(not(os.path.exists(observed_cores_path))):
            return
        with open(derived_cores_path, "r", encoding="utf-8") as file:
            derived_lines = file.readlines()

        border_car = ["!","|"]

        def _search_for_property(name, lines):
            for i, line in enumerate(observed_lines):
                if not(line[0] in border_car):
                    continue
                if not(name.lower() in line.lower()):
                    continue
                match = re.search(r"\((\d+)\)", line)
                if match:
                    return int(match.group(1))
                else:
                    LOGGER.error("Can't get cores properties, there is no int in () for property {name} in cores files.")
                    return None
            LOGGER.error("Can't get cores properties, there is no match for property {name} in cores files.")
            return None


        observed_cores = []
        for i, line in enumerate(observed_lines):
            if not(line[0] in border_car):
                properties = line.strip().split()
                if len(properties) < 3:
                    continue
                offset_index = 4
                try:
                    ra_str = f"{properties[2]} {properties[3]} {properties[4]}"
                    dec_str = f"{properties[5]} {properties[6]} {properties[7]}"
                    coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg)) 
                except ValueError:
                    ra_str = f"{properties[2]}"
                    dec_str = f"{properties[3]}"
                    offset_index = 0
                    coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg)) 
                pc = {
                    "name": properties[1],
                    "peak_ncol": float(properties[54+offset_index])*1e21,
                    "radius": float(properties[58+offset_index]),
                    "ra": coord.ra.deg,
                    "dec": coord.dec.deg
                }
                observed_cores.append(pc)
        derived_cores = []
        for i, line in enumerate(derived_lines):
            if not(line[0] in border_car):
                properties = line.strip().split()
                if len(properties) < 3:
                    continue
                pc = {
                    "name": properties[1],
                    "peak_n": float(properties[13+offset_index])*1e4,
                    "average_n": float(properties[14+offset_index])*1e4,
                    "radius_pc": float(properties[4+offset_index]),
                    "comment": properties[18+offset_index] if len(properties) > (18+offset_index) else "" 
                }

                try:
                    pc["type"] = int(properties[17+offset_index])
                except:
                    pc["type"] = properties[17+offset_index]

                derived_cores.append(pc)
        
        cores = []
        #Disgusting code, TODO
        for obs_dict in observed_cores:
            for der_dict in derived_cores:
                if obs_dict["name"] != der_dict["name"]:
                    continue
                if (type(der_dict["type"]) is int and der_dict["type"] < 2) or (type(der_dict["type"]) is str and der_dict["type"] in ["prestellar", "protostellar"]):
                    continue
                if "tentative" in der_dict["comment"] or "SED" in der_dict["comment"]:
                    continue
                core = {**obs_dict, **der_dict}
                cores.append(core)
                break

        self.cores = cores

        return cores

    def plotCores(self,ax,cores=None,norm=None,vol_density=False,show_text=False):
        if cores is None:
            cores = self.cores
        if cores is None:
            cores = self.getCores()
            if cores is None:
                LOGGER.warn("Can't get the dense cores")
                return
        ra = [c["ra"] for c in cores]
        dec = [c["dec"] for c in cores]
        if vol_density:
            values = np.array([c["peak_n"] for c in cores])
        else:
            values = np.array([c["peak_ncol"] for c in cores])

        world_coords = SkyCoord(ra, dec, unit="deg", frame="fk5")
        x_pix, y_pix = skycoord_to_pixel(world_coords, ax.wcs)

        radius = np.array([c["radius"] for c in cores]) / 3600

        pixel_scale = np.mean(np.abs(ax.wcs.pixel_scale_matrix.diagonal()))

        if norm is None:
            colors = 'none'
        else:
            colors = plt.cm.rainbow(norm(values))


        ax.scatter(x_pix, y_pix, s=radius/pixel_scale, facecolors=colors, edgecolors="black")
        if show_text:
            for i,c in enumerate(cores):
                ax.text(x_pix[i], y_pix[i], f"${values[i]:.2e}$", color='black')

        return ax
    
    def getPredictedDensityAtCore(self, column_density=False):
        if self.cores is None:
            self.getCores()
        if self.cores is None:
            LOGGER.error("No cores found")
            return None
        
        densities = self.prediction
        if column_density:
            densities = self.data

        if densities is None:
            LOGGER.error("No predicted density found")
            return None

        coords = SkyCoord(
            [core["ra"] for core in self.cores],
            [core["dec"] for core in self.cores],
            unit="deg"
        )
        x_pix, y_pix = skycoord_to_pixel(coords, self.wcs)

        values = []
        for x, y in zip(x_pix, y_pix):
            x_int, y_int = int(round(x)), int(round(y))
            if (0 <= y_int < densities.shape[0]) and (0 <= x_int < densities.shape[1]):
                values.append(densities[y_int, x_int])
            else:
                values.append(np.nan)

        for core, val in zip(self.cores, values):
            core["data_value"] = val

        return values
        
    
    def plot(self, data=None, norm=None, plotCores=False, crop=None, force_vol=False, force_col=False):
        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(projection=self.wcs)
        data = self.data if data is None else data
        flag_vol_density = False
        label = r"$N_H(cm^{-2})$"
        norm = norm if not(norm is None) else LogNorm()
        if not(force_col) and (np.nanpercentile(data,50) < 1e10 or force_vol):
            flag_vol_density = True
            label=r"$n_H(cm^{-3})$"
        im = ax.imshow(data, cmap="rainbow", norm=norm)
        overlay = ax.get_coords_overlay('fk5')
        overlay.grid(color='black', ls='dotted')
        overlay[0].set_axislabel('Right Ascension (J2000)')
        #overlay[1].set_axislabel('Declination (J2000)')
        plt.colorbar(im, label=label)
        fig.tight_layout()

        if plotCores:
            self.plotCores(ax, norm=norm, vol_density=flag_vol_density)

        if not(crop is None):
            x_min, x_max, y_min, y_max = _crop(self.wcs, crop)
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))

        return fig, ax
    
    def _get_cores_predicted_values(self, region=None, return_ncol=False):
        predicted_densities = np.array(self.getPredictedDensityAtCore())
        derived_densities =  np.array([c["average_n"] for c in self.getCores()])
        mask = (~np.isnan(predicted_densities)) & (predicted_densities > 0) & (derived_densities > 0)
        if region is not None:
            ra_max, ra_min, dec_min, dec_max = region
            ra = np.array([c["ra"] for c in self.getCores()])
            dec = np.array([c["dec"] for c in self.getCores()])
            region_mask = (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max)
            mask = mask & region_mask
        predicted_densities = predicted_densities[mask]
        derived_densities = derived_densities[mask]
        predicted_densities = np.log10(predicted_densities)
        derived_densities = np.log10(derived_densities)
        if return_ncol:
            column_densities = np.array(self.getPredictedDensityAtCore(column_density=True))
            column_densities = np.log10(column_densities[mask])
            sorted_indexes = np.argsort(column_densities)
            return (predicted_densities[sorted_indexes], derived_densities[sorted_indexes], column_densities[sorted_indexes])
        return (predicted_densities, derived_densities)
    
    def plot_cores_hist(self, ax=None, region=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        predicted_densities, derived_densities = self._get_cores_predicted_values(region=region)

        ax.hist(predicted_densities, bins=10, alpha=0.5, label="Predicted Densities")
        ax.hist(derived_densities, bins=10, alpha=0.5, label="Derived Densities")
        
        ax.set_xlabel("$\log_{10}(n_H) [cm^{-3}]$")

        ax.legend()

        return fig, ax
    
    def plot_cores_hist2d(self, ax=None, region=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        predicted_densities, derived_densities = self._get_cores_predicted_values(region=region)

        _,_,_, hist= ax.hist2d(predicted_densities, derived_densities, bins=(10,10), norm=LogNorm())
        plt.colorbar(hist, ax=ax, label="counts")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Derived")

        return fig, ax
    
    def plot_cores_space(self, ax=None, region=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        predicted_densities, derived_densities, column_densities = self._get_cores_predicted_values(region=region, return_ncol=True)

        bins_number = 10
        contour_levels = 10

        def _plot(c1, c2, color="black", label="", linestyles="solid"):
            hist, xedges, yedges = np.histogram2d(c1, c2, bins=(bins_number, bins_number))
            hist = np.where(hist == 0, np.nan, hist)

            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])
            X, Y = np.meshgrid(xcenters, ycenters)

            contour = ax.contour(X, Y, hist.T, levels=contour_levels, colors=color, linestyles=linestyles, label=label)
            #ax.clabel(contour, fmt=lambda x: r"$10^{{{:.0f}}}$".format(np.log10(x)), inline=True, fontsize=8)

        _plot(column_densities, predicted_densities, linestyles="dashed", label="predicted")
        _plot(column_densities, derived_densities, label="derived")

        #ax.legend()

        merged_list = [d for d in derived_densities]
        merged_list.extend([c for c in predicted_densities])
        plot_lines(column_densities, merged_list, ax)

        #ax.grid()

        return fig, ax
    
    def plot_cores_error(self, ax=None, region=None, alpha=1., movAverage=5, show_errors=True):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        predicted_densities, derived_densities, column_densities = self._get_cores_predicted_values(region=region, return_ncol=True)

        ax.grid()

        residuals = predicted_densities-derived_densities
        if movAverage > 1:
            residuals, residuals_std = movingAverage(residuals, n=movAverage, return_std=True) 
            column_densities = movingAverage(column_densities, n=movAverage)
        line, = ax.plot(column_densities,residuals, marker="+", alpha=alpha, label=self.name)
        if movAverage > 1 and show_errors:
            ax.fill_between(column_densities,residuals-residuals_std,residuals+residuals_std, color=line.get_color(), alpha=0.2)
        
        ax.set_xlabel("$\log_{10}(N_{col}) [cm^{-2}]$")
        ax.set_ylabel("$\log_{10}(n_{pred})-\log_{10}(n_{derived}) [cm^{-3}]$")

        ax.legend()

        return fig, ax
    
    def save(self,replace=False):
        if self.prediction is None:
            LOGGER.error(f"Can't save cache for prediction on {self.name} because there has no prediction on this observation, use .predict(model)")
            return
        if not(os.path.exists(CACHES_FOLDER)):
            os.mkdir(CACHES_FOLDER)
        path = os.path.join(CACHES_FOLDER,self.name+".npy")
        if os.path.exists(path):
            if not(replace):
                LOGGER.error(f"Can't save cache for prediction on {self.name} because there is already a cache and replace is set to False")
                return
            os.remove(path)
        LOGGER.log(f"Observation prediction {self.name} saved")
        np.save(path,self.prediction)

    def load(self):
        path = os.path.join(CACHES_FOLDER,self.name.split(".npy")[0]+".npy")
        if not(os.path.exists(path)):
            return
        self.prediction = np.load(path) 
        return self.prediction
    
def script_data_and_figures(name,crop=None,suffix=None,save_fig=False,plotCores=True,normcol=[None,None],normvol=[None,None], show=True):
    obs = Observation(name, "column_density_map")
    name = name.replace("_","")
    fig, ax = obs.plot(norm=LogNorm(vmin=normcol[0], vmax= normcol[1]),plotCores=plotCores,crop=crop, force_col=True)
    suff = '_'+suffix if not(suffix is None) else ""
    if save_fig:
        fig.savefig(FIGURE_FOLDER+f"obs_{name.lower()}_columndensity{suff}.jpg")

    from networks.Trainer import load_trainer
    obs.load()
    if obs.prediction is None:
        trainer = load_trainer("Unet_highres_2outputs_Adam")
        obs.predict(trainer,patch_size=(128,128), overlap=0., downsample_factor=1, apply_baseline=False)
        obs.save()
    fig, ax = obs.plot(obs.prediction,plotCores=plotCores,norm=LogNorm(vmin=normvol[0], vmax=normvol[1]),crop=crop, force_vol=True)
    if save_fig:
        fig.savefig(FIGURE_FOLDER+f"obs_{name.lower()}_volumedensity{suff}.jpg")
    
    from training_batch import plot_batch_correlation
    fig, ax = plot_batch_correlation([(obs.data,obs.prediction)],show_yx=False)
    ax.set_xlabel(r"Column density ($log_{10}(cm^{-2})$)")
    ax.set_ylabel(r"Mass-weighted density ($log_{10}(cm^{-3})$)")
    fig.tight_layout()
    if save_fig:
        fig.savefig(FIGURE_FOLDER+f"obs_{name.lower()}_correlation.jpg")

    print(f"Max: {np.nanmax(obs.prediction)}, percentiles(10%,50%,90%,95%): {np.nanpercentile(obs.prediction,[10,50,90,95])}")

    if show:
        plt.show()

if __name__ == "__main__":

    #Orion A cropped_region = [Angle("5h36m20s").deg, Angle("5h33m30s").deg, Angle("-6d03m").deg, Angle("-4d55").deg]

    # Orion B cropped_regions
    #cropped_region = [Angle("5h49m").deg, Angle("5h45").deg, Angle("-0d19m").deg, Angle("0d53m").deg]
    #script_data_and_figures("OrionB", suffix="NGC2024_cores", normcol=[1e21,None], normvol=[0.5e1,1e5], save_fig=True, crop=cropped_region, show=False, plotCores=True)
    #cropped_region = [Angle("5h42m56s").deg, Angle("5h40m28s").deg, Angle("-2d32m").deg, Angle("-1d28m").deg]
    #script_data_and_figures("OrionB", suffix="NGC2023_cores", normcol=[1e21,None], normvol=[0.5e1,1e5], save_fig=True, crop=cropped_region, show=False, plotCores=True)

    #script_data_and_figures("Taurus_L1495", normcol=[0.5e21,3e22], normvol=[1e1,2.5e4], save_fig=True, plotCores=False, show=True)

    fig, ax = plt.subplots()

    names = ["OrionB", "Taurus_L1495", "Aquila"]

    

    for n in names:
        obs = Observation(n,"column_density_map")
        obs.load()
        #obs.plot_cores_hist(ax=ax)
        obs.plot_cores_error(ax=ax, alpha=0.75, movAverage=50)

    #fig.savefig(FIGURE_FOLDER+"observation_errors.jpg")

    plt.show()