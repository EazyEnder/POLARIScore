import os
import sys
import time
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from batch_utils import *
from config import *
import uuid
from networks.nn_UNet import *
from networks.nn_FCN import FCN
from networks.nn_MultiNet import MultiNet
from networks.nn_PPV import PPV, Test
from networks.nn_KNet import *
from networks.utils.nn_utils import compute_batch_accuracy
from utils import movingAverage, applyBaseline
import json
from objects.Dataset import getDataset, Dataset
import shutil
from typing import List, Dict, Union, Tuple

class Trainer():
    """
    Allows training of models and experiments with them.
    """
    def __init__(self,network=None,training_set:Dataset=None,validation_set:Dataset=None,learning_rate:float=0.001,model_name:str=None,auto_save:int=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """device: gpu or cpu"""

        self.auto_save:int = auto_save
        """The model is saved each 'auto_save' epochs during training."""

        self.model_name:str = model_name
        if model_name is None:
            self.model_name = str(uuid.uuid4())

        LOGGER.log(f"{self.device} is used for {self.model_name}")
        
        self.network_type:str = "None"
        """Network class name used for the model"""
        if not(network is None):
            self.network_type = network.__name__
        self.learning_rate:float = learning_rate
        """Not constant step size at each iteration during training while moving toward a minimum of a loss function."""
        self.training_set:Dataset = training_set
        """Dataset with the data used for training, i.e used for training/rectify the model weights."""
        self.validation_set:Dataset = validation_set
        """Dataset with the data used for validation, i.e not seen during training."""

        self.prediction_batch:Tuple[List[np.ndarray],List[np.ndarray]] = None
        """data channel 0, prediction channel 0"""

        self.network = network
        """Network class used"""
        self.network_settings: Dict ={}
        """Network settings"""

        self.target_names:Union[List[str],str] = ["vdens"]
        self.input_names:Union[List[str],str] = ["cdens"]

        self.training_random_transform:bool = False

        self.model = None
        """Instance of self.network"""
        self.optimizer = None
        """Instance of an optimizer, by default Adam if optimizer_name was not changed."""
        self.optimizer_name:str = str(type(torch.optim.Adam))
        self.scheduler = None
        """Instance of a scheduler"""
        self.weight_decay:float = 0.
        self.cache_threshold:float = 2.
        """If validation error is lower that this value and if cache option is enable then the model is saved as 'cache_model'
        (and each time the validation error is lower than the previous minimum)."""
        
        self.baseline:Tuple[List[float],List[float]] = None
        """Baseline : (*predictions, *residuals)"""

        if not(self.network is None):
            self.model = network(**self.network_settings).to(self.device)
            if self.optimizer_name in (str(type(torch.optim.Adam)),"Adam"):
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            elif self.optimizer_name in (str(type(torch.optim.SGD)),"SGD"):
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.5)
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.75, threshold=0.005)

        self.loss_method = nn.MSELoss()
        self.training_losses:List[float] = []
        self.validation_losses:List[float] = []
        self.last_epoch:int = 0

    def init(self, model=None)->bool:
        """
        Init the model, use this function if you changed settings as self.network,self.optimizer or self.scheduler. 
        By default, when creating a Trainer instance a model is created with default settings as Adam for optimizer.
        Args:
            model: Network instance, can be None if you want to let the code create the instance using self.network_settings.
        Returns:
            bool: If a model was created.
        """
        if(self.network is None and model is None):
            LOGGER.warn(f"Can't init model {self.model_name}, check if network is defined or model is not None.")
            return False
        self.network_type = self.network.__name__
        if model is None:
            self.model = self.network(**self.network_settings).to(self.device)
        else:
            self.model = model.to(self.device)
        if self.optimizer_name in (str(type(torch.optim.Adam)),"Adam") or "Adam" in self.optimizer_name:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name in (str(type(torch.optim.SGD)),"SGD") or "SGD" in self.optimizer_name:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.75, threshold=0.005)
        return True

    def train(self, epoch_number:int=50, batch_number:int=32, compute_validation:int=10, cache:bool=True):
        """
        Train the model (check trainer variables for settings)

        Args:
            epoch_number(int, default: 50): train the model for x epochs.
            batch_number(int, default: 32): How many images will be processed at a time in the GPU/CPU.
            compute_validation(int, default:10): compute validation losses each x epochs.
            cache(bool, default:True): If the validation loss is less than a previous epoch, the model will be saved in a cache.
        """
        LOGGER.log(f"Training started with {str(epoch_number)} epochs on network {self.network_type} with mini-batch of size {batch_number}, model has {sum(p.numel() for p in self.model.parameters())} parameters.")
   
        self.model.train()

        def _format_time(seconds:float)->str:
            hours, rem = divmod(seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        def _random_transform(tensors_input, tensors_target):
            k = torch.randint(0, 4, (1,)).item()
            #Maybe need to change this: See if it works ! 
            tensors_input = [torch.rot90(t, k, [2, 3]) for t in tensors_input]
            tensors_target = [torch.rot90(t, k, [2, 3]) for t in tensors_target]

            if torch.rand(1).item() > 0.5:
                tensors_input = [torch.flip(t, [2]) for t in tensors_input]
                tensors_target = [torch.flip(t, [2]) for t in tensors_target]

            if torch.rand(1).item() > 0.5:
                tensors_input = [torch.flip(t, [3]) for t in tensors_input]
                tensors_target = [torch.flip(t, [3]) for t in tensors_target]

            return tensors_input, tensors_target


        total_epoch = self.last_epoch
        l_ep = self.last_epoch
        batch_size = len(self.training_set.batch)
        validation_batch_size = len(self.validation_set.batch)
        start_time = time.process_time()

        minimimum_validation_loss = self.cache_threshold

        for epoch in range(epoch_number):
            total_epoch += 1
            epoch_loss = 0
            shuffled_indices = torch.randperm(batch_size)  
            self.optimizer.zero_grad()
            minbatch_nbr = int(np.floor(batch_size/batch_number))
            epoch_time = time.process_time()
            for b in range(minbatch_nbr if minbatch_nbr > 1 else 1):
                printProgressBar(b, minbatch_nbr, length=10, prefix=f"{b}/{minbatch_nbr}")
                used_batch = self.training_set.get(indexes=shuffled_indices[b*batch_number:(b+1)*batch_number if minbatch_nbr > 1 else -1])
                if batch_number == 1:
                    used_batch = [used_batch]
                t_input, t_target = self.model.shape_data(used_batch, self.training_set.get_element_index(self.target_names), self.training_set.get_element_index(self.input_names))
                if not(type(t_input) is list):
                    t_input = [t_input]

                if self.training_random_transform:
                    t_input, t_target = _random_transform(t_input, t_target)
                output = self.model(*t_input)
                loss = 0
                try:
                    loss = self.loss_method(output, t_target)
                except:
                    for tt in range(len(t_target)):
                        if type(output) is list:
                            loss += self.loss_method(output[tt], t_target[tt])
                        else:
                            loss += self.loss_method(output, t_target[tt])
                loss.backward()
                epoch_loss += loss.item()
            self.optimizer.step()
            epoch_loss /= minbatch_nbr if minbatch_nbr != 0 else 1 
            self.scheduler.step(epoch_loss)
            self.training_losses.append((total_epoch, epoch_loss))
            val_total_loss = None
            if compute_validation>0 and total_epoch % compute_validation == 0:
                minbatch_nbr = int(np.floor(validation_batch_size/batch_number))
                with torch.no_grad():
                    #self.model.eval()
                    val_total_loss = 0
                    for b in range(minbatch_nbr if minbatch_nbr > 1 else 1):
                        printProgressBar(b, minbatch_nbr, length=10, prefix=f"{b}/{minbatch_nbr}")
                        used_batch = self.validation_set.get(indexes=list(range(len(self.validation_set.batch)))[b*batch_number:(b+1)*batch_number if minbatch_nbr > 1 else -1])
                        if batch_number == 1:
                            used_batch = [used_batch]
                        v_input_tensor, v_target_tensor = self.model.shape_data(used_batch, self.validation_set.get_element_index(self.target_names), self.validation_set.get_element_index(self.input_names))
                        if not(type(v_input_tensor) is list):
                            v_input_tensor = [v_input_tensor]
                        validation_output = self.model(*v_input_tensor)
                        v_loss = 0
                        try:
                            v_loss = self.loss_method(validation_output,v_target_tensor).item()
                        except:
                            for tt in range(len(v_target_tensor)):
                                if type(validation_output) is list:
                                    v_loss += self.loss_method(validation_output[tt],v_target_tensor[tt]).item()
                                else:
                                    v_loss += self.loss_method(validation_output,v_target_tensor[tt]).item()
                        val_total_loss += v_loss
                    #self.model.train()
                val_total_loss /= minbatch_nbr if minbatch_nbr > 0 else 1
                self.validation_losses.append((total_epoch,val_total_loss))
            if self.auto_save > 0 and total_epoch % self.auto_save == 0:
                self.last_epoch = total_epoch
                self.save()
            if cache and not(val_total_loss is None) and val_total_loss < minimimum_validation_loss:
                minimimum_validation_loss = val_total_loss
                self.last_epoch = total_epoch
                self.save(is_cache=True)

            actual_time = time.process_time()
            epoch_time = actual_time - epoch_time
            time_left = (actual_time-start_time) / (epoch+1) * (epoch_number-(epoch+1))
            LOGGER.print(f'Epoch {total_epoch}/{l_ep + epoch_number} | Elapsed: {_format_time(actual_time-start_time)} | Time Left: {_format_time(time_left)} | Training Loss: {epoch_loss}, Validation loss: {val_total_loss if val_total_loss else "Not computed"}', type="training", level=1, color="34m")
            
        self.last_epoch = total_epoch
        self.learning_rate = self.scheduler.get_last_lr()[0]

    def create_baseline(self,n:int=1000, force_compute:bool=False, log:bool=True)->Tuple[List[float],List[float]]:
        """
        Create a baseline using moving average method.
        Args:
            n(int): Moving average step
            force_compute(bool): Erase the previous computed baseline and compute a new one.
            log(bool): Log to the console.
        Returns:
            (List of predictions, List of residuals)
        """
        if not(force_compute) and not(self.baseline is None):
            return self.baseline
        if log:
            LOGGER.log(f"Computing baseline for model: {self.network_type}")
        batch = self.get_prediction_batch()
        d_target = np.array([np.log10(b[0]) for b in batch]).flatten()
        d_prediction = np.array([np.log10(b[1]) for b in batch]).flatten()
        residuals = d_prediction-d_target

        sorted_indexes = np.argsort(d_target)
        d_prediction = d_prediction[sorted_indexes]
        residuals = residuals[sorted_indexes]

        mresiduals = movingAverage(residuals, n=n)
        mx = movingAverage(d_prediction, n=n)

        self.baseline = (mx,mresiduals)

        return self.baseline
    
    def apply_baseline(self, prediction:np.ndarray, log:bool=True)->np.ndarray:
        """
        Apply the baseline to data (prediction).
        Args:
            prediction: data to be processed
            log (bool): log output in console.
        Returns:
            modified prediction: Prediction minus residuals.
        """
        if log:
            LOGGER.log(f"Applying baseline for prediction {prediction.shape[0]}x{prediction.shape[1]}")
        if self.baseline is None:
            self.create_baseline(log=log)

        H,W = prediction.shape
        d_prediction = np.array(np.log10(prediction)).flatten()

        d_prediction = applyBaseline(self.baseline[0],self.baseline[1],d_prediction,d_prediction)
        d_prediction = d_prediction.reshape((H,W))

        d_prediction = np.exp(d_prediction * np.log(10))

        return d_prediction

    def plot_losses(self, ax=None ,log10=True):
        """
        Plot training and validation losses.

        Args:
            ax (matplotlib.axes.Axes, default:None): Axis to plot on. If None, a new figure is created.
            log10 (bool, default:True): Whether to express losses in log10 scale. Default is True.

        Returns:
            figure and ax
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        x_training = [i[0] for i in self.training_losses]
        x_validation = [i[0] for i in self.validation_losses]
        y_training = [i[1] for i in self.training_losses]
        y_validation = [i[1] for i in self.validation_losses]
        
        loss_method_name = getattr(self.loss_method, "_get_name", lambda: "Custom")()
        ax.set_ylabel(loss_method_name)
        if log10:
            y_training = np.log10(y_training)
            y_validation = np.log10(y_validation)
            ax.set_ylabel(loss_method_name + " (log10)")

        ax.scatter(x_training, y_training, label="training losses")
        ax.scatter(x_validation, y_validation, label="validation losses")
        ax.plot(x_training, y_training)
        ax.plot(x_validation, y_validation)

        ax.set_xlabel("epoch")
        ax.legend()
        
        return fig, ax

    def get_prediction_batch(self,force_compute=False):
        """
        Args:
            force_compute(bool): If trainer has already a prediction batch computed then if this is True, this will be computed again.
        Returns:
            prediction_batch:[(target_img1, prediction_img1), (target_img2, prediction_img2), ...]
        """
        if not(self.prediction_batch is None or force_compute):
            return self.prediction_batch
        
        self.prediction_batch = self.predict(self.validation_set)
    
        return self.prediction_batch

    def plot_validation(self, inter=(None,None),number_per_row=8,same_limits=True):
        """
        Show target and model prediction images
        """
        plot_batch(self.get_prediction_batch()[0 if inter[0] is None else inter[0]: -1 if inter[1] is None else inter[1]], same_limits=same_limits, number_per_row=number_per_row)

    def plot_residuals(self, batch=None, ax=None, plot_distribution=True, color="blue", bins_inter=(None,None)):
        """
        Plot model predictions residuals

        Args:
            ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, a new figure is created.
            plot_distribution (bool, optional): Whether to plot the residuals distribution. Default is True.
            color (str, optional): Residuals distribution color. Default is blue
            bins_inter (tuple (x,x), optional): Set the plot min and max (min,max), min can be None when max takes a value.

        Returns:
            figure and ax
        """

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        if batch is None:
            batch = self.get_prediction_batch()
        d_target = np.array([np.log(b[0])/np.log(10) for b in batch]).flatten()
        d_prediction = np.array([np.log(b[1])/np.log(10) for b in batch]).flatten()

        residuals = d_prediction-d_target
        violin_num_bins = 5
        bins = np.linspace(min(d_target) if bins_inter[0] is None else bins_inter[0], max(d_target) if bins_inter[1] is None else bins_inter[1], violin_num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2 
        bin_indices = np.digitize(d_target, bins) - 1 
        binned_residuals = [residuals[bin_indices == i] for i in range(violin_num_bins)]

        re_bin_centers = []
        re_binned_residuals = []
        mean_residuals = []
        for i, res in enumerate(binned_residuals):
            if res.size <= 0:
                continue
            re_bin_centers.append(bin_centers[i])
            re_binned_residuals.append(res)
            mean_residuals.append(np.mean(residuals[bin_indices == i]))
        bin_centers = re_bin_centers
        binned_residuals = re_binned_residuals

        if plot_distribution:
            vp = ax.violinplot(binned_residuals, positions=bin_centers, showmeans=False, showmedians=True)
            for i, body in enumerate(vp['bodies']):
                body.set_facecolor(color)
                body.set_alpha(0.5)
            for part in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
                vp[part].set_edgecolor('black')
                vp[part].set_linewidth(1.0)
        ax.axhline(0, color='red', linestyle='--')
        ax.plot(bin_centers, mean_residuals, marker='o', linestyle='-', color='black', alpha=0.7)

        ax.set_xticks(bin_centers)
        ax.set_xlabel("Pixel value (log10)")
        ax.set_ylabel("Residuals (prediction-target)")
        ax.grid(True, linestyle="--", alpha=0.5)

        return fig, ax
    
    def predict(self, dataset:Dataset, batch_number:int=1)->List[Tuple[List[np.ndarray],List[np.ndarray]]]:
        """Apply the model on a dataset
        Args:
            dataset: the dataset
            batch_number(int): How many pairs of images/arrays send to the gpu and computed at the same time.
        Returns:
            List: list of 
        """
        #TODO prendre en compte si il y a plusieurs outputs, idem sur les autres fonctions. Car là c'est pas clair et pas fou. 

        self.model.eval()

        result_batch = []

        batch_size = len(dataset.batch)
        minbatch_nbr = int(np.floor(batch_size/batch_number))
        for b in range(minbatch_nbr if minbatch_nbr > 1 else 1):
            printProgressBar(b, minbatch_nbr, length=10, prefix=f"{b}/{minbatch_nbr}")
            used_batch = dataset.get(indexes=list(range(batch_size))[b*batch_number:(b+1)*batch_number if minbatch_nbr > 1 else -1])
            if batch_number == 1:
                used_batch = [used_batch]
            input_tensor, target_tensor = self.model.shape_data(used_batch, dataset.get_element_index(self.target_names), dataset.get_element_index(self.input_names))
            if not(type(input_tensor) is list):
                input_tensor = [input_tensor]
            if not(type(target_tensor) is list):
                target_tensor = [target_tensor]
            output = self.model(*input_tensor)
            target_tensor = [t.cpu().detach().numpy() for t in target_tensor] 
            if type(output) is list:
                output = [o.cpu().detach().numpy() for o in output]
            else:
                output = [output.cpu().detach().numpy()]
            result_batch.append((*(np.exp(target_tensor[0][:,0])), *(np.exp(output[0][:,0]))))

        return result_batch
    
    def predict_image(self, image):
        self.model.eval()

        #TODO: generalize this function for image of nparray type and torch image
        input_tensor = torch.log(image).float().to(self.device)
        output = self.model(input_tensor)
        if type(output) is list:
            output = output[0]
        output = output.cpu().detach().numpy()
        return np.exp(output)

    def plot_sim_validation(self, simulation, plot_total=False):
        sim_col_dens = simulation._compute_c_density()
        sim_mass_dens = simulation._compute_v_density(method=compute_mass_weighted_density)
        raw_sim_batch =  [(sim_col_dens,sim_mass_dens)]
        d_m_s_col = divide_matrix_to_sub(sim_col_dens)
        d_m_s_mass = divide_matrix_to_sub(sim_mass_dens)
        divided_sim_batch = rebuild_batch(d_m_s_col, d_m_s_mass)
        pred_raw_batch = self.predict(raw_sim_batch)
        pred_divided_batch = self.predict(divided_sim_batch)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})
        fig.suptitle("Simulation Validation")

        raw_fig, raw_axes = plot_batch(pred_raw_batch, number_per_row=1, same_limits=True)
        raw_axes = np.array(raw_axes).flatten()

        divided_fig, divided_axes = plot_batch(rebuild_batch([group_matrix([p[0] for p in pred_divided_batch])],[group_matrix([p[1] for p in pred_divided_batch])]), number_per_row=1, same_limits=True)
        divided_axes = np.array(divided_axes).flatten()

        if plot_total:
            raw_fig.canvas.draw()
            raw_axes_image = np.array(raw_fig.canvas.renderer.buffer_rgba())

            divided_fig.canvas.draw()
            divided_axes_image = np.array(divided_fig.canvas.renderer.buffer_rgba())

            axes[0].imshow(raw_axes_image)
            axes[0].set_title("Raw Data")
            axes[0].axis("off")

            axes[1].imshow(divided_axes_image)
            axes[1].set_title("Divided Data")
            axes[1].axis("off")
        else:
            fig = None
    
    def plot_validation_spatial_error(self,number_per_row=4,log=True):
        batch = self.get_prediction_batch()
        if log:
            error = (np.log10(np.array([b[0] for b in batch]))-np.array(np.log10([b[1] for b in batch])))
        else:
            error = np.abs(np.array([b[0] for b in batch])-np.array([b[1] for b in batch]))
        fig, axes = plt.subplots(int(np.ceil(len(error)/number_per_row)),number_per_row,figsize=(14, 9))
        for i,e in enumerate(error):
            if len(axes.shape) > 1:
                im = axes[(i//number_per_row)][i%number_per_row].imshow(e, cmap="jet")
            else:
                im = axes[i].imshow(e, cmap="jet")
            plt.colorbar(im)
        fig.subplots_adjust( left=None, bottom=None,  right=None, top=None, wspace=None, hspace=None)
        return fig, axes

    def plot_prediction_correlation(self,ax=None,factors=[0]):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        batch = self.get_prediction_batch()
        target_imgs = np.array([np.log10(b[0]) for b in batch]).flatten()
        min_t, max_t = np.min(target_imgs), np.max(target_imgs)
        plot_batch_correlation(batch, ax=ax)
        ax.set_xlabel("Target (log10)")
        ax.set_ylabel("Prediction (log10)")
        
        X = np.linspace(min_t,max_t,10)
        colors = FIGURE_CMAP(range(len(factors)))
        for i,f in enumerate(factors):
            if f == 0:
                continue
            f = np.abs(f)
        
            ax.plot(X,np.log10(f)+X,color=colors[i],linestyle="dashdot",label=fr"y=${f}\times x$")
            ax.plot(X,np.log10(1/f)+X,color=colors[i],linestyle="dashdot",label=fr"y=${1/f:.1f}\times x$")

        plt.legend()

        return fig, ax

    def plot(self):
        """
        Plot all in one figure: validation correlation, residuals and losses.
        """
        plt.figure()
        plt.suptitle(self.model_name)
        
        ax1 = plt.subplot(2,2,1)
        self.plot_prediction_correlation(ax=ax1)

        ax2 = plt.subplot(2,2,2)
        self. plot_residuals(ax=ax2)

        ax3 = plt.subplot(2,2,3)
        self.plot_losses(ax=ax3)

        plt.tight_layout()

    def save(self, model_name=None, is_cache=False):
        """
        Save model and model settings in a new folder

        Args:
            model_name (str, optional): set model name, if None it will take the trainer model_name if user had set it or a random uuid.
            is_cache (bool, default:False): Save the model as a cache (just one cached model can exists).

        Returns:
            bool: If this is a success
        """

        if(not(model_name is None)):
            self.model_name = model_name
        
        if not(os.path.exists(MODEL_FOLDER)):
            os.mkdir(MODEL_FOLDER)

        #while os.path.exists(os.path.join(MODEL_FOLDER,self.model_name)):
        #    self.model_name = str(uuid.uuid4())

        model_path = os.path.join(MODEL_FOLDER,self.model_name.rsplit("_epoch",1)[0])
        if is_cache:
            model_path = os.path.join(MODEL_FOLDER, "cached_model")
        if not(os.path.exists(model_path)):
            os.mkdir(model_path)
        elif is_cache:
            LOGGER.warn(f"A previous cached model was removed.")
            shutil.rmtree(model_path)
            os.mkdir(model_path)


        ep = self.last_epoch
        model_name = self.model_name.rsplit("_epoch",1)[0]+"_epoch"+str(ep) if not(is_cache) else "cached_model"
        if os.path.exists(os.path.join(model_path,model_name+".pth")):
            LOGGER.warn(f"Can't save {model_name} with epoch: {ep}")
            return
        
        torch.save(self.model.state_dict(),os.path.join(model_path,model_name+".pth"))

        loss_method_name = ""
        try:
            loss_method_name = self.loss_method._get_name()
        except AttributeError:
            loss_method_name = "Custom"

        cloned_network_settings = self.network_settings.copy()

        if "convBlock" in self.network_settings:
            cloned_network_settings["convBlock"] = self.network_settings["convBlock"].__name__ if not(type(self.network_settings["convBlock"]) is str) else self.network_settings["convBlock"]

        settings = {
            "model_name": self.model_name,
            "network": self.network_type,
            "network_settings": cloned_network_settings,
            "loss_method": loss_method_name,
            "optimizer": str(type(self.optimizer)),
            "learning_rate": str(self.learning_rate),
            "scheduler": str(type(self.scheduler)),
            "total_epoch": str(self.last_epoch),
            "input_names": self.input_names,
            "target_names": self.target_names,
            "training_set": str(self.training_set.name),
            "validation_set": str(self.validation_set.name),
            "system": get_system_info(),
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
        }

        with open(os.path.join(model_path,'settings.json'), 'w') as file:
            json.dump(settings, file, indent=4)

        LOGGER.log(f"{self.model_name} saved.")

        return True

def load_trainer(model_name, load_model=True):

    folder_model_name = model_name

    model_path = os.path.join(MODEL_FOLDER, model_name)
    if(not(os.path.exists(model_path))):
        LOGGER.error(f"Can't load {model_name}, file {model_path} doesn't exist.")
        return
    
    settings = {}
    with open(os.path.join(model_path,'settings.json')) as file:
        settings = json.load(file)
    
    trainer = Trainer(model_name=settings["model_name"])
    if "training_set" in settings and not(settings["training_set"] is None):
        trainer.training_set = getDataset(settings["training_set"])
    if "validation_set" in settings and not(settings["validation_set"] is None):
        trainer.validation_set = getDataset(settings["validation_set"])

    network_options = {"UNet" : UNet,
           "FCN" : FCN,
           "KNet" : KNet,
           "UneK": UneK,
           "MultiNet": MultiNet,
           "PPV": PPV,
           "JustKAN": JustKAN,
           "Test": Test,
           "None": None
    }
    network_convblock_options = {"DoubleConvBlock": DoubleConvBlock,"ResConvBlock":ResConvBlock, "KanConvBlock":KanConvBlock, "ConvBlock":ConvBlock}
    network_settings = settings["network_settings"] if "network_settings" in settings else {}
    if "convBlock" in network_settings:
        network_settings["convBlock"] = network_convblock_options[network_settings["convBlock"]]

    trainer.network_type = settings["network"]
    trainer.network_settings = network_settings
    trainer.network = network_options[settings["network"]]
    trainer.learning_rate = float(settings["learning_rate"])
    trainer.last_epoch = int(settings["total_epoch"])
    trainer.training_losses = settings["training_losses"]
    trainer.validation_losses = settings["validation_losses"]
    trainer.input_names = settings["input_names"] if "input_names" in settings else ["cdens"]
    trainer.target_names = settings["target_names"] if "target_names" in settings else (settings["target_name"] if "target_name" in settings else "vdens")

    trainer.optimizer_name = settings["optimizer"]

    if load_model:
        model = trainer.network(**trainer.network_settings)
        try:
            model.load_state_dict(torch.load(os.path.join(model_path,trainer.model_name+".pth")))
        except FileNotFoundError:
            try:
                model.load_state_dict(torch.load(os.path.join(model_path,folder_model_name+".pth")))
            except FileNotFoundError:
                model.load_state_dict(torch.load(os.path.join(model_path,trainer.model_name+f"_epoch{trainer.last_epoch}.pth")))
        model.to(trainer.device)
        trainer.init(model=model)

    LOGGER.log(f"{model_name} loaded")

    return trainer

def plot_models_residuals(trainers = [], ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    models_residuals = []
    models_targets = []
    for t in trainers:
        batch = t.get_prediction_batch()
        d_target = np.array([np.log(b[0])/np.log(10) for b in batch]).flatten()
        d_prediction = np.array([np.log(b[1])/np.log(10) for b in batch]).flatten()
        models_residuals.append(d_prediction-d_target)
        models_targets.append(d_target)
    models_name = [t.model_name for t in trainers]

    positions = np.arange(len(models_name))

    vp = ax.violinplot(models_residuals, positions=positions, showmeans=True, showmedians=True)
    colors = FIGURE_CMAP(np.linspace(FIGURE_CMAP_MIN, FIGURE_CMAP_MAX, len(models_name)))
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.8)
    vp["cmedians"].set_edgecolor('black')
    vp["cmedians"].set_linestyle('dashed')
    vp["cmedians"].set_linewidth(1.2)
    for part in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
        vp[part].set_edgecolor('black')
        vp[part].set_linewidth(1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels(models_name)
    ax.set_ylabel("Residuals (prediction-target)")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig, ax, colors

def plot_models_residuals_extended(trainers = []):
    fig = plt.figure()

    ax1 = plt.subplot2grid((len(trainers), 2), (0, 0), rowspan=len(trainers))
    _,_, colors = plot_models_residuals(trainers=trainers, ax=ax1)
    ax1.set_title("Comparison of different models")

    for i,t in enumerate(trainers):
        ax = plt.subplot(len(trainers),2, (i+1)*2)
        t.plot_residuals(ax=ax, color=colors[i], bins_inter=(1.5,7.5))
        ax.set_ylim((-1.25, 1.25))
        ax.set_ylabel("")

    plt.tight_layout()
    return fig, ax

def plot_models_accuracy(trainers = [], ax = None, sigmas = (0.,1.,20), show_errors = False):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    colors = FIGURE_CMAP(np.linspace(FIGURE_CMAP_MIN, FIGURE_CMAP_MAX, len(trainers)))
    ax = plt.subplot(1,1,1)
    sigmas = np.linspace(sigmas[0],sigmas[1],sigmas[2])
    for i,t in enumerate(trainers):
        accuracies = []
        accuracies_error = []
        for s in sigmas:
            acc_mean, acc_std = compute_batch_accuracy(t.get_prediction_batch(),sigma=s)
            accuracies.append(acc_mean)
            accuracies_error.append(acc_std)
        accuracies = np.array(accuracies)
        accuracies_error = np.array(accuracies_error)
        if show_errors:
            ax.fill_between(sigmas,np.clip(accuracies-accuracies_error,0.,1.),np.clip(accuracies+accuracies_error,0.,1.), color=colors[i], alpha=0.2)
        ax.scatter(sigmas, accuracies, color=colors[i], label=t.model_name)
        #ax.errorbar(sigmas, accuracies, yerr=accuracies_error, color=colors[i])
        ax.plot(sigmas, accuracies, color=colors[i])
    ax.set_xlabel("Error allowed (in log10)")
    ax.set_ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    return fig, ax

import re
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
def heatmap(root_name, validation_batch, X, Y, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    Z = []
    X_flat = []
    Y_flat = []
    for l in Y:
        for bf in X:
            trainer = load_trainer(root_name+f"_l{str(l)}_bf{str(bf)}")
            if trainer is None:
                continue
            trainer.validation_batch = validation_batch
            acc, std = compute_batch_accuracy(trainer.get_prediction_batch(),sigma=0.3)
            Z.append(acc)
            Y_flat.append(l)
            X_flat.append(bf)
            trainer = None

    X = np.array(X_flat)
    Y = np.array(Y_flat)
    Z = np.array(Z)
    X_unique = np.sort(np.unique(X))
    Y_unique = np.sort(np.unique(Y))
    Z_grid = np.full((len(X_unique), len(Y_unique)), np.nan)
    for x, y, z in zip(X, Y, Z):
        x_idx = np.where(X_unique == x)[0][0]
        y_idx = np.where(Y_unique == y)[0][0] 
        Z_grid[x_idx, y_idx] = z
    
    
    grid_x, grid_y = np.meshgrid(X_unique, Y_unique, indexing='ij')
    all_points = np.array([(x, y) for x, y in zip(grid_x.ravel(), grid_y.ravel())])

    known_points = np.array([(x, y) for x, y in zip(X, Y)])
    known_values = Z

    interpolation_method = 'nearest'

    interpolated_values = griddata(known_points, known_values, all_points, method=interpolation_method)

    Z_grid = interpolated_values.reshape(len(X_unique), len(Y_unique))
    for i in range(len(Z_grid)):
        for j in range(len(Z_grid[i])):
            if np.isnan(Z_grid[i,j]):
                Z_grid[i,j] = 0

    cmap = plt.get_cmap("viridis", 100)
    norm = mcolors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    #cm = plt.pcolormesh(grid_x, grid_y, Z_grid.T, shading='auto', cmap=cmap, alpha=0.75, norm=norm)
    cm = ax.pcolormesh(X_unique, Y_unique, Z_grid.T, shading='auto', cmap=cmap, alpha=0.75, norm=norm)
    for i in range(len(X)):
        sc = ax.scatter(X[i], Y[i], color=cmap(norm(Z[i])), marker="o" , edgecolors='k', norm=norm)
    cbar = plt.colorbar(cm,label=r"Accuracy for $\sigma=0.3$",ax=ax)
    ax.set_xlabel("Base Filters")
    ax.set_ylabel("Layers")

    return fig, ax

import glob
import re
def plot_modelset(root_name, validation_batch=None, prefix="t", ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    models_paths = glob.glob(os.path.join(MODEL_FOLDER,root_name)+f"_{prefix}*")
    X = []
    names = []
    for p in models_paths:
        p = p.split('/')[-1]
        match = re.search(r"_t(\d+)", p)
        if match:
            X.append(int(match.group(1)))
            names.append(p)
        else:
            LOGGER.warn(f"Can't read property in model name: {p}.")
            continue

    indexes = np.argsort(X)
    X = np.array(X)[indexes]
    names = np.array(names)[indexes]

    Y = []
    for n in names:
        trainer = load_trainer(n)
        if trainer is None:
            continue
        if not(validation_batch is None):
            trainer.validation_set = validation_batch
        acc, std = compute_batch_accuracy(trainer.get_prediction_batch(),sigma=0.3)
        Y.append(acc)
        del trainer
    
    ax.plot(X, Y)
    ax.scatter(X, Y)
    ax.grid(True)
    ax.set_xlabel("Size of the training set")
    ax.set_ylabel(r"Accuracy for $\sigma=0.3$")
    fig.tight_layout()

    return fig, ax

def generate_model_map(root_name, train_batch, validation_batch, network=UNet, layers=[2,3,4,5], base_filters=[8,16,32,48,64,80]):

    for i,l in enumerate(layers):
        for j,bf in enumerate(base_filters):
            model_path = os.path.join(MODEL_FOLDER, root_name+f"_l{str(l)}_bf{str(bf)}")
            if(os.path.exists(model_path)):
                LOGGER.warn(root_name+f"_l{str(l)}_bf{str(bf)}"+f" already exists, delete the folder if you want to traina new model with these settings.")
                continue
            LOGGER.log(f"Now training: {l}, {bf} ({str(np.round((i*len(base_filters)+j)/(len(base_filters)*len(layers))*100,3))}%)")
            trainer  = Trainer(network, train_batch,validation_batch, model_name=root_name+f"_l{str(l)}_bf{str(bf)}")
            trainer.network_settings["base_filters"] = bf
            trainer.network_settings["num_layers"] = l
            trainer.training_random_transform = True
            trainer.network_settings["attention"] = True
            trainer.init()
            trainer.train(int(2000+1000*(1-i/len(layers))*(1-i/len(base_filters))))
            trainer.save()

def generate_model_training_map(root_name, train_batch, validation_batch, network=UNet, training=[8,16,32,48,64,80]):

    for i,l in enumerate(training):
        if l > 1:
            l = l/len(train_batch.batch)
        dataset, _ = train_batch.split(l)
        l = len(dataset.batch)
        model_path = os.path.join(MODEL_FOLDER, root_name+f"_t{str(l)}")
        if(os.path.exists(model_path)):
            LOGGER.warn(root_name+f"_t{str(l)}"+f" already exists, delete the folder if you want to train a new model with these settings.")
            continue
        LOGGER.log(f"Now training: {l}({str(np.round((i)/(len(training))*100,3))}%)")
        trainer  = Trainer(network, dataset ,validation_batch, model_name=root_name+f"_t{str(l)}")
        trainer.network_settings["base_filters"] = 64
        trainer.network_settings["num_layers"] = 4
        #trainer.network_settings["convBlock"] = DoubleConvBlock
        trainer.training_random_transform = True
        trainer.network_settings["attention"] = True
        trainer.target_names = "vdens"
        trainer.input_names = ["cdens"]
        trainer.optimizer_name = "SGD"
        trainer.init()
        trainer.train(1000, batch_number=16, cache=False)
        trainer.save()

if __name__ == "__main__":

    def binned_loss(output, target, bin_edges=[0,2,4,6,8,10]):
        loss = 0.0
        for i in range(len(bin_edges) - 1):
            mask = (target >= bin_edges[i]) & (target < bin_edges[i + 1])
            
            if mask.any():
                bin_loss = torch.mean((output[mask] - target[mask]) ** 2)
                loss += bin_loss
        return loss
    
    class WeightedMSELoss(nn.Module):
        def __init__(self, bin_edges, bin_weights):
            super(WeightedMSELoss, self).__init__()
            self.bin_edges = torch.tensor(bin_edges, dtype=torch.float32, device="cuda")
            self.bin_weights = torch.tensor(bin_weights, dtype=torch.float32, device="cuda")

        def forward(self, y_pred, y_true):
            flag = type(y_pred) is list
            if flag:
                column_y_pred = y_pred[1]
                y_pred  = y_pred[0]
                column_y_true = y_true[1]
                y_true  = y_true[0]

            y_true_flat = y_true.reshape(-1)
            y_pred_flat = y_pred.reshape(-1)
            
            bin_indices = torch.bucketize(y_true_flat, self.bin_edges[:-1], right=False)
            bin_indices = torch.clamp(bin_indices, 0, len(self.bin_weights) - 1)
            weights = self.bin_weights[bin_indices]

            loss = torch.mean(weights * (y_true_flat - y_pred_flat) ** 2)
            if flag:
                loss += 0.1* torch.mean((column_y_true - column_y_pred)**2)
            return loss
        
    def batch_loss(output, target):
        per_image_loss = torch.mean((output - target) ** 2, dim=[1, 2, 3])
        weights = per_image_loss / (torch.max(per_image_loss) + 1e-6)
        weighted_loss = torch.sum(per_image_loss * weights)
        return torch.sum(weighted_loss)

    def column_density_loss(output, target):
        
        loss = torch.mean(torch.log(torch.abs(torch.sum(torch.exp(output), dim=4)-torch.sum(torch.exp(target), dim=4))+1)**2)   

        return loss
    
    def MSELoss2outputs(output, target):
        o1 = output[0]
        o2 = output[1]
        t1 = target[0]
        t2 = target[1]
        return torch.mean((o1 - t1) ** 2)+0.1*torch.mean((o2 - t2) ** 2)
    
    ds = getDataset("batch_orionMHD_lowB_0.39_512_13CO_max")
    #ds = ds.downsample(channel_names=["cospectra"], target_depths=[128], methods=["mean"])
    ds1, ds2 = ds.split(cutoff=0.7)


    """
    trainer = Trainer(MultiNet, ds1, ds2, model_name="MultiNet_13CO_max_wout")
    trainer.training_set = ds1
    trainer.validation_set = ds2
    trainer.network_settings["base_filters"] = 64
    trainer.network_settings["convBlock"] = DoubleConvBlock
    trainer.network_settings["num_layers"] = 4
    #trainer.network_settings["out_channels"] = 2
    #trainer.network_settings["deeper_skips"] = True
    trainer.network_settings["channel_dimensions"] = [2]
    trainer.network_settings["channel_modes"] = [None]
    trainer.training_random_transform = False
    trainer.network_settings["attention"] = True
    trainer.optimizer_name = "Adam"
    trainer.target_names = ["vdens"]
    trainer.input_names = ["cdens"]
    import numpy as np
    #y_train = np.array(ds1.get_element_index("vdens"))
    #num_bins = 100 
    #hist, bin_edges = np.histogram(y_train, bins=num_bins, density=False)
    #bin_weights = 1.0 / (hist + 1)
    #trainer.loss_method = WeightedMSELoss(bin_edges,bin_weights)
    trainer.init()
    trainer.train(1500,batch_number=4,compute_validation=10)
    trainer.save()
    #trainer.plot()
    #trainer.plot_validation()
    #plot_models_accuracy([trainer,trainer2])
    """

    lis = ["MultiNet_13CO_max","MultiNet_13CO_max_proj","MultiNet_13CO_max_moments","MultiNet_13CO_max_wout"]
    ts = [load_trainer(l) for l in lis]
    for t in ts:
        t.model_name = t.model_name.replace("_max","")
    fig, ax = plot_models_accuracy(ts, show_errors=True)
    fig2, ax2, _ = plot_models_residuals(ts)
        
    #fig.savefig(os.path.join(FIGURE_FOLDER,"multinet_accuracy_max.jpg"))
    #fig2.savefig(os.path.join(FIGURE_FOLDER,"multinet_residuals_max.jpg"))

    #trainer = load_trainer("UneK_highres_fingercrossed")
    #fig, ax=  trainer.plot_residuals()
    #fig.savefig(os.path.join(FIGURE_FOLDER,"unek_residuals_notfitted.jpg"))    


    """
    #trainer = load_trainer("UneK_highres_fingercrossed")  
    #trainer.validation_set = ds2  
    #trainer.plot()
    trainer.create_baseline()

    batch = trainer.get_prediction_batch()
    reconstructed_batch = []
    for b in batch:
        pred = b[1]
        pred = trainer.apply_baseline(pred)
        reconstructed_batch.append((b[0], pred))
    trainer.prediction_batch = reconstructed_batch
    #trainer.plot()
    #trainer.plot_validation()
    fig, ax=  trainer.plot_residuals()
    fig.savefig(os.path.join(FIGURE_FOLDER,"unek_residuals_fitted.jpg"))

    """

    plt.show()