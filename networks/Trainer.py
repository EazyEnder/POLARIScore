import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from sympy import limit_seq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from training_batch import *
from config import *
import uuid
from nn_UNet import *
from nn_FCN import FCN
from nn_KNet import FullKNet, KNet
from networks.utils.nn_utils import compute_batch_accuracy
import json
class Trainer():
    def __init__(self,network=None,training_batch=None,validation_batch=None,training_batch_name=None,validation_batch_name=None,learning_rate=0.001,model_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = model_name
        if model_name is None:
            self.model_name = str(uuid.uuid4())

        LOGGER.log(f"{self.device} is used for {self.model_name}")
        
        self.network_type = "None"
        if not(network is None):
            self.network_type = network.__name__
        self.learning_rate = learning_rate
        self.training_batch = training_batch
        self.validation_batch = validation_batch
        self.training_batch_name = training_batch_name
        self.validation_batch_name = validation_batch_name
        if not(training_batch_name is None or training_batch_name=="None"):
            self.training_batch = open_batch(training_batch_name)
        if not(validation_batch_name is None or validation_batch_name=="None"):
            self.validation_batch_name = open_batch(validation_batch_name)

        self.prediction_batch = None


        self.network = network
        self.network_settings ={}

        self.model = None
        self.optimizer = None
        self.scheduler = None
        

        if not(self.network is None):
            self.model = network(**self.network_settings).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.75, threshold=0.005)

        self.loss_method = nn.MSELoss()
        self.training_losses = []
        self.validation_losses = []
        self.last_epoch = 0

    def init(self, model=None):
        if not(self.network is None) or not(model is None):
            self.network_type = self.network.__name__
            if model is None:
                self.model = self.network(**self.network_settings).to(self.device)
            else:
                self.model = model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.75, threshold=0.005)
            return True
        LOGGER.warn(f"Can't init model {self.model_name}, check if network is defined or model is not None.")
        return False

    def train(self, epoch_number=50, compute_validation=10, auto_stop=0., auto_stop_min_loss=2.):
        """
        Train the model (check trainer variables for settings)

        Args:
            epoch_number(int, default: 50): train the model for x epochs.
            compute_validation(int, default:10): compute validation losses each x epochs.

        """
        LOGGER.log(f"Training started with {str(epoch_number)} epochs on network {self.network_type}")

        training_input_tensor = torch.from_numpy(np.pad(np.log(np.array([b[0] for b in self.training_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        training_target_tensor = torch.from_numpy(np.pad(np.log(np.array([b[1] for b in self.training_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        validation_input_tensor = torch.from_numpy(np.pad(np.log(np.array([b[0] for b in self.validation_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        validation_target_tensor = torch.from_numpy(np.pad(np.log(np.array([b[1] for b in self.validation_batch])),(0,0))).float().unsqueeze(1).to(self.device)        
        self.model.train()

        def random_transform(tensors_input, tensors_target):
            k = torch.randint(0, 4, (1,)).item()
            tensors_input = torch.rot90(tensors_input, k, [2, 3])
            tensors_target = torch.rot90(tensors_target, k, [2, 3])

            # Apply random vertical flip
            if torch.rand(1).item() > 0.5:
                tensors_input = torch.flip(tensors_input, [2])
                tensors_target = torch.flip(tensors_target, [2])

            # Apply random horizontal flip
            if torch.rand(1).item() > 0.5:
                tensors_input = torch.flip(tensors_input, [3])
                tensors_target = torch.flip(tensors_target, [3])

            return tensors_input, tensors_target

        total_epoch = self.last_epoch
        break_flag = False
        for epoch in range(epoch_number):
            total_epoch += 1
            self.optimizer.zero_grad()
            #training_input_tensor, training_target_tensor = random_transform(training_input_tensor, training_target_tensor)
            output = self.model(training_input_tensor)
            loss = self.loss_method(output, training_target_tensor)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            self.training_losses.append((total_epoch, loss.item()))
            v_loss = None
            if compute_validation>0 and total_epoch % compute_validation == 0:
                self.model.eval()
                validation_output = self.model(validation_input_tensor)
                v_loss = self.loss_method(validation_output,validation_target_tensor).item()
                self.validation_losses.append((total_epoch,v_loss))
                self.model.train()
                if(auto_stop > 0 and np.abs(v_loss-loss.item()) < auto_stop and v_loss < auto_stop_min_loss):
                    break_flag = True
            LOGGER.print(f'Epoch {total_epoch}/{self.last_epoch + epoch_number}, Training Loss: {loss.item()}, Validation loss: {v_loss if v_loss else "Not computed"}', type="training", level=1)
            if break_flag:
                break
        self.last_epoch = total_epoch
        self.learning_rate = self.scheduler.get_last_lr()[0]

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

        self.model.eval()
        validation_input_tensor = torch.from_numpy(np.pad(np.log(np.array([b[0] for b in self.validation_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        validation_target_tensor = torch.from_numpy(np.pad(np.log(np.array([b[1] for b in self.validation_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        validation_output = self.model(validation_input_tensor)

        validation_target_tensor, validation_output = validation_target_tensor.cpu().detach().numpy(), validation_output.cpu().detach().numpy()
        validation_batch = rebuild_batch(np.exp(validation_target_tensor[:,0,:,:]), np.exp(validation_output[:,0,:,:]))

        self.prediction_batch = validation_batch
    
        return validation_batch

    def plot_validation(self):
        """
        Show target and model prediction images
        """
        plot_batch(self.get_prediction_batch(), same_limits=True)

    def plot_residuals(self, ax, plot_distribution=True, color="blue", bins_inter=(None,None)):
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
    
    def predict(self, batch):
        self.model.eval()
        input_tensor = torch.from_numpy(np.pad(np.log(np.array([b[0] for b in batch])),(0,0))).float().unsqueeze(1).to(self.device)
        target_tensor = torch.from_numpy(np.pad(np.log(np.array([b[1] for b in batch])),(0,0))).float().unsqueeze(1).to(self.device)
        output = self.model(input_tensor)

        target_tensor, output = target_tensor.cpu().detach().numpy(), output.cpu().detach().numpy()
        result_batch = rebuild_batch(np.exp(target_tensor[:,0,:,:]), np.exp(output[:,0,:,:]))
        return result_batch

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

        #line = plt.Line2D([0.5, 0.5], [0, 1], transform=fig.transFigure, color="black", linewidth=2)
        #fig.add_artist(line)

    def plot(self):
        """
        Plot all in one figure: validation correlation, residuals and losses.
        """
        plt.figure()
        plt.suptitle(self.model_name)
        batch = self.get_prediction_batch()
        plt.subplot(2,2,1)
        ax1 = plt.subplot(2,2,1)
        plot_batch_correlation(batch, ax=ax1)
        ax1.set_xlabel("Target")
        ax1.set_ylabel("Prediction")

        ax2 = plt.subplot(2,2,2)
        self. plot_residuals(ax=ax2)

        ax3 = plt.subplot(2,2,3)
        self.plot_losses(ax=ax3)

        plt.tight_layout()

    def save(self, model_name=None):
        """
        Save model and model settings in a new folder

        Args:
            model_name (str, optional): set model name, if None it will take the trainer model_name if user had set it or a random uuid

        Returns:
            bool: If this is a success
        """
        if(not(model_name is None)):
            self.model_name = model_name
        
        if not(os.path.exists(MODEL_FOLDER)):
            os.mkdir(MODEL_FOLDER)

        while os.path.exists(os.path.join(MODEL_FOLDER,str(uuid.uuid4()))):
            self.model_name = str(uuid.uuid4())

        model_path = os.path.join(MODEL_FOLDER,self.model_name.rsplit("_ver",1)[0])
        if not(os.path.exists(model_path)):
            os.mkdir(model_path)

        v = 1
        while(os.path.exists(os.path.join(model_path,self.model_name+".pth"))):
            self.model_name = self.model_name.rsplit("_ver",1)[0]+"_ver"+str(v)
            v += 1
        torch.save(self.model.state_dict(),os.path.join(model_path,self.model_name+".pth"))

        loss_method_name = ""
        try:
            loss_method_name = self.loss_method._get_name()
        except AttributeError:
            loss_method_name = "Custom"

        if "convBlock" in self.network_settings:
            self.network_settings["convBlock"] = self.network_settings["convBlock"].__name__

        settings = {
            "model_name": self.model_name,
            "network": self.network_type,
            "network_settings": self.network_settings,
            "loss_method": loss_method_name,
            "optimizer": str(type(self.optimizer)),
            "learning_rate": str(self.learning_rate),
            "scheduler": str(type(self.scheduler)),
            "total_epoch": str(self.last_epoch),
            "training_batch": str(self.training_batch_name),
            "validation_batch": str(self.validation_batch_name),
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
        }

        with open(os.path.join(model_path,'settings.json'), 'w') as file:
            json.dump(settings, file, indent=4)

        LOGGER.log(f"{self.model_name} saved")

        return True

def load_trainer(model_name, load_model=True):

    model_path = os.path.join(MODEL_FOLDER, model_name)
    if(not(os.path.exists(model_path))):
        LOGGER.error(f"Can't load {model_name}, file {model_path} doesn't exist.")
        return
    
    settings = {}
    with open(os.path.join(model_path,'settings.json')) as file:
        settings = json.load(file)
    
    trainer = Trainer(model_name=settings["model_name"],validation_batch_name=settings["validation_batch"] if "validation_batch" in settings else None,training_batch_name=settings["training_batch"] if "training_batch" in settings else None)

    network_options = {"UNet" : UNet,
           "FCN" : FCN,
           "KNet" : KNet,
           "FullKNet": FullKNet,
           "None": None
    }
    network_convblock_options = {"DoubleConvBlock": DoubleConvBlock,"ResConvBlock":ResConvBlock}
    network_settings = settings["network_settings"] if "network_settings" in settings else {}
    if "convBlock" in network_settings:
        network_settings["convBlock"] = network_convblock_options[network_settings["convBlock"]]

    trainer.network_type = settings["network"]
    trainer.network_settings = settings["network_settings"] if "network_settings" in settings else {}
    trainer.network = network_options[settings["network"]]
    trainer.learning_rate = float(settings["learning_rate"])
    trainer.last_epoch = int(settings["total_epoch"])
    trainer.training_losses = settings["training_losses"]
    trainer.validation_losses = settings["validation_losses"]

    if load_model:
        model = trainer.network(**trainer.network_settings)
        model.load_state_dict(torch.load(os.path.join(model_path,trainer.model_name+".pth")))
        model.to(trainer.device)
        model.eval()
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
    plt.figure()

    ax1 = plt.subplot2grid((len(trainers), 2), (0, 0), rowspan=len(trainers))
    _,_, colors = plot_models_residuals(trainers=trainers, ax=ax1)
    ax1.set_title("Comparison of different models")

    for i,t in enumerate(trainers):
        ax = plt.subplot(len(trainers),2, (i+1)*2)
        t.plot_residuals(ax=ax, color=colors[i], bins_inter=(1.5,7.5))
        ax.set_ylim((-1.25, 1.25))
        ax.set_ylabel("")

    plt.tight_layout()

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

if __name__ == "__main__":
    batch = open_batch("batch_37392b55-be04-4e8c-aa49-dca42fa684fc")
    train_batch = batch
    validation_batch = open_batch("batch_92b49d92-369a-45a0-b4eb-385658b05f41")

    trainer_knet = load_trainer("KNet")

    trainer_list = [trainer_knet]
    for t in trainer_list:
        t.training_batch = train_batch
        t.validation_batch = validation_batch

    def custom_loss(output, target):
        weights = target.clone()/torch.max(target)
        return nn.HuberLoss()(output,target)+torch.mean((output - target) ** 2)

    
    trainer = Trainer(KNet, train_batch, validation_batch, model_name="KNet_Test")
    trainer.network_settings["base_filters"] = 64
    #trainer.network_settings["convBlock"] = Kan
    trainer.network_settings["num_layers"] = 4
    trainer.init()
    trainer.train(500)
    trainer_list.append(trainer)

    """from objects.Simulation_DC import Simulation_DC
    #sim_MHD = Simulation_DC(name="orionMHD_lowB_0.39_512", global_size=66.0948)
    #sim_HD = Simulation_DC(name="orionHD_all_512", global_size=66.0948)
    #sim_MHD.plot_correlation(method=compute_mass_weighted_density)
    #sim_HD.plot_correlation(method=compute_mass_weighted_density)
    #sim_HD.generate_batch(,number=64,force_size=128,)
    #sim_HD.plot()"""

    trainer.plot()
    trainer.plot_validation()
    plot_models_accuracy(trainer_list, show_errors=True)
    plot_models_residuals_extended(trainer_list)
    plt.show()