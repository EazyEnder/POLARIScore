import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from training_batch import *
from config import *
import uuid
from nn_UNet import UNet
from nn_FCN import FCN
from nn_KNet import KNet
import json

class Trainer():
    def __init__(self,network=None,training_batch=None,validation_batch=None,training_batch_name=None,validation_batch_name=None,learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = str(uuid.uuid4())
        
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


        self.network = network
        self.model = None
        self.optimizer = None
        self.scheduler = None
        

        if not(self.network is None):
            self.model = network().to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.75, threshold=0.005)

        self.loss_method = nn.MSELoss()
        self.training_losses = []
        self.validation_losses = []
        self.last_epoch = 0

    def init(self, model=None):
        if self.network or not(model is None):
            self.network_type = self.network.__name__
            if model is None:
                self.model = self.network().to(self.device)
            else:
                self.model = model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.75, threshold=0.005)
            return True
        return False

    def train(self, epoch_number=50, compute_validation=10):
        training_input_tensor = torch.from_numpy(np.pad(np.log(np.array([b[0] for b in self.training_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        training_target_tensor = torch.from_numpy(np.pad(np.log(np.array([b[1] for b in self.training_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        validation_input_tensor = torch.from_numpy(np.pad(np.log(np.array([b[0] for b in self.validation_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        validation_target_tensor = torch.from_numpy(np.pad(np.log(np.array([b[1] for b in self.validation_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        
        self.model.train()

        total_epoch = self.last_epoch
        for epoch in range(epoch_number):
            total_epoch += 1
            self.optimizer.zero_grad()
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
            print(f'Epoch {total_epoch}/{self.last_epoch + epoch_number}, Training Loss: {loss.item()}, Validation loss: {v_loss if v_loss else "Not computed"}')
        self.last_epoch = total_epoch
        self.learning_rate = self.scheduler.get_last_lr()[0]

    def plot_losses(self, ax=None ,log10=True):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        x_training = [i[0] for i in self.training_losses]
        x_validation = [i[0] for i in self.validation_losses]
        y_training = [i[1] for i in self.training_losses]
        y_validation = [i[1] for i in self.validation_losses]
        
        if log10:
            y_training = np.log10(y_training)
            y_validation = np.log10(y_validation)

        ax.scatter(x_training, y_training, label="training losses")
        ax.scatter(x_validation, y_validation, label="validation losses")
        ax.plot(x_training, y_training)
        ax.plot(x_validation, y_validation)

        ax.set_xlabel("epoch")

        loss_method_name = getattr(self.loss_method, "_get_name", lambda: "Custom")()
        ax.set_ylabel(loss_method_name)

        ax.legend()
        
        return fig, ax

    def get_validation_batch(self):
        self.model.eval()
        validation_input_tensor = torch.from_numpy(np.pad(np.log(np.array([b[0] for b in self.validation_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        validation_target_tensor = torch.from_numpy(np.pad(np.log(np.array([b[1] for b in self.validation_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        validation_output = self.model(validation_input_tensor)

        validation_target_tensor, validation_output = validation_target_tensor.cpu().detach().numpy(), validation_output.cpu().detach().numpy()
        validation_batch = rebuild_batch(np.exp(validation_target_tensor[:,0,:,:]), np.exp(validation_output[:,0,:,:]))
    
        return validation_batch

    def plot_validation(self):
        plot_batch(self.get_validation_batch(), same_limits=True)

    def plot_residuals(self, ax, violinplot=True, color="blue", bins_inter=(None,None)):
        if ax is None:
                    fig, ax = plt.subplots()
        else:
            fig = ax.figure
        batch = self.get_validation_batch()
        d_prediction = np.array([np.log(b[0])/np.log(10) for b in batch]).flatten()
        d_target = np.array([np.log(b[1])/np.log(10) for b in batch]).flatten()

        residuals = d_prediction-d_target
        violin_num_bins = 5
        bins = np.linspace(min(d_target) if bins_inter[0] is None else bins_inter[0], max(d_target) if bins_inter[1] is None else bins_inter[1], violin_num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2 
        bin_indices = np.digitize(d_target, bins) - 1 
        binned_residuals = [residuals[bin_indices == i] for i in range(violin_num_bins)]
        if violinplot:
            vp = ax.violinplot(binned_residuals, positions=bin_centers, showmeans=False, showmedians=True)
            for i, body in enumerate(vp['bodies']):
                body.set_facecolor(color)
                body.set_alpha(0.5)
            for part in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
                vp[part].set_edgecolor('black')
                vp[part].set_linewidth(1.0)
        mean_residuals = [np.mean(residuals[bin_indices == i]) for i in range(violin_num_bins)]
        ax.axhline(0, color='red', linestyle='--')
        ax.plot(bin_centers, mean_residuals, marker='o', linestyle='-', color='black', alpha=0.7)

        ax.set_xticks(bin_centers)
        ax.set_xlabel("Pixel value (log10)")
        ax.set_ylabel("Residuals (prediction-target)")
        ax.grid(True, linestyle="--", alpha=0.5)

        return fig, ax

    def plot(self):
        batch = self.get_validation_batch()
        plt.subplot(2,2,1)
        ax1 = plt.subplot(2,2,1)
        plot_batch_correlation(batch, ax=ax1)
        ax1.set_xlabel("Prediction")
        ax1.set_ylabel("Target")

        ax2 = plt.subplot(2,2,2)
        self. plot_residuals(ax=ax2)

        ax3 = plt.subplot(2,2,3)
        self.plot_losses(ax=ax3)

        plt.tight_layout()

    def save(self, model_name=None):
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

        settings = {
            "model_name": self.model_name,
            "network": self.network_type,
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

        return True

def load_trainer(model_name):

    model_path = os.path.join(MODEL_FOLDER, model_name)
    if(not(os.path.exists(model_path))):
        return
    
    settings = {}
    with open(os.path.join(model_path,'settings.json')) as file:
        settings = json.load(file)
    
    trainer = Trainer(validation_batch_name=settings["validation_batch"] if "validation_batch" in settings else None,training_batch_name=settings["training_batch"] if "training_batch" in settings else None)
    trainer.model_name = settings["model_name"]

    network_options = {"UNet" : UNet,
           "FCN" : FCN,
           "KNet" : KNet,
           "None": None
    }
    trainer.network = network_options[settings["network"]]
    trainer.learning_rate = float(settings["learning_rate"])
    trainer.last_epoch = int(settings["total_epoch"])
    trainer.training_losses = settings["training_losses"]
    trainer.validation_losses = settings["validation_losses"]

    model = trainer.network()
    model.load_state_dict(torch.load(os.path.join(model_path,trainer.model_name+".pth")))
    model.to(trainer.device)
    model.eval()
    trainer.init(model=model)

    return trainer

def plot_models_violinplot(trainers = [], ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    models_residuals = []
    models_targets = []
    for t in trainers:
        batch = t.get_validation_batch()
        d_prediction = np.array([np.log(b[0])/np.log(10) for b in batch]).flatten()
        d_target = np.array([np.log(b[1])/np.log(10) for b in batch]).flatten()
        models_residuals.append(d_prediction-d_target)
        models_targets.append(d_target)
    models_name = [t.model_name for t in trainers]

    positions = np.arange(len(models_name))

    vp = ax.violinplot(models_residuals, positions=positions, showmeans=True, showmedians=True)
    colors = cm.jet(np.linspace(0, 1, len(models_name)))
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

def plot_models_comparisons(trainers = []):
    ax1 = plt.subplot(len(trainers),2, (1,len(trainers)))
    _,_, colors = plot_models_violinplot(trainers=trainers, ax=ax1)
    ax1.set_title("Comparison of different models")

    for i,t in enumerate(trainers):
        ax = plt.subplot(len(trainers),2, (i+1)*2)
        t.plot_residuals(ax=ax, color=colors[i], bins_inter=(1.5,7.5))
        ax.set_ylim((-1.25, 1.25))

    plt.tight_layout()

if __name__ == "__main__":
    #batch = open_batch("batch_37392b55-be04-4e8c-aa49-dca42fa684fc")
    #train_b, validation_b = split_batch(batch, cutoff=0.7)

    trainer_batch = open_batch("batch_37392b55-be04-4e8c-aa49-dca42fa684fc")
    validation_b = open_batch("batch_92b49d92-369a-45a0-b4eb-385658b05f41")
    
    trainer_knet = load_trainer("KNet")
    trainer__unet = load_trainer("UNet")
    trainer_knet_customloss = load_trainer("KNet_customloss")

    trainer_list = [trainer__unet, trainer_knet, trainer_knet_customloss]
    for t in trainer_list:
        t.training_batch = trainer_batch
        t.validation_batch = validation_b
    #trainer = Trainer(KNet, train_b, validation_b)
    #trainer.model_name = "test_kan_cutoffbatch"

    def custom_loss(output, target):
        weights = torch.ones_like(target)      
        return torch.mean((output - target) ** 2)+torch.var((output - target) ** 2)
    
    #trainer.loss_method = custom_loss

    #trainer.train(500)
    #trainer.save()

    #plot_batch(train_b)
    #plot_batch_correlation(trainer.get_validation_batch())
    #trainer.plot_validation()
    #trainer.plot()
    plot_models_comparisons(trainer_list)
    plt.show()