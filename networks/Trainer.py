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
from training_batch import *
from config import *
import uuid
from nn_UNet import UNet
from nn_FCN import FCN
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

    def plot_losses(self, log10=True):
        plt.figure()

        x_training = [i[0] for i in self.training_losses]
        x_validation = [i[0] for i in self.validation_losses]
        y_training = [i[1] for i in self.training_losses]
        y_validation = [i[1] for i in self.validation_losses]
        if log10:
            y_training = np.log(y_training)/np.log(10)
            y_validation = np.log(y_validation)/np.log(10)
        plt.scatter(x_training, y_training, label="training losses")
        plt.scatter(x_validation, y_validation, label="validation losses")
        plt.plot(x_training, y_training)
        plt.plot(x_validation, y_validation)

        plt.xlabel("epoch")
        loss_method_name = ""
        try:
            loss_method_name = self.loss_method._get_name()
        except AttributeError:
            loss_method_name = "Custom"
        plt.ylabel(loss_method_name)

        plt.legend()

    def plot_validation(self):

        self.model.eval()
        validation_input_tensor = torch.from_numpy(np.pad(np.log(np.array([b[0] for b in self.validation_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        validation_target_tensor = torch.from_numpy(np.pad(np.log(np.array([b[1] for b in self.validation_batch])),(0,0))).float().unsqueeze(1).to(self.device)
        validation_output = self.model(validation_input_tensor)

        validation_target_tensor, validation_output = validation_target_tensor.cpu().detach().numpy(), validation_output.cpu().detach().numpy()
        validation_batch = rebuild_batch(np.exp(validation_target_tensor[:,0,:,:]), np.exp(validation_output[:,0,:,:]))

        plot_batch(validation_batch, same_limits=True)

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

if __name__ == "__main__":
    train_b = open_batch("batch_37392b55-be04-4e8c-aa49-dca42fa684fc")
    validation_b = open_batch("batch_92b49d92-369a-45a0-b4eb-385658b05f41")
    trainer = load_trainer("test")
    trainer.training_batch = train_b
    trainer.validation_batch = validation_b
    #trainer = Trainer(UNet, train_b, validation_b)
    #trainer.model_name = "test"

    def custom_loss(output, target):
        weights = torch.ones_like(target)      
        return torch.mean((output - target) ** 2)+torch.var((output - target) ** 2)
    
    trainer.loss_method = custom_loss

    trainer.train(500)
    trainer.save()

    #trainer.plot_validation()
    trainer.plot_losses()
    plt.show()