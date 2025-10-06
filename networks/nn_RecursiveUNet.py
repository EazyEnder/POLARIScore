import os
import sys

from ..config import LOGGER
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_UNet import UNet
from .nn_BaseModule import BaseModule
from .nn_MultiNet import MultiNet

class Recursive_UNet(BaseModule):
    def __init__(self,num_steps=5,**kwargs):
        super(Recursive_UNet, self).__init__()

        self.unet = MultiNet(channel_dimensions=[2,2], attention=True)
        self.num_steps = max(int(num_steps),1)

    def forward(self, x):
        
        input = [x,torch.zeros_like(x)]
        for i in range(self.num_steps):
            pred = self.unet(*input)
            input = [x, pred]

        return pred