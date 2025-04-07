import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LOGGER
from nn_BaseModule import BaseModule
from nn_UNet import UNet, AttentionBlock
import numpy as np
class PPV(BaseModule):
    def __init__(self, **kwargs):
        super(PPV, self).__init__()
        
        self.unet_density = UNet(**kwargs, is3D=False)
        self.unet_velocity = UNet(**kwargs, is3D=True)

        self.fusion_block = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(1, 1, kernel_size=1)
        )
    
    def forward(self, x, v):
        
        x = self.unet_density(x).unsqueeze(-1).expand(-1,-1,-1, -1, v.shape[-3])
        v = self.unet_velocity(v)

        combined = torch.cat([x, v], dim=1)
        n = self.fusion_block(combined) 
    
        return n

    def shape_data(self, batch, target_index=1, input_indexes=[0,2]):
        density_input_tensor = torch.from_numpy(np.log(np.array([b[input_indexes[0]] for b in batch]))).float().unsqueeze(1).to(self.device)
        velocity_input_tensor = torch.from_numpy((np.array([b[input_indexes[1]] for b in batch]))).float().unsqueeze(1).to(self.device)
        target_tensor = torch.from_numpy(np.log(np.array([b[target_index] for b in batch]))).float().unsqueeze(1).to(self.device)
        return [density_input_tensor,velocity_input_tensor], target_tensor
    
if __name__ == "__main__":
    model = PPV()
    x = torch.randn(1, 1, 128, 128)
    v = torch.randn(1, 1, 128, 128, 128)
    print(model(*[x, v]).shape)