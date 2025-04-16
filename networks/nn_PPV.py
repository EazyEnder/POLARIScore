import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LOGGER
from networks.nn_BaseModule import BaseModule
from networks.nn_UNet import UNet, AttentionBlock
import numpy as np

class PPV(BaseModule):
    def __init__(self, **kwargs):
        super(PPV, self).__init__()
        
        self.unet2d_col = UNet(**kwargs, is3D=False)
        self.unet2d_moment_0 = UNet(**kwargs, is3D=False)
        self.unet2d_moment_1 = UNet(**kwargs, is3D=False)
        self.unet2d_moment_2 = UNet(**kwargs, is3D=False)

        self.velocity_projection = nn.Conv2d(128,1, kernel_size=1)
        #self.velocity_unet = UNet(**kwargs, is3D=False)

        #self.unet2d_final = UNet(**kwargs, is3D=False, in_channels=5, out_channels=1)

        self.tile_channels = nn.Conv2d(4, 1, kernel_size=1)
        #with torch.no_grad():
        #    self.tile_channels.weight.fill_(0.25)
         #   self.tile_channels.bias.zero_()

    def compute_moments(self, v, v_axis):
        moment0 = v.sum(dim=-1)
        moment1 = (v * v_axis).sum(dim=-1) / (moment0 + 1e-6)
        moment2 = ((v_axis - moment1.unsqueeze(-1)) ** 2 * v).sum(dim=-1) / (moment0 + 1e-6)
        return [moment0, moment1, moment2]
    
    def forward(self, x, v):
        
        B, C, H, W, V = v.shape
        device = v.device

        v_axis = torch.linspace(-32, 32, V, device=device).view(1, 1, 1, 1, V)
        moments = self.compute_moments(v, v_axis)

        col_feat = F.relu(self.unet2d_col(x))
        #vel_feat = self.velocity_unet(self.velocity_projection(v.view(B, C*V, H, W)))
        moment0_feat = F.relu(self.unet2d_moment_0(moments[0]))
        moment1_feat = F.relu(self.unet2d_moment_1(moments[1]))
        moment2_feat = F.relu(self.unet2d_moment_2(moments[2]))

        out =  self.tile_channels(torch.cat([col_feat, moment0_feat, moment1_feat, moment2_feat], dim=1))
        #out = self.unet2d_final(features_2d)
        #probs = torch.sigmoid(out)
        #print(probs.min().item(), probs.max().item(), probs.mean().item(), end="")
        #print("Weights:", self.tile_channels.weight.view(-1).detach().cpu().numpy(), end="")
        return out

    def shape_data(self, batch, target_index=1, input_indexes=[0,2]):
        density_input_tensor = torch.from_numpy(np.log(np.array([b[input_indexes[0]] for b in batch]))).float().unsqueeze(1).to(self.device)
        velocity_input_tensor = torch.from_numpy((np.array([b[input_indexes[1]] for b in batch]))).float().unsqueeze(1).to(self.device)
        t_tensor = torch.from_numpy(np.log(np.array([b[target_index] for b in batch]))).float().unsqueeze(1).to(self.device)
        #target_tensor = (t_tensor > 4.).float().to(self.device)
        return [density_input_tensor,velocity_input_tensor], t_tensor

"""
class Test(BaseModule):
    def __init__(self, **kwargs):
        super(Test, self).__init__()
        
        self.unet = UNet(**kwargs, is3D=False)
    
    def forward(self, x):

        x = torch.sum(x, dim=-1)

        out = self.unet(x)
        return out

    def shape_data(self, batch, target_index=1, input_indexes=2):
        input_tensor = torch.from_numpy((np.array([b[input_indexes] for b in batch]))).float().unsqueeze(1).to(self.device)
        t_tensor = torch.from_numpy(np.log(np.array([b[target_index] for b in batch]))).float().unsqueeze(1).to(self.device)
        #target_tensor = (t_tensor > 4.).float().to(self.device)
        return [input_tensor], t_tensor
"""
    
class Test(BaseModule):
    def __init__(self, **kwargs):
        super(Test, self).__init__()
        
        self.unet = UNet(**kwargs, is3D=True)
    
    def forward(self, x, v):

        W = torch.sum(v, dim=-1, keepdim=True)
        out = F.relu(self.unet(x.unsqueeze(-1)*v/(W+1e-8)))
        out = out
        return out

    def shape_data(self, batch, target_index=1, input_indexes=[0,2]):
        density_input_tensor = torch.from_numpy(np.log(np.array([b[input_indexes[0]] for b in batch]))).float().unsqueeze(1).to(self.device)
        velocity_input_tensor = torch.from_numpy((np.array([b[input_indexes[1]] for b in batch]))).float().unsqueeze(1).to(self.device)
        t_tensor = torch.from_numpy(np.log(np.array([b[target_index] for b in batch]))).float().unsqueeze(1).to(self.device)
        target_tensor = (t_tensor > 3.).float().to(self.device)
        return [density_input_tensor, velocity_input_tensor], target_tensor
    
if __name__ == "__main__":
    model = PPV().to("cuda")
    x = torch.randn(1, 1, 128, 128).to("cuda")
    v = torch.randn(1, 1, 128, 128, 128).to("cuda")
    print(model(*[x, v]).device)