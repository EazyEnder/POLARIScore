import os
import sys

from config import LOGGER
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.utils.fastkanconv import FastKANConvLayer
from networks.nn_UNet import ConvBlock, UNet
from networks.nn_BaseModule import BaseModule

    
class KanConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,is3D=False):
        super(KanConvBlock, self).__init__()
        self.conv = nn.Sequential(
            FastKANConvLayer(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.conv(x)

class KNet(UNet):
    """Same as a UNet network but the final convolution layer is a Kolmogorov Arnold layer"""
    def __init__(self, convBlock=ConvBlock, num_layers=4, base_filters=64,**kwargs):
        super(KNet, self).__init__(convBlock=convBlock, num_layers=num_layers, base_filters=base_filters, **kwargs)
        self.final_conv = FastKANConvLayer(base_filters, 1, kernel_size=1)
        
from kan import KAN
class UneK(BaseModule):
    def __init__(self, multiply=False, **kwargs):
        super(UneK, self).__init__()
        self.multiply = multiply
        self.unet = UNet(**kwargs)
        #old one is just two layers of [10] and [10]
        self.kan = KAN(width=[1,[5,5],[5,5],[5,5],[5,5],1], grid=5, k=3, seed=1, device='cuda' if torch.cuda.is_available() else 'cpu', auto_save=False)
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.reshape(B*C*H*W, 1)
        x_kan = self.kan(x_flat)
        x_kan = x_kan.reshape(B, C, H, W)
        x_unet = self.unet(x)
        if self.multiply:
            x = x_kan * x_unet
        else:
            x = x_kan + x_unet
        return x
    def getKAN(self):
        kan = JustKAN(num_layers=4, op_per_layers=5)
        kan.kan = self.kan
        return kan
    
class JustKAN(BaseModule):
    def __init__(self, in_channels=1, out_channels=1, num_layers=2, op_per_layers=5):
        super(JustKAN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        width = [in_channels]
        width.extend([[op_per_layers, op_per_layers] for n in range(num_layers)])
        width.extend([out_channels])
        self.kan = KAN(width=width, grid=5, k=3, seed=1, device='cuda' if torch.cuda.is_available() else 'cpu', auto_save=False)
    def forward(self, x):
        if len(x.shape) < 5:
            B, C, H, W = x.shape
            assert C == self.in_channels, LOGGER.error(f"KAN can't work because mismatch in channel number ({C} instead of {self.in_channels})")
            x_flat = x.reshape(B*H*W, C)
            x_kan = self.kan(x_flat)
            x_kan = x_kan.reshape(B, self.out_channels, H, W)
        else:
            B, C, H, W, D = x.shape
            assert C == self.in_channels, LOGGER.error("KAN can't work because mismatch in channel number")
            x_flat = x.reshape(B*H*W*D, C)
            x_kan = self.kan(x_flat)
            x_kan = x_kan.reshape(B, self.out_channels, H, W, D)
        return x_kan


