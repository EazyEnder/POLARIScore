import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LOGGER
from torch.nn import init
from networks.utils.nn_utils import xavier_init
from typing import Union, Literal

"""
Some saved models use the old version of early May 2025.
Check https://github.com/EazyEnder/POLARIScore/blob/e247372b4617c842894e4d91e0b4e1ec54261505/networks/nn_UNet.py 
If you want to load and use the models, just replace this file by the github file.
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels:int=1, out_channels:int=1, is3D:bool=False):
        super(ConvBlock, self).__init__()
        c = nn.Conv2d
        b = nn.BatchNorm2d
        d = nn.Dropout2d
        if is3D:
            c = nn.Conv3d
            b = nn.BatchNorm3d
            d = nn.Dropout3d
        self.conv = nn.Sequential(
            c(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            b(out_channels),
            d(p=0.05),
        )

        self.initialize()

    def initialize(self):
        xavier_init(self)
    def forward(self, x):
        return self.conv(x)
    
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels:int=1, out_channels:int=1, is3D:bool=False):
        super(DoubleConvBlock, self).__init__()
        c = nn.Conv2d
        b = nn.BatchNorm2d
        d = nn.Dropout2d
        if is3D:
            c = nn.Conv3d
            b = nn.BatchNorm3d
            d = nn.Dropout3d
        self.conv = nn.Sequential(
            c(in_channels, out_channels, kernel_size=3, padding=1),
            b(out_channels),
            nn.ReLU(),
            c(out_channels, out_channels, kernel_size=3, padding=1),
            b(out_channels),
            nn.ReLU(),
            d(p=0.05)
        )

        self.initialize()

    def initialize(self):
        xavier_init(self)
    
    def forward(self, x):
        return self.conv(x)
    
class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is3D):
        super(ResConvBlock, self).__init__()
        c = nn.Conv2d
        b = nn.BatchNorm2d
        if is3D:
            c = nn.Conv3d
            b = nn.BatchNorm3d
        self.conv = nn.Sequential(
            c(in_channels, out_channels//2, kernel_size=3, padding=1),
            b(out_channels//2),
            nn.ReLU(inplace=False),
            c(out_channels//2, out_channels, kernel_size=3, padding=1),
            b(out_channels),
        )
        self.match_dim = c(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else None
    
    def forward(self, x):
        res = x.clone()
        x = self.conv(x)
        if self.match_dim:
            res = self.match_dim(res)
        x = x+res
        return F.relu(x)
    
class AttentionBlock(nn.Module):
    def __init__(self, F_g:int, F_l:int, F_int:int, is3D=False):
        """
        Args:
            F_g: dimensions of upsampled features
            F_l: dimensions of skip features
            F_int: number of hidden intermediary parameters in the block
        """
        super(AttentionBlock, self).__init__()

        c = nn.Conv2d
        b = nn.BatchNorm2d
        if is3D:
            c = nn.Conv3d
            b = nn.BatchNorm3d
        
        self.W_g = nn.Sequential(
            c(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            b(F_int)
        )

        self.W_x = nn.Sequential(
            c(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            b(F_int)
        )

        self.psi = nn.Sequential(
            c(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            b(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

        self.initialize()

    def initialize(self):
        xavier_init(self)

    def forward(self, g, x):
        g1 = self.W_g(g)  # Apply 1x1 Conv on upsampled feature
        x1 = self.W_x(x)  # Apply 1x1 Conv on encoder feature
        psi = self.relu(g1 + x1)  # Element-wise sum
        psi = self.psi(psi)  # Apply sigmoid activation
        return x * psi  # Scale encoder features

from networks.nn_BaseModule import BaseModule
class UNet(BaseModule):
    def __init__(self, convBlock:'nn.Module'=ConvBlock, deeper_skips:bool=False, num_layers:int=4, base_filters:int=64, in_channels:int=1, out_channels:Union[None,int]=None, convBlock_layer:int=None, filter_function:Literal['constant']='constant', k:float=2., attention:bool=True, is3D:bool=False):
        """
        UNet implementation using torch
        Args:
            convBlock: convblock used (like ConvBlock classic, DoubleConv, ResConv, KAN) 
            convBlock_layer (int): if not None: if depth > convBlock_layer use convBlock else use classic convBlock (default: use convBlock).
            num_layers (int): Number of layers in the UNet
            deeper_skips (bool): if enable deeper skips (not recommanded)
            base_filters (int): how many features per layer = base_filters*k^layer if layer begins at 0 where k is algo an arg(default 2) (and if filter_function is 'constant')
            k (float): for base_filters laws.
            in_channels (int): How many inputs (encoders branchs), for example if only column density, 'in_channels' is 1
            out_channels (int, default:None): How many outputs (decoders branchs)
            attention (bool): Enable attention blocks
            is3D (bool): is3D
        """
        
        super(UNet, self).__init__()

        self.num_layers = num_layers
        self.deeperskips = deeper_skips
        self.attention = attention
        self.is3D = is3D
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        convBlock_layer = num_layers if convBlock_layer is None else convBlock_layer
        convBlock_layer = num_layers - convBlock_layer
        self.convBlock_layer = convBlock_layer

        if filter_function == 'constant':
            filter_sizes = [int(base_filters * k**i) for i in range(num_layers+1)]
        else:
            raise ValueError("Invalid filter function type.")
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2) if is3D else nn.MaxPool2d(2,2)

        in_channels =  self.in_channels
        for i in range(num_layers):
            out_channels = filter_sizes[i]
            if i >= convBlock_layer:
                self.encoders.append(convBlock(in_channels, out_channels, is3D=is3D))
            else:
                self.encoders.append(DoubleConvBlock(in_channels, out_channels, is3D=is3D))
            in_channels = out_channels
        out_channels = filter_sizes[-1]

        # Bottleneck
        self.bottleneck = convBlock(in_channels, out_channels, is3D=is3D)
        in_channels = out_channels

        # (Multi) Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        reversed_filters = filter_sizes[::-1]

        for j in range(self.out_channels):
            self.upconvs.append(nn.ModuleList())
            self.decoders.append(nn.ModuleList())
            self.attentions.append(nn.ModuleList())

            in_channels = filter_sizes[-1]
            for i in range(num_layers):
                out_channels = reversed_filters[1+i]
                upconv = nn.ConvTranspose3d if is3D else nn.ConvTranspose2d
                self.upconvs[j].append(upconv(in_channels, out_channels, kernel_size=2, stride=2))
                if self.attention:
                    self.attentions[j].append(AttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels//2, is3D=is3D))
                concat_ch = out_channels * 2
                if i+2 <= num_layers and self.deeperskips:
                    concat_ch += reversed_filters[2+i]
                block = convBlock(concat_ch, out_channels, is3D=is3D) if num_layers-i >= convBlock_layer else DoubleConvBlock(concat_ch, out_channels, is3D=is3D)
                self.decoders[j].append(block)

                in_channels = out_channels

        # Output layer
        final_conv = nn.Conv3d if is3D else nn.Conv2d
        self.final_conv = nn.ModuleList()
        for o in range(self.out_channels):
            self.final_conv.append(final_conv(base_filters, 1, kernel_size=1))

        self.initialize()

    def initialize(self):
        for f in self.final_conv:
            init.xavier_uniform_(f.weight)
            init.zeros_(f.bias)
    
    def forward(self, x):

        x = super().forward(x)

        assert (self.is3D and len(x.shape) > 4) or (not(self.is3D) and len(x.shape) < 5), LOGGER.error(f"U-Net is defined as {'3D' if self.is3D else '2D'} but input has {len(x.shape)-2} dimensions")

        # Encoder forward pass
        enc_features = []
        for i in range(self.num_layers):
            x = self.encoders[i](x)
            enc_features.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder forward pass
        decoded_x = []
        interp_mode = 'trilinear' if self.is3D else 'bilinear'
        for j in range(self.out_channels):
            xj = x#.clone()
            for i in range(self.num_layers):
                xj = self.upconvs[j][i](xj)
                enc_feat = enc_features[-(i+1)]
                if self.attention:
                    enc_feat = self.attentions[j][i](xj, enc_feat)
                    
                skip_feats = [xj, enc_feat]
                if i+2 <= self.num_layers and self.deeperskips:
                    deeper_skip = enc_features[-(i+2)]
                    if deeper_skip.shape[2:] != xj.shape[2:]:
                        deeper_skip = F.interpolate(deeper_skip, size=xj.shape[2:], mode=interp_mode, align_corners=False)
                    skip_feats.append(deeper_skip)
            
                xj = torch.cat(skip_feats, dim=1)
                xj = self.decoders[j][i](xj)
            decoded_x.append(xj)
        
        # Output
        if self.out_channels > 1:
            return [self.final_conv[j](decoded_x[j]) for j in range(self.out_channels)]
        else:
            return self.final_conv[0](decoded_x[0])
    
if __name__ == "__main__":
    model = UNet(is3D=True)
    x = torch.randn(1, 1, 128, 128, 128)
    print(model(x).shape)