import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LOGGER

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is3D):
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
            nn.ReLU(inplace=True),
            b(out_channels),
            d(p=0.05),
        )
    
    def forward(self, x):
        return self.conv(x)
    
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is3D):
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
            nn.ReLU(inplace=True),
            c(out_channels, out_channels, kernel_size=3, padding=1),
            b(out_channels),
            nn.ReLU(inplace=True),
            d(p=0.05)
        )
    
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
    def __init__(self, F_g, F_l, F_int, is3D):
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

    def forward(self, g, x):
        g1 = self.W_g(g)  # Apply 1x1 Conv on upsampled feature
        x1 = self.W_x(x)  # Apply 1x1 Conv on encoder feature
        psi = self.relu(g1 + x1)  # Element-wise sum
        psi = self.psi(psi)  # Apply sigmoid activation
        return x * psi  # Scale encoder features

from networks.nn_BaseModule import BaseModule
class UNet(BaseModule):
    def __init__(self, convBlock=ConvBlock, deeper_skips=False, num_layers=4, base_filters=64, in_channels=1, out_channels=None, convBlock_layer=None, filter_function='constant', k=2., attention = False, is3D = False):
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
        if is3D:
            self.pool = nn.MaxPool3d(2, 2)
        else:
            self.pool = nn.MaxPool2d(2, 2)

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

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        reversed_filters = filter_sizes[::-1]

        for i in range(num_layers):
            out_channels = reversed_filters[1+i]
            upconv = nn.ConvTranspose3d if is3D else nn.ConvTranspose2d
            self.upconvs.append(upconv(in_channels, out_channels, kernel_size=2, stride=2))
            if self.attention:
                self.attentions.append(AttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels//2, is3D=is3D))
            concat_ch = out_channels * 2
            if i+2 <= num_layers and self.deeperskips:
                concat_ch += reversed_filters[2+i]
            block = convBlock(concat_ch, out_channels, is3D=is3D) if num_layers-i >= convBlock_layer else DoubleConvBlock(concat_ch, out_channels, is3D=is3D)
            self.decoders.append(block)

            in_channels = out_channels

        # Output layer
        final_conv = nn.Conv3d if is3D else nn.Conv2d
        self.final_conv = final_conv(base_filters, self.out_channels, kernel_size=1)
    
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
        for i in range(self.num_layers):
            x = self.upconvs[i](x)
            enc_feat = enc_features[-(i+1)]
            if self.attention:
                enc_feat = self.attentions[i](x, enc_feat)
                
            skip_feats = [x, enc_feat]
            if i+2 <= self.num_layers and self.deeperskips:
                deeper_skip = enc_features[-(i+2)]
                if deeper_skip.shape[2:] != x.shape[2:]:
                    deeper_skip = F.interpolate(deeper_skip, size=x.shape[2:], mode='bilinear' if not self.is3D else 'trilinear', align_corners=False)
                skip_feats.append(deeper_skip)
            
            x = torch.cat(skip_feats, dim=1)
            x = self.decoders[i](x)
        
        # Output
        return self.final_conv(x)
    
if __name__ == "__main__":
    #model = UNet(is3D=True)
    #x = torch.randn(1, 1, 128, 128, 128)
    #print(model(x).shape)

    import numpy as np
    nc = 5e4
    nd = 3e1
    Lc = 6e-2
    nm = 3e2

    L = Lc*(nc**2-nd**2-nm*(nc-nd))/(nd*(nm-nd))
    print(L)