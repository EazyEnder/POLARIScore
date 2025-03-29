import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(out_channels),
            #nn.Dropout2d(p=0.1),
        )
    
    def forward(self, x):
        return self.conv(x)
    
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(out_channels//2),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.conv(x)
    
class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.match_dim = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else None
    
    def forward(self, x):
        res = x.clone()
        x = self.conv(x)
        if self.match_dim:
            res = self.match_dim(res)
        x = x+res
        return F.relu(x)
    
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)  # Apply 1x1 Conv on upsampled feature
        x1 = self.W_x(x)  # Apply 1x1 Conv on encoder feature
        psi = self.relu(g1 + x1)  # Element-wise sum
        psi = self.psi(psi)  # Apply sigmoid activation
        return x * psi  # Scale encoder features

import math
class UNet(nn.Module):
    def __init__(self, convBlock=ConvBlock, num_layers=4, base_filters=64, filter_function='constant', k=2., attention = False):
        super(UNet, self).__init__()

        self.num_layers = num_layers
        self.attention = attention

        if filter_function == 'constant':
            filter_sizes = [int(base_filters * k**i) for i in range(num_layers+1)]
        else:
            raise ValueError("Invalid filter function type.")
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        in_channels = 1
        for i in range(num_layers):
            out_channels = filter_sizes[i]
            self.encoders.append(convBlock(in_channels, out_channels))
            in_channels = out_channels
        out_channels = filter_sizes[-1]

        # Bottleneck
        self.bottleneck = convBlock(in_channels, out_channels)
        in_channels = out_channels

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        reversed_filters = filter_sizes[::-1]

        for i in range(num_layers):
            out_channels = reversed_filters[1+i]
            self.upconvs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            if self.attention:
                self.attentions.append(AttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels//2))
            self.decoders.append(convBlock(2*out_channels, out_channels))
            in_channels = out_channels

        # Output layer
        self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)
    
    def forward(self, x):
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
            if self.attention:
                enc_features[-(i+1)] = self.attentions[i](x, enc_features[-(i+1)])
            x = torch.cat([x, enc_features[-(i+1)]], dim=1)  # Skip connection
            x = self.decoders[i](x)
        
        # Output
        return self.final_conv(x)
    
if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 1, 128, 128)
    print(model(x).shape)