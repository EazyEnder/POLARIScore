import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config import LOGGER

import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_UNet import ConvBlock, AttentionBlock

class MultiNet(nn.Module):
    def __init__(self, convBlock=ConvBlock, num_channels=1 , num_layers=4, base_filters=64, attention = False):
        super(MultiNet, self).__init__()

        self.num_channels = num_channels
        self.num_layers = num_layers
        self.attention = attention

        filter_sizes = [int(base_filters * 2**i) for i in range(num_layers+1)]

        self.pool = nn.MaxPool2d(2, 2)

        #Channels Encoder
        self.channels_encoder = []
        self.channels_catconv = []
        for _ in range(num_channels): 
            encoders = nn.ModuleList()
            catconvs = nn.ModuleList()

            in_channels = 1
            catconvs.append(convBlock(2, 1))
            for i in range(num_layers):
                out_channels = filter_sizes[i]
                encoders.append(convBlock(in_channels, out_channels))
                catconvs.append(convBlock(2*out_channels, out_channels))

                in_channels = out_channels

            self.channels_encoder.append(encoders)
            self.channels_catconv.append(catconvs)

        #Main Encoder
        self.encoders = nn.ModuleList()
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
        B,C,W,H = x.shape
        channels = [x[:, i, :, :] for i in range(C)]
        assert C == self.num_channels, LOGGER.error(f"Model not trained on {C} channels")

        # list of enc_features for each channel
        channels_features=[]
        for i in range(C):
            xc = channels[i].unsqueeze(1)
            channels_features.append([])
            for j in range(self.num_layers):
                xc = self.channels_encoder[i][j](xc)
                channels_features[i].append(xc)
                xc = self.pool(xc)

        # Encoder
        enc_features = []

        x = channels[0].unsqueeze(1)
        #This can be improved by making a C->1 conv instead of Cx 2->1 convs bcs why rebuild x when we already have x...
        for i in range(1,C):
            xc = channels[i].unsqueeze(1)
            x = torch.cat([x, xc], dim=1)
            x = self.channels_catconv[i][0](x)

        for i in range(self.num_layers):
            x = self.encoders[i](x)
            for j in range(0,C):
                xc = channels_features[j][i]
                x = torch.cat([x, xc], dim=1)
                x = self.channels_catconv[j][i+1](x)
            enc_features.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i in range(self.num_layers):
            x = self.upconvs[i](x)
            if self.attention:
                enc_features[-(i+1)] = self.attentions[i](x, enc_features[-(i+1)])
            x = torch.cat([x, enc_features[-(i+1)]], dim=1)
            x = self.decoders[i](x)
        
        # Output
        return self.final_conv(x)
    
if __name__ == "__main__":
    model = MultiNet(num_channels=2)
    x = torch.randn(1, 2, 128, 128)
    print(x.shape)
    print(model(x).shape)