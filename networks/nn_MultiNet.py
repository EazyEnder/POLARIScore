import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config import LOGGER

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.nn_UNet import ConvBlock, AttentionBlock
from networks.nn_BaseModule import BaseModule
import numpy as np

class MultiNet(BaseModule):
    def __init__(self, convBlock=ConvBlock, channel_dimensions=[2] , num_layers=4, base_filters=64, attention = False, is3D=None):
        super(MultiNet, self).__init__()

        self.num_channels = len(channel_dimensions) if type(channel_dimensions) is list else 1
        self.channel_dimensions = channel_dimensions if type(channel_dimensions) is list else [channel_dimensions]
        channel_is3D = []
        for dim in self.channel_dimensions:
            channel_is3D.append(True if dim == 3 else False)
        self.is3D = (True if True in channel_is3D else False) if is3D is None else is3D

        self.num_layers = num_layers
        self.attention = attention

        filter_sizes = [int(base_filters * 2**i) for i in range(num_layers+1)]

        self.pool2D = nn.MaxPool2d(2, 2)
        self.pool3D = nn.MaxPool3d(2,2)

        #Channels Encoder
        self.channels_encoder = nn.ModuleList()
        self.channels_catconv = nn.ModuleList()
        for is3D in channel_is3D: 
            encoders = nn.ModuleList()
            catconvs = nn.ModuleList()

            in_channels = 1
            catconvs.append(convBlock(2, 1, is3D=self.is3D))
            for i in range(num_layers):
                out_channels = filter_sizes[i]
                encoders.append(convBlock(in_channels, out_channels, is3D=is3D))
                catconvs.append(convBlock(2*out_channels, out_channels, is3D=self.is3D))

                in_channels = out_channels

            self.channels_encoder.append(encoders)
            self.channels_catconv.append(catconvs)

        #Main Encoder
        self.encoders = nn.ModuleList()
        in_channels = 1
        for i in range(num_layers):
            out_channels = filter_sizes[i]
            self.encoders.append(convBlock(in_channels, out_channels, is3D=self.is3D))
            in_channels = out_channels

        out_channels = filter_sizes[-1]

        # Bottleneck
        self.bottleneck = convBlock(in_channels, out_channels, is3D=self.is3D)
        in_channels = out_channels

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        reversed_filters = filter_sizes[::-1]

        for i in range(num_layers):
            out_channels = reversed_filters[1+i]
            if self.is3D:
                self.upconvs.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2))
            else:
                self.upconvs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))

            if self.attention:
                self.attentions.append(AttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels//2, is3D=self.is3D))
            self.decoders.append(convBlock(2*out_channels, out_channels, is3D=self.is3D))
            in_channels = out_channels

        # Output layer
        if self.is3D:
            self.final_conv = nn.Conv3d(base_filters, 1, kernel_size=1)
        else:
            self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)

        
    def forward(self, *x):

        C = len(x)
        channels = x
        assert C == self.num_channels, LOGGER.error(f"Model trained with {self.num_channels} inputs but received {C} inputs")

        def _is3D(t):
            return len(t.shape) > 4
        def _convertTo3D(t):
            if not self.is3D:
                return t
            if not _is3D(t):
                t = t.unsqueeze(-1)
                t = t.expand(-1,-1,-1, -1, t.shape[-2])
                t = t.permute(0, 1, 4, 2, 3)
            elif t.shape[-1] != t.shape[-3]:
                t = torch.nn.functional.pad(t, (0, t.shape[-2] - t.shape[-1]))
            return t

        # list of enc_features for each channel
        channels_features=[]
        for i in range(C):
            xc = channels[i]
            channels_features.append([])
            for j in range(self.num_layers):
                xc = self.channels_encoder[i][j](xc)
                channels_features[i].append(xc)
                xc = self.pool3D(xc) if _is3D(xc) else self.pool2D(xc)

        #Multi Encoder
        enc_features = []

        x = channels[0]
        x = _convertTo3D(x)

        for i in range(1,C):
            xc = channels[i]
            xc = _convertTo3D(xc)
            x = torch.cat([x, xc], dim=1)
            x = self.channels_catconv[i][0](x)

        for i in range(self.num_layers):
            x = self.encoders[i](x)
            x = _convertTo3D(x)
            for j in range(0,C):
                xc = channels_features[j][i]
                xc = _convertTo3D(xc)
                x = torch.cat([x, xc], dim=1)
                x = self.channels_catconv[j][i+1](x)
            enc_features.append(x)
            x = self.pool3D(x) if self.is3D else self.pool2D(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Main Decoder
        for i in range(self.num_layers):
            x = self.upconvs[i](x)
            if self.attention:
                enc_features[-(i+1)] = self.attentions[i](x, enc_features[-(i+1)])
            x = torch.cat([x, enc_features[-(i+1)]], dim=1)
            x = self.decoders[i](x)
        
        # Output
        f_x = self.final_conv(x)
        return f_x.permute(0, 1, 3, 4, 2) if self.is3D else f_x
    
    def shape_data(self, batch, target_index=3, input_indexes=[0,2]):
        input_tensors = []
        input_indexes = input_indexes if type(input_indexes) is list else [input_indexes]

        assert (len(batch)) > 0, LOGGER.error("Can't apply the model on the batch, the batch is empty.")
        for i in range(len(batch[0])):
            if not(i in input_indexes):
                continue
            if i == target_index:
                continue
            xi = self.shape_image(np.array([b[i] for b in batch])).to(self.device)
            if len(xi.shape) > 4:
                xi = xi.permute(0, 1, 4, 2, 3)
            input_tensors.append(xi)

        target_tensor = self.shape_image(np.array([b[target_index] for b in batch]))
        return input_tensors, target_tensor
    
if __name__ == "__main__":
    model = MultiNet(channel_dimensions=[2,3])
    x = torch.randn(1, 1, 128, 128)
    y = torch.randn(1, 1, 128, 128, 128)
    print(model(x,y).shape)