import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from ...config import LOGGER
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_UNet import DoubleConvBlock, GatedAttentionBlock
from .nn_BaseModule import BaseModule
from ..utils.nn_utils import init_network
from torch.nn import init
from typing import List, Literal, Tuple
from kan import KAN
from ..addons.FiLM import FiLMGenerator

#tensors shape B,C,H,W ; no third axis
class ContextAwareUNet(BaseModule):
    """
    Basic Attention U-Net with context given
    """
    #Local refers to wanted Region
    #Global refers to Context
    def __init__(self,num_layers:int=4, base_filters:int=64, init_method=init.kaiming_uniform_):
        super(ContextAwareUNet, self).__init__()

        self.num_layers = num_layers
        self.in_channels = 1
        self.out_channels = 1
        self.init_method = init_method

        filter_sizes = [int(base_filters * 2**i) for i in range(num_layers+1)]
        self.pool = nn.MaxPool2d(2,2)

        #Global & Local Encoders
        self.g_encoders = nn.ModuleList()
        self.l_encoders = nn.ModuleList()
        in_channels =  self.in_channels
        for i in range(num_layers):
            out_channels = filter_sizes[i]
            self.l_encoders.append(DoubleConvBlock(in_channels, out_channels, init_method=self.init_method))
            self.g_encoders.append(DoubleConvBlock(2 if i==0 else in_channels, out_channels, init_method=self.init_method))
            in_channels = out_channels
        out_channels = filter_sizes[-1]

        # Bottleneck
        self.g_bottleneck = DoubleConvBlock(in_channels, out_channels, init_method=self.init_method)
        self.l_bottleneck = DoubleConvBlock(in_channels, out_channels, init_method=self.init_method)
        in_channels = out_channels

        #FiLM learned using Global latent space
        self.film = FiLMGenerator(in_channels, filter_sizes[:-1])
        #FiLM learned using physical size
        #self.film_physize = FiLMGenerator(1, filter_sizes[:-1])

        #Local Decoder with FiLM learned using Global
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
                self.upconvs[j].append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
                self.attentions[j].append(GatedAttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels//2, init_method=self.init_method))
                concat_ch = out_channels * 2
                block = DoubleConvBlock(concat_ch, out_channels, init_method=self.init_method)
                self.decoders[j].append(block)

                in_channels = out_channels

        # Output layer
        self.final_conv = nn.ModuleList()
        for o in range(self.out_channels):
            self.final_conv.append(nn.Conv2d(base_filters, 1, kernel_size=1))

        self.initialize()
    
    def initialize(self):
        for f in self.final_conv:
            init_network(self,self.init_method)
            init.zeros_(f.bias)

    def forward(self, x:List[torch.tensor]):
        """
        Args
            x: tensor shape [(B,1,H,W),(B,2,H,W),(B,1)] ; i.e [region, context, physical_size]
        """
        gl_x = x[1]
        lo_x = x[0]

        # Encoders forward pass
        l_enc_features = []
        for i in range(self.num_layers):
            gl_x = self.g_encoders[i](gl_x)
            lo_x = self.l_encoders[i](lo_x)
            l_enc_features.append(lo_x)
            gl_x = self.pool(gl_x)
            lo_x = self.pool(lo_x)
        
        # Bottleneck
        gl_x = self.g_bottleneck(gl_x)
        lo_x = self.l_bottleneck(lo_x)

        film_params = self.film(gl_x.mean(dim=-1).mean(dim=-1))
        #film_physize_params = self.film_physize(x[2])
        
        # Decoder forward pass with film modulation
        decoded_x = []
        for j in range(self.out_channels):
            xj = lo_x#.clone()
            for i in range(self.num_layers):
                xj = self.upconvs[j][i](xj)
                enc_feat = l_enc_features[-(i+1)]
                enc_feat = self.attentions[j][i](xj, enc_feat)
                skip_feats = [xj, enc_feat]
                xj = torch.cat(skip_feats, dim=1)
                gamma, beta = film_params[::-1][i]
                #gamma_physize, beta_physize = film_physize_params[::-1][i]
                xj = gamma*self.decoders[j][i](xj)+beta
                
            decoded_x.append(xj)
        
        # Output
        return self.final_conv[0](decoded_x[0])
    
if __name__ == "__main__":
    #Test shape
    model = ContextAwareUNet(num_layers=2)
    x = [torch.randn(2, 1, 32, 32), torch.randn(2, 2, 32, 32), torch.randn(2,1)]
    print(model(x).shape)