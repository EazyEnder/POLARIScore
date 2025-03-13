import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.utils.fastkanconv import FastKANConvLayer
from nn_UNet import ConvBlock, UNet
    
class KanConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(KanConvBlock, self).__init__()
        self.conv = nn.Sequential(
            FastKANConvLayer(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(out_channels),
            #nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class KNet(UNet):
    """Same as a UNet network but the final convolution layer is a Kolmogorov Arnold layer"""
    def __init__(self, convBlock=ConvBlock, num_layers=4, base_filters=64, kan_type="RBF", num_grids=4, **kwargs):
        super(KNet, self).__init__(convBlock=convBlock, num_layers=num_layers, base_filters=base_filters, **kwargs)
        self.final_conv = FastKANConvLayer(base_filters, 1, kernel_size=1, kan_type=kan_type, num_grids=num_grids)

class FullKNet(KNet):
    def __init__(self, **kwargs):
        super(FullKNet, self).__init__(convBlock=KanConvBlock, **kwargs)
