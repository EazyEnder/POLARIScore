import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.conv(x)
    
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels//2),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.conv(x)
    
class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.match_dim = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else None
    
    def forward(self, x):
        res = x
        x = self.conv(x)
        if self.match_dim:
            res = self.match_dim(res)
        x += res
        return F.relu(x)

class UNet(nn.Module):
    def __init__(self, convBlock=ConvBlock, num_layers=4, base_filters=64):
        super(UNet, self).__init__()
        
        self.num_layers = num_layers
        self.base_filters = base_filters
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        in_channels = 1
        out_channels = base_filters
        for _ in range(num_layers):
            self.encoders.append(convBlock(in_channels, out_channels))
            in_channels = out_channels
            out_channels *= 2
        
        # Bottleneck
        self.bottleneck = convBlock(in_channels, out_channels)
        
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for _ in range(num_layers):
            self.upconvs.append(nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2))
            self.decoders.append(convBlock(out_channels, out_channels // 2))
            out_channels //= 2
        
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
            x = torch.cat([x, enc_features[-(i+1)]], dim=1)  # Skip connection
            x = self.decoders[i](x)
        
        # Output
        return self.final_conv(x)