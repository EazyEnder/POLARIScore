import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.utils.fastkanconv import FastKANConvLayer

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        #x = F.relu(self.bn(self.conv2(x)))
        return x

class KNet(nn.Module):
    def __init__(self):
        super(KNet, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = ConvBlock(1, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder (Upsampling + Nested Skip Connections)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)  # Skip connection from enc4
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)  # Skip connection from enc3
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)  # Skip connection from enc2
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)  # Skip connection from enc1
        
        # Output layer
        #self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.final_conv = FastKANConvLayer(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        
        # Bottleneck
        x_b = self.bottleneck(self.pool(x4))
        
        # Decoder with Skip Connections
        x_d4 = self.dec4(torch.cat([self.upconv4(x_b), x4], dim=1))
        x_d3 = self.dec3(torch.cat([self.upconv3(x_d4), x3], dim=1))
        x_d2 = self.dec2(torch.cat([self.upconv2(x_d3), x2], dim=1))
        x_d1 = self.dec1(torch.cat([self.upconv1(x_d2), x1], dim=1))
        
        # Output
        out = self.final_conv(x_d1)
        return out