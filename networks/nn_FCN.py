import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        # Encoder (Downsampling)
        self.conv_e_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv_e_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_e_3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2) 

        # Bottleneck
        self.conv_b = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Decoder (Upsampling)
        self.upconv_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_d_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upconv_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_d_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_d_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
       # Encoding
        x1 = F.relu(self.conv_e_1(x))  # 1 → 64
        x2 = self.pool(F.relu(self.conv_e_2(x1)))  # 64 → 128
        x3 = self.pool(F.relu(self.conv_e_3(x2)))  # 128 → 256
        x4 = self.pool(x3)  # 256 → 512 (Bottleneck input)

        # Bottleneck
        x_b = F.relu(self.conv_b(x4))  # Still 512

        # Decoding
        x = F.relu(self.upconv_1(x_b))  # 512 → 256
        x = torch.cat((x, x3), dim=1)  # Skip connection
        x = F.relu(self.conv_d_1(x))  # Process after concatenation

        x = F.relu(self.upconv_2(x))  # 256 → 128
        x = torch.cat((x, x2), dim=1)  # Skip connection
        x = F.relu(self.conv_d_2(x))

        x = F.relu(self.upconv_3(x))  # 128 → 64
        x = torch.cat((x, x1), dim=1)  # Skip connection
        x = F.relu(self.conv_d_3(x))

        x = self.final_conv(x)  # Final output (1 channel)

        return x