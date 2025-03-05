import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        
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
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
    
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

if __name__ == "__main__":
    import sys
    import os

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FCN().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.75, threshold=0.005)


    from training_batch import *
    train_b = open_batch("batch_37392b55-be04-4e8c-aa49-dca42fa684fc")

    def custom_loss(output, target):
        weights = torch.ones_like(target)      
        return torch.mean(weights * (output - target) ** 2)

    input_tensor = torch.from_numpy(np.pad(np.log(np.array([b[0] for b in train_b])),(0,0))).float().unsqueeze(1)
    output_tensor = torch.from_numpy(np.pad(np.log(np.array([b[1] for b in train_b])),(0,0))).float().unsqueeze(1)

    input_tensor, output_tensor = input_tensor.to(device), output_tensor.to(device)

    
    test_batch = open_batch("batch_92b49d92-369a-45a0-b4eb-385658b05f41")
    test_input_tensor = torch.from_numpy(np.pad(np.log(np.array([b[0] for b in test_batch])),(0,0))).float().unsqueeze(1).to(device)
    test_output_tensor = torch.from_numpy(np.pad(np.log(np.array([b[1] for b in test_batch])),(0,0))).float().unsqueeze(1).to(device)

    training_losses = []
    validation_losses = []

    for epoch in range(500):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = custom_loss(output, output_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        test_output = model(test_input_tensor)
        training_losses.append(loss.item())
        v_loss = custom_loss(test_output, test_output_tensor).item()
        validation_losses.append(v_loss)
        print(f'Epoch {epoch}, Training Loss: {loss.item()}, Validation loss: {v_loss}')

    model.eval()
    test_output = model(test_input_tensor)
    print(f'Validation loss: {custom_loss(test_output, test_output_tensor).item()}')

    plt.figure()
    plt.plot(training_losses, label="training losses")
    plt.plot(validation_losses, label="validation losses")
    plt.legend()

    test_output_tensor, test_output = test_output_tensor.cpu().detach().numpy(), test_output.cpu().detach().numpy()
    plot_batch(rebuild_batch(np.exp(test_output_tensor[:,0,:,:]), np.exp(test_output[:,0,:,:])), same_limits=True)
    plt.show()