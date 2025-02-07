"""
Module that contains different architectures that can be further trained and tested with our custom dataset
"""
import torch.nn as nn
import torch.nn.functional as F
from swin_transformer_pytorch import SwinTransformer  

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x += residual
        return x

class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=5, padding=2),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoCNN(nn.Module):
    def __init__(self):
        super(AutoCNN, self).__init__()
        self.encoder = SimpleAE().encoder
        self.channel_reducer = nn.Conv2d(64, 3, kernel_size=1)
        self.cnn = CNN()

    def forward(self, x):
        x = self.encoder(x)
        x = self.channel_reducer(x)
        x = self.cnn(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x + residual

class SwinBlock(nn.Module):
    def __init__(self, in_channels):
        super(SwinBlock, self).__init__()
        self.swin = SwinTransformer(
            hidden_dim=96,
            layers=[2, 2, 6, 2],
            heads=[3, 6, 12, 24],
            channels=in_channels,
            window_size=7,
            downscaling_factors=(4, 2, 2, 2),
            relative_pos_embedding=True
        )

    def forward(self, x):
        return self.swin(x)

class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        self.encoder1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(64)
        self.encoder2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.res2 = ResidualBlock(128)
        self.encoder3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.res3 = ResidualBlock(256)
        self.bottleneck = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.decoder3 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.res_dec3 = ResidualBlock(128)
        self.decoder2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.res_dec2 = ResidualBlock(64)
        self.decoder1 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.relu(self.res1(self.encoder1(x)))
        x2 = F.relu(self.res2(self.encoder2(x1)))
        x3 = F.relu(self.res3(self.encoder3(x2)))
        x_bottleneck = self.bottleneck(x3)
        x = F.relu(self.res_dec3(self.decoder3(x_bottleneck))) + x2
        x = F.relu(self.res_dec2(self.decoder2(x))) + x1
        x = self.decoder1(x)
        return x
