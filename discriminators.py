import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from generators import SelfAttention

class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(
                0.2,
            ),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(
                0.2,
            ),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(
                0.2,
            ),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(
                0.2,
            ),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class SpectralNormDiscriminator(nn.Module):
    """Discriminator with spectral normalization for training stability"""
    def __init__(self, input_channels=3):
        super(SpectralNormDiscriminator, self).__init__()

        def spectral_norm_conv(in_ch, out_ch, kernel_size, stride, padding):
            return nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
            )

        self.layers = nn.Sequential(
            spectral_norm_conv(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm_conv(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm_conv(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm_conv(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # Self-attention at this resolution
            SelfAttention(512),

            spectral_norm_conv(512, 1024, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(1024, 1)),
        )

    def forward(self, x):
        return self.layers(x)