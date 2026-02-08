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


# Missing ResidualBlock class
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += residual
        # add without inplace operation
        out = out + residual
        return self.relu(out)

# TODO: Allow for color images
class SimpleGenerator(nn.Module):
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


class ResidualGenerator(nn.Module):
    def __init__(self, num_residual_blocks=4):
        super(ResidualGenerator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.ReLU()
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        out = self.initial(x)
        residual = out
        out = self.residual_blocks(out)
        # out += residual
        # add without inplace operation
        out = out + residual
        out = self.upsample(out)
        out = torch.tanh(self.final(out))
        return out


class SelfAttention(nn.Module):
    """Self-attention mechanism for focusing on important regions"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        # Tools for step 1 (computing query, key, and values)
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # predefine some helpful things for output
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Step 1: Compute q, k, v
        # Generate Q, K, V
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        proj_value = self.value(x).view(batch_size, -1, width * height)

        # Attention computation

        # Step 2: Compute the dot product scores
        scores = torch.bmm(proj_query, proj_key)

        # Step 3: Convert scores to weights
        weights = self.softmax(scores)

        # Step 4: multiply weights with values
        # Apply attention to values
        out = torch.bmm(proj_value, weights.permute(0, 2, 1))

        # Step 5: reshape (to get attention output)
        out = out.view(batch_size, channels, height, width)

        # Step 6: recombine
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out




class ChannelAttention(nn.Module):
    """Channel attention for feature recalibration"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # step 1: extract information from channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # step 2: setup a scoring function
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      	# Get scores
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # combine and weight scores
        attention = self.sigmoid(avg_out + max_out)
        # apply weights to channels
        return x * attention


class AttentionResidualBlock(nn.Module):
    """Residual block with attention mechanisms"""
    def __init__(self, channels):
        super(AttentionResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.channel_attention = ChannelAttention(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        # get resid block output
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply channel attention
        out = self.channel_attention(out)

        # Residual connection
        out += residual
        return self.relu(out)


# TODO: Allow for color images
class AttentionMedicalGenerator(nn.Module):
    """State-of-the-art generator with attention mechanisms"""
    def __init__(self, num_residual_blocks=8):
        super(AttentionMedicalGenerator, self).__init__()

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Attention-enhanced residual blocks
        self.residual_blocks = nn.ModuleList([
            AttentionResidualBlock(64) for _ in range(num_residual_blocks)
        ])

        # Self-attention after residual blocks
        self.self_attention = SelfAttention(64)

        # Upsampling path with attention
        self.upsample1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.attention1 = ChannelAttention(64)

        self.upsample2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.attention2 = ChannelAttention(64)

        # Final refinement
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        # Initial features
        features = self.initial(x)
        residual = features

        # Attention-enhanced residual processing
        for block in self.residual_blocks:
            features = block(features)

        # Global skip connection
        features = features + residual

        # Self-attention
        features = self.self_attention(features)

        # Upsampling with attention
        features = self.upsample1(features)
        features = self.attention1(features)

        features = self.upsample2(features)
        features = self.attention2(features)

        # Final output
        output = self.final(features)
        return output



