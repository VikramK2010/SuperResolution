import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader


class PixelLoss(nn.Module):
    def __init__(self, loss_type="mse"):
    super(PixelLoss, self).__init__()

    # quadratic loss - (p_1 - y_1) ^ 2 +....+(p_n - y_n)^2
    if loss_type == "mse":
        self.loss_fn = nn.MSELoss()
        # direct subtraction - (p_1 - y_1) +....+(p_n - y_n)
    elif loss_type == "l1":
        self.loss_fn = nn.L1Loss()
    else:
        raise ValueError("loss_type must be mse or l1")

    def forward(self, generated, target):
        # compute the loss with our chosen loss function
        return self.loss_fn(generated, target)

class PerceptualLoss(nn.Module):
    def __init__(self):
    super(PerceptualLoss, self).__init__()

    # load in an existing model
    vgg = models.vgg16(pretrained=True).features
    # create a representation model from the existing model by only using a subset of layers
    self.feature_extractor = nn.Sequential(*list(vgg.children())[:16])

    for param in self.feature_extractor.parameters():
        param.requires_grad = False

    self.mse_loss = nn.MSELoss()

def forward(self, generated, target):
    # convert black and white images to psuedo color images
    # (go from 1 to 3 channels)
    if generated.size(1) == 1:
        generated = generated.repeat(1,3,1,1)
    target = target.repeat(1,3,1,1)

    gen_features = self.feature_extractor(generated)
    target_feature = self.feature_extractor(target)

    return self.mse_loss(gen_features, target_features)


class CombinedLoss(nn.Module):
    def __init__(self, pixel_weight=0.2, perceptual_weight=0.8, pixel_type='mse'):
        super(CombinedLoss, self).__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight

        self.pixel_loss = PixelLoss(pixel_type)
        self.perceptual_loss = PerceptualLoss()

    def update_weights(self, pixel_weight, perceptual_weight):
        """Update loss weights during training"""
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, generated, target):
        pixel_loss = self.pixel_loss(generated, target)
        perceptual_loss = self.perceptual_loss(generated, target)

        # then compute the weighted average
        total_loss = (self.pixel_weight * pixel_loss +
            self.perceptual_weight * perceptual_loss)

        return total_loss, pixel_loss, perceptual_loss


class LossWeightScheduler:
    def __init__(self, initial_pixel_weight=0.2, initial_perceptual_weight=0.8,
                 final_pixel_weight=0.5, final_perceptual_weight=0.5, total_epochs=10):
    self.initial_pixel = initial_pixel_weight
    self.initial_perceptual = initial_perceptual_weight
    self.final_pixel = final_pixel_weight
    self.final_perceptual = final_perceptual_weight
    self.total_epochs = total_epochs


def get_weights(self, epoch):
    """Get current weights based on epoch (linear interpolation)"""
    progress = epoch / self.total_epochs

    pixel_weight = self.initial_pixel + progress * (self.final_pixel - self.initial_pixel)
    perceptual_weight = self.initial_perceptual + progress * (self.final_perceptual - self.initial_perceptual)

    return pixel_weight, perceptual_weight


class EnhancedTrainer:
    def __init__(self, generator, discriminator, device, loss_type='combined'):
    self.generator = generator
    self.discriminator = discriminator
    self.device = device

    # Adaptive learning rates
    self.g_lr = 0.0002
    self.d_lr = 0.0002
    self.d_train_freq = 1
    self.g_train_freq = 1
    self.d_loss_history = []
    self.g_loss_history = []

    # Loss functions
    self.adversarial_criterion = nn.BCELoss()
    self.content_weight = 100.0

    # Initialize content loss based on type
    if loss_type == 'pixel':
        self.content_loss = PixelLoss('mse').to(device)
        self.loss_scheduler = None
    elif loss_type == 'perceptual':
        self.content_loss = PerceptualLoss().to(device)
        self.loss_scheduler = None
    elif loss_type == 'combined':
        self.content_loss = CombinedLoss(pixel_weight=1.0, perceptual_weight=0.1).to(device)
    self.loss_scheduler = LossWeightScheduler()

    self.loss_type = loss_type
    self.update_optimizers()

def update_optimizers(self):
    self.g_optimizer = optim.Adam(self.generator.parameters(),
                                  lr=self.g_lr, betas=(0.5, 0.999))
    self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                  lr=self.d_lr, betas=(0.5, 0.999))

def update_loss_weights(self, epoch, total_epochs):
    """Update loss weights if using combined loss"""
    if self.loss_type == 'combined' and self.loss_scheduler:
        pixel_weight, perceptual_weight = self.loss_scheduler.get_weights(epoch)
    self.content_loss.update_weights(pixel_weight, perceptual_weight)
    print(f"Updated weights - Pixel: {pixel_weight:.3f}, Perceptual: {perceptual_weight:.3f}")

def adapt_training_schedule(self, d_loss, g_loss):
    self.d_loss_history.append(d_loss)
    self.g_loss_history.append(g_loss)

    if len(self.d_loss_history) < 10:
        return

    recent_d_loss = np.mean(self.d_loss_history[-10:])
    recent_g_loss = np.mean(self.g_loss_history[-10:])
    loss_ratio = recent_d_loss / (recent_g_loss + 1e-8)

    if loss_ratio < 0.5:
        self.d_train_freq = 2
        self.g_train_freq = 1
        self.d_lr *= 0.95
        self.g_lr *= 1.05
        print("Adapting: Discriminator too strong")
    elif loss_ratio > 2.0:
        self.d_train_freq = 1
    self.g_train_freq = 2
    self.d_lr *= 1.05
    self.g_lr *= 0.95
    print("Adapting: Generator too strong")
    else:
    self.d_train_freq = 1
    self.g_train_freq = 1

    self.update_optimizers()

def train_step(self, low_res, high_res, step):
    batch_size = low_res.size(0)
    real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9
    fake_labels = torch.ones(batch_size, 1).to(self.device) * 0.1

    d_loss, g_loss = 0, 0

    if step % self.d_train_freq == 0:
        d_loss = self.train_discriminator(low_res, high_res, real_labels, fake_labels)
    if step % self.g_train_freq == 0:
        g_loss = self.train_generator(low_res, high_res, real_labels)

    return d_loss, g_loss

def train_discriminator(self, low_res, high_res, real_labels, fake_labels):
    self.d_optimizer.zero_grad()
    real_pred = self.discriminator(high_res)
    d_real_loss = self.adversarial_criterion(real_pred, real_labels)
    fake_images = self.generator(low_res)
    fake_pred = self.discriminator(fake_images.detach())
    d_fake_loss = self.adversarial_criterion(fake_pred, fake_labels)
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
    self.d_optimizer.step()
    return d_loss.item()

def train_generator(self, low_res, high_res, real_labels):
    self.g_optimizer.zero_grad()

    fake_images = self.generator(low_res)

    # Adversarial loss
    fake_pred = self.discriminator(fake_images)
    adversarial_loss = self.adversarial_criterion(fake_pred, real_labels)

    # Content loss with detailed reporting
    if self.loss_type == 'combined':
        content_loss, pixel_loss, perceptual_loss = self.content_loss(fake_images, high_res)
    else:
        content_loss = self.content_loss(fake_images, high_res)
    pixel_loss = perceptual_loss = 0

    # Total generator loss
    g_loss = adversarial_loss + self.content_weight * content_loss

    g_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
    self.g_optimizer.step()

    return g_loss.item()

