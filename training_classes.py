import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.models as models
from torch import nn, optim
from scipy import ndimage
import cv2


# Medical-specific loss functions
class MedicalPerceptualLoss(nn.Module):
    """Perceptual loss using medical-relevant features"""
    def __init__(self):
        super(MedicalPerceptualLoss, self).__init__()
        # Use ResNet instead of VGG for better medical feature extraction
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layers

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, generated, target):
        if generated.size(1) == 1:
            generated = generated.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)

        return self.mse_loss(gen_features, target_features)

class EdgePreservationLoss(nn.Module):
    """Preserves important edges/boundaries in medical images"""
    def __init__(self):
        super(EdgePreservationLoss, self).__init__()
        # Sobel edge detection kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        self.mse_loss = nn.MSELoss()

    def get_edges(self, x):
        """Extract edges using Sobel operators"""
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        edges = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        return edges

    def forward(self, generated, target):
        gen_edges = self.get_edges(generated)
        target_edges = self.get_edges(target)
        return self.mse_loss(gen_edges, target_edges)

class FrequencyLoss(nn.Module):
    """Preserves frequency domain characteristics important for medical imaging"""
    def __init__(self):
        super(FrequencyLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, generated, target):
        # Convert to frequency domain
        gen_fft = torch.fft.fft2(generated)
        target_fft = torch.fft.fft2(target)

        # Compare magnitude spectra
        gen_mag = torch.abs(gen_fft)
        target_mag = torch.abs(target_fft)

        return self.mse_loss(gen_mag, target_mag)

class StructuralSimilarityLoss(nn.Module):
    """SSIM loss for structural preservation"""
    def __init__(self, window_size=11, sigma=1.5, channels=3): 
        super(StructuralSimilarityLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels
        self.register_buffer('window', self._create_window(window_size, sigma, channels))

    def _create_window(self, window_size, sigma, channels):
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        window = g.outer(g).unsqueeze(0).unsqueeze(0)
        # replicate window for all channels
        # window = window.expand(channels, 1, window_size, window_size).contiguous()
        window = window.repeat(channels, 1, 1, 1)
        return window

    def _ssim(self, img1, img2):
        if self.channels != img1.size(1):
            # raise ValueError("Number of channels in input images does not match the initialized channels")
            print("Failed SSIM channel check...")

        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=img1.size(1))
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=img2.size(1))

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=img1.size(1)) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def forward(self, generated, target):
        return 1 - self._ssim(generated, target)

class MedicalCombinedLoss(nn.Module):
    """Comprehensive loss for medical image super-resolution"""
    def __init__(self, pixel_weight=1.0, perceptual_weight=0.1, edge_weight=0.5,
                 freq_weight=0.2, ssim_weight=0.3):
        super(MedicalCombinedLoss, self).__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.freq_weight = freq_weight
        self.ssim_weight = ssim_weight

        self.pixel_loss = PixelLoss('l1')  # L1 better for medical images
        self.perceptual_loss = MedicalPerceptualLoss()
        self.edge_loss = EdgePreservationLoss()
        self.freq_loss = FrequencyLoss()
        self.ssim_loss = StructuralSimilarityLoss()

    def update_weights(self, pixel_weight, perceptual_weight, edge_weight, freq_weight, ssim_weight):
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.freq_weight = freq_weight
        self.ssim_weight = ssim_weight

    def forward(self, generated, target):
        pixel_loss = self.pixel_loss(generated, target)
        perceptual_loss = self.perceptual_loss(generated, target)
        edge_loss = self.edge_loss(generated, target)
        freq_loss = self.freq_loss(generated, target)
        ssim_loss = self.ssim_loss(generated, target)

        total_loss = (self.pixel_weight * pixel_loss +
                     self.perceptual_weight * perceptual_loss +
                     self.edge_weight * edge_loss +
                     self.freq_weight * freq_loss +
                     self.ssim_weight * ssim_loss)

        return total_loss, pixel_loss, perceptual_loss, edge_loss, freq_loss, ssim_loss

# Medical-specific weight scheduler
class MedicalLossScheduler:
    """Scheduler optimized for medical image training progression"""
    def __init__(self, total_epochs=10):
        self.total_epochs = total_epochs

    def get_weights(self, epoch):
        """Progressive weight scheduling for medical images"""
        progress = epoch / self.total_epochs

        # Early: Focus on pixel accuracy and edges
        # Later: Add perceptual and structural quality
        pixel_weight = 1.0
        edge_weight = 0.8 - 0.3 * progress      # Start high, decrease
        freq_weight = 0.1 + 0.3 * progress      # Start low, increase
        ssim_weight = 0.2 + 0.4 * progress      # Start low, increase
        perceptual_weight = 0.05 + 0.15 * progress  # Start very low, increase

        return pixel_weight, perceptual_weight, edge_weight, freq_weight, ssim_weight




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
        def forward(self, generated, target):
        # Only repeat if the input is grayscale (1 channel)
        if generated.size(1) == 1:
            generated = generated.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1) # <--- Fixed: now inside the if-block

        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)

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



class AdaptiveTrainer:
    def __init__(self, generator, discriminator, device):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.g_lr = 0.0002
        self.d_lr = 0.0002
        self.d_train_freq = 1
        self.g_train_freq = 1
        self.d_loss_history = []
        self.g_loss_history = []
        self.criterion = nn.BCELoss()
        self.update_optimizers()

    def update_optimizers(self):
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), lr=self.g_lr, betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.d_lr, betas=(0.5, 0.999)
        )

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
            d_loss = self.train_discriminator(
                low_res, high_res, real_labels, fake_labels
            )
        if step % self.g_train_freq == 0:
            g_loss = self.train_generator(low_res, real_labels)

        return d_loss, g_loss


    def train_discriminator(self, low_res, high_res, real_labels, fake_labels):
        self.d_optimizer.zero_grad()
        # print("Training Discriminator")
        real_pred = self.discriminator(high_res)
        # print("Real images shape:", real_pred.shape)
        d_real_loss = self.criterion(real_pred, real_labels)
        # print("Real loss:", d_real_loss.item())
        fake_images = self.generator(low_res)
        # print("Fake images shape:", fake_images.shape)
        fake_pred = self.discriminator(fake_images.detach())
        # print("Fake images prediction shape:", fake_pred.shape)
        d_fake_loss = self.criterion(fake_pred, fake_labels)
        # print("Fake loss:", d_fake_loss.item())
        d_loss = d_real_loss + d_fake_loss
        # print("Total Discriminator loss:", d_loss.item())
        d_loss.backward()
        # print("Discriminator gradients computed")
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        # print("Clipped gradients for Discriminator")
        self.d_optimizer.step()
        return d_loss.item()

    def train_generator(self, low_res, real_labels):
        self.g_optimizer.zero_grad()
        # print("Training Generator")
        fake_images = self.generator(low_res)
        # print("Generated images shape:", fake_images.shape)
        fake_pred = self.discriminator(fake_images)
        # print("Fake images prediction shape:", fake_pred.shape)
        g_loss = self.criterion(fake_pred, real_labels)
        # print("Generator loss:", g_loss.item())
        g_loss.backward()
        # print("Generator gradients computed")
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        # print("Clipped gradients for Generator")
        if torch.isnan(g_loss) or torch.isinf(g_loss):
            print(f"Invalid loss detected: {g_loss}")
            return g_loss
        self.g_optimizer.step()
        return g_loss.item()
    


class EnhancedTrainer:
    def __init__(self, generator, discriminator, device, loss_type='combined', total_epochs=10):
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
        self.loss_scheduler = LossWeightScheduler(total_epochs=total_epochs)

        self.loss_type = loss_type
        self.update_optimizers()

    def update_optimizers(self):
        self.g_optimizer = optim.Adam(self.generator.parameters(),
                                      lr=self.g_lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                      lr=self.d_lr, betas=(0.5, 0.999))

    def update_loss_weights(self, epoch):
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
        pixel_loss = perceptual_loss = 0 # Move initialization to the top
        if self.loss_type == 'combined':
            content_loss, pixel_loss, perceptual_loss = self.content_loss(fake_images, high_res)
        else:
            content_loss = self.content_loss(fake_images, high_res)

        # Total generator loss
        g_loss = adversarial_loss + self.content_weight * content_loss

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.g_optimizer.step()

        return g_loss.item()

    def print_loss_summary(self, epoch):
        """Print detailed loss breakdown"""
        if self.loss_type == 'combined' and len(self.d_loss_history) >= 10:
            recent_d = np.mean(self.d_loss_history[-10:])
            recent_g = np.mean(self.g_loss_history[-10:])

            print(f"Loss Summary Epoch {epoch}:")
            print(f"    Discriminator Loss: {recent_d:.4f}")
            print(f"    Generator Loss: {recent_g:.4f}")
            print(f"    Content Weight: {self.content_weight:.2f}")


# Enhanced trainer for medical images
class MedicalTrainer(EnhancedTrainer):
    """Specialized trainer for medical image super-resolution"""
    def __init__(self, generator, discriminator, device, total_epochs=10):
        # Initialize with medical-specific loss
        super().__init__(generator, discriminator, device, loss_type='medical')

        # Override with medical loss
        self.content_loss = MedicalCombinedLoss().to(device)
        self.loss_scheduler = MedicalLossScheduler(total_epochs=total_epochs)
        self.content_weight = 50.0  # Lower weight for medical images

        # Medical-specific tracking
        self.edge_loss_history = []
        self.freq_loss_history = []
        self.ssim_loss_history = []

    def update_loss_weights(self, epoch):
        """Update medical-specific loss weights"""
        weights = self.loss_scheduler.get_weights(epoch)
        pixel_w, perceptual_w, edge_w, freq_w, ssim_w = weights

        self.content_loss.update_weights(pixel_w, perceptual_w, edge_w, freq_w, ssim_w)
        print(f"Medical weights - Pixel:{pixel_w:.2f} Edge:{edge_w:.2f} "
              f"Freq:{freq_w:.2f} SSIM:{ssim_w:.2f} Perceptual:{perceptual_w:.2f}")

    def train_step(self, low_res, high_res, step):
        batch_size = low_res.size(0)
        real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9
        fake_labels = torch.ones(batch_size, 1).to(self.device) * 0.1

        d_loss, g_tot_loss, g_adv_loss, g_cont_loss = 0, 0, 0, 0

        if step % self.d_train_freq == 0:
            d_loss = self.train_discriminator(low_res, high_res, real_labels, fake_labels)
        if step % self.g_train_freq == 0:
            g_tot_loss, g_adv_loss, g_cont_loss = self.train_generator(low_res, high_res, real_labels)

        return d_loss, g_tot_loss, g_adv_loss, g_cont_loss

    def train_generator(self, low_res, high_res, real_labels):
        self.g_optimizer.zero_grad()

        fake_images = self.generator(low_res)

        # Adversarial loss
        fake_pred = self.discriminator(fake_images)
        adversarial_loss = self.adversarial_criterion(fake_pred, real_labels)

        # Medical content loss with detailed breakdown
        losses = self.content_loss(fake_images, high_res)
        content_loss, pixel_loss, perceptual_loss, edge_loss, freq_loss, ssim_loss = losses

        # Track medical-specific losses
        self.edge_loss_history.append(edge_loss.item())
        self.freq_loss_history.append(freq_loss.item())
        self.ssim_loss_history.append(ssim_loss.item())

        # Total generator loss
        weighted_content_loss = self.content_weight * content_loss
        g_total_loss = adversarial_loss + weighted_content_loss

        g_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.g_optimizer.step()

        return g_total_loss.item(), adversarial_loss.item(), content_loss.item()


    def print_loss_summary(self, epoch):
        """Print detailed medical loss breakdown"""
        if len(self.edge_loss_history) >= 10:
            recent_edge = np.mean(self.edge_loss_history[-10:])
            recent_freq = np.mean(self.freq_loss_history[-10:])
            recent_ssim = np.mean(self.ssim_loss_history[-10:])

            print(f"Medical Metrics:")
            print(f"    Edge Preservation: {recent_edge:.4f}")
            print(f"    Frequency Domain: {recent_freq:.4f}")
            print(f"    Structural Similarity: {recent_ssim:.4f}")
