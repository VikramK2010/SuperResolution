"""
agricultural_adaptations.py

Domain-specific adaptations for crop disease super-resolution.
This module replaces medical-specific components with agricultural-optimized versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


# ============================================================================
# AGRICULTURAL-SPECIFIC GENERATORS
# ============================================================================

class AgriculturalAttentionGenerator(nn.Module):
    """
    Adapted from AttentionMedicalGenerator for RGB crop images.
    Key changes:
    - 3 input/output channels instead of 1 (RGB vs grayscale)
    - Optimized for color variation and texture patterns in crops
    """
    def __init__(self, num_residual_blocks=8):
        super(AgriculturalAttentionGenerator, self).__init__()

        # Initial feature extraction (3 channels for RGB)
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Attention-enhanced residual blocks
        self.residual_blocks = nn.ModuleList([
            AgriculturalAttentionBlock(64) for _ in range(num_residual_blocks)
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

        # Final refinement (3 output channels for RGB)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.initial(x)
        residual = features

        # Process through attention blocks
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


class AgriculturalAttentionBlock(nn.Module):
    """Residual block optimized for crop texture patterns"""
    def __init__(self, channels):
        super(AgriculturalAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.channel_attention = ChannelAttention(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.channel_attention(out)
        out += residual
        return self.relu(out)


class SelfAttention(nn.Module):
    """Self-attention mechanism (reused from generators.py)"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        proj_value = self.value(x).view(batch_size, -1, width * height)

        scores = torch.bmm(proj_query, proj_key)
        weights = self.softmax(scores)
        out = torch.bmm(proj_value, weights.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out


class ChannelAttention(nn.Module):
    """Channel attention (reused from generators.py)"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


# ============================================================================
# AGRICULTURAL-SPECIFIC DISCRIMINATOR
# ============================================================================

class AgriculturalDiscriminator(nn.Module):
    """
    Discriminator optimized for RGB crop images.
    Adapted from SpectralNormDiscriminator with 3 input channels.
    """
    def __init__(self):
        super(AgriculturalDiscriminator, self).__init__()

        def spectral_norm_conv(in_ch, out_ch, kernel_size, stride, padding):
            return nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
            )

        self.layers = nn.Sequential(
            spectral_norm_conv(3, 64, 4, 2, 1),  # 3 channels for RGB
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm_conv(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm_conv(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm_conv(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            SelfAttention(512),

            spectral_norm_conv(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(1024, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================================
# AGRICULTURAL-SPECIFIC LOSS FUNCTIONS
# ============================================================================

class ColorPreservationLoss(nn.Module):
    """
    Preserves color information critical for crop disease identification.
    Agricultural images rely heavily on color for disease symptoms.
    """
    def __init__(self):
        super(ColorPreservationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, generated, target):
        # RGB color loss
        color_loss = self.mse_loss(generated, target)
        
        # HSV color space loss (important for disease hue variations)
        gen_hsv = self.rgb_to_hsv(generated)
        target_hsv = self.rgb_to_hsv(target)
        hsv_loss = self.mse_loss(gen_hsv, target_hsv)
        
        return color_loss + 0.5 * hsv_loss

    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space"""
        # Denormalize from [-1, 1] to [0, 1]
        rgb = (rgb + 1) / 2
        
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        
        max_rgb, argmax_rgb = rgb.max(1)
        min_rgb, _ = rgb.min(1)
        
        diff = max_rgb - min_rgb
        
        # Hue calculation
        h = torch.zeros_like(max_rgb)
        mask = diff > 0
        
        # Red is max
        h[mask & (argmax_rgb == 0)] = (60 * ((g[mask & (argmax_rgb == 0)] - b[mask & (argmax_rgb == 0)]) / diff[mask & (argmax_rgb == 0)]) + 360) % 360
        # Green is max
        h[mask & (argmax_rgb == 1)] = (60 * ((b[mask & (argmax_rgb == 1)] - r[mask & (argmax_rgb == 1)]) / diff[mask & (argmax_rgb == 1)]) + 120)
        # Blue is max
        h[mask & (argmax_rgb == 2)] = (60 * ((r[mask & (argmax_rgb == 2)] - g[mask & (argmax_rgb == 2)]) / diff[mask & (argmax_rgb == 2)]) + 240)
        
        # Saturation
        s = torch.zeros_like(max_rgb)
        s[max_rgb > 0] = diff[max_rgb > 0] / max_rgb[max_rgb > 0]
        
        # Value
        v = max_rgb
        
        return torch.stack([h/360, s, v], dim=1)


class TexturePreservationLoss(nn.Module):
    """
    Preserves leaf texture patterns critical for disease identification.
    Uses Gabor-like filters to capture texture at multiple orientations.
    """
    def __init__(self):
        super(TexturePreservationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        
        # Create texture filters at different orientations
        self.register_buffer('filters', self._create_texture_filters())

    def _create_texture_filters(self):
        """Create oriented texture filters"""
        filters = []
        for angle in [0, 45, 90, 135]:
            theta = np.radians(angle)
            kernel = self._gabor_kernel(theta)
            filters.append(kernel)
        return torch.stack(filters).unsqueeze(1)

    def _gabor_kernel(self, theta, sigma=3.0, lambd=10.0, gamma=0.5):
        """Create Gabor filter kernel"""
        kernel_size = 15
        x = np.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        y = np.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        x, y = np.meshgrid(x, y)
        
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        kernel = np.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2) * \
                 np.cos(2 * np.pi * x_theta / lambd)
        
        return torch.FloatTensor(kernel)

    def get_texture_features(self, x):
        """Extract texture features using filters"""
        # Apply to each RGB channel and average
        features = []
        for c in range(x.size(1)):
            channel = x[:, c:c+1, :, :]
            texture = F.conv2d(channel, self.filters, padding=7)
            features.append(texture)
        return torch.cat(features, dim=1)

    def forward(self, generated, target):
        gen_texture = self.get_texture_features(generated)
        target_texture = self.get_texture_features(target)
        return self.mse_loss(gen_texture, target_texture)


class AgriculturalPerceptualLoss(nn.Module):
    """
    Perceptual loss using features relevant for agricultural images.
    Uses pretrained ResNet which captures natural image features better than VGG.
    """
    def __init__(self):
        super(AgriculturalPerceptualLoss, self).__init__()
        # ResNet pretrained on ImageNet (includes many plant categories)
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, generated, target):
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        return self.mse_loss(gen_features, target_features)


class AgriculturalCombinedLoss(nn.Module):
    """
    Comprehensive loss function for agricultural super-resolution.
    Balances pixel accuracy, perceptual quality, color preservation, and texture.
    """
    def __init__(self, pixel_weight=1.0, perceptual_weight=0.2, 
                 color_weight=0.6, texture_weight=0.4, ssim_weight=0.3):
        super(AgriculturalCombinedLoss, self).__init__()
        
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.color_weight = color_weight
        self.texture_weight = texture_weight
        self.ssim_weight = ssim_weight

        # Import PixelLoss and StructuralSimilarityLoss from training_classes.py
        from training_classes import PixelLoss, StructuralSimilarityLoss
        
        self.pixel_loss = PixelLoss('l1')  # L1 works well for natural images
        self.perceptual_loss = AgriculturalPerceptualLoss()
        self.color_loss = ColorPreservationLoss()
        self.texture_loss = TexturePreservationLoss()
        self.ssim_loss = StructuralSimilarityLoss(channels=3)  # 3 channels for RGB

    def update_weights(self, pixel_weight, perceptual_weight, color_weight, 
                      texture_weight, ssim_weight):
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.color_weight = color_weight
        self.texture_weight = texture_weight
        self.ssim_weight = ssim_weight

    def forward(self, generated, target):
        pixel_loss = self.pixel_loss(generated, target)
        perceptual_loss = self.perceptual_loss(generated, target)
        color_loss = self.color_loss(generated, target)
        texture_loss = self.texture_loss(generated, target)
        ssim_loss = self.ssim_loss(generated, target)

        total_loss = (self.pixel_weight * pixel_loss +
                     self.perceptual_weight * perceptual_loss +
                     self.color_weight * color_loss +
                     self.texture_weight * texture_loss +
                     self.ssim_weight * ssim_loss)

        return total_loss, pixel_loss, perceptual_loss, color_loss, texture_loss, ssim_loss


# ============================================================================
# AGRICULTURAL-SPECIFIC SCHEDULER
# ============================================================================

class AgriculturalLossScheduler:
    """
    Loss weight scheduler optimized for crop disease training.
    
    Training progression:
    - Early: Focus on pixel accuracy and color preservation
    - Middle: Add texture details
    - Late: Refine perceptual quality and structure
    """
    def __init__(self, total_epochs=10):
        self.total_epochs = total_epochs

    def get_weights(self, epoch):
        """Progressive weight scheduling for agricultural images"""
        progress = epoch / self.total_epochs

        # Pixel weight: constant, foundational
        pixel_weight = 1.0
        
        # Color weight: high throughout (critical for disease identification)
        color_weight = 0.7 - 0.1 * progress  # Slight decrease
        
        # Texture weight: increase gradually
        texture_weight = 0.2 + 0.4 * progress
        
        # SSIM weight: increase for structural quality
        ssim_weight = 0.2 + 0.3 * progress
        
        # Perceptual weight: increase slowly
        perceptual_weight = 0.1 + 0.2 * progress

        return pixel_weight, perceptual_weight, color_weight, texture_weight, ssim_weight


# ============================================================================
# AGRICULTURAL-SPECIFIC TRAINER
# ============================================================================

class AgriculturalTrainer:
    """
    Specialized trainer for agricultural image super-resolution.
    Adapted from MedicalTrainer with agricultural-specific modifications.
    """
    def __init__(self, generator, discriminator, device, total_epochs=10):
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
        self.content_weight = 60.0  # Balanced weight for agricultural images

        # Agricultural-specific loss
        self.content_loss = AgriculturalCombinedLoss().to(device)
        self.loss_scheduler = AgriculturalLossScheduler(total_epochs=total_epochs)

        # Agricultural-specific tracking
        self.color_loss_history = []
        self.texture_loss_history = []
        self.ssim_loss_history = []

        self.update_optimizers()

    def update_optimizers(self):
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.g_lr, betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.d_lr, betas=(0.5, 0.999)
        )

    def update_loss_weights(self, epoch):
        """Update agricultural-specific loss weights"""
        weights = self.loss_scheduler.get_weights(epoch)
        pixel_w, perceptual_w, color_w, texture_w, ssim_w = weights

        self.content_loss.update_weights(pixel_w, perceptual_w, color_w, texture_w, ssim_w)
        print(f"Agricultural weights - Pixel:{pixel_w:.2f} Color:{color_w:.2f} "
              f"Texture:{texture_w:.2f} SSIM:{ssim_w:.2f} Perceptual:{perceptual_w:.2f}")

    def adapt_training_schedule(self, d_loss, g_loss):
        """Adaptive training schedule"""
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
        """Single training step"""
        batch_size = low_res.size(0)
        real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9
        fake_labels = torch.ones(batch_size, 1).to(self.device) * 0.1

        d_loss, g_tot_loss, g_adv_loss, g_cont_loss = 0, 0, 0, 0

        if step % self.d_train_freq == 0:
            d_loss = self.train_discriminator(low_res, high_res, real_labels, fake_labels)
        if step % self.g_train_freq == 0:
            g_tot_loss, g_adv_loss, g_cont_loss = self.train_generator(low_res, high_res, real_labels)

        return d_loss, g_tot_loss, g_adv_loss, g_cont_loss

    def train_discriminator(self, low_res, high_res, real_labels, fake_labels):
        """Train discriminator"""
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
        """Train generator"""
        self.g_optimizer.zero_grad()

        fake_images = self.generator(low_res)

        # Adversarial loss
        fake_pred = self.discriminator(fake_images)
        adversarial_loss = self.adversarial_criterion(fake_pred, real_labels)

        # Agricultural content loss with detailed breakdown
        losses = self.content_loss(fake_images, high_res)
        content_loss, pixel_loss, perceptual_loss, color_loss, texture_loss, ssim_loss = losses

        # Track agricultural-specific losses
        self.color_loss_history.append(color_loss.item())
        self.texture_loss_history.append(texture_loss.item())
        self.ssim_loss_history.append(ssim_loss.item())

        # Total generator loss
        weighted_content_loss = self.content_weight * content_loss
        g_total_loss = adversarial_loss + weighted_content_loss

        g_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.g_optimizer.step()

        return g_total_loss.item(), adversarial_loss.item(), content_loss.item()

    def print_loss_summary(self, epoch):
        """Print detailed agricultural loss breakdown"""
        if len(self.color_loss_history) >= 10:
            recent_color = np.mean(self.color_loss_history[-10:])
            recent_texture = np.mean(self.texture_loss_history[-10:])
            recent_ssim = np.mean(self.ssim_loss_history[-10:])

            print(f"Agricultural Metrics:")
            print(f"    Color Preservation: {recent_color:.4f}")
            print(f"    Texture Quality: {recent_texture:.4f}")
            print(f"    Structural Similarity: {recent_ssim:.4f}")
