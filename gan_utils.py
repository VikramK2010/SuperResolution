"""
gan_utils.py

Relativistic average GAN (RaGAN) loss functions and Exponential Moving Average
for stabilizing super-resolution GAN training.

References:
    - Jolicoeur-Martineau, "The relativistic discriminator", ICLR 2019
    - Wang et al., "ESRGAN", ECCVW 2018
"""

import copy

import torch
import torch.nn.functional as F


def relativistic_discriminator_loss(real_logits, fake_logits):
    """RaGAN discriminator loss: real should be more real than average fake."""
    d_real = F.binary_cross_entropy_with_logits(
        real_logits - fake_logits.mean(), torch.ones_like(real_logits))
    d_fake = F.binary_cross_entropy_with_logits(
        fake_logits - real_logits.mean(), torch.zeros_like(fake_logits))
    return (d_real + d_fake) / 2


def relativistic_generator_loss(real_logits, fake_logits):
    """RaGAN generator loss: fake should be more real than average real."""
    g_real = F.binary_cross_entropy_with_logits(
        real_logits - fake_logits.mean(), torch.zeros_like(real_logits))
    g_fake = F.binary_cross_entropy_with_logits(
        fake_logits - real_logits.mean(), torch.ones_like(fake_logits))
    return (g_real + g_fake) / 2


class EMAModel:
    """Exponential moving average of generator weights.

    Usage:
        ema = EMAModel(generator)
        # after each g_optimizer.step():
        ema.update(generator)
        # before eval/save:
        ema.apply_shadow(generator)
        # after eval/save (restore training weights):
        ema.restore(generator)
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for key, param in model.state_dict().items():
            self.shadow[key] = self.decay * self.shadow[key] + (1 - self.decay) * param

    def apply_shadow(self, model):
        self.backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model):
        if self.backup is not None:
            model.load_state_dict(self.backup)
            self.backup = None
