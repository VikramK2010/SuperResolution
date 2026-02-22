"""
Dry-run smoke test for GAN training pipeline.
Uses synthetic tensors on CPU — no real data, no GPU, no network downloads needed.
Tests: RaGAN losses, EMA, discriminator/generator forward passes,
       trainer train_step, pre-training loop, LR schedulers.
"""

import torch
import torch.nn as nn

# ── 1. gan_utils: RaGAN losses + EMA ──────────────────────────────────────────

print("=" * 60)
print("TEST 1: gan_utils — RaGAN losses + EMA")
print("=" * 60)

from gan_utils import relativistic_discriminator_loss, relativistic_generator_loss, EMAModel

# Synthetic logits (batch=4, output=1)
real_logits = torch.randn(4, 1)
fake_logits = torch.randn(4, 1)

d_loss = relativistic_discriminator_loss(real_logits, fake_logits)
g_loss = relativistic_generator_loss(real_logits, fake_logits)

assert d_loss.dim() == 0, f"d_loss should be scalar, got shape {d_loss.shape}"
assert g_loss.dim() == 0, f"g_loss should be scalar, got shape {g_loss.shape}"
assert d_loss.item() > 0, f"d_loss should be positive, got {d_loss.item()}"
assert g_loss.item() > 0, f"g_loss should be positive, got {g_loss.item()}"
print(f"  d_loss = {d_loss.item():.4f}")
print(f"  g_loss = {g_loss.item():.4f}")

# EMA on a tiny model
tiny = nn.Linear(4, 1)
ema = EMAModel(tiny, decay=0.99)
old_weight = tiny.weight.data.clone()
# Simulate a param update
tiny.weight.data += 0.1
ema.update(tiny)
# Shadow should be between old and new
shadow_weight = ema.shadow['weight']
assert not torch.equal(shadow_weight, old_weight), "EMA shadow should have moved"
assert not torch.equal(shadow_weight, tiny.weight.data), "EMA shadow should lag behind current"
# apply_shadow / restore round-trip
ema.apply_shadow(tiny)
assert torch.equal(tiny.weight.data, shadow_weight), "apply_shadow should set model to shadow"
ema.restore(tiny)
assert not torch.equal(tiny.weight.data, shadow_weight), "restore should revert to training weights"
print("  EMA update / apply_shadow / restore: OK")
print("  PASSED\n")


# ── 2. Discriminators: no BatchNorm, no Sigmoid, raw logits ──────────────────

print("=" * 60)
print("TEST 2: Discriminators — raw logits (no Sigmoid)")
print("=" * 60)

from discriminators import SpectralNormDiscriminator
from agricultural_sr import AgriculturalDiscriminator

# Check no BatchNorm2d or Sigmoid in either discriminator
for name, DiscClass in [("SpectralNormDiscriminator", SpectralNormDiscriminator),
                         ("AgriculturalDiscriminator", AgriculturalDiscriminator)]:
    disc = DiscClass()
    modules = list(disc.modules())
    bn_modules = [m for m in modules if isinstance(m, nn.BatchNorm2d)]
    sig_modules = [m for m in modules if isinstance(m, nn.Sigmoid)]
    assert len(bn_modules) == 0, f"{name} still has {len(bn_modules)} BatchNorm2d layers"
    assert len(sig_modules) == 0, f"{name} still has {len(sig_modules)} Sigmoid layers"
    print(f"  {name}: no BatchNorm2d, no Sigmoid")

# Forward pass — output should be unbounded logits
x = torch.randn(2, 3, 256, 256)
for name, DiscClass in [("SpectralNormDiscriminator", SpectralNormDiscriminator),
                         ("AgriculturalDiscriminator", AgriculturalDiscriminator)]:
    disc = DiscClass()
    disc.eval()
    with torch.no_grad():
        out = disc(x)
    assert out.shape == (2, 1), f"{name} output shape wrong: {out.shape}"
    # Raw logits can be negative (Sigmoid would clamp to [0,1])
    print(f"  {name} output range: [{out.min().item():.3f}, {out.max().item():.3f}]")

print("  PASSED\n")


# ── 3. EnhancedTrainer (pixel loss — avoids VGG download) ───────────────────

print("=" * 60)
print("TEST 3: EnhancedTrainer — TTUR + RaGAN train_step")
print("=" * 60)

from generators import ResidualGenerator
from training_classes import EnhancedTrainer

gen = ResidualGenerator(num_residual_blocks=2)
disc = SpectralNormDiscriminator()

trainer = EnhancedTrainer(gen, disc, device=torch.device('cpu'),
                          loss_type='pixel', total_epochs=5)

# Verify TTUR LRs
assert trainer.g_lr == 1e-4, f"g_lr should be 1e-4, got {trainer.g_lr}"
assert trainer.d_lr == 4e-4, f"d_lr should be 4e-4, got {trainer.d_lr}"
print(f"  TTUR LRs: g={trainer.g_lr}, d={trainer.d_lr}")

# Verify adversarial/content weights
assert trainer.adversarial_weight == 0.01, f"adversarial_weight should be 0.01, got {trainer.adversarial_weight}"
assert trainer.content_weight == 1.0, f"content_weight should be 1.0, got {trainer.content_weight}"
print(f"  Weights: adv={trainer.adversarial_weight}, content={trainer.content_weight}")

# Verify no BCELoss attribute
assert not hasattr(trainer, 'adversarial_criterion'), "Should not have adversarial_criterion (BCELoss removed)"
print("  No BCELoss: OK")

# Verify no adapt_training_schedule
assert not hasattr(trainer, 'adapt_training_schedule'), "adapt_training_schedule should be removed"
print("  No adapt_training_schedule: OK")

# Verify schedulers exist
assert hasattr(trainer, 'g_scheduler'), "Should have g_scheduler"
assert hasattr(trainer, 'd_scheduler'), "Should have d_scheduler"
print("  Cosine schedulers: OK")

# Run a train_step with synthetic data
low_res = torch.randn(2, 3, 64, 64)
high_res = torch.randn(2, 3, 256, 256)

d_loss, g_total, g_adv, g_content = trainer.train_step(low_res, high_res, step=0)
assert isinstance(d_loss, float), f"d_loss should be float, got {type(d_loss)}"
assert isinstance(g_total, float), f"g_total should be float, got {type(g_total)}"
assert d_loss > 0, f"d_loss should be positive, got {d_loss}"
assert g_total > 0, f"g_total should be positive, got {g_total}"
print(f"  train_step: d_loss={d_loss:.4f}, g_total={g_total:.4f}, g_adv={g_adv:.4f}, g_content={g_content:.4f}")

# Run a second step — make sure losses change (params updated)
d_loss2, g_total2, g_adv2, g_content2 = trainer.train_step(low_res, high_res, step=1)
print(f"  train_step2: d_loss={d_loss2:.4f}, g_total={g_total2:.4f}")

# Step schedulers
old_g_lr = trainer.g_optimizer.param_groups[0]['lr']
trainer.step_schedulers()
new_g_lr = trainer.g_optimizer.param_groups[0]['lr']
print(f"  LR scheduler step: g_lr {old_g_lr:.6f} -> {new_g_lr:.6f}")

print("  PASSED\n")


# ── 4. AgriculturalTrainer ───────────────────────────────────────────────────

print("=" * 60)
print("TEST 4: AgriculturalTrainer — TTUR + RaGAN train_step")
print("=" * 60)

from agricultural_sr import AgriculturalAttentionGenerator, AgriculturalTrainer

ag_gen = AgriculturalAttentionGenerator(num_residual_blocks=2)
ag_disc = AgriculturalDiscriminator()

ag_trainer = AgriculturalTrainer(ag_gen, ag_disc, device=torch.device('cpu'), total_epochs=5)

# Verify TTUR
assert ag_trainer.g_lr == 1e-4, f"g_lr should be 1e-4, got {ag_trainer.g_lr}"
assert ag_trainer.d_lr == 4e-4, f"d_lr should be 4e-4, got {ag_trainer.d_lr}"
print(f"  TTUR LRs: g={ag_trainer.g_lr}, d={ag_trainer.d_lr}")

# Verify weights
assert ag_trainer.adversarial_weight == 0.01
assert ag_trainer.content_weight == 1.0
print(f"  Weights: adv={ag_trainer.adversarial_weight}, content={ag_trainer.content_weight}")

# No BCELoss, no adapt_training_schedule
assert not hasattr(ag_trainer, 'adversarial_criterion')
assert not hasattr(ag_trainer, 'adapt_training_schedule')
print("  No BCELoss, no adapt_training_schedule: OK")

# Train step
low_res = torch.randn(2, 3, 64, 64)
high_res = torch.randn(2, 3, 256, 256)

d_loss, g_total, g_adv, g_content = ag_trainer.train_step(low_res, high_res, step=0)
print(f"  train_step: d_loss={d_loss:.4f}, g_total={g_total:.4f}, g_adv={g_adv:.4f}, g_content={g_content:.4f}")

# Check agricultural-specific tracking populated
assert len(ag_trainer.color_loss_history) == 1, "color_loss_history should have 1 entry"
assert len(ag_trainer.texture_loss_history) == 1, "texture_loss_history should have 1 entry"
assert len(ag_trainer.ssim_loss_history) == 1, "ssim_loss_history should have 1 entry"
print(f"  Agricultural tracking: color={ag_trainer.color_loss_history[-1]:.4f}, "
      f"texture={ag_trainer.texture_loss_history[-1]:.4f}, "
      f"ssim={ag_trainer.ssim_loss_history[-1]:.4f}")

# Weight scheduling
ag_trainer.update_loss_weights(0)
ag_trainer.step_schedulers()
print("  PASSED\n")


# ── 5. Pre-training loop simulation ─────────────────────────────────────────

print("=" * 60)
print("TEST 5: Pre-training loop (content-only, 2 batches)")
print("=" * 60)

# Simulate what training_loop.py does for pretrain_epochs
pt_gen = ResidualGenerator(num_residual_blocks=2)
pt_disc = SpectralNormDiscriminator()
pt_trainer = EnhancedTrainer(pt_gen, pt_disc, device=torch.device('cpu'),
                             loss_type='pixel', total_epochs=5)
pt_ema = EMAModel(pt_gen)

# Grab initial weights
w_before = pt_gen.initial[0].weight.data.clone()

# 2 pretrain steps
for i in range(2):
    pt_trainer.g_optimizer.zero_grad()
    fake = pt_gen(low_res)
    content_loss = pt_trainer.content_loss(fake, high_res)
    if isinstance(content_loss, tuple):
        content_loss = content_loss[0]
    content_loss.backward()
    pt_trainer.g_optimizer.step()
    pt_ema.update(pt_gen)
    print(f"  pretrain step {i}: content_loss={content_loss.item():.4f}")

w_after = pt_gen.initial[0].weight.data.clone()
assert not torch.equal(w_before, w_after), "Generator weights should have changed during pre-training"
print("  Generator weights updated: OK")

# EMA shadow should differ from current weights
pt_ema.apply_shadow(pt_gen)
w_ema = pt_gen.initial[0].weight.data.clone()
assert not torch.equal(w_ema, w_after), "EMA weights should differ from training weights"
pt_ema.restore(pt_gen)
print("  EMA shadow differs from training weights: OK")
print("  PASSED\n")


# ── 6. Multi-step loss convergence check ─────────────────────────────────────

print("=" * 60)
print("TEST 6: Multi-step training — losses stay finite and update")
print("=" * 60)

ms_gen = ResidualGenerator(num_residual_blocks=2)
ms_disc = SpectralNormDiscriminator()
ms_trainer = EnhancedTrainer(ms_gen, ms_disc, device=torch.device('cpu'),
                             loss_type='pixel', total_epochs=10)

d_losses = []
g_losses = []
for step in range(10):
    d_loss, g_total, g_adv, g_content = ms_trainer.train_step(low_res, high_res, step=step)
    d_losses.append(d_loss)
    g_losses.append(g_total)

import math
for i, (d, g) in enumerate(zip(d_losses, g_losses)):
    assert math.isfinite(d), f"Step {i}: d_loss is {d} (not finite)"
    assert math.isfinite(g), f"Step {i}: g_total is {g} (not finite)"

print(f"  10 steps, all losses finite")
print(f"  d_loss: {d_losses[0]:.4f} -> {d_losses[-1]:.4f}")
print(f"  g_loss: {g_losses[0]:.4f} -> {g_losses[-1]:.4f}")

# Verify losses aren't stuck at exactly the same value
d_unique = len(set(f"{x:.6f}" for x in d_losses))
g_unique = len(set(f"{x:.6f}" for x in g_losses))
assert d_unique > 1, f"d_loss stuck at same value across all 10 steps"
assert g_unique > 1, f"g_loss stuck at same value across all 10 steps"
print(f"  d_loss has {d_unique} unique values, g_loss has {g_unique} unique values")
print("  PASSED\n")


# ── Summary ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
