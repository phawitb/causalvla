"""Reward shaping and clean-task protection for PACER-VLA."""

from __future__ import annotations

import torch
from torch import Tensor, nn


@torch.no_grad()
def productive_difficulty_reward(
    clean_loss: Tensor,
    augmented_loss: Tensor,
    disagreement: Tensor,
    max_loss_ratio: float,
    overhard_penalty: float,
    disagreement_clip: float,
) -> tuple[Tensor, Tensor]:
    """Reward action sensitivity only while the augmented target stays learnable."""
    if clean_loss.shape != augmented_loss.shape or clean_loss.shape != disagreement.shape:
        raise ValueError("clean_loss, augmented_loss, and disagreement must have equal shapes")
    if max_loss_ratio <= 0:
        raise ValueError("max_loss_ratio must be positive")
    if overhard_penalty <= 0:
        raise ValueError("overhard_penalty must be positive")
    if disagreement_clip <= 0:
        raise ValueError("disagreement_clip must be positive")

    clean = clean_loss.detach()
    augmented = augmented_loss.detach()
    sensitivity = disagreement.detach().clamp(min=0, max=disagreement_clip)
    epsilon = torch.finfo(clean.dtype).eps
    ratio = augmented / clean.clamp_min(epsilon)
    finite = torch.isfinite(clean) & torch.isfinite(augmented) & torch.isfinite(sensitivity)
    penalty = torch.exp(-overhard_penalty * torch.relu(ratio - max_loss_ratio))
    reward = sensitivity * penalty
    reward = torch.where(finite & torch.isfinite(reward), reward, torch.zeros_like(reward))
    ratio = torch.nan_to_num(
        ratio,
        nan=0.0,
        posinf=torch.finfo(ratio.dtype).max,
        neginf=0.0,
    )
    return reward, ratio


class CleanSafetyController(nn.Module):
    """Adapt augmented supervision weight when clean loss degrades."""

    def __init__(
        self,
        max_weight: float = 0.50,
        min_weight: float = 0.10,
        tolerance: float = 0.05,
        weight_decay: float = 0.90,
        weight_recovery: float = 0.01,
        fast_decay: float = 0.90,
        slow_decay: float = 0.99,
        warmup_steps: int = 1_000,
    ):
        super().__init__()
        if not 0 <= min_weight <= max_weight <= 0.5:
            raise ValueError("weights must satisfy 0 <= min_weight <= max_weight <= 0.5")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if not 0 < weight_decay <= 1:
            raise ValueError("weight_decay must be in (0, 1]")
        if weight_recovery < 0:
            raise ValueError("weight_recovery must be non-negative")
        if not 0 <= fast_decay < 1 or not 0 <= slow_decay < 1:
            raise ValueError("EMA decays must be in [0, 1)")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

        self.max_weight = max_weight
        self.min_weight = min_weight
        self.tolerance = tolerance
        self.weight_decay = weight_decay
        self.weight_recovery = weight_recovery
        self.fast_decay = fast_decay
        self.slow_decay = slow_decay
        self.warmup_steps = warmup_steps
        self.register_buffer("fast_ema", torch.zeros(()))
        self.register_buffer("slow_ema", torch.zeros(()))
        self.register_buffer("augmented_weight", torch.tensor(float(max_weight)))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, clean_loss: Tensor) -> tuple[Tensor, Tensor]:
        value = clean_loss.detach().mean().to(self.fast_ema.device, self.fast_ema.dtype)
        false = torch.tensor(False, device=self.fast_ema.device)
        if not torch.isfinite(value):
            return self.augmented_weight.detach().clone(), false

        if not self.initialized.item():
            self.fast_ema.copy_(value)
            self.slow_ema.copy_(value)
            self.initialized.fill_(True)
        else:
            self.fast_ema.mul_(self.fast_decay).add_(value * (1 - self.fast_decay))
            self.slow_ema.mul_(self.slow_decay).add_(value * (1 - self.slow_decay))
        self.steps.add_(1)

        if self.steps.item() <= self.warmup_steps:
            self.augmented_weight.fill_(self.max_weight)
            return self.augmented_weight.detach().clone(), false

        triggered = self.fast_ema > self.slow_ema * (1 + self.tolerance)
        if triggered.item():
            updated = self.augmented_weight * self.weight_decay
        else:
            updated = self.augmented_weight + self.weight_recovery
        self.augmented_weight.copy_(updated.clamp(self.min_weight, self.max_weight))
        return self.augmented_weight.detach().clone(), triggered.detach().clone()
