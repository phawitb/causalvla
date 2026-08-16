"""Coverage-constrained loss allocation for COVER-VLA."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


COVER_GROUPS = (
    "clean",
    "brightness",
    "color",
    "noise",
    "blur",
    "shadow",
    "geometry",
    "composed",
)

_FLOORS = torch.tensor([0.50] + [0.025] * 6 + [0.15])
_ADAPTIVE_MASS = 0.20


class CoverCleanController(nn.Module):
    """Protect clean learning while retaining every augmentation floor."""

    def __init__(
        self,
        fast_decay: float = 0.90,
        slow_decay: float = 0.99,
        tolerance: float = 0.05,
        minimum_strength: float = 0.25,
        strength_decay: float = 0.90,
        recovery: float = 0.01,
        warmup_steps: int = 1000,
    ) -> None:
        super().__init__()
        if not 0.0 <= fast_decay < 1.0:
            raise ValueError("fast_decay must be in [0, 1)")
        if not 0.0 <= slow_decay < 1.0:
            raise ValueError("slow_decay must be in [0, 1)")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        if not 0.0 <= minimum_strength <= 1.0:
            raise ValueError("minimum_strength must be in [0, 1]")
        if not 0.0 < strength_decay <= 1.0:
            raise ValueError("strength_decay must be in (0, 1]")
        if not 0.0 <= recovery <= 1.0:
            raise ValueError("recovery must be in [0, 1]")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

        self.fast_decay = fast_decay
        self.slow_decay = slow_decay
        self.tolerance = tolerance
        self.minimum_strength = minimum_strength
        self.strength_decay = strength_decay
        self.recovery = recovery
        self.warmup_steps = warmup_steps
        self.register_buffer("fast_ema", torch.zeros(()))
        self.register_buffer("slow_ema", torch.zeros(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("robust_strength", torch.ones(()))

    @torch.no_grad()
    def update(self, clean_loss: Tensor) -> tuple[Tensor, Tensor]:
        value = clean_loss.detach().mean().to(self.fast_ema.device)
        if not bool(torch.isfinite(value)):
            return self.robust_strength.clone(), torch.zeros((), dtype=torch.bool, device=value.device)
        if not self.initialized:
            self.fast_ema.copy_(value)
            self.slow_ema.copy_(value)
            self.initialized.fill_(True)
        else:
            self.fast_ema.mul_(self.fast_decay).add_(value * (1.0 - self.fast_decay))
            self.slow_ema.mul_(self.slow_decay).add_(value * (1.0 - self.slow_decay))
        self.step.add_(1)
        if self.step.item() <= self.warmup_steps:
            return self.robust_strength.clone(), torch.zeros((), dtype=torch.bool, device=value.device)

        triggered = self.fast_ema > self.slow_ema * (1.0 + self.tolerance)
        if triggered:
            self.robust_strength.mul_(self.strength_decay).clamp_(min=self.minimum_strength)
        else:
            self.robust_strength.add_(self.recovery).clamp_(max=1.0)
        return self.robust_strength.clone(), triggered.clone()


class CoverageController(nn.Module):
    """Track per-group loss and allocate robust mass without losing coverage."""

    def __init__(
        self,
        ema_decay: float = 0.95,
        warmup_steps: int = 1000,
        temperature: float = 0.5,
        update_interval: int = 100,
        weight_min: float = 0.5,
        weight_max: float = 2.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if weight_min <= 0.0 or weight_max <= 0.0 or weight_min > weight_max:
            raise ValueError("weight bounds must be positive and ordered")
        if not weight_min <= 1.0 <= weight_max:
            raise ValueError("weight bounds must contain 1.0")

        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.temperature = temperature
        self.update_interval = update_interval
        self.weight_min = weight_min
        self.weight_max = weight_max

        uniform = _FLOORS.clone()
        uniform[1:] += _ADAPTIVE_MASS / 7
        self.register_buffer("loss_ema", torch.zeros(len(COVER_GROUPS)))
        self.register_buffer("initialized", torch.zeros(len(COVER_GROUPS), dtype=torch.bool))
        self.register_buffer("selection_counts", torch.zeros(len(COVER_GROUPS), dtype=torch.long))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("cached_mass", uniform)
        self.register_buffer("fallback", torch.ones((), dtype=torch.bool))

    @torch.no_grad()
    def target_mass(self, robust_strength: Tensor | float = 1.0) -> Tensor:
        floors = _FLOORS.to(device=self.loss_ema.device, dtype=self.loss_ema.dtype)
        uniform_scores = torch.full((7,), 1.0 / 7, device=floors.device, dtype=floors.dtype)
        ready = self.step.item() >= self.warmup_steps and bool(self.initialized[1:].all())
        scores = uniform_scores
        fallback = True
        if ready:
            augmented = self.loss_ema[1:]
            finite = torch.isfinite(augmented)
            if bool(finite.all()) and augmented.mean().abs().item() > torch.finfo(augmented.dtype).eps:
                normalized = augmented / augmented.mean().detach()
                scores = torch.softmax(normalized / self.temperature, dim=0)
                fallback = False

        strength = torch.as_tensor(robust_strength, device=floors.device, dtype=floors.dtype).clamp(0, 1)
        scores = uniform_scores + strength * (scores - uniform_scores)
        mass = floors.clone()
        mass[1:] += _ADAPTIVE_MASS * scores
        self.fallback.fill_(fallback)
        return mass

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device | str) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        mass = self.cached_mass.to(device=device)
        ids = torch.multinomial(mass, batch_size, replacement=True)
        self.selection_counts.add_(torch.bincount(ids.to(self.selection_counts.device), minlength=len(COVER_GROUPS)))
        return ids

    @torch.no_grad()
    def update(self, losses: Tensor, group_ids: Tensor, robust_strength: Tensor | float = 1.0) -> None:
        if losses.shape != group_ids.shape:
            raise ValueError("losses and group_ids must have the same shape")
        if losses.ndim != 1:
            raise ValueError("losses and group_ids must be one-dimensional")
        if group_ids.numel() and (group_ids.min().item() < 0 or group_ids.max().item() >= len(COVER_GROUPS)):
            raise ValueError("group_ids must be in [0, 7]")

        for group_id in range(len(COVER_GROUPS)):
            selected = (group_ids == group_id) & torch.isfinite(losses)
            if not bool(selected.any()):
                continue
            value = losses[selected].mean().detach().to(self.loss_ema.device)
            if self.initialized[group_id]:
                self.loss_ema[group_id].mul_(self.ema_decay).add_(value * (1.0 - self.ema_decay))
            else:
                self.loss_ema[group_id].copy_(value)
                self.initialized[group_id] = True

        self.step.add_(1)
        if self.step.item() % self.update_interval == 0:
            self.cached_mass.copy_(self.target_mass(robust_strength))

    @torch.no_grad()
    def importance_weights(self, group_ids: Tensor) -> Tensor:
        if group_ids.ndim != 1:
            raise ValueError("group_ids must be one-dimensional")
        if group_ids.numel() == 0:
            return torch.empty_like(group_ids, dtype=self.loss_ema.dtype)
        if group_ids.min().item() < 0 or group_ids.max().item() >= len(COVER_GROUPS):
            raise ValueError("group_ids must be in [0, 7]")
        counts = torch.bincount(group_ids, minlength=len(COVER_GROUPS)).to(dtype=self.loss_ema.dtype)
        frequency = counts / group_ids.numel()
        desired = self.cached_mass.to(group_ids.device)
        raw = desired[group_ids] / frequency[group_ids].clamp_min(torch.finfo(frequency.dtype).eps)

        low, high = 0.0, 16.0
        for _ in range(32):
            scale = (low + high) / 2
            mean = (raw * scale).clamp(self.weight_min, self.weight_max).mean().item()
            if mean < 1.0:
                low = scale
            else:
                high = scale
        return (raw * ((low + high) / 2)).clamp(self.weight_min, self.weight_max).detach()

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        total = self.selection_counts.sum().clamp_min(1)
        fractions = self.selection_counts.float() / total
        metrics: dict[str, float] = {"cover/fallback": float(self.fallback.item())}
        for index, name in enumerate(COVER_GROUPS):
            metrics[f"cover/group/{name}_fraction"] = fractions[index].item()
            metrics[f"cover/group/{name}_ema"] = self.loss_ema[index].item()
            metrics[f"cover/group/{name}_target_mass"] = self.cached_mass[index].item()
        return metrics
