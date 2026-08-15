"""Branch sampling and residual intervention composition for Residual RAPID."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .risk_sampler import RiskWeightedInterventionSampler


class ResidualBranchSampler(nn.Module):
    """Sample clean, broad, and broad-plus-risk training branches."""

    def __init__(
        self,
        augmentation_probability: float,
        overlay_probability: float,
        risk_temperature: float = 1.0,
        exploration_floor: float = 0.10,
    ):
        super().__init__()
        if not 0 <= augmentation_probability <= 1:
            raise ValueError("augmentation_probability must be in [0, 1]")
        if not 0 <= overlay_probability <= 1:
            raise ValueError("overlay_probability must be in [0, 1]")
        self.augmentation_probability = augmentation_probability
        self.overlay_probability = overlay_probability
        self.risk_sampler = RiskWeightedInterventionSampler(
            temperature=risk_temperature,
            exploration_floor=exploration_floor,
        )

    def sample(self, batch_size: int, device: torch.device | str) -> tuple[Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        augmented = torch.rand(batch_size, device=device) < self.augmentation_probability
        overlaid = augmented & (torch.rand(batch_size, device=device) < self.overlay_probability)
        branch = torch.zeros(batch_size, dtype=torch.long, device=device)
        branch[augmented] = 1
        branch[overlaid] = 2
        choices = torch.multinomial(
            self.risk_sampler.probabilities(device), batch_size, replacement=True
        )
        return branch, choices
