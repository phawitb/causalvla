"""Branch sampling and residual intervention composition for Residual RAPID."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .intervention_bank import RAPID_LITE_CANDIDATES
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

    def compose(
        self, images: list[Tensor], broad_images: list[Tensor]
    ) -> tuple[list[Tensor], Tensor, Tensor]:
        if not images or not broad_images:
            raise ValueError("camera views must not be empty")
        if len(images) != len(broad_images):
            raise ValueError("camera view counts must match")
        for clean, broad in zip(images, broad_images, strict=True):
            if clean.shape != broad.shape:
                raise ValueError("clean and broad camera shapes must match")

        branch, choices = self.sample(images[0].shape[0], images[0].device)
        broad_mask = (branch >= 1)[:, None, None, None]
        mixed = [
            torch.where(broad_mask, broad, clean).clone()
            for clean, broad in zip(images, broad_images, strict=True)
        ]
        for index, (family, intensity, _) in enumerate(RAPID_LITE_CANDIDATES):
            mask = (branch == 2) & (choices == index)
            if not mask.any():
                continue
            overlaid = self.risk_sampler.bank.apply(broad_images, family, intensity)
            broadcast = mask[:, None, None, None]
            mixed = [
                torch.where(broadcast, overlay, current)
                for current, overlay in zip(mixed, overlaid, strict=True)
            ]
        return mixed, branch, choices
