"""Static policy-risk curriculum used by the RAPID-Lite ablation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .intervention_bank import RAPID_LITE_CANDIDATES, InterventionBank


class RiskWeightedInterventionSampler(nn.Module):
    """Sample guarded intervention candidates in proportion to policy risk.

    RAPID-Lite deliberately freezes the distribution obtained by the Phase 8
    profiler. An exploration floor prevents a candidate from receiving zero
    probability, while temperature controls concentration. Only training calls
    this module; it adds no parameters and no inference cost.
    """

    def __init__(self, temperature: float = 1.0, exploration_floor: float = 0.10):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= exploration_floor <= 1:
            raise ValueError("exploration_floor must be in [0, 1]")
        self.temperature = temperature
        self.exploration_floor = exploration_floor
        self.bank = InterventionBank()

    def probabilities(self, device: torch.device | str = "cpu") -> Tensor:
        risks = torch.tensor([item[2] for item in RAPID_LITE_CANDIDATES], device=device)
        exploited = torch.softmax(torch.log(risks) / self.temperature, dim=0)
        uniform = torch.full_like(exploited, 1 / len(RAPID_LITE_CANDIDATES))
        return (1 - self.exploration_floor) * exploited + self.exploration_floor * uniform

    def forward(self, images: list[Tensor]) -> tuple[list[Tensor], Tensor]:
        if not images:
            raise ValueError("images must contain at least one camera view")
        batch_size = images[0].shape[0]
        choices = torch.multinomial(
            self.probabilities(images[0].device), batch_size, replacement=True
        )
        mixed = [image.clone() for image in images]
        # Apply only candidates selected by this batch. This retains a single VLA
        # forward while avoiding work for absent curriculum arms.
        for index, (family, intensity, _) in enumerate(RAPID_LITE_CANDIDATES):
            mask = choices == index
            if not mask.any():
                continue
            augmented = self.bank.apply(images, family, intensity)
            broadcast = mask[:, None, None, None]
            mixed = [
                torch.where(broadcast, candidate, current)
                for current, candidate in zip(mixed, augmented, strict=True)
            ]
        return mixed, choices
