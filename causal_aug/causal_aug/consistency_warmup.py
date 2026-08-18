"""Training-step schedules for consistency regularization."""

import torch
from torch import nn


class LinearConsistencyWarmup(nn.Module):
    """Increase a consistency weight linearly from zero to a fixed target."""

    def __init__(self, target: float, warmup_steps: int) -> None:
        super().__init__()
        if target < 0:
            raise ValueError("target must be non-negative")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        self.target = float(target)
        self.warmup_steps = int(warmup_steps)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def value(self) -> float:
        progress = min(self.step.item() / self.warmup_steps, 1.0)
        return self.target * progress

    @torch.no_grad()
    def advance(self) -> None:
        self.step.add_(1)
