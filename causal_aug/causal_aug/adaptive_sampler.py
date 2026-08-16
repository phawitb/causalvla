"""Online contextual intervention selection for PACER-VLA."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .intervention_bank import INTERVENTION_FAMILIES, InterventionBank


class PacerContextualBandit(nn.Module):
    """Select interventions from policy-loss context and online reward feedback."""

    N_CONTEXTS = 3

    def __init__(
        self,
        temperature: float = 1.0,
        exploration_floor: float = 0.20,
        ema_decay: float = 0.95,
        warmup_steps: int = 1_000,
        families: tuple[str, ...] = INTERVENTION_FAMILIES,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= exploration_floor <= 1:
            raise ValueError("exploration_floor must be in [0, 1]")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not families:
            raise ValueError("families must not be empty")
        if any(family not in INTERVENTION_FAMILIES for family in families):
            raise ValueError("families contains an unknown intervention")

        self.temperature = temperature
        self.exploration_floor = exploration_floor
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.families = tuple(families)
        self.bank = InterventionBank()
        shape = (self.N_CONTEXTS, len(self.families))
        self.register_buffer("reward_ema", torch.zeros(shape))
        self.register_buffer("counts", torch.zeros(shape, dtype=torch.long))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def assign_context(self, clean_losses: Tensor) -> Tensor:
        if clean_losses.ndim != 1 or clean_losses.numel() == 0:
            raise ValueError("clean_losses must be a non-empty 1D tensor")
        if not torch.isfinite(clean_losses).all():
            raise ValueError("clean_losses must be finite")

        order = torch.argsort(clean_losses, stable=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(clean_losses.numel(), device=clean_losses.device)
        return torch.clamp(ranks * self.N_CONTEXTS // clean_losses.numel(), max=2)

    def probabilities(self, contexts: Tensor) -> Tensor:
        if contexts.ndim != 1:
            raise ValueError("contexts must be a 1D tensor")
        if contexts.numel() and (contexts.min() < 0 or contexts.max() >= self.N_CONTEXTS):
            raise ValueError("contexts contains an invalid context index")

        n_arms = len(self.families)
        uniform = torch.full(
            (contexts.numel(), n_arms),
            1 / n_arms,
            device=contexts.device,
            dtype=self.reward_ema.dtype,
        )
        if self.steps.item() < self.warmup_steps or contexts.numel() == 0:
            return uniform

        rewards = self.reward_ema.to(contexts.device)[contexts]
        observed = self.counts.to(contexts.device)[contexts].sum(dim=1) > 0
        mean = rewards.mean(dim=1, keepdim=True)
        scale = rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        exploited = torch.softmax((rewards - mean) / scale / self.temperature, dim=1)
        exploited = torch.where(observed[:, None], exploited, uniform)
        return (1 - self.exploration_floor) * exploited + self.exploration_floor * uniform

    def sample(self, clean_losses: Tensor) -> tuple[Tensor, Tensor]:
        contexts = self.assign_context(clean_losses)
        choices = torch.multinomial(self.probabilities(contexts), 1).squeeze(1)
        return contexts, choices

    def apply(self, images: list[Tensor], choices: Tensor, intensity: float) -> list[Tensor]:
        if not images:
            raise ValueError("images must contain at least one camera view")
        batch_size = images[0].shape[0]
        if any(image.shape[0] != batch_size for image in images):
            raise ValueError("all camera views must have the same batch size")
        if choices.ndim != 1 or choices.numel() != batch_size:
            raise ValueError("choices must contain one arm index per sample")
        if choices.numel() and (choices.min() < 0 or choices.max() >= len(self.families)):
            raise ValueError("choices contains an invalid arm index")
        if not 0 <= intensity <= 1:
            raise ValueError("intensity must be in [0, 1]")

        mixed = [image.clone() for image in images]
        for index, family in enumerate(self.families):
            mask = choices == index
            if not mask.any():
                continue
            augmented = self.bank.apply(images, family, intensity)
            broadcast = mask[:, None, None, None]
            mixed = [
                torch.where(broadcast, candidate, current)
                for current, candidate in zip(mixed, augmented, strict=True)
            ]
        return mixed

    @torch.no_grad()
    def update(self, contexts: Tensor, choices: Tensor, rewards: Tensor) -> Tensor:
        if not (contexts.ndim == choices.ndim == rewards.ndim == 1):
            raise ValueError("contexts, choices, and rewards must be 1D tensors")
        if not (contexts.numel() == choices.numel() == rewards.numel()):
            raise ValueError("contexts, choices, and rewards must have equal length")
        if contexts.numel() and (contexts.min() < 0 or contexts.max() >= self.N_CONTEXTS):
            raise ValueError("contexts contains an invalid context index")
        if choices.numel() and (choices.min() < 0 or choices.max() >= len(self.families)):
            raise ValueError("choices contains an invalid arm index")

        finite = torch.isfinite(rewards)
        rejected = (~finite).sum()
        contexts = contexts[finite].to(self.counts.device)
        choices = choices[finite].to(self.counts.device)
        rewards = rewards[finite].to(self.reward_ema.device, self.reward_ema.dtype)

        for context in range(self.N_CONTEXTS):
            for arm in range(len(self.families)):
                mask = (contexts == context) & (choices == arm)
                if not mask.any():
                    continue
                group_mean = rewards[mask].mean()
                previous_count = self.counts[context, arm].item()
                if previous_count == 0:
                    self.reward_ema[context, arm] = group_mean
                else:
                    self.reward_ema[context, arm].mul_(self.ema_decay).add_(
                        group_mean * (1 - self.ema_decay)
                    )
                self.counts[context, arm].add_(mask.sum())
        self.steps.add_(1)
        return rejected
