from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("pacer_lite")
@dataclass
class PacerLiteConfig(SmolVLAConfig):
    """Two-forward policy-adaptive counterfactual training."""

    aug_intensity: float = 1.0
    bandit_temperature: float = 1.0
    exploration_floor: float = 0.20
    bandit_ema_decay: float = 0.95
    bandit_warmup_steps: int = 1_000
    max_loss_ratio: float = 2.0
    overhard_penalty: float = 2.0
    disagreement_clip: float = 1.0
    max_augmented_weight: float = 0.50
    min_augmented_weight: float = 0.10
    clean_tolerance: float = 0.05
    clean_weight_decay: float = 0.90
    clean_weight_recovery: float = 0.01
    fast_ema_decay: float = 0.90
    slow_ema_decay: float = 0.99

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.aug_intensity <= 1:
            raise ValueError("aug_intensity must be in [0, 1]")
        if self.bandit_temperature <= 0:
            raise ValueError("bandit_temperature must be positive")
        if not 0 <= self.exploration_floor <= 1:
            raise ValueError("exploration_floor must be in [0, 1]")
        if not 0 <= self.bandit_ema_decay < 1:
            raise ValueError("bandit_ema_decay must be in [0, 1)")
        if self.bandit_warmup_steps < 0:
            raise ValueError("bandit_warmup_steps must be non-negative")
        for name, value in (
            ("max_loss_ratio", self.max_loss_ratio),
            ("overhard_penalty", self.overhard_penalty),
            ("disagreement_clip", self.disagreement_clip),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.min_augmented_weight <= self.max_augmented_weight <= 0.5:
            raise ValueError(
                "weights must satisfy 0 <= min_augmented_weight <= max_augmented_weight <= 0.5"
            )
        if self.clean_tolerance < 0:
            raise ValueError("clean_tolerance must be non-negative")
        if not 0 < self.clean_weight_decay <= 1:
            raise ValueError("clean_weight_decay must be in (0, 1]")
        if self.clean_weight_recovery < 0:
            raise ValueError("clean_weight_recovery must be non-negative")
        if not 0 <= self.fast_ema_decay < 1 or not 0 <= self.slow_ema_decay < 1:
            raise ValueError("EMA decays must be in [0, 1)")
