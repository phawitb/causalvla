from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("cover_base")
@dataclass
class CoverBaseConfig(SmolVLAConfig):
    """Single-forward coverage-constrained group-robust training."""

    aug_intensity: float = 1.0
    cover_ema_decay: float = 0.95
    cover_warmup_steps: int = 1_000
    cover_temperature: float = 0.5
    cover_update_interval: int = 100
    cover_weight_min: float = 0.5
    cover_weight_max: float = 2.0
    enable_clean_safety: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.aug_intensity <= 1.0:
            raise ValueError("aug_intensity must be in [0, 1]")
        if not 0.0 <= self.cover_ema_decay < 1.0:
            raise ValueError("cover_ema_decay must be in [0, 1)")
        if self.cover_warmup_steps < 0:
            raise ValueError("cover_warmup_steps must be non-negative")
        if self.cover_temperature <= 0.0:
            raise ValueError("cover_temperature must be positive")
        if self.cover_update_interval <= 0:
            raise ValueError("cover_update_interval must be positive")
        if (
            self.cover_weight_min <= 0.0
            or self.cover_weight_max <= 0.0
            or self.cover_weight_min > self.cover_weight_max
            or not self.cover_weight_min <= 1.0 <= self.cover_weight_max
        ):
            raise ValueError("cover weight bounds must be positive, ordered, and contain 1")
