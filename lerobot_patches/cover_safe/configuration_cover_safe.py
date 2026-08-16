from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from ..cover_base.configuration_cover_base import CoverBaseConfig


@PreTrainedConfig.register_subclass("cover_safe")
@dataclass
class CoverSafeConfig(CoverBaseConfig):
    """COVER-Base with clean-retention control over robust strength."""

    enable_clean_safety: bool = True
    clean_fast_decay: float = 0.90
    clean_slow_decay: float = 0.99
    clean_tolerance: float = 0.05
    minimum_robust_strength: float = 0.25
    robust_strength_decay: float = 0.90
    robust_strength_recovery: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        if not self.enable_clean_safety:
            raise ValueError("cover_safe requires enable_clean_safety=True")
        if not 0.0 <= self.clean_fast_decay < 1.0:
            raise ValueError("clean_fast_decay must be in [0, 1)")
        if not 0.0 <= self.clean_slow_decay < 1.0:
            raise ValueError("clean_slow_decay must be in [0, 1)")
        if self.clean_tolerance < 0.0:
            raise ValueError("clean_tolerance must be non-negative")
        if not 0.0 <= self.minimum_robust_strength <= 1.0:
            raise ValueError("minimum_robust_strength must be in [0, 1]")
        if not 0.0 < self.robust_strength_decay <= 1.0:
            raise ValueError("robust_strength_decay must be in (0, 1]")
        if not 0.0 <= self.robust_strength_recovery <= 1.0:
            raise ValueError("robust_strength_recovery must be in [0, 1]")
