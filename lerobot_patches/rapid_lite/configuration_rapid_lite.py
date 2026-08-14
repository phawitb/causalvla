from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("rapid_lite")
@dataclass
class RapidLiteConfig(SmolVLAConfig):
    """Static risk-weighted online intervention curriculum."""

    aug_probability: float = 0.5
    risk_temperature: float = 1.0
    exploration_floor: float = 0.10
    profile_revision: str = "phase8-3seed-256samples-robust-risk-v1"

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.aug_probability <= 1:
            raise ValueError("aug_probability must be in [0, 1]")
        if self.risk_temperature <= 0:
            raise ValueError("risk_temperature must be positive")
        if not 0 <= self.exploration_floor <= 1:
            raise ValueError("exploration_floor must be in [0, 1]")
