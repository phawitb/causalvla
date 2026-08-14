from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("rapid_mix")
@dataclass
class RapidMixConfig(SmolVLAConfig):
    """Coverage-preserving mixture of clean, broad DR and risk-guided views."""

    broad_probability: float = 0.25
    risk_probability: float = 0.25
    broad_intensity: float = 1.0
    risk_temperature: float = 1.0
    exploration_floor: float = 0.10
    profile_revision: str = "phase8-3seed-256samples-robust-risk-v1"

    def __post_init__(self):
        super().__post_init__()
        for name, probability in (
            ("broad_probability", self.broad_probability),
            ("risk_probability", self.risk_probability),
        ):
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.broad_probability + self.risk_probability > 1:
            raise ValueError("broad_probability + risk_probability must be <= 1")
        if self.broad_intensity < 0:
            raise ValueError("broad_intensity must be non-negative")
        if self.risk_temperature <= 0:
            raise ValueError("risk_temperature must be positive")
        if not 0 <= self.exploration_floor <= 1:
            raise ValueError("exploration_floor must be in [0, 1]")

    @property
    def clean_probability(self) -> float:
        return 1 - self.broad_probability - self.risk_probability
