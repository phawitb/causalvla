from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("online_dr")
@dataclass
class OnlineDRConfig(SmolVLAConfig):
    """Online domain-randomization baseline for CausalVLA-v2 ablations."""

    aug_probability: float = 0.5
    aug_intensity: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.aug_probability <= 1.0:
            raise ValueError("aug_probability must be in [0, 1]")
        if self.aug_intensity < 0.0:
            raise ValueError("aug_intensity must be non-negative")
