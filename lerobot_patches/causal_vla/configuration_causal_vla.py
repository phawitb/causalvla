from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("causal_vla")
@dataclass
class CausalVLAConfig(SmolVLAConfig):
    """CausalVLA-v2 configuration for paired clean/augmented supervision."""

    # Counterfactual augmentation
    n_counterfactual: int = 1        # V2 uses one paired augmented view per clean sample
    aug_intensity: float = 1.0       # Augmentation strength (0.0 = none, 1.0 = full)

    # Paired supervised task losses (CausalVLA-v2)
    clean_task_weight: float = 0.5
    augmented_task_weight: float = 0.5

    # Loss weights
    lambda_latent: float = 0.0      # Disabled until paired supervision is validated
    lambda_action: float = 0.0      # Disabled: Phase 6 showed action consistency hurt success
    lambda_smooth: float = 0.0      # Disabled: velocity smoothness is not trajectory smoothness

    # Ablation switches
    use_latent_loss: bool = False   # Enable only for a controlled v2 ablation
    use_action_loss: bool = False   # Enable only with a small weight and stop-gradient teacher

    def __post_init__(self):
        super().__post_init__()
        if self.n_counterfactual < 1:
            raise ValueError("n_counterfactual must be at least 1 for paired supervision")
        if self.clean_task_weight < 0 or self.augmented_task_weight < 0:
            raise ValueError("task loss weights must be non-negative")
        if self.clean_task_weight + self.augmented_task_weight <= 0:
            raise ValueError("at least one task loss weight must be positive")
