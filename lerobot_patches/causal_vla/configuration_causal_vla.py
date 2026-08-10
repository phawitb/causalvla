from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("causal_vla")
@dataclass
class CausalVLAConfig(SmolVLAConfig):
    """CausalVLA extends SmolVLA with counterfactual augmentation and dual invariance losses."""

    # Counterfactual augmentation
    n_counterfactual: int = 3        # K: number of counterfactual images per sample
    aug_intensity: float = 1.0       # Augmentation strength (0.0 = none, 1.0 = full)

    # Loss weights
    lambda_latent: float = 5.0      # Weight for L_latent (counterfactual latent invariance)
    lambda_action: float = 1.0      # Weight for L_action (action consistency)
    lambda_smooth: float = 0.01     # Weight for L_smooth (temporal action smoothing)

    # Ablation switches
    use_latent_loss: bool = True    # False → Model D (ablation w/o L_latent)
    use_action_loss: bool = True    # False → Model E (ablation w/o L_action)
