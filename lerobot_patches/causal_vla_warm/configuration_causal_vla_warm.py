from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..causal_vla.configuration_causal_vla import CausalVLAConfig


@PreTrainedConfig.register_subclass("causal_vla_warm")
@dataclass
class CausalVLAWarmConfig(CausalVLAConfig):
    """CausalVLA-v2 with continuous action-consistency warmup."""

    use_action_loss: bool = True
    lambda_action: float = 0.05
    action_warmup_steps: int = 10_000

    def __post_init__(self):
        super().__post_init__()
        if self.lambda_action < 0:
            raise ValueError("lambda_action must be non-negative")
        if self.action_warmup_steps < 1:
            raise ValueError("action_warmup_steps must be positive")
