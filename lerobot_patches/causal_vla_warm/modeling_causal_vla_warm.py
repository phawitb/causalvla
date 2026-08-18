"""CausalVLA-v2 with continuous action-consistency warmup."""

from ..causal_vla.modeling_causal_vla import CausalVLAPolicy
from .configuration_causal_vla_warm import CausalVLAWarmConfig


class CausalVLAWarmPolicy(CausalVLAPolicy):
    """Ramp action consistency linearly from zero during early training."""

    config_class = CausalVLAWarmConfig
    name = "causal_vla_warm"

    def __init__(self, config: CausalVLAWarmConfig, **kwargs):
        super().__init__(config, **kwargs)

        from causal_aug import LinearConsistencyWarmup

        self.consistency_schedule = LinearConsistencyWarmup(
            target=config.lambda_action,
            warmup_steps=config.action_warmup_steps,
        )

    def _action_consistency_weight(self) -> float:
        return self.consistency_schedule.value()

    def _after_training_forward(self) -> None:
        self.consistency_schedule.advance()
