from types import SimpleNamespace

import pytest

from lerobot.policies.causal_vla.modeling_causal_vla import CausalVLAPolicy


def test_v2_default_action_consistency_weight_remains_constant():
    policy = SimpleNamespace(config=SimpleNamespace(lambda_action=0.0))

    assert CausalVLAPolicy._action_consistency_weight(policy) == 0.0


def test_v2_warm_uses_current_schedule_value_and_advances_after_forward():
    from causal_aug import LinearConsistencyWarmup
    from lerobot.policies.causal_vla_warm.modeling_causal_vla_warm import CausalVLAWarmPolicy

    schedule = LinearConsistencyWarmup(target=0.05, warmup_steps=10_000)
    schedule.step.fill_(5_000)
    policy = SimpleNamespace(consistency_schedule=schedule)

    assert CausalVLAWarmPolicy._action_consistency_weight(policy) == pytest.approx(0.025)

    CausalVLAWarmPolicy._after_training_forward(policy)

    assert schedule.step.item() == 5_001
