import pytest
import torch

from causal_aug import LinearConsistencyWarmup


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, 0.0),
        (1_000, 0.005),
        (2_500, 0.0125),
        (5_000, 0.025),
        (7_500, 0.0375),
        (10_000, 0.05),
        (25_000, 0.05),
    ],
)
def test_linear_warmup_increases_continuously_and_caps_at_target(step, expected):
    schedule = LinearConsistencyWarmup(target=0.05, warmup_steps=10_000)

    schedule.step.fill_(step)

    assert schedule.value() == pytest.approx(expected)


def test_linear_warmup_advances_one_training_step_and_survives_checkpoint_roundtrip():
    schedule = LinearConsistencyWarmup(target=0.05, warmup_steps=10_000)
    schedule.step.fill_(4_321)
    state = schedule.state_dict()

    restored = LinearConsistencyWarmup(target=0.05, warmup_steps=10_000)
    restored.load_state_dict(state)
    restored.advance()

    assert restored.step.item() == 4_322
    assert restored.value() == pytest.approx(0.02161)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": -0.01, "warmup_steps": 10_000},
        {"target": 0.05, "warmup_steps": 0},
    ],
)
def test_linear_warmup_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LinearConsistencyWarmup(**kwargs)
