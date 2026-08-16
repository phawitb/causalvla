import pytest

from lerobot.policies.pacer_lite.configuration_pacer_lite import PacerLiteConfig


def test_pacer_config_serializes_preregistered_defaults():
    cfg = PacerLiteConfig(device="mps", push_to_hub=False)

    assert cfg.type == "pacer_lite"
    assert cfg.aug_intensity == 1.0
    assert cfg.bandit_temperature == 1.0
    assert cfg.exploration_floor == 0.2
    assert cfg.bandit_ema_decay == 0.95
    assert cfg.bandit_warmup_steps == 1000
    assert cfg.max_loss_ratio == 2.0
    assert cfg.overhard_penalty == 2.0
    assert cfg.disagreement_clip == 1.0
    assert cfg.max_augmented_weight == 0.5
    assert cfg.min_augmented_weight == 0.1
    assert cfg.clean_tolerance == 0.05
    assert cfg.clean_weight_decay == 0.9
    assert cfg.clean_weight_recovery == 0.01
    assert cfg.fast_ema_decay == 0.9
    assert cfg.slow_ema_decay == 0.99


@pytest.mark.parametrize(
    "overrides",
    [
        {"aug_intensity": -0.1}, {"aug_intensity": 1.1},
        {"bandit_temperature": 0.0}, {"exploration_floor": -0.1},
        {"exploration_floor": 1.1}, {"bandit_ema_decay": -0.1},
        {"bandit_ema_decay": 1.0}, {"bandit_warmup_steps": -1},
        {"max_loss_ratio": 0.0}, {"overhard_penalty": 0.0},
        {"disagreement_clip": 0.0}, {"min_augmented_weight": -0.1},
        {"max_augmented_weight": 0.6},
        {"min_augmented_weight": 0.4, "max_augmented_weight": 0.3},
        {"clean_tolerance": -0.1}, {"clean_weight_decay": 0.0},
        {"clean_weight_decay": 1.1}, {"clean_weight_recovery": -0.1},
        {"fast_ema_decay": 1.0}, {"slow_ema_decay": -0.1},
    ],
)
def test_pacer_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        PacerLiteConfig(device="mps", push_to_hub=False, **overrides)
