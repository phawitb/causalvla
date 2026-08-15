import inspect

import pytest

from lerobot.policies.residual_rapid.configuration_residual_rapid import ResidualRapidConfig
from lerobot.policies.residual_rapid.modeling_residual_rapid import ResidualRapidPolicy


def test_config_defaults_preserve_model_f_coverage():
    cfg = ResidualRapidConfig(device="mps", push_to_hub=False)

    assert cfg.type == "residual_rapid"
    assert cfg.augmentation_probability == 0.5
    assert cfg.risk_overlay_probability == 0.25
    assert cfg.broad_intensity == 1.0
    assert cfg.risk_temperature == 1.0
    assert cfg.exploration_floor == 0.1
    assert cfg.profile_revision == "phase8-3seed-256samples-robust-risk-v1"
    assert cfg.clean_probability == pytest.approx(0.5)
    assert cfg.broad_only_probability == pytest.approx(0.375)
    assert cfg.residual_probability == pytest.approx(0.125)


@pytest.mark.parametrize(
    "overrides",
    [
        {"augmentation_probability": -0.1},
        {"augmentation_probability": 1.1},
        {"risk_overlay_probability": -0.1},
        {"risk_overlay_probability": 1.1},
        {"broad_intensity": -0.1},
        {"risk_temperature": 0.0},
        {"exploration_floor": -0.1},
        {"exploration_floor": 1.1},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        ResidualRapidConfig(device="mps", push_to_hub=False, **overrides)


def test_policy_uses_one_standard_model_forward():
    source = inspect.getsource(ResidualRapidPolicy.forward)

    assert source.count("self.model.forward(") == 1
    assert "forward_with_latent" not in source
    assert "loss_latent" not in source
    assert "loss_action" not in source
