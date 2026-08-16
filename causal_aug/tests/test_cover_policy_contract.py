import inspect
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

import lerobot.policies

lerobot.policies.__path__.append(
    str(__import__("pathlib").Path(__file__).resolve().parents[2] / "lerobot_patches")
)

from lerobot.policies.cover_base.configuration_cover_base import CoverBaseConfig
from lerobot.policies.cover_base.modeling_cover_base import CoverBasePolicy
from lerobot.policies.cover_safe.configuration_cover_safe import CoverSafeConfig
from lerobot.policies.cover_safe.modeling_cover_safe import CoverSafePolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def test_cover_base_config_serializes_preregistered_defaults():
    cfg = CoverBaseConfig(device="mps", push_to_hub=False)
    data = asdict(cfg)

    assert cfg.type == "cover_base"
    assert data["aug_intensity"] == 1.0
    assert data["cover_ema_decay"] == 0.95
    assert data["cover_warmup_steps"] == 1000
    assert data["cover_temperature"] == 0.5
    assert data["cover_update_interval"] == 100
    assert data["cover_weight_min"] == 0.5
    assert data["cover_weight_max"] == 2.0
    assert data["enable_clean_safety"] is False


def test_cover_safe_config_serializes_preregistered_defaults():
    cfg = CoverSafeConfig(device="mps", push_to_hub=False)
    data = asdict(cfg)

    assert cfg.type == "cover_safe"
    assert data["enable_clean_safety"] is True
    assert data["clean_fast_decay"] == 0.90
    assert data["clean_slow_decay"] == 0.99
    assert data["clean_tolerance"] == 0.05
    assert data["minimum_robust_strength"] == 0.25
    assert data["robust_strength_decay"] == 0.90
    assert data["robust_strength_recovery"] == 0.01


@pytest.mark.parametrize(
    "overrides",
    [
        {"aug_intensity": -0.1}, {"aug_intensity": 1.1},
        {"cover_ema_decay": -0.1}, {"cover_ema_decay": 1.0},
        {"cover_warmup_steps": -1}, {"cover_temperature": 0.0},
        {"cover_update_interval": 0}, {"cover_weight_min": 0.0},
        {"cover_weight_max": 0.0},
        {"cover_weight_min": 2.1, "cover_weight_max": 2.0},
    ],
)
def test_cover_base_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        CoverBaseConfig(device="mps", push_to_hub=False, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"clean_fast_decay": -0.1}, {"clean_fast_decay": 1.0},
        {"clean_slow_decay": -0.1}, {"clean_slow_decay": 1.0},
        {"clean_tolerance": -0.1}, {"minimum_robust_strength": -0.1},
        {"minimum_robust_strength": 1.1}, {"robust_strength_decay": 0.0},
        {"robust_strength_decay": 1.1}, {"robust_strength_recovery": -0.1},
        {"robust_strength_recovery": 1.1},
    ],
)
def test_cover_safe_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        CoverSafeConfig(device="mps", push_to_hub=False, **overrides)


def test_per_sample_task_loss_excludes_extra_dimensions_and_padding():
    policy = SimpleNamespace(config=SimpleNamespace(action_feature=torch.zeros(2)))
    losses = torch.tensor(
        [
            [[1.0, 3.0, 100.0], [5.0, 7.0, 100.0]],
            [[2.0, 4.0, 100.0], [20.0, 20.0, 100.0]],
        ]
    )
    padding = torch.tensor([[False, False], [False, True]])

    result = CoverBasePolicy._per_sample_task_loss(policy, losses, padding)

    assert result.tolist() == pytest.approx([4.0, 3.0])


class _FakeFlowModel:
    def __init__(self):
        self.calls = 0

    def forward(self, images, image_masks, tokens, token_masks, state, actions, noise, time):
        self.calls += 1
        batch, steps = actions.shape[:2]
        values = torch.tensor([1.0, 3.0], device=actions.device)[:batch, None, None]
        return values.expand(batch, steps, 3).clone().requires_grad_()


class _FakeCoverage:
    def __init__(self):
        self.updated = False
        self.cached_mass = torch.tensor([0.5, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.35])

    def sample(self, batch_size, device):
        return torch.tensor([0, 7], device=device)[:batch_size]

    def update(self, losses, group_ids, robust_strength=1.0):
        self.updated = True

    def importance_weights(self, group_ids):
        return torch.ones(group_ids.shape[0], device=group_ids.device)

    def metrics(self):
        return {"cover/fallback": 0.0}


def test_training_forward_runs_exactly_once_and_logs_groups():
    flow_model = _FakeFlowModel()
    coverage = _FakeCoverage()
    actions = torch.zeros(2, 2, 2)
    fake_policy = SimpleNamespace(
        config=SimpleNamespace(
            adapt_to_pi_aloha=False,
            action_feature=torch.zeros(2),
            aug_intensity=0.0,
            enable_clean_safety=False,
        ),
        model=flow_model,
        coverage=coverage,
        clean_controller=None,
        training=True,
        prepare_images=lambda batch: ([torch.zeros(2, 3, 8, 8)], [torch.ones(2)]),
        prepare_state=lambda batch: torch.zeros(2, 4),
        prepare_action=lambda batch: actions,
    )
    batch = {
        ACTION: actions,
        OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        "action_is_pad": torch.zeros(2, 2, dtype=torch.bool),
    }

    loss, info = CoverBasePolicy.forward(fake_policy, batch)

    assert flow_model.calls == 1
    assert loss.item() == pytest.approx(2.0)
    assert coverage.updated
    assert info["cover/forward_count"] == 1.0
    assert info["cover/group/clean_fraction"] == pytest.approx(0.5)
    assert info["cover/group/composed_fraction"] == pytest.approx(0.5)
    assert info["cover/group/clean_loss"] == pytest.approx(1.0)
    assert info["cover/group/composed_loss"] == pytest.approx(3.0)


def test_policy_source_uses_one_standard_forward_and_no_auxiliary_losses():
    source = inspect.getsource(CoverBasePolicy.forward)
    assert source.count("self.model.forward(") == 1
    assert "forward_with_latent" not in source
    assert "loss_latent" not in source
    assert "loss_action" not in source


def test_policy_inference_is_inherited_from_smolvla():
    for policy in (CoverBasePolicy, CoverSafePolicy):
        assert "select_action" not in policy.__dict__
        assert "predict_action_chunk" not in policy.__dict__
