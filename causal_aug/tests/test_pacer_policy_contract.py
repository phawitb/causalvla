from types import SimpleNamespace

import pytest
import torch

from causal_aug import CleanSafetyController, PacerContextualBandit
from lerobot.policies.pacer_lite.configuration_pacer_lite import PacerLiteConfig
from lerobot.policies.pacer_lite.modeling_pacer_lite import PacerLitePolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


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


def test_per_sample_task_loss_excludes_extra_dimensions_and_padding():
    policy = SimpleNamespace(config=SimpleNamespace(action_feature=torch.zeros(2)))
    losses = torch.tensor(
        [
            [[1.0, 3.0, 100.0], [5.0, 7.0, 100.0]],
            [[2.0, 4.0, 100.0], [20.0, 20.0, 100.0]],
        ]
    )
    padding = torch.tensor([[False, False], [False, True]])

    result = PacerLitePolicy._per_sample_task_loss(policy, losses, padding)

    assert result.tolist() == pytest.approx([4.0, 3.0])


def test_action_disagreement_excludes_padding_and_extra_dimensions():
    policy = SimpleNamespace(config=SimpleNamespace(action_feature=torch.zeros(2)))
    clean = torch.zeros(2, 2, 3)
    augmented = torch.tensor(
        [
            [[1.0, 1.0, 100.0], [3.0, 3.0, 100.0]],
            [[2.0, 2.0, 100.0], [50.0, 50.0, 100.0]],
        ]
    )
    padding = torch.tensor([[False, False], [False, True]])

    result = PacerLitePolicy._action_disagreement(policy, clean, augmented, padding)

    assert result.tolist() == pytest.approx([5.0, 4.0])


class _FakeFlowModel:
    def __init__(self):
        self.calls = []

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def sample_time(self, batch_size, device):
        return torch.zeros(batch_size, device=device)

    def forward_with_latent(self, *args):
        noise, time = args[-2:]
        self.calls.append((noise, time))
        batch, steps = args[5].shape[:2]
        value = float(len(self.calls))
        losses = torch.full((batch, steps, 3), value)
        latent = torch.zeros(batch, steps, 4)
        velocity = torch.full((batch, steps, 3), value - 1)
        return losses, latent, velocity


def test_training_forward_runs_exactly_two_branches_with_shared_flow_target():
    flow_model = _FakeFlowModel()
    actions = torch.zeros(2, 2, 2)
    supplied_noise = torch.randn_like(actions)
    supplied_time = torch.rand(2)
    bandit = PacerContextualBandit(warmup_steps=10, families=("brightness",))
    safety = CleanSafetyController(warmup_steps=10)
    fake_policy = SimpleNamespace(
        config=SimpleNamespace(
            adapt_to_pi_aloha=False,
            action_feature=torch.zeros(2),
            aug_intensity=0.5,
            max_loss_ratio=2.0,
            overhard_penalty=2.0,
            disagreement_clip=1.0,
        ),
        model=flow_model,
        bandit=bandit,
        safety=safety,
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

    loss, info = PacerLitePolicy.forward(
        fake_policy, batch, noise=supplied_noise, time=supplied_time
    )

    assert len(flow_model.calls) == 2
    assert all(call[0] is supplied_noise and call[1] is supplied_time for call in flow_model.calls)
    assert loss.item() == pytest.approx(1.5)
    assert info["loss_task_clean"] == pytest.approx(1.0)
    assert info["loss_task_augmented"] == pytest.approx(2.0)
    assert info["pacer/augmented_weight"] == pytest.approx(0.5)
    assert info["pacer/action_disagreement"] == pytest.approx(1.0)
    assert info["pacer/select/brightness"] == pytest.approx(1.0)
    assert bandit.steps.item() == 1


def test_policy_inference_is_inherited_from_smolvla():
    assert "select_action" not in PacerLitePolicy.__dict__
    assert "predict_action_chunk" not in PacerLitePolicy.__dict__
