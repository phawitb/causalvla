import pytest
import torch

from causal_aug import CleanSafetyController, productive_difficulty_reward


def test_reward_prefers_learnable_disagreement():
    clean = torch.tensor([1.0, 1.0])
    augmented = torch.tensor([1.5, 4.0])
    disagreement = torch.tensor([0.4, 0.4])

    reward, ratio = productive_difficulty_reward(
        clean, augmented, disagreement, max_loss_ratio=2.0, overhard_penalty=2.0,
        disagreement_clip=1.0,
    )

    assert ratio.tolist() == pytest.approx([1.5, 4.0])
    assert reward[0] > reward[1]


def test_reward_clips_disagreement_and_is_detached():
    disagreement = torch.tensor([0.5, 5.0], requires_grad=True)

    reward, _ = productive_difficulty_reward(
        torch.ones(2), torch.ones(2), disagreement, 2.0, 2.0, 1.0
    )

    assert reward.tolist() == pytest.approx([0.5, 1.0])
    assert not reward.requires_grad


def test_reward_is_finite_with_zero_clean_loss_and_nonfinite_input():
    reward, ratio = productive_difficulty_reward(
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, float("nan")]),
        torch.tensor([0.5, float("inf")]),
        2.0,
        2.0,
        1.0,
    )

    assert torch.isfinite(reward).all()
    assert torch.isfinite(ratio).all()
    assert reward[1].item() == 0.0


@pytest.mark.parametrize(
    ("max_loss_ratio", "overhard_penalty", "disagreement_clip"),
    [(0.0, 2.0, 1.0), (2.0, 0.0, 1.0), (2.0, 2.0, 0.0)],
)
def test_reward_rejects_invalid_hyperparameters(
    max_loss_ratio, overhard_penalty, disagreement_clip
):
    with pytest.raises(ValueError):
        productive_difficulty_reward(
            torch.ones(1), torch.ones(1), torch.ones(1),
            max_loss_ratio, overhard_penalty, disagreement_clip,
        )


def test_reward_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        productive_difficulty_reward(
            torch.ones(2), torch.ones(1), torch.ones(2), 2.0, 2.0, 1.0
        )


def test_safety_controller_decays_and_recovers_weight():
    controller = CleanSafetyController(
        warmup_steps=0, tolerance=0.05, weight_decay=0.5, weight_recovery=0.1
    )
    controller.fast_ema.fill_(2.0)
    controller.slow_ema.fill_(1.0)
    controller.initialized.fill_(True)

    weight, triggered = controller.update(torch.tensor(2.0))

    assert triggered.item() is True
    assert weight.item() == pytest.approx(0.25)

    controller.fast_ema.fill_(1.0)
    controller.slow_ema.fill_(1.0)
    weight, triggered = controller.update(torch.tensor(1.0))

    assert triggered.item() is False
    assert weight.item() == pytest.approx(0.35)


def test_safety_weight_never_leaves_bounds():
    controller = CleanSafetyController(
        min_weight=0.1, max_weight=0.5, warmup_steps=0, weight_decay=0.5
    )
    for _ in range(100):
        controller.fast_ema.fill_(10.0)
        controller.slow_ema.fill_(1.0)
        controller.initialized.fill_(True)
        controller.update(torch.tensor(10.0))

    assert controller.augmented_weight.item() == pytest.approx(0.1)


def test_safety_holds_maximum_weight_during_warmup():
    controller = CleanSafetyController(warmup_steps=2, weight_decay=0.5)

    first_weight, first_trigger = controller.update(torch.tensor(1.0))
    controller.fast_ema.fill_(10.0)
    controller.slow_ema.fill_(1.0)
    second_weight, second_trigger = controller.update(torch.tensor(10.0))

    assert first_weight.item() == pytest.approx(0.5)
    assert second_weight.item() == pytest.approx(0.5)
    assert not first_trigger.item() and not second_trigger.item()


def test_safety_ignores_nonfinite_loss():
    controller = CleanSafetyController(warmup_steps=0)
    before = {key: value.clone() for key, value in controller.state_dict().items()}

    _, triggered = controller.update(torch.tensor(float("nan")))

    assert not triggered.item()
    for key, value in controller.state_dict().items():
        assert torch.equal(value, before[key])


def test_safety_state_dict_round_trip():
    source = CleanSafetyController(warmup_steps=0)
    source.update(torch.tensor(2.0))
    target = CleanSafetyController(warmup_steps=0)

    target.load_state_dict(source.state_dict())

    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_weight": -0.1}, {"max_weight": 0.6},
        {"min_weight": 0.4, "max_weight": 0.3}, {"tolerance": -0.1},
        {"weight_decay": 0.0}, {"weight_decay": 1.1},
        {"weight_recovery": -0.1}, {"fast_decay": 1.0},
        {"slow_decay": -0.1}, {"warmup_steps": -1},
    ],
)
def test_safety_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CleanSafetyController(**kwargs)
