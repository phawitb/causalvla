import pytest
import torch

from causal_aug import COVER_GROUPS, CoverCleanController, CoverageController


def test_cover_groups_are_stable():
    assert COVER_GROUPS == (
        "clean",
        "brightness",
        "color",
        "noise",
        "blur",
        "shadow",
        "geometry",
        "composed",
    )


def test_target_mass_preserves_exact_floors_and_probability():
    controller = CoverageController(warmup_steps=0, update_interval=1)
    controller.loss_ema.copy_(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0]))
    controller.initialized.fill_(True)

    mass = controller.target_mass()

    assert mass.sum().item() == pytest.approx(1.0)
    assert mass[0].item() == pytest.approx(0.50)
    assert torch.all(mass[1:7] >= 0.025)
    assert mass[7].item() >= 0.15
    assert mass[7] > mass[1]


def test_uniform_warmup_mass_preserves_floors():
    controller = CoverageController(warmup_steps=10)

    mass = controller.target_mass()

    expected = torch.tensor([0.50] + [0.025 + 0.20 / 7] * 6 + [0.15 + 0.20 / 7])
    assert torch.allclose(mass, expected)


def test_sample_updates_selection_counts():
    torch.manual_seed(1000)
    controller = CoverageController()

    ids = controller.sample(128, "cpu")

    assert ids.shape == (128,)
    assert ids.dtype == torch.long
    assert ids.min() >= 0 and ids.max() < len(COVER_GROUPS)
    assert controller.selection_counts.sum().item() == 128


def test_importance_weights_are_detached_bounded_and_normalized():
    controller = CoverageController(warmup_steps=0, update_interval=1)
    ids = torch.tensor([0, 0, 0, 1, 7])

    weights = controller.importance_weights(ids)

    assert not weights.requires_grad
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights.min().item() >= 0.5
    assert weights.max().item() <= 2.0


def test_update_skips_absent_and_nonfinite_groups():
    controller = CoverageController(ema_decay=0.5, warmup_steps=0, update_interval=1)

    controller.update(torch.tensor([2.0, float("nan"), 4.0]), torch.tensor([1, 2, 1]))

    assert controller.initialized[1]
    assert controller.loss_ema[1].item() == pytest.approx(3.0)
    assert not controller.initialized[2]


def test_update_uses_ema_after_initial_value():
    controller = CoverageController(ema_decay=0.5, warmup_steps=0, update_interval=1)
    controller.update(torch.tensor([2.0]), torch.tensor([3]))

    controller.update(torch.tensor([4.0]), torch.tensor([3]))

    assert controller.loss_ema[3].item() == pytest.approx(3.0)
    assert controller.step.item() == 2


def test_metrics_expose_every_group_and_weight_range():
    controller = CoverageController()
    controller.sample(16, "cpu")
    metrics = controller.metrics()

    for name in COVER_GROUPS:
        assert f"cover/group/{name}_fraction" in metrics
        assert f"cover/group/{name}_ema" in metrics
        assert f"cover/group/{name}_target_mass" in metrics
    assert "cover/fallback" in metrics


def test_state_dict_round_trip():
    source = CoverageController(warmup_steps=0, update_interval=1)
    source.update(torch.tensor([1.0, 2.0]), torch.tensor([0, 7]))
    source.sample(8, "cpu")
    target = CoverageController(warmup_steps=0, update_interval=1)

    target.load_state_dict(source.state_dict())

    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ema_decay": -0.1},
        {"ema_decay": 1.0},
        {"warmup_steps": -1},
        {"temperature": 0.0},
        {"update_interval": 0},
        {"weight_min": 0.0},
        {"weight_max": 0.0},
        {"weight_min": 2.1, "weight_max": 2.0},
    ],
)
def test_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CoverageController(**kwargs)


def test_rejects_invalid_batch_inputs():
    controller = CoverageController()
    with pytest.raises(ValueError, match="same shape"):
        controller.update(torch.ones(2), torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match="group_ids"):
        controller.update(torch.ones(1), torch.tensor([8]))
    with pytest.raises(ValueError, match="batch_size"):
        controller.sample(0, "cpu")


def test_cover_clean_controller_decays_and_recovers_strength():
    controller = CoverCleanController(warmup_steps=0, strength_decay=0.9, recovery=0.01)
    controller.fast_ema.fill_(2.0)
    controller.slow_ema.fill_(1.0)
    controller.initialized.fill_(True)

    strength, triggered = controller.update(torch.tensor(2.0))

    assert triggered.item() is True
    assert strength.item() == pytest.approx(0.9)
    controller.fast_ema.fill_(1.0)
    controller.slow_ema.fill_(1.0)
    strength, triggered = controller.update(torch.tensor(1.0))
    assert triggered.item() is False
    assert strength.item() == pytest.approx(0.91)


def test_cover_clean_controller_holds_full_strength_during_warmup():
    controller = CoverCleanController(warmup_steps=2, strength_decay=0.5)
    first, first_trigger = controller.update(torch.tensor(1.0))
    controller.fast_ema.fill_(10.0)
    controller.slow_ema.fill_(1.0)
    second, second_trigger = controller.update(torch.tensor(10.0))

    assert first.item() == pytest.approx(1.0)
    assert second.item() == pytest.approx(1.0)
    assert not first_trigger and not second_trigger


def test_cover_clean_controller_respects_minimum_strength():
    controller = CoverCleanController(warmup_steps=0, minimum_strength=0.25, strength_decay=0.5)
    for _ in range(20):
        controller.fast_ema.fill_(10.0)
        controller.slow_ema.fill_(1.0)
        controller.initialized.fill_(True)
        controller.update(torch.tensor(10.0))
    assert controller.robust_strength.item() == pytest.approx(0.25)


def test_cover_clean_controller_ignores_nonfinite_loss():
    controller = CoverCleanController(warmup_steps=0)
    before = {key: value.clone() for key, value in controller.state_dict().items()}

    _, triggered = controller.update(torch.tensor(float("nan")))

    assert not triggered
    for key, value in controller.state_dict().items():
        assert torch.equal(value, before[key])


def test_robust_strength_interpolates_only_adaptive_mass():
    controller = CoverageController(warmup_steps=0)
    controller.loss_ema.copy_(torch.tensor([1., 1., 1., 1., 1., 1., 1., 5.]))
    controller.initialized.fill_(True)

    uniform = controller.target_mass(0.0)
    robust = controller.target_mass(1.0)

    assert uniform[0].item() == robust[0].item() == pytest.approx(0.5)
    assert torch.all(uniform[1:7] >= 0.025)
    assert robust[7] > uniform[7]
    assert uniform.sum().item() == pytest.approx(1.0)
    assert robust.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fast_decay": -0.1}, {"fast_decay": 1.0},
        {"slow_decay": -0.1}, {"slow_decay": 1.0},
        {"tolerance": -0.1}, {"minimum_strength": -0.1},
        {"minimum_strength": 1.1}, {"strength_decay": 0.0},
        {"strength_decay": 1.1}, {"recovery": -0.1},
        {"recovery": 1.1}, {"warmup_steps": -1},
    ],
)
def test_cover_clean_controller_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CoverCleanController(**kwargs)
