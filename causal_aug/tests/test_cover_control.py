import pytest
import torch

from causal_aug import COVER_GROUPS, CoverageController


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
