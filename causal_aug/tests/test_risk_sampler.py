import pytest
import torch

from causal_aug import RAPID_LITE_CANDIDATES, RiskWeightedInterventionSampler


def test_probabilities_are_normalized_and_risk_ordered():
    sampler = RiskWeightedInterventionSampler(temperature=1.0, exploration_floor=0.1)
    probabilities = sampler.probabilities()

    assert probabilities.shape == (len(RAPID_LITE_CANDIDATES),)
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert torch.all(probabilities > 0)
    assert torch.all(probabilities[:-1] > probabilities[1:])


def test_exploration_floor_one_is_uniform():
    probabilities = RiskWeightedInterventionSampler(exploration_floor=1.0).probabilities()
    assert torch.allclose(probabilities, torch.full_like(probabilities, 1 / len(probabilities)))


def test_sampler_preserves_camera_shapes_and_range():
    torch.manual_seed(7)
    images = [torch.zeros(8, 3, 32, 32), torch.zeros(8, 3, 32, 32)]
    outputs, choices = RiskWeightedInterventionSampler()(images)

    assert choices.shape == (8,)
    assert choices.min() >= 0 and choices.max() < len(RAPID_LITE_CANDIDATES)
    for source, output in zip(images, outputs, strict=True):
        assert output.shape == source.shape
        assert torch.isfinite(output).all()
        assert output.min() >= -1 and output.max() <= 1


@pytest.mark.parametrize(
    ("temperature", "floor"),
    [(0.0, 0.1), (-1.0, 0.1), (1.0, -0.1), (1.0, 1.1)],
)
def test_rejects_invalid_hyperparameters(temperature, floor):
    with pytest.raises(ValueError):
        RiskWeightedInterventionSampler(temperature, floor)
