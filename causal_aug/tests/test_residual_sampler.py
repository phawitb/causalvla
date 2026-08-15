import pytest
import torch

from causal_aug import ResidualBranchSampler


@pytest.mark.parametrize(
    ("augmentation_probability", "overlay_probability"),
    [(-0.1, 0.25), (1.1, 0.25), (0.5, -0.1), (0.5, 1.1)],
)
def test_rejects_invalid_probabilities(augmentation_probability, overlay_probability):
    with pytest.raises(ValueError):
        ResidualBranchSampler(augmentation_probability, overlay_probability)


def test_samples_expected_clean_broad_and_residual_distribution():
    torch.manual_seed(1000)
    sampler = ResidualBranchSampler(0.5, 0.25)

    branch, choices = sampler.sample(100_000, "cpu")

    clean = branch == 0
    broad = branch == 1
    residual = branch == 2
    assert torch.all(clean | broad | residual)
    assert clean.float().mean().item() == pytest.approx(0.50, abs=0.01)
    assert broad.float().mean().item() == pytest.approx(0.375, abs=0.01)
    assert residual.float().mean().item() == pytest.approx(0.125, abs=0.01)
    assert choices.shape == (100_000,)
    assert choices.min() >= 0
    assert choices.max() <= 2
