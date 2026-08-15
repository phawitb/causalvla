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


def test_apply_preserves_clean_images_when_augmentation_is_disabled():
    sampler = ResidualBranchSampler(0.0, 0.0)
    images = [torch.linspace(-1, 1, 8 * 3 * 32 * 32).reshape(8, 3, 32, 32)]
    broad = [images[0].flip(-1)]

    output, branch, _ = sampler.compose(images, broad)

    assert torch.equal(output[0], images[0])
    assert torch.all(branch == 0)
    assert output[0].data_ptr() != images[0].data_ptr()


def test_apply_overlays_risk_on_top_of_broad_images():
    torch.manual_seed(7)
    sampler = ResidualBranchSampler(1.0, 1.0)
    clean = torch.linspace(-1, 1, 64 * 3 * 32 * 32).reshape(64, 3, 32, 32)
    broad = clean.flip(-1).mul(0.8)

    output, branch, choices = sampler.compose([clean], [broad])

    assert torch.all(branch == 2)
    assert output[0].shape == clean.shape
    assert torch.isfinite(output[0]).all()
    assert output[0].min() >= -1 and output[0].max() <= 1
    assert not torch.equal(output[0], clean)
    for index in range(3):
        mask = choices == index
        assert mask.any()
        assert not torch.equal(output[0][mask], broad[mask])


def test_apply_rejects_empty_or_mismatched_camera_views():
    sampler = ResidualBranchSampler(0.5, 0.25)
    image = torch.zeros(2, 3, 32, 32)

    with pytest.raises(ValueError, match="camera"):
        sampler.compose([], [])
    with pytest.raises(ValueError, match="camera"):
        sampler.compose([image], [image, image])
    with pytest.raises(ValueError, match="shape"):
        sampler.compose([image], [torch.zeros(3, 3, 32, 32)])
