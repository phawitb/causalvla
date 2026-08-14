import pytest
import torch

from causal_aug import INTERVENTION_FAMILIES, InterventionBank


@pytest.fixture
def images():
    first = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    return [first, first.flip(-1)]


@pytest.mark.parametrize("family", INTERVENTION_FAMILIES)
def test_intervention_preserves_shape_range_and_finiteness(images, family):
    outputs = InterventionBank().apply(images, family, 0.5)

    assert len(outputs) == len(images)
    for source, output in zip(images, outputs, strict=True):
        assert output.shape == source.shape
        assert torch.isfinite(output).all()
        assert output.min() >= -1
        assert output.max() <= 1


@pytest.mark.parametrize("family", INTERVENTION_FAMILIES)
def test_zero_intensity_is_identity_without_aliasing(images, family):
    outputs = InterventionBank().apply(images, family, 0.0)

    for source, output in zip(images, outputs, strict=True):
        assert torch.equal(output, source)
        assert output.data_ptr() != source.data_ptr()


def test_rejects_invalid_family_or_intensity(images):
    bank = InterventionBank()

    with pytest.raises(ValueError, match="Unknown intervention family"):
        bank.apply(images, "unknown", 0.5)
    with pytest.raises(ValueError, match="intensity"):
        bank.apply(images, "noise", 1.1)
