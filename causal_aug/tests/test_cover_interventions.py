import pytest
import torch

from causal_aug import apply_cover_groups


def _images(batch_size: int = 8) -> list[torch.Tensor]:
    base = torch.linspace(-1, 1, batch_size * 3 * 32 * 32).reshape(batch_size, 3, 32, 32)
    return [base.clone(), base.clone()]


def test_clean_group_is_exact_identity():
    images = _images()

    output = apply_cover_groups(images, torch.zeros(8, dtype=torch.long))

    assert all(torch.equal(source, result) for source, result in zip(images, output))


@pytest.mark.parametrize("group_id", range(1, 8))
def test_each_augmented_group_changes_pixels_and_preserves_shape(group_id):
    torch.manual_seed(1000)
    images = _images()

    output = apply_cover_groups(images, torch.full((8,), group_id), intensity=1.0)

    assert output[0].shape == images[0].shape
    assert torch.isfinite(output[0]).all()
    assert not torch.equal(output[0], images[0])


def test_parameters_are_shared_across_identical_camera_views():
    torch.manual_seed(1000)
    images = _images()
    ids = torch.ones(8, dtype=torch.long)

    output = apply_cover_groups(images, ids)

    assert torch.equal(output[0], output[1])


def test_group_application_is_reproducible_from_torch_seed():
    images = _images()
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    torch.manual_seed(2000)
    first = apply_cover_groups(images, ids)
    torch.manual_seed(2000)
    second = apply_cover_groups(images, ids)

    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_rejects_empty_images():
    with pytest.raises(ValueError, match="images"):
        apply_cover_groups([], torch.zeros(1, dtype=torch.long))


def test_rejects_mismatched_image_batches():
    with pytest.raises(ValueError, match="batch"):
        apply_cover_groups([torch.zeros(2, 3, 8, 8), torch.zeros(3, 3, 8, 8)], torch.zeros(2, dtype=torch.long))


def test_rejects_mismatched_group_batch():
    with pytest.raises(ValueError, match="group_ids"):
        apply_cover_groups([torch.zeros(2, 3, 8, 8)], torch.zeros(1, dtype=torch.long))


@pytest.mark.parametrize("ids", [torch.tensor([-1]), torch.tensor([8])])
def test_rejects_out_of_range_group_ids(ids):
    with pytest.raises(ValueError, match="group_ids"):
        apply_cover_groups([torch.zeros(1, 3, 8, 8)], ids)


def test_rejects_nonfloating_images():
    with pytest.raises(ValueError, match="floating"):
        apply_cover_groups([torch.zeros(1, 3, 8, 8, dtype=torch.uint8)], torch.zeros(1, dtype=torch.long))


@pytest.mark.parametrize("intensity", [-0.1, 1.1])
def test_rejects_invalid_intensity(intensity):
    with pytest.raises(ValueError, match="intensity"):
        apply_cover_groups([torch.zeros(1, 3, 8, 8)], torch.zeros(1, dtype=torch.long), intensity)
