import pytest
import torch

from causal_aug import PairedBatchSampler, exact_half_mask


def test_exact_half_mask_is_balanced_and_reproducible():
    first = exact_half_mask(16, "cpu", torch.Generator().manual_seed(1000))
    second = exact_half_mask(16, "cpu", torch.Generator().manual_seed(1000))
    assert first.dtype == torch.bool
    assert first.sum().item() == 8
    assert torch.equal(first, second)


@pytest.mark.parametrize("batch_size", [0, 1, 3])
def test_exact_half_mask_rejects_invalid_batch_sizes(batch_size):
    with pytest.raises(ValueError, match="positive even"):
        exact_half_mask(batch_size, "cpu")


def test_paired_batch_sampler_emits_equal_domains_reproducibly():
    first = list(PairedBatchSampler(range(8), range(8, 16), batch_size=4, seed=1000))
    second = list(PairedBatchSampler(range(8), range(8, 16), batch_size=4, seed=1000))
    assert first == second
    assert len(first) == 4
    for batch in first:
        assert sum(index < 8 for index in batch) == 2
        assert sum(index >= 8 for index in batch) == 2


def test_paired_batch_sampler_rejects_unequal_pools():
    with pytest.raises(ValueError, match="equal size"):
        PairedBatchSampler(range(4), range(4, 10), batch_size=4, seed=1000)
