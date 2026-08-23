import torch

from causal_aug.fixed_ood import FixedOODIdentity, apply_fixed_ood_record, derive_fixed_ood_record


def test_record_is_reproducible_and_independent_of_global_rng():
    identity = FixedOODIdentity(4000, 3, 7, "level_2")
    torch.manual_seed(1)
    first = derive_fixed_ood_record(identity)
    torch.manual_seed(999)
    assert derive_fixed_ood_record(identity) == first


def test_different_episode_identity_changes_record():
    left = derive_fixed_ood_record(FixedOODIdentity(4000, 3, 7, "level_2"))
    right = derive_fixed_ood_record(FixedOODIdentity(4000, 3, 8, "level_2"))
    assert left != right


def test_same_record_repeats_pixels():
    record = derive_fixed_ood_record(FixedOODIdentity(4000, 1, 2, "level_2"))
    image = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    assert torch.equal(apply_fixed_ood_record(image, record), apply_fixed_ood_record(image, record))


def test_level_zero_is_identity():
    image = torch.rand(2, 3, 16, 16)
    record = derive_fixed_ood_record(FixedOODIdentity(4000, 0, 0, "level_0"))
    assert torch.equal(apply_fixed_ood_record(image, record), image)
