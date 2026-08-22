"""Deterministic exact-balanced sampling for Fair Protocol v1."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import Sampler


def exact_half_mask(
    batch_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Tensor:
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("batch_size must be a positive even integer")
    order = torch.randperm(batch_size, generator=generator, device=device)
    mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    mask[order[: batch_size // 2]] = True
    return mask


class PairedBatchSampler(Sampler[list[int]]):
    """Yield batches containing equal clean and augmented indices."""

    def __init__(
        self,
        clean_indices: Sequence[int],
        augmented_indices: Sequence[int],
        batch_size: int,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("batch_size must be a positive even integer")
        if len(clean_indices) != len(augmented_indices):
            raise ValueError("clean and augmented pools must have equal size")
        self.clean_indices = tuple(clean_indices)
        self.augmented_indices = tuple(augmented_indices)
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last

    def __len__(self) -> int:
        half = self.batch_size // 2
        if self.drop_last:
            return len(self.clean_indices) // half
        return (len(self.clean_indices) + half - 1) // half

    def __iter__(self) -> Iterator[list[int]]:
        half = self.batch_size // 2
        clean_order = torch.randperm(
            len(self.clean_indices), generator=torch.Generator().manual_seed(self.seed)
        ).tolist()
        aug_order = torch.randperm(
            len(self.augmented_indices), generator=torch.Generator().manual_seed(self.seed + 1)
        ).tolist()
        for offset in range(0, len(clean_order), half):
            clean = clean_order[offset : offset + half]
            augmented = aug_order[offset : offset + half]
            if len(clean) < half and self.drop_last:
                break
            batch = [self.clean_indices[index] for index in clean]
            batch.extend(self.augmented_indices[index] for index in augmented)
            yield batch
