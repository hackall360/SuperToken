"""Dataset utilities for GPU tokenization trainers."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence, Tuple

import random

import torch

from .dtypes import length_storage_dtype


class PackedBatcher:
    """Stream padded batches of integer token sequences using double buffers."""

    def __init__(
        self,
        sequences: Iterable[Sequence[int]] | Sequence[Sequence[int]],
        batch_size: int = 1024,
        seed: int = 1337,
    ):
        self.sequences: List[List[int]] = [list(seq) for seq in sequences]
        self.bs = batch_size
        rng = random.Random(seed)
        rng.shuffle(self.sequences)

        max_len = max((len(seq) for seq in self.sequences), default=0)
        # Allocate storage width of at least 1 to keep tensor shapes valid even when empty.
        self._storage_width = max(1, max_len)
        self._length_dtype = length_storage_dtype(self._storage_width)
        self._buffers = [self._allocate_buffer() for _ in range(2)]

    def _allocate_buffer(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.full(
            (self.bs, self._storage_width),
            -1,
            dtype=torch.int32,
            pin_memory=True,
        )
        valid = torch.zeros(
            (self.bs, self._storage_width), dtype=torch.uint8, pin_memory=True
        )
        lengths = torch.zeros((self.bs,), dtype=self._length_dtype, pin_memory=True)
        return tokens, valid, lengths

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not self.sequences:
            return

        buf_idx = 0
        for start in range(0, len(self.sequences), self.bs):
            chunk = self.sequences[start : start + self.bs]
            if not chunk:
                break

            tokens, valid, lengths = self._buffers[buf_idx]
            rows = len(chunk)
            tokens[:rows].fill_(-1)
            valid[:rows].zero_()
            lengths[:rows].zero_()

            max_len = 0
            for row, seq in enumerate(chunk):
                L = len(seq)
                if L == 0:
                    lengths[row] = 0
                    continue
                max_len = max(max_len, L)
                lengths[row] = L
                vals = torch.as_tensor(seq, dtype=torch.long)
                tokens[row, :L] = vals.to(torch.int32)
                valid[row, :L] = 1

            # Ensure we always provide a view with at least one column to avoid zero-width tensors
            width = max(1, max_len)
            yield (
                tokens[:rows, :width],
                valid[:rows, :width],
                lengths[:rows],
            )
            buf_idx = 1 - buf_idx
