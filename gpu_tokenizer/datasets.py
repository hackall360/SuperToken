"""Dataset utilities for GPU tokenization trainers."""

from __future__ import annotations

from typing import Iterator, List, Tuple

import random

import torch


class PackedBatcher:
    """Stream padded batches of integer token sequences using pinned memory."""

    def __init__(self, sequences: List[list[int]], batch_size: int = 1024, seed: int = 1337):
        self.sequences = sequences
        self.bs = batch_size
        rng = random.Random(seed)
        rng.shuffle(self.sequences)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for i in range(0, len(self.sequences), self.bs):
            chunk = self.sequences[i : i + self.bs]
            max_len = max(len(x) for x in chunk)
            x = torch.full((len(chunk), max_len), -1, dtype=torch.long, pin_memory=True)
            v = torch.zeros((len(chunk), max_len), dtype=torch.long, pin_memory=True)
            for r, seq in enumerate(chunk):
                L = len(seq)
                x[r, :L] = torch.tensor(seq, dtype=torch.long)
                v[r, :L] = 1
            yield x, v
