"""Dataset utilities for GPU tokenization trainers."""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, List, Sequence, Tuple

import random

import torch

_PIN_MEMORY = torch.cuda.is_available()

from .dtypes import length_storage_dtype
from .io import CorpusStreamer, DecodedShard


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
            pin_memory=_PIN_MEMORY,
        )
        valid = torch.zeros(
            (self.bs, self._storage_width),
            dtype=torch.uint8,
            pin_memory=_PIN_MEMORY,
        )
        lengths = torch.zeros(
            (self.bs,), dtype=self._length_dtype, pin_memory=_PIN_MEMORY
        )
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

    def iter_device(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = True,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield batches staged on ``device`` without reallocating host buffers."""

        device_obj = torch.device(device)
        for tokens, valid, lengths in self:
            yield (
                tokens.to(device=device_obj, non_blocking=non_blocking),
                valid.to(device=device_obj, non_blocking=non_blocking),
                lengths.to(device=device_obj, non_blocking=non_blocking),
            )


class StreamingPackedBatcher:
    """Pack batches from a :class:`CorpusStreamer` without preloading shards."""

    def __init__(
        self,
        streamer: CorpusStreamer,
        encode_view: Callable[[memoryview], Iterator[int]],
        *,
        batch_size: int = 1024,
    ) -> None:
        self.streamer = streamer
        self.encode_view = encode_view
        self.bs = batch_size
        self._storage_width = 1
        self._length_dtype = length_storage_dtype(self._storage_width)
        self._buffers = [self._allocate_buffer() for _ in range(2)]

    def _allocate_buffer(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.full(
            (self.bs, self._storage_width),
            -1,
            dtype=torch.int32,
            pin_memory=_PIN_MEMORY,
        )
        valid = torch.zeros(
            (self.bs, self._storage_width),
            dtype=torch.uint8,
            pin_memory=_PIN_MEMORY,
        )
        lengths = torch.zeros(
            (self.bs,), dtype=self._length_dtype, pin_memory=_PIN_MEMORY
        )
        return tokens, valid, lengths

    def _ensure_width(self, width: int) -> None:
        width = max(1, width)
        if width <= self._storage_width:
            return
        self._storage_width = width
        self._length_dtype = length_storage_dtype(self._storage_width)
        self._buffers = [self._allocate_buffer() for _ in range(2)]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        buf_idx = 0
        rows: list[list[int]] = []
        shards: list[DecodedShard] = []
        for decoded in self.streamer:
            seq = list(self.encode_view(decoded.view))
            rows.append(seq)
            shards.append(decoded)
            if len(rows) == self.bs:
                yield from self._emit_batch(rows, shards, buf_idx)
                buf_idx = 1 - buf_idx
                rows = []
                shards = []
        if rows:
            yield from self._emit_batch(rows, shards, buf_idx)

    def _emit_batch(
        self,
        rows: list[list[int]],
        shards: list[DecodedShard],
        buf_idx: int,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        max_len = max((len(r) for r in rows), default=0)
        self._ensure_width(max_len)
        tokens, valid, lengths = self._buffers[buf_idx]
        count = len(rows)
        tokens[:count].fill_(-1)
        valid[:count].zero_()
        lengths[:count].zero_()

        for row_idx, seq in enumerate(rows):
            if not seq:
                lengths[row_idx] = 0
                continue
            L = len(seq)
            lengths[row_idx] = L
            vals = torch.as_tensor(seq, dtype=torch.long)
            tokens[row_idx, :L] = vals.to(torch.int32)
            valid[row_idx, :L] = 1

        width = max(1, max_len)
        try:
            yield tokens[:count, :width], valid[:count, :width], lengths[:count]
        finally:
            for shard in shards:
                shard.release()
