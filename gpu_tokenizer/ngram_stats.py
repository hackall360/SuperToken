"""Helpers for computing n-gram histograms on GPU batches."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple, Union

import torch

from .bpe_trainer import GPUBatchRecord
from .utils import aggregate_pair_keys

BatchLike = Union[
    GPUBatchRecord,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def _coerce_batch_tensors(
    batch: BatchLike, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(tokens, valid)`` tensors on ``device`` for ``batch``."""

    if isinstance(batch, GPUBatchRecord):
        tokens = batch.tokens
        valid = batch.valid
    else:
        tokens = batch[0]
        valid = batch[1]

    if tokens.device != device:
        tokens = tokens.to(device=device)
    if valid.device != device:
        valid = valid.to(device=device)

    return tokens.contiguous(), valid.contiguous()


def _pack_ngrams(windows: torch.Tensor, order: int, bits_per_symbol: int) -> torch.Tensor:
    """Pack ``windows`` representing n-grams into ``torch.int64`` keys."""

    if windows.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=windows.device)

    shifts = torch.arange(
        order - 1, -1, -1, dtype=torch.int64, device=windows.device
    ) * bits_per_symbol
    packed = torch.bitwise_left_shift(windows.to(torch.int64), shifts)
    return torch.sum(packed, dim=-1, dtype=torch.int64)


def compute_ngram_histograms(
    batches: Iterable[BatchLike],
    max_order: int = 4,
    device: torch.device | None = None,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute aggregated n-gram histograms for ``batches`` up to ``max_order``.

    Args:
        batches: Iterable of GPU batches. Each element can be a
            :class:`GPUBatchRecord` or a tuple containing ``(tokens, valid)``
            tensors (optionally followed by lengths). ``tokens`` should be of
            shape ``(B, L)`` and ``valid`` is a mask with the same shape where a
            non-zero value marks that position as valid.
        max_order: Maximum n-gram length to aggregate. Must be at least 1.
        device: Device used for aggregation. If ``None`` the device is inferred
            from the batches, defaulting to the current CUDA device when
            available.

    Returns:
        Dictionary mapping each order ``n`` to a tuple ``(keys, counts)`` where
        ``keys`` is a one-dimensional ``torch.int64`` tensor of packed n-gram
        identifiers and ``counts`` is a ``torch.int32`` tensor containing the
        corresponding frequencies.
    """

    if max_order < 1:
        raise ValueError("max_order must be at least 1")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bits_per_symbol = max(16, 64 // max_order)

    collected_keys: Dict[int, list[torch.Tensor]] = {n: [] for n in range(1, max_order + 1)}

    for batch in batches:
        tokens, valid = _coerce_batch_tensors(batch, device)
        B, width = tokens.shape
        if B == 0 or width == 0:
            continue

        valid_bool = valid.to(torch.bool)

        for order in range(1, max_order + 1):
            if width < order:
                break

            windows = tokens.unfold(1, order, 1)
            valid_windows = valid_bool.unfold(1, order, 1)
            mask = torch.all(valid_windows, dim=-1)
            if not torch.any(mask):
                continue

            selected = windows[mask]
            if selected.numel() == 0:
                continue

            packed = _pack_ngrams(selected, order, bits_per_symbol)
            collected_keys[order].append(packed)

    histograms: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for order in range(1, max_order + 1):
        if collected_keys[order]:
            keys = torch.cat(collected_keys[order])
            counts = torch.ones((keys.numel(),), dtype=torch.int32, device=device)
            aggregated_keys, aggregated_counts = aggregate_pair_keys(keys, counts)
        else:
            aggregated_keys = torch.empty((0,), dtype=torch.long, device=device)
            aggregated_counts = torch.empty((0,), dtype=torch.int32, device=device)
        histograms[order] = (aggregated_keys, aggregated_counts)

    return histograms
