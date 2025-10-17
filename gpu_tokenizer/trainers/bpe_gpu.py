"""GPU orchestration helpers for the BPE trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch

from ..triton_kernels import count_pairs_histogram
from ..utils import aggregate_pair_keys


@dataclass(frozen=True)
class PairHistogramResult:
    """Container for packed pair histogram tensors."""

    keys: torch.Tensor
    counts: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.keys.device

    @staticmethod
    def empty(device: torch.device | str) -> "PairHistogramResult":
        dev = torch.device(device)
        return PairHistogramResult(
            torch.empty((0,), dtype=torch.long, device=dev),
            torch.empty((0,), dtype=torch.int64, device=dev),
        )

    def is_empty(self) -> bool:
        return self.keys.numel() == 0 or self.counts.numel() == 0


def count_pairs_on_device(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    pair_keys_buffer: torch.Tensor,
    pair_counts_buffer: torch.Tensor,
    pair_count_length: torch.Tensor,
) -> PairHistogramResult:
    """Run the Triton histogram kernel and return packed keys/counts."""

    packed_keys, packed_counts, total = count_pairs_histogram(
        tokens, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length
    )
    if total <= 0:
        return PairHistogramResult.empty(tokens.device)

    keys = packed_keys.narrow(0, 0, total).to(torch.long)
    counts = packed_counts.narrow(0, 0, total).to(torch.int64)
    return PairHistogramResult(keys=keys, counts=counts)


def _resolve_target_device(
    results: Sequence[PairHistogramResult], target_device: Optional[torch.device | str]
) -> torch.device:
    if target_device is not None:
        return torch.device(target_device)
    for result in results:
        if result.keys.device.type == "cuda":
            return result.keys.device
    return results[0].keys.device


def combine_histogram_results(
    results: Iterable[PairHistogramResult],
    *,
    target_device: Optional[torch.device | str] = None,
) -> PairHistogramResult:
    """Aggregate per-shard histograms on ``target_device`` without host copies."""

    materialized = [res for res in results if not res.is_empty()]
    if not materialized:
        device = torch.device(target_device) if target_device is not None else torch.device("cpu")
        return PairHistogramResult.empty(device)

    device = _resolve_target_device(materialized, target_device)

    moved_keys = [res.keys.to(device=device, non_blocking=True) for res in materialized]
    moved_counts = [res.counts.to(device=device, non_blocking=True) for res in materialized]

    if len(moved_keys) == 1:
        stacked_keys = moved_keys[0]
        stacked_counts = moved_counts[0]
    else:
        stacked_keys = torch.cat(moved_keys, dim=0)
        stacked_counts = torch.cat(moved_counts, dim=0)

    aggregated_keys, aggregated_counts = aggregate_pair_keys(stacked_keys, stacked_counts)
    return PairHistogramResult(keys=aggregated_keys, counts=aggregated_counts)


def select_best_pair(
    histogram: PairHistogramResult,
) -> tuple[Optional[int], int, Optional[int]]:
    """Return the best packed pair key, its count, and the tensor index."""

    if histogram.is_empty():
        return None, 0, None

    counts = histogram.counts
    best_tensor_count = torch.max(counts)
    best_count = int(best_tensor_count.item())
    if best_count <= 0:
        return None, 0, None

    candidate_indices = torch.nonzero(counts == best_tensor_count, as_tuple=False).flatten()
    if candidate_indices.numel() == 1:
        best_idx = candidate_indices[0]
    else:
        keys = histogram.keys
        best_idx = candidate_indices[torch.argmin(keys[candidate_indices])]

    best_key = histogram.keys[best_idx]
    return int(best_key.item()), best_count, int(best_idx.item())


__all__ = [
    "PairHistogramResult",
    "count_pairs_on_device",
    "combine_histogram_results",
    "select_best_pair",
]
