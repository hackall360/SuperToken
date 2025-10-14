"""CPU fast path helpers for tiny batches.

These helpers provide a vectorized implementation that mirrors the CUDA
kernels but executes on the host.  They primarily target the extremely small
batches that would otherwise suffer from kernel launch overheads.  The
implementation relies on PyTorch tensor primitives which internally leverage
SIMD instructions (AVX2/AVX-512 when available) and makes the logic easy to
exercise inside the unit test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .utils import apply_merge_once, count_pairs


@dataclass
class FastPathWorkspaces:
    """Reusable buffers for the CPU fast path."""

    pair_workspace: Optional[torch.Tensor] = None
    prefix_workspace: Optional[torch.Tensor] = None
    span_workspace: Optional[torch.Tensor] = None
    overflow_workspace: Optional[torch.Tensor] = None

    def resize(self, batch: torch.Tensor, lengths: torch.Tensor) -> None:
        rows, width = batch.shape
        width = max(width - 1, 0)
        device = batch.device
        if width > 0:
            if self.pair_workspace is None or self.pair_workspace.shape != (rows, width):
                self.pair_workspace = torch.zeros((rows, width), dtype=torch.bool, device=device)
            if self.span_workspace is None or self.span_workspace.shape != (rows, width):
                self.span_workspace = torch.zeros((rows, width), dtype=torch.bool, device=device)
        else:
            self.pair_workspace = torch.empty((rows, 0), dtype=torch.bool, device=device)
            self.span_workspace = torch.empty((rows, 0), dtype=torch.bool, device=device)
        if self.prefix_workspace is None or self.prefix_workspace.shape[0] != rows:
            self.prefix_workspace = torch.zeros((rows,), dtype=lengths.dtype, device=device)
        if self.overflow_workspace is None or self.overflow_workspace.shape != (rows,):
            self.overflow_workspace = torch.zeros((rows,), dtype=torch.bool, device=device)
        else:
            self.overflow_workspace.zero_()


def should_route_to_cpu(batch_rows: int, width: int) -> bool:
    """Heuristic to determine when the CPU fast path is preferable."""

    if batch_rows <= 0 or width <= 0:
        return True
    pair_slots = batch_rows * width
    return batch_rows <= 2 or width <= 4 or pair_slots <= 512


def count_pairs_fastpath(
    tokens: torch.Tensor, valid: torch.Tensor, workspaces: Optional[FastPathWorkspaces] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute adjacent pair histograms using the CPU fast path."""

    if tokens.is_cuda:
        raise RuntimeError("count_pairs_fastpath expects CPU tensors")
    rows, width = tokens.shape
    width = max(width - 1, 0)
    if rows == 0 or width <= 0:
        return (
            torch.empty((0,), dtype=torch.long, device=tokens.device),
            torch.empty((0,), dtype=torch.int32, device=tokens.device),
        )

    capacity = max(rows * width, 1)
    pair_workspace = torch.empty((capacity, 2), dtype=tokens.dtype, device=tokens.device)
    count_workspace = torch.empty((capacity,), dtype=torch.int32, device=tokens.device)
    length_tensor = torch.zeros((1,), dtype=torch.long, device=tokens.device)
    count_pairs(tokens, valid, pair_workspace, count_workspace, length_tensor)
    length = int(length_tensor.item())
    if length <= 0:
        return (
            torch.empty((0,), dtype=torch.long, device=tokens.device),
            torch.empty((0,), dtype=torch.int32, device=tokens.device),
        )
    pairs = pair_workspace[:length]
    counts = count_workspace[:length].to(torch.int32)
    a_ids = pairs[:, 0].to(torch.long)
    b_ids = pairs[:, 1].to(torch.long)
    keys = (a_ids << 32) | b_ids
    return keys, counts


def apply_merge_fastpath(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    lengths: torch.Tensor,
    a_id: int,
    b_id: int,
    new_id: int,
    workspaces: Optional[FastPathWorkspaces] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a merge on CPU tensors and return the updated span mask."""

    if tokens.is_cuda:
        raise RuntimeError("apply_merge_fastpath expects CPU tensors")
    if workspaces is None:
        workspaces = FastPathWorkspaces()
    workspaces.resize(tokens, lengths)
    assert workspaces.pair_workspace is not None
    assert workspaces.prefix_workspace is not None
    assert workspaces.span_workspace is not None
    assert workspaces.overflow_workspace is not None
    return apply_merge_once(
        tokens,
        valid,
        lengths,
        a_id,
        b_id,
        new_id,
        workspaces.pair_workspace,
        workspaces.prefix_workspace,
        workspaces.span_workspace,
        workspaces.overflow_workspace,
    )


__all__ = [
    "FastPathWorkspaces",
    "should_route_to_cpu",
    "count_pairs_fastpath",
    "apply_merge_fastpath",
]
