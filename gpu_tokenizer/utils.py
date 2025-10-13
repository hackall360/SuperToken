"""Utility functions for GPU-based tokenization trainers."""

from __future__ import annotations

from typing import Optional

import torch


def device_of(x: torch.Tensor) -> torch.device:
    """Return the device hosting ``x``."""

    return x.device


@torch.jit.script
def compact_tokens_inplace(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    lengths: torch.Tensor,
    prefix_workspace: torch.Tensor,
) -> None:
    """Compact ``tokens``/``valid`` in place using ``prefix_workspace``.

    The helper performs a stable compaction along the last dimension.  For
    every valid position the token value is written to the index recorded by
    the running prefix sum stored in ``prefix_workspace``.  Surplus capacity is
    zero‑filled to preserve the original buffer shapes.
    """

    B, L = tokens.shape
    if B == 0 or L == 0:
        lengths.zero_()
        return

    rows = torch.arange(B, device=tokens.device)
    prefix_workspace.zero_()

    for col in range(L):
        keep_col = valid[:, col].to(torch.bool)
        src_vals = tokens[:, col]
        tokens[:, col] = 0
        valid[:, col] = 0
        if keep_col.any():
            row_ids = rows[keep_col]
            dst = prefix_workspace[row_ids]
            tokens[row_ids, dst] = src_vals[keep_col]
            valid[row_ids, dst] = 1
            prefix_workspace[row_ids] = dst + 1

    if lengths.dtype == prefix_workspace.dtype:
        lengths.copy_(prefix_workspace)
    else:
        lengths.copy_(prefix_workspace.to(lengths.dtype))


@torch.jit.script
def apply_merge_once(
    seqs: torch.Tensor,
    valid: torch.Tensor,
    lengths: torch.Tensor,
    a_id: int,
    b_id: int,
    new_id: int,
    pair_workspace: Optional[torch.Tensor] = None,
    prefix_workspace: Optional[torch.Tensor] = None,
):
    """Apply a single BPE merge directly within ``seqs`` and ``valid``.

    The tensors are updated in place; ``pair_workspace`` and ``prefix_workspace``
    are optional reusable scratch buffers used to avoid reallocation during
    repeated merges.  ``lengths`` is mutated to reflect the number of valid
    tokens remaining per sequence while preserving the original capacity of the
    buffers.
    """

    B, L = seqs.shape
    device = seqs.device

    if B == 0:
        return seqs, valid, lengths

    if L <= 1:
        if lengths.numel() == B:
            lengths.copy_(valid.sum(dim=-1, dtype=lengths.dtype))
        return seqs, valid, lengths

    if pair_workspace is None or pair_workspace.size(0) != B or pair_workspace.size(1) != L - 1:
        pair_workspace = torch.zeros((B, L - 1), dtype=torch.bool, device=device)
    else:
        pair_workspace.zero_()

    lhs = seqs[:, :-1]
    rhs = seqs[:, 1:]
    v_l = valid[:, :-1]
    v_r = valid[:, 1:]

    pair_workspace.copy_(lhs == a_id)
    pair_workspace &= rhs == b_id
    pair_workspace &= v_l.to(torch.bool)
    pair_workspace &= v_r.to(torch.bool)

    lhs.masked_fill_(pair_workspace, torch.as_tensor(new_id, device=device, dtype=seqs.dtype))
    valid[:, 1:].masked_fill_(pair_workspace, 0)
    seqs[:, 1:].masked_fill_(pair_workspace, 0)

    if prefix_workspace is None or prefix_workspace.size(0) != B:
        prefix_workspace = torch.zeros((B,), dtype=torch.long, device=device)

    compact_tokens_inplace(seqs, valid, lengths, prefix_workspace)
    return seqs, valid, lengths


@torch.jit.script
def count_pairs(seqs: torch.Tensor, valid: torch.Tensor):
    """Return unique adjacent token pairs and their counts."""

    lhs = seqs[:, :-1]
    rhs = seqs[:, 1:]
    mask = valid[:, :-1].to(torch.bool) & valid[:, 1:].to(torch.bool)
    if not mask.any():
        return seqs.new_empty((0, 2)), seqs.new_empty((0,), dtype=torch.long)
    device = seqs.device
    lhs_vals = lhs[mask].to(torch.long)
    rhs_vals = rhs[mask].to(torch.long)

    pair_keys = (lhs_vals << 32) | rhs_vals
    sorted_keys, _ = torch.sort(pair_keys)

    num_keys = sorted_keys.numel()
    if num_keys == 0:
        return seqs.new_empty((0, 2)), seqs.new_empty((0,), dtype=torch.long)

    run_start = torch.ones(1, dtype=torch.bool, device=device)
    run_start = torch.cat([run_start, sorted_keys[1:] != sorted_keys[:-1]])
    run_indices = torch.nonzero(run_start).flatten()

    next_indices = torch.cat(
        [
            run_indices[1:],
            torch.as_tensor([num_keys], device=device, dtype=run_indices.dtype),
        ]
    )
    counts = next_indices - run_indices

    unique_keys = sorted_keys[run_indices]
    lower_mask = torch.as_tensor((1 << 32) - 1, device=device, dtype=torch.long)
    a_vals = (unique_keys >> 32).to(seqs.dtype)
    b_vals = (unique_keys & lower_mask).to(seqs.dtype)
    pairs = torch.stack([a_vals, b_vals], dim=1)

    return pairs, counts
