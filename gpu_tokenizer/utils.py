"""Utility functions for GPU-based tokenization trainers."""

from __future__ import annotations

import torch


def device_of(x: torch.Tensor) -> torch.device:
    """Return the device hosting ``x``."""

    return x.device


@torch.jit.script
def prefix_sum_int(mask: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix sum over the last dimension of ``mask``."""

    return torch.cumsum(mask, dim=-1)


@torch.jit.script
def compact_by_mask(vals: torch.Tensor, keep: torch.Tensor, max_len: int) -> torch.Tensor:
    """Compact ``vals`` along the last dim using ``keep`` as a mask."""

    B, L = vals.shape
    idx = prefix_sum_int(keep) - 1
    out = vals.new_full((B, max_len), -1)
    b_ids = torch.arange(B, device=vals.device).unsqueeze(1).expand(B, L)
    take = keep.to(torch.bool)
    if take.any():
        out[b_ids[take], idx[take]] = vals[take]
    return out


@torch.jit.script
def apply_merge_once(
    seqs: torch.Tensor,
    valid: torch.Tensor,
    a_id: int,
    b_id: int,
    new_id: int,
):
    """Apply a single BPE merge on ``seqs`` and return the compacted tensors."""

    B, L = seqs.shape
    lhs = seqs[:, :-1]
    rhs = seqs[:, 1:]
    v_l = valid[:, :-1]
    v_r = valid[:, 1:]
    pair_match = (lhs == a_id) & (rhs == b_id) & v_l.to(torch.bool) & v_r.to(torch.bool)

    keep = valid.clone()
    keep[:, 1:] = keep[:, 1:] & (~pair_match)

    seqs = seqs.clone()
    seqs[:, :-1] = torch.where(
        pair_match,
        torch.as_tensor(new_id, device=seqs.device, dtype=seqs.dtype),
        lhs,
    )

    new_lens = keep.sum(dim=-1)
    max_new = int(new_lens.max().item())
    new_seqs = compact_by_mask(seqs, keep, max_new)
    new_valid = (new_seqs != -1).long()
    return new_seqs, new_valid


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
