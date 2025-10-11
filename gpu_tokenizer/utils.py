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
    take = keep.bool()
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
    pair_match = (lhs == a_id) & (rhs == b_id) & v_l.bool() & v_r.bool()

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
    mask = valid[:, :-1].bool() & valid[:, 1:].bool()
    if not mask.any():
        return seqs.new_empty((0, 2)), seqs.new_empty((0,), dtype=torch.long)
    pairs = torch.stack([lhs[mask], rhs[mask]], dim=1)
    uniq, inv = torch.unique(pairs, dim=0, return_inverse=True)
    counts = torch.bincount(inv, minlength=uniq.size(0))
    return uniq, counts
