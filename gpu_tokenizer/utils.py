"""Utility functions for GPU-based tokenization trainers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from .cuda_kernels import apply_merge_and_compact as cuda_apply_merge_and_compact


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


def apply_merge_once(
    seqs: torch.Tensor,
    valid: torch.Tensor,
    lengths: torch.Tensor,
    a_id: int,
    b_id: int,
    new_id: int,
    pair_workspace: Optional[torch.Tensor] = None,
    prefix_workspace: Optional[torch.Tensor] = None,
    span_workspace: Optional[torch.Tensor] = None,
):
    """Apply a single BPE merge directly within ``seqs`` and ``valid``.

    The tensors are updated in place; ``pair_workspace`` and ``prefix_workspace``
    are optional reusable scratch buffers used to avoid reallocation during
    repeated merges.  ``span_workspace`` may be provided to capture the active
    left indices of merged pairs prior to compaction.  ``lengths`` is mutated to
    reflect the number of valid tokens remaining per sequence while preserving
    the original capacity of the buffers.  The function returns the mutated
    tensors along with the boolean mask identifying merged left indices.
    """

    B, L = seqs.shape
    device = seqs.device
    width = max(L - 1, 0)
    empty_span = torch.zeros((B, width), dtype=torch.bool, device=device)

    if B == 0:
        if lengths.numel() == B:
            lengths.zero_()
        return seqs, valid, lengths, empty_span

    if L <= 1:
        if lengths.numel() == B:
            lengths.copy_(valid.sum(dim=-1, dtype=lengths.dtype))
        if span_workspace is not None and span_workspace.shape == empty_span.shape:
            span_workspace.zero_()
            return seqs, valid, lengths, span_workspace
        return seqs, valid, lengths, empty_span

    if seqs.is_cuda:
        if pair_workspace is None or pair_workspace.shape != (B, width):
            pair_workspace = torch.zeros((B, width), dtype=torch.bool, device=device)
        if prefix_workspace is None or prefix_workspace.shape[0] != B:
            prefix_workspace = torch.zeros((B,), dtype=torch.long, device=device)

        cuda_apply_merge_and_compact(
            seqs,
            valid,
            prefix_workspace,
            pair_workspace,
            int(a_id),
            int(b_id),
            int(new_id),
        )

        if lengths.dtype == prefix_workspace.dtype:
            lengths.copy_(prefix_workspace)
        else:
            lengths.copy_(prefix_workspace.to(lengths.dtype))

        spans: torch.Tensor = pair_workspace
        if span_workspace is not None and span_workspace.shape == pair_workspace.shape:
            if span_workspace.dtype != torch.bool:
                span_workspace.copy_(pair_workspace.to(span_workspace.dtype))
            else:
                span_workspace.copy_(pair_workspace)
            spans = span_workspace
        return seqs, valid, lengths, spans

    if pair_workspace is None or pair_workspace.size(0) != B or pair_workspace.size(1) != width:
        pair_workspace = torch.zeros((B, width), dtype=torch.bool, device=device)
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

    spans = pair_workspace
    if span_workspace is not None and span_workspace.shape == pair_workspace.shape:
        if span_workspace.dtype != torch.bool:
            span_workspace.copy_(pair_workspace.to(span_workspace.dtype))
        else:
            span_workspace.copy_(pair_workspace)
        spans = span_workspace

    if prefix_workspace is None or prefix_workspace.size(0) != B:
        prefix_workspace = torch.zeros((B,), dtype=torch.long, device=device)

    compact_tokens_inplace(seqs, valid, lengths, prefix_workspace)
    return seqs, valid, lengths, spans


@torch.jit.script
def count_pairs(
    seqs: torch.Tensor,
    valid: torch.Tensor,
    pair_keys_buffer: torch.Tensor,
    pair_counts_buffer: torch.Tensor,
    pair_count_length: torch.Tensor,
) -> None:
    """Populate ``pair_*`` workspaces with unique adjacent token pairs."""

    B = seqs.size(0)
    L = seqs.size(1) if seqs.dim() > 1 else 0
    width = L - 1 if L > 0 else 0
    capacity = pair_counts_buffer.numel()

    if pair_count_length.numel() != 1:
        raise RuntimeError("pair_count_length must be a singleton tensor")

    pair_count_length.zero_()

    if B == 0 or width <= 0 or capacity == 0:
        if pair_keys_buffer.numel() > 0:
            pair_keys_buffer.fill_(-1)
        if pair_counts_buffer.numel() > 0:
            pair_counts_buffer.zero_()
        return

    lhs = seqs[:, :-1]
    rhs = seqs[:, 1:]
    mask = valid[:, :-1].to(torch.bool) & valid[:, 1:].to(torch.bool)

    if not mask.any():
        pair_keys_buffer.fill_(-1)
        pair_counts_buffer.zero_()
        return

    lhs_vals = lhs[mask].to(torch.long)
    rhs_vals = rhs[mask].to(torch.long)

    pair_keys = (lhs_vals << 32) | rhs_vals
    sorted_keys, _ = torch.sort(pair_keys)

    num_keys = int(sorted_keys.numel())
    if num_keys == 0:
        pair_keys_buffer.fill_(-1)
        pair_counts_buffer.zero_()
        return

    device = seqs.device
    run_start = torch.ones((num_keys,), dtype=torch.bool, device=device)
    if num_keys > 1:
        run_start[1:] = sorted_keys[1:] != sorted_keys[:-1]
    run_indices = torch.nonzero(run_start, as_tuple=False).flatten()

    next_indices = torch.empty_like(run_indices)
    next_indices[:-1] = run_indices[1:]
    next_indices[-1] = num_keys
    counts = next_indices - run_indices

    unique_keys = sorted_keys[run_indices]
    length = int(unique_keys.numel())
    if length > capacity:
        raise RuntimeError("pair workspace capacity exceeded")

    a_vals = (unique_keys >> 32).to(seqs.dtype)
    b_vals = (unique_keys & ((1 << 32) - 1)).to(seqs.dtype)

    if length > 0:
        pair_keys_buffer[:length, 0].copy_(a_vals)
        pair_keys_buffer[:length, 1].copy_(b_vals)
        pair_counts_buffer[:length].copy_(counts)

    if length < pair_keys_buffer.size(0):
        pair_keys_buffer[length:].fill_(-1)
    if length < pair_counts_buffer.size(0):
        pair_counts_buffer[length:].zero_()

    pair_count_length[0] = length


def aggregate_pair_keys(
    keys: torch.Tensor, counts: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate duplicate packed pair ``keys`` by summing ``counts``.

    Both tensors are expected to be one-dimensional and reside on the same
    device.  The helper returns unique keys sorted in ascending order along with
    their aggregated counts.
    """

    if keys.numel() == 0:
        if counts.dtype != torch.long:
            counts = counts.to(torch.long)
        if keys.dtype != torch.long:
            keys = keys.to(torch.long)
        return keys, counts

    device = keys.device
    keys = keys.to(torch.long)
    counts = counts.to(torch.long)

    order = torch.argsort(keys)
    sorted_keys = keys[order]
    sorted_counts = counts[order]

    diff = torch.ones_like(sorted_keys, dtype=torch.bool)
    diff[1:] = sorted_keys[1:] != sorted_keys[:-1]
    run_starts = torch.nonzero(diff, as_tuple=False).flatten()

    prefix = torch.cumsum(sorted_counts, dim=0)
    prefix = torch.cat(
        [torch.zeros((1,), dtype=sorted_counts.dtype, device=device), prefix]
    )

    next_indices = torch.cat(
        [
            run_starts[1:],
            torch.as_tensor([sorted_keys.numel()], dtype=run_starts.dtype, device=device),
        ]
    )

    aggregated_counts = prefix[next_indices] - prefix[run_starts]
    aggregated_keys = sorted_keys[run_starts]
    return aggregated_keys, aggregated_counts


def reduce_pair_histograms(
    keys: torch.Tensor, counts: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce pair histograms across distributed workers using NCCL.

    The helper first performs a local aggregation of duplicate keys and then
    all-gathers the padded histograms so every rank materializes the global
    histogram.  When ``torch.distributed`` is unavailable or uninitialized the
    function simply returns the locally aggregated histogram.
    """

    keys, counts = aggregate_pair_keys(keys, counts)

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return keys, counts

    device = keys.device
    world_size = dist.get_world_size()

    local_length = torch.tensor([keys.numel()], dtype=torch.long, device=device)
    gathered_lengths = [torch.zeros_like(local_length) for _ in range(world_size)]
    dist.all_gather(gathered_lengths, local_length)
    lengths = [int(val.item()) for val in gathered_lengths]

    max_len = max(lengths, default=0)
    if max_len == 0:
        return keys.new_empty((0,), dtype=torch.long), counts.new_empty((0,), dtype=torch.long)

    pad_value = torch.iinfo(torch.long).max
    padded_keys = torch.full((max_len,), pad_value, dtype=torch.long, device=device)
    padded_counts = torch.zeros((max_len,), dtype=torch.long, device=device)
    if keys.numel() > 0:
        padded_keys[: keys.numel()] = keys
        padded_counts[: counts.numel()] = counts

    gathered_keys = [torch.empty_like(padded_keys) for _ in range(world_size)]
    gathered_counts = [torch.empty_like(padded_counts) for _ in range(world_size)]
    dist.all_gather(gathered_keys, padded_keys)
    dist.all_gather(gathered_counts, padded_counts)

    slices: list[torch.Tensor] = []
    slice_counts: list[torch.Tensor] = []
    for rank, length in enumerate(lengths):
        if length == 0:
            continue
        slices.append(gathered_keys[rank][:length])
        slice_counts.append(gathered_counts[rank][:length])

    if not slices:
        return keys.new_empty((0,), dtype=torch.long), counts.new_empty((0,), dtype=torch.long)

    all_keys = torch.cat(slices, dim=0)
    all_counts = torch.cat(slice_counts, dim=0)
    return aggregate_pair_keys(all_keys, all_counts)
