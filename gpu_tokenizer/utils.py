"""Utility functions for GPU-based tokenization trainers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import logging
import os
import socket
import torch
import torch.distributed as dist

from .cuda_kernels import apply_merge_and_compact as cuda_apply_merge_and_compact
from .dtypes import clamp_lengths_to_dtype, promote_length_sum_dtype
from .triton_kernels import (
    can_use_triton_count_pairs,
    count_pairs_histogram,
)


UINT32_MAX = (1 << 32) - 1
logger = logging.getLogger(__name__)


_NODE_GROUP_CACHE: Dict[str, object] = {}


def _clear_cached_process_groups() -> None:
    """Reset cached process-group metadata (primarily for tests)."""

    _NODE_GROUP_CACHE.clear()


def _layout_from_env(
    world_size: int, rank: int
) -> Optional[Tuple[List[int], int, List[List[int]]]]:
    """Infer node layout using distributed environment variables."""

    try:
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    except (KeyError, ValueError):
        return None

    if local_world_size <= 0 or world_size % local_world_size != 0:
        return None

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", rank % local_world_size))
    except ValueError:
        local_rank = rank % local_world_size

    base_rank = rank - local_rank
    node_ranks = list(range(base_rank, base_rank + local_world_size))

    payload = (base_rank, local_world_size)
    gathered: List[Tuple[int, int]] = [payload] * world_size
    dist.all_gather_object(gathered, payload)

    nodes: List[List[int]] = []
    seen: set[Tuple[int, ...]] = set()
    for other_base, other_size in gathered:
        ranks_tuple = tuple(range(other_base, other_base + other_size))
        if ranks_tuple not in seen:
            seen.add(ranks_tuple)
            nodes.append(list(ranks_tuple))

    nodes.sort(key=lambda seq: seq[0])
    return node_ranks, local_rank, nodes


def _layout_from_hostnames(
    world_size: int, rank: int
) -> Tuple[List[int], int, List[List[int]]]:
    """Gather hostnames to deduce node-local rank groupings."""

    hostname = socket.gethostname()
    gathered: List[Optional[str]] = [None] * world_size
    dist.all_gather_object(gathered, hostname)

    host_to_ranks: Dict[str, List[int]] = {}
    for idx, host in enumerate(gathered):
        key = host or ""
        host_to_ranks.setdefault(key, []).append(idx)

    nodes = [sorted(ranks) for ranks in host_to_ranks.values()]
    nodes.sort(key=lambda seq: seq[0])
    node_ranks = next(seq for seq in nodes if rank in seq)
    local_rank = node_ranks.index(rank)
    return node_ranks, local_rank, nodes


def _get_or_create_node_groups() -> Dict[str, object]:
    """Cache process group metadata for hierarchical histogram reductions."""

    cached = _NODE_GROUP_CACHE.get("info")
    if cached is not None:
        return cached  # type: ignore[return-value]

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before calling reduce_pair_histograms")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    layout = _layout_from_env(world_size, rank)
    if layout is None:
        node_ranks, local_rank, nodes = _layout_from_hostnames(world_size, rank)
    else:
        node_ranks, local_rank, nodes = layout

    backend = dist.get_backend()
    local_group: Optional[dist.ProcessGroup] = None
    for ranks in nodes:
        group = dist.new_group(ranks=ranks, backend=backend)
        if rank in ranks:
            local_group = group
    if local_group is None:
        raise RuntimeError("Failed to construct local process group")

    node_leader = node_ranks[0]
    leader_ranks = [ranks[0] for ranks in nodes]

    leader_group: Optional[dist.ProcessGroup] = None
    if len(leader_ranks) > 1:
        inter_backend = "gloo"
        if inter_backend == backend:
            inter_backend = backend
        try:
            group = dist.new_group(ranks=leader_ranks, backend=inter_backend)
        except RuntimeError:
            if inter_backend != backend:
                logger.warning(
                    "Falling back to %s backend for inter-node reduction", backend
                )
                group = dist.new_group(ranks=leader_ranks, backend=backend)
            else:
                raise
        if rank in leader_ranks:
            leader_group = group
    info = {
        "local_group": local_group,
        "local_ranks": node_ranks,
        "local_rank": local_rank,
        "node_leader": node_leader,
        "leader_ranks": leader_ranks,
        "leader_group": leader_group,
    }
    _NODE_GROUP_CACHE["info"] = info
    return info


def _broadcast_histogram_to_local_ranks(
    keys_cpu: torch.Tensor,
    counts_cpu: torch.Tensor,
    local_group: dist.ProcessGroup,
    node_leader: int,
    device: torch.device,
    local_rank: int,
    local_world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Broadcast CPU reduced histograms from the node leader to local ranks."""

    if local_world_size == 1:
        return (
            keys_cpu.to(device=device, dtype=torch.long),
            counts_cpu.to(device=device, dtype=torch.int64),
        )

    if local_rank == 0:
        keys_device = keys_cpu.to(device=device, dtype=torch.long)
        counts_device = counts_cpu.to(device=device, dtype=torch.int64)
        length_tensor = torch.tensor([keys_device.numel()], dtype=torch.long, device=device)
    else:
        keys_device = torch.empty((0,), dtype=torch.long, device=device)
        counts_device = torch.empty((0,), dtype=torch.int64, device=device)
        length_tensor = torch.zeros((1,), dtype=torch.long, device=device)

    dist.broadcast(length_tensor, src=node_leader, group=local_group)
    total_len = int(length_tensor.item())

    if local_rank != 0:
        keys_device = torch.empty((total_len,), dtype=torch.long, device=device)
        counts_device = torch.empty((total_len,), dtype=torch.int64, device=device)
    elif keys_device.numel() != total_len:
        keys_device = keys_device[:total_len]
        counts_device = counts_device[:total_len]

    if total_len > 0:
        dist.broadcast(keys_device, src=node_leader, group=local_group)
        dist.broadcast(counts_device, src=node_leader, group=local_group)
    else:
        keys_device = keys_device.new_empty((0,), dtype=torch.long)
        counts_device = counts_device.new_empty((0,), dtype=torch.int64)

    return keys_device, counts_device


def _ensure_counts_int64(counts: torch.Tensor, context: str) -> torch.Tensor:
    """Promote ``counts`` to ``torch.int64`` while logging dtype changes."""

    if counts.dtype == torch.int64:
        return counts

    promoted = counts.to(torch.int64)
    if counts.numel() > 0 and counts.dtype != torch.int64:
        logger.debug("%s promoted %d counts to int64", context, counts.numel())
    return promoted


def device_of(x: torch.Tensor) -> torch.device:
    """Return the device hosting ``x``."""

    return x.device


@torch.jit.script
def compact_tokens_inplace(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    lengths: torch.Tensor,
    prefix_workspace: torch.Tensor,
    overflow_workspace: Optional[torch.Tensor] = None,
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

    rows = torch.arange(B, device=tokens.device, dtype=torch.int64)
    prefix_workspace.zero_()
    if overflow_workspace is not None:
        if overflow_workspace.numel() != B:
            raise RuntimeError("overflow_workspace shape mismatch")
        overflow_workspace.zero_()

    for col in range(L):
        keep_col = valid[:, col].to(torch.bool)
        src_vals = tokens[:, col]
        tokens[:, col] = 0
        valid[:, col] = 0
        if keep_col.any():
            row_ids = rows[keep_col]
            dst_long = prefix_workspace[row_ids].to(torch.long)
            tokens[row_ids, dst_long] = src_vals[keep_col]
            valid[row_ids, dst_long] = 1
            next_counts = dst_long + 1
            if prefix_workspace.dtype == torch.uint16:
                max_val = 65535
                next_counts = torch.clamp_max(next_counts, max_val)
            prefix_workspace[row_ids] = next_counts.to(prefix_workspace.dtype)

    prefix_values = prefix_workspace.to(torch.int64)
    if lengths.dtype == torch.uint16:
        max_val = 65535
        clipped = torch.clamp_max(prefix_values, max_val)
        lengths.copy_(clipped.to(torch.uint16))
        if overflow_workspace is not None:
            overflow_workspace.copy_(prefix_values > max_val)
    elif lengths.dtype == torch.int32:
        lengths.copy_(prefix_values.to(torch.int32))
        if overflow_workspace is not None:
            overflow_workspace.zero_()
    else:
        lengths.copy_(prefix_values.to(lengths.dtype))
        if overflow_workspace is not None:
            overflow_workspace.zero_()


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
    overflow_workspace: Optional[torch.Tensor] = None,
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

    if max(a_id, b_id, new_id) > UINT32_MAX:
        raise OverflowError(
            f"Token id overflow: encountered id above UINT32_MAX (limit {UINT32_MAX})."
        )

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
            sum_dtype = promote_length_sum_dtype(lengths.dtype)
            computed = valid.sum(dim=-1, dtype=sum_dtype)
            coerced, overflow = clamp_lengths_to_dtype(computed, lengths.dtype)
            lengths.copy_(coerced)
            if overflow_workspace is not None:
                if overflow_workspace.shape != (B,):
                    raise RuntimeError("overflow_workspace shape mismatch")
                if overflow is not None:
                    overflow_workspace.copy_(overflow.to(overflow_workspace.dtype))
                else:
                    overflow_workspace.zero_()
        if span_workspace is not None and span_workspace.shape == empty_span.shape:
            span_workspace.zero_()
            return seqs, valid, lengths, span_workspace
        return seqs, valid, lengths, empty_span

    if seqs.is_cuda:
        if pair_workspace is None or pair_workspace.shape != (B, width):
            pair_workspace = torch.zeros((B, width), dtype=torch.bool, device=device)
        expected_dtype = lengths.dtype
        if (
            prefix_workspace is None
            or prefix_workspace.shape[0] != B
            or prefix_workspace.dtype != expected_dtype
        ):
            prefix_workspace = torch.zeros((B,), dtype=expected_dtype, device=device)
        if overflow_workspace is not None:
            if overflow_workspace.shape != (B,):
                raise RuntimeError("overflow_workspace shape mismatch")
            overflow_workspace.zero_()

        cuda_apply_merge_and_compact(
            seqs,
            valid,
            prefix_workspace,
            pair_workspace,
            int(a_id),
            int(b_id),
            int(new_id),
        )

        prefix_values = prefix_workspace.to(torch.int64)
        if lengths.dtype == torch.uint16:
            max_val = 65535
            clipped = torch.clamp_max(prefix_values, max_val)
            lengths.copy_(clipped.to(torch.uint16))
            if overflow_workspace is not None:
                overflow_workspace.copy_(prefix_values > max_val)
        else:
            lengths.copy_(prefix_values.to(lengths.dtype))
            if overflow_workspace is not None:
                overflow_workspace.zero_()

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

    expected_dtype = lengths.dtype
    if (
        prefix_workspace is None
        or prefix_workspace.size(0) != B
        or prefix_workspace.dtype != expected_dtype
    ):
        prefix_workspace = torch.zeros((B,), dtype=expected_dtype, device=device)
    if overflow_workspace is not None:
        if overflow_workspace.shape != (B,):
            raise RuntimeError("overflow_workspace shape mismatch")
        overflow_workspace.zero_()

    compact_tokens_inplace(seqs, valid, lengths, prefix_workspace, overflow_workspace)
    return seqs, valid, lengths, spans


def _count_pairs_pytorch(
    seqs: torch.Tensor,
    valid: torch.Tensor,
    pair_keys_buffer: torch.Tensor,
    pair_counts_buffer: torch.Tensor,
    pair_count_length: torch.Tensor,
) -> None:
    """Packed-key implementation executed with PyTorch ops."""

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
    counts = _ensure_counts_int64(counts, "count_pairs")

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


def _count_pairs_triton(
    seqs: torch.Tensor,
    valid: torch.Tensor,
    pair_keys_buffer: torch.Tensor,
    pair_counts_buffer: torch.Tensor,
    pair_count_length: torch.Tensor,
) -> bool:
    if not can_use_triton_count_pairs(
        seqs, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length
    ):
        return False

    try:
        packed_keys, packed_counts, total = count_pairs_histogram(
            seqs, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length
        )
    except RuntimeError:
        return False

    capacity = pair_counts_buffer.numel()
    pair_count_length.zero_()

    if total == 0 or capacity == 0:
        if pair_keys_buffer.numel() > 0:
            pair_keys_buffer.fill_(-1)
        if pair_counts_buffer.numel() > 0:
            pair_counts_buffer.zero_()
        return True

    keys = packed_keys[:total]
    counts = packed_counts[:total]

    unique_keys, unique_counts = aggregate_pair_keys(keys, counts)
    length = int(unique_keys.numel())
    if length > capacity:
        raise RuntimeError("pair workspace capacity exceeded")

    lhs_vals = (unique_keys >> 32).to(seqs.dtype)
    rhs_vals = (unique_keys & UINT32_MAX).to(seqs.dtype)

    pair_keys_buffer[:length, 0].copy_(lhs_vals)
    pair_keys_buffer[:length, 1].copy_(rhs_vals)
    pair_counts_buffer[:length].copy_(unique_counts)

    if length < pair_keys_buffer.size(0):
        pair_keys_buffer[length:].fill_(-1)
    if length < pair_counts_buffer.size(0):
        pair_counts_buffer[length:].zero_()

    pair_count_length[0] = length
    return True


def count_pairs(
    seqs: torch.Tensor,
    valid: torch.Tensor,
    pair_keys_buffer: torch.Tensor,
    pair_counts_buffer: torch.Tensor,
    pair_count_length: torch.Tensor,
) -> None:
    """Populate ``pair_*`` workspaces with unique adjacent token pairs."""

    used_triton = _count_pairs_triton(
        seqs, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length
    )
    if not used_triton:
        _count_pairs_pytorch(
            seqs, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length
        )


def aggregate_pair_keys(
    keys: torch.Tensor, counts: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate duplicate packed pair ``keys`` by summing ``counts``.

    Both tensors are expected to be one-dimensional and reside on the same
    device.  The helper returns unique keys sorted in ascending order along with
    their aggregated counts.
    """

    if keys.numel() == 0:
        if keys.dtype != torch.long:
            keys = keys.to(torch.long)
        counts = _ensure_counts_int64(counts, "aggregate_pair_keys input")
        return keys, counts

    device = keys.device
    keys = keys.to(torch.long)
    counts = _ensure_counts_int64(counts, "aggregate_pair_keys input")

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
    """Reduce pair histograms across distributed workers using hierarchical groups.

    The helper performs a local aggregation of duplicate keys and then
    participates in an intra-node reduction (NCCL when available) to shrink the
    histogram prior to any cross-node communication.  Node leaders then exchange
    the compacted histograms over a CPU/Gloo group before broadcasting the
    merged result back to their local peers.  When ``torch.distributed`` is
    unavailable or uninitialized the function simply returns the locally
    aggregated histogram.
    """

    keys, counts = aggregate_pair_keys(keys, counts)

    if not dist.is_available() or not dist.is_initialized():
        return keys, counts

    if dist.get_world_size() == 1:
        return keys, counts

    info = _get_or_create_node_groups()
    device = keys.device

    local_group: dist.ProcessGroup = info["local_group"]  # type: ignore[index]
    local_ranks: List[int] = info["local_ranks"]  # type: ignore[index]
    local_rank: int = info["local_rank"]  # type: ignore[index]
    node_leader: int = info["node_leader"]  # type: ignore[index]
    leader_group: Optional[dist.ProcessGroup] = info["leader_group"]  # type: ignore[index]
    leader_ranks: List[int] = info["leader_ranks"]  # type: ignore[index]

    local_world_size = len(local_ranks)

    # Intra-node reduction (typically NCCL-backed)
    if local_world_size > 1:
        local_length = torch.tensor([keys.numel()], dtype=torch.long, device=device)
        gathered_lengths = [torch.zeros_like(local_length) for _ in range(local_world_size)]
        dist.all_gather(gathered_lengths, local_length, group=local_group)
        lengths = [int(val.item()) for val in gathered_lengths]
        max_len = max(lengths, default=0)
        if max_len == 0:
            node_keys = keys.new_empty((0,), dtype=torch.long)
            node_counts = counts.new_empty((0,), dtype=counts.dtype)
        else:
            pad_value = torch.iinfo(torch.long).max
            padded_keys = torch.full((max_len,), pad_value, dtype=torch.long, device=device)
            if keys.numel() > 0:
                padded_keys[: keys.numel()] = keys
            gathered_keys = [torch.empty_like(padded_keys) for _ in range(local_world_size)]
            dist.all_gather(gathered_keys, padded_keys, group=local_group)

            valid_slices = [gathered_keys[idx][:lengths[idx]] for idx in range(local_world_size) if lengths[idx] > 0]
            if valid_slices:
                union_keys = torch.unique(torch.cat(valid_slices, dim=0), sorted=True)
            else:
                union_keys = keys.new_empty((0,), dtype=torch.long)

            if union_keys.numel() == 0:
                node_keys = keys.new_empty((0,), dtype=torch.long)
                node_counts = counts.new_empty((0,), dtype=counts.dtype)
            else:
                local_counts64 = torch.zeros((union_keys.numel(),), dtype=torch.int64, device=device)
                if keys.numel() > 0:
                    indices = torch.searchsorted(union_keys, keys)
                    local_counts64.index_add_(0, indices, counts.to(torch.int64))
                dist.all_reduce(local_counts64, op=dist.ReduceOp.SUM, group=local_group)
                node_keys, node_counts = aggregate_pair_keys(union_keys, local_counts64)
    else:
        node_keys, node_counts = keys, counts

    # Inter-node reduction (CPU/Gloo) with broadcast back to local ranks
    if len(leader_ranks) == 1:
        return node_keys, node_counts

    if local_rank == 0:
        node_keys_cpu = node_keys.to(dtype=torch.long, device="cpu")
        node_counts_cpu = node_counts.to(dtype=torch.int64, device="cpu")
        payload = (node_keys_cpu.tolist(), node_counts_cpu.tolist())
        gathered: List[Tuple[List[int], List[int]]] = [([], []) for _ in leader_ranks]
        assert leader_group is not None
        dist.all_gather_object(gathered, payload, group=leader_group)

        union_set = {key for keys_list, _ in gathered for key in keys_list}
        if union_set:
            sorted_keys = torch.tensor(sorted(union_set), dtype=torch.long)
            reduced_counts64 = torch.zeros((sorted_keys.numel(),), dtype=torch.int64)
            if node_keys_cpu.numel() > 0:
                indices = torch.searchsorted(sorted_keys, node_keys_cpu)
                reduced_counts64.index_add_(0, indices, node_counts_cpu.to(torch.int64))
            dist.all_reduce(reduced_counts64, op=dist.ReduceOp.SUM, group=leader_group)
            final_keys_cpu = sorted_keys
            final_counts_cpu = reduced_counts64
        else:
            final_keys_cpu = torch.empty((0,), dtype=torch.long)
            final_counts_cpu = torch.empty((0,), dtype=torch.int64)
    else:
        final_keys_cpu = torch.empty((0,), dtype=torch.long)
        final_counts_cpu = torch.empty((0,), dtype=torch.int64)

    final_keys, final_counts = _broadcast_histogram_to_local_ranks(
        final_keys_cpu,
        final_counts_cpu,
        local_group,
        node_leader,
        device,
        local_rank,
        local_world_size,
    )
    return final_keys, final_counts
