"""Triton kernels for GPU tokenizers."""

from __future__ import annotations

from typing import Optional

import torch

try:  # pragma: no cover - guarded import
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional at runtime
    triton = None
    tl = None


def is_triton_available() -> bool:
    """Return ``True`` when Triton is available for use."""

    return triton is not None


def can_use_triton_apply_merge(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    prefix_workspace: Optional[torch.Tensor],
    pair_workspace: Optional[torch.Tensor],
) -> bool:
    """Check whether the Triton merge kernel can be used."""

    if triton is None:
        return False
    if tokens.dim() != 2 or valid.dim() != 2:
        return False
    if tokens.shape != valid.shape:
        return False
    if not (tokens.is_cuda and valid.is_cuda):
        return False
    if prefix_workspace is None or not prefix_workspace.is_cuda:
        return False
    if pair_workspace is None or not pair_workspace.is_cuda:
        return False
    B, L = tokens.shape
    width = max(L - 1, 0)
    if prefix_workspace.dim() != 1 or prefix_workspace.shape[0] != B:
        return False
    if width > 0:
        if pair_workspace.dim() != 2:
            return False
        if pair_workspace.shape[0] != B or pair_workspace.shape[1] != width:
            return False
    else:
        if pair_workspace.dim() == 0:
            return False
        if pair_workspace.shape[0] != B:
            return False
    if tokens.dtype != torch.int32:
        return False
    if valid.dtype not in (torch.uint8, torch.bool):
        return False
    if prefix_workspace.dtype not in (torch.int32, torch.uint16):
        return False
    if pair_workspace.dtype != torch.bool:
        return False
    if not tokens.is_contiguous() or not valid.is_contiguous():
        return False
    if not prefix_workspace.is_contiguous():
        return False
    if pair_workspace.numel() > 0 and not pair_workspace.is_contiguous():
        return False
    return True


if triton is not None:

    @triton.jit
    def _apply_merge_and_compact_kernel(
        tokens_ptr,
        valid_ptr,
        prefix_ptr,
        pair_ptr,
        stride_tokens_row: tl.constexpr,
        stride_valid_row: tl.constexpr,
        stride_pair_row: tl.constexpr,
        a_id: tl.constexpr,
        b_id: tl.constexpr,
        new_id: tl.constexpr,
        prefix_max: tl.constexpr,
        L: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_PAIR: tl.constexpr,
        VALID_IS_BOOL: tl.constexpr,
        PREFIX_IS_UINT16: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        tokens_row = tokens_ptr + row_idx * stride_tokens_row
        valid_row = valid_ptr + row_idx * stride_valid_row
        if HAS_PAIR:
            pair_row = pair_ptr + row_idx * stride_pair_row
        else:
            pair_row = pair_ptr

        cols = tl.arange(0, BLOCK_SIZE)

        one_valid = (
            tl.full([BLOCK_SIZE], 1, dtype=tl.int1)
            if VALID_IS_BOOL
            else tl.full([BLOCK_SIZE], 1, dtype=tl.uint8)
        )
        zero_valid = (
            tl.zeros([BLOCK_SIZE], dtype=tl.int1)
            if VALID_IS_BOOL
            else tl.zeros([BLOCK_SIZE], dtype=tl.uint8)
        )

        # Stage 1: compute pair mask and apply merges.
        for start in range(0, WIDTH, BLOCK_SIZE):
            col = start + cols
            mask = col < WIDTH

            left_tokens = tl.load(tokens_row + col, mask=mask, other=0)
            right_tokens = tl.load(tokens_row + col + 1, mask=mask, other=0)

            left_valid = tl.load(valid_row + col, mask=mask, other=0)
            right_valid = tl.load(valid_row + col + 1, mask=mask, other=0)

            left_valid_bool = left_valid != 0
            right_valid_bool = right_valid != 0

            matches = left_valid_bool & right_valid_bool & (left_tokens == a_id) & (right_tokens == b_id)

            if HAS_PAIR:
                tl.store(pair_row + col, matches, mask=mask)

            merged_left = tl.where(matches, new_id, left_tokens)
            merged_right = tl.where(matches, 0, right_tokens)
            merged_right_valid = tl.where(matches, 0, right_valid)

            left_valid_store = (
                left_valid_bool.to(tl.int1) if VALID_IS_BOOL else left_valid_bool.to(tl.uint8)
            )
            right_valid_store = (
                (merged_right_valid != 0).to(tl.int1)
                if VALID_IS_BOOL
                else merged_right_valid.to(tl.uint8)
            )

            tl.store(tokens_row + col, merged_left, mask=mask)
            tl.store(tokens_row + col + 1, merged_right, mask=mask)
            tl.store(valid_row + col, left_valid_store, mask=mask)
            tl.store(valid_row + col + 1, right_valid_store, mask=mask)

        # Stage 2: compact surviving tokens to the left using a running prefix sum.
        running = tl.zeros((), dtype=tl.int32)
        for start in range(0, L, BLOCK_SIZE):
            col = start + cols
            mask = col < L

            staged_tokens = tl.load(tokens_row + col, mask=mask, other=0)
            staged_valid = tl.load(valid_row + col, mask=mask, other=0)
            keep = (staged_valid != 0) & mask
            keep_i32 = keep.to(tl.int32)

            local_prefix = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
            prefix_acc = running
            for idx in range(BLOCK_SIZE):
                take = keep_i32[idx]
                local_prefix[idx] = prefix_acc
                prefix_acc += take
            running = prefix_acc

            tl.store(tokens_row + col, tl.zeros([BLOCK_SIZE], dtype=tl.int32), mask=mask)
            tl.store(valid_row + col, zero_valid, mask=mask)

            destinations = local_prefix
            tl.store(tokens_row + destinations, staged_tokens, mask=keep)
            tl.store(valid_row + destinations, one_valid, mask=keep)

        if PREFIX_IS_UINT16:
            clipped = tl.minimum(running, prefix_max)
            tl.store(prefix_ptr + row_idx, clipped.to(tl.uint16))
        else:
            tl.store(prefix_ptr + row_idx, running.to(tl.int32))


def apply_merge_and_compact(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    prefix_workspace: torch.Tensor,
    pair_workspace: torch.Tensor,
    a_id: int,
    b_id: int,
    new_id: int,
) -> None:
    """Apply a BPE merge and compact tokens using the Triton kernel."""

    if not can_use_triton_apply_merge(tokens, valid, prefix_workspace, pair_workspace):
        raise RuntimeError("Triton kernel is not available for the provided tensors")

    B, L = tokens.shape
    width = max(L - 1, 0)

    if B == 0:
        prefix_workspace.zero_()
        if pair_workspace.numel() > 0:
            pair_workspace.zero_()
        return

    if L == 0:
        prefix_workspace.zero_()
        if pair_workspace.numel() > 0:
            pair_workspace.zero_()
        return

    stride_tokens = tokens.stride(0)
    stride_valid = valid.stride(0)
    stride_pair = pair_workspace.stride(0) if width > 0 and pair_workspace.numel() > 0 else 0

    block_size = 128

    pair_ptr = pair_workspace
    if pair_workspace.numel() == 0:
        pair_ptr = pair_workspace.new_empty(1)

    prefix_max = 65535
    grid = (B,)
    _apply_merge_and_compact_kernel[grid](
        tokens,
        valid,
        prefix_workspace,
        pair_ptr,
        stride_tokens,
        stride_valid,
        stride_pair,
        int(a_id),
        int(b_id),
        int(new_id),
        prefix_max,
        L,
        width,
        block_size,
        width > 0 and pair_workspace.numel() > 0,
        valid.dtype == torch.bool,
        prefix_workspace.dtype == torch.uint16,
        num_warps=4,
        num_stages=2,
    )


if triton is not None:

    @triton.jit
    def _count_pairs_histogram_kernel(
        tokens_ptr,
        valid_ptr,
        packed_keys_ptr,
        packed_counts_ptr,
        total_length_ptr,
        error_flag_ptr,
        stride_tokens: tl.constexpr,
        stride_valid: tl.constexpr,
        capacity,
        width,
        PAD_WIDTH: tl.constexpr,
        TOKENS_IS_INT64: tl.constexpr,
        VALID_IS_BOOL: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        tokens_row = tokens_ptr + row_idx * stride_tokens
        valid_row = valid_ptr + row_idx * stride_valid

        cols = tl.arange(0, PAD_WIDTH)
        active_cols = cols < width

        lhs = tl.load(tokens_row + cols, mask=active_cols, other=0)
        rhs = tl.load(tokens_row + cols + 1, mask=active_cols, other=0)

        lhs_i64 = lhs.to(tl.int64) if TOKENS_IS_INT64 else lhs.to(tl.int32).to(tl.int64)
        rhs_i64 = rhs.to(tl.int64) if TOKENS_IS_INT64 else rhs.to(tl.int32).to(tl.int64)

        left_valid = tl.load(valid_row + cols, mask=active_cols, other=0)
        right_valid = tl.load(valid_row + cols + 1, mask=active_cols, other=0)

        if VALID_IS_BOOL:
            left_valid_bool = left_valid != 0
            right_valid_bool = right_valid != 0
        else:
            left_valid_bool = left_valid.to(tl.int32) != 0
            right_valid_bool = right_valid.to(tl.int32) != 0

        pair_active = active_cols & left_valid_bool & right_valid_bool

        UINT32_MASK = 0xFFFFFFFF
        lhs_u32 = lhs_i64 & UINT32_MASK
        rhs_u32 = rhs_i64 & UINT32_MASK
        packed_keys = (lhs_u32 << 32) | rhs_u32

        sentinel = tl.full([PAD_WIDTH], -1, dtype=tl.int64)
        key_bucket = tl.shared_memory((PAD_WIDTH,), dtype=tl.int64)
        tl.store(key_bucket + cols, tl.where(pair_active, packed_keys, sentinel))

        sorted_keys = tl.sort(tl.load(key_bucket + cols))
        tl.store(key_bucket + cols, sorted_keys)

        num_valid = tl.sum(pair_active.to(tl.int32), axis=0)
        num_valid = tl.minimum(num_valid, width)
        start_idx = PAD_WIDTH - num_valid

        agg_keys = tl.shared_memory((PAD_WIDTH,), dtype=tl.int64)
        agg_counts = tl.shared_memory((PAD_WIDTH,), dtype=tl.int64)

        unique_count = tl.zeros((), dtype=tl.int32)
        run_length = tl.zeros((), dtype=tl.int32)
        prev_key = tl.zeros((), dtype=tl.int64)
        have_prev = tl.zeros((), dtype=tl.int1)

        for idx in range(PAD_WIDTH):
            key = tl.load(key_bucket + idx)
            idx_scalar = tl.full((), idx, dtype=tl.int32)
            active = (idx_scalar >= start_idx) & (num_valid > 0)

            same_key = have_prev & (key == prev_key)
            run_length = tl.where(active & same_key, run_length + 1, run_length)

            new_run = active & tl.logical_not(same_key)
            finalize = new_run & have_prev

            tl.store(agg_keys + unique_count, prev_key, mask=finalize)
            tl.store(agg_counts + unique_count, run_length.to(tl.int64), mask=finalize)
            unique_count = tl.where(finalize, unique_count + 1, unique_count)

            prev_key = tl.where(new_run, key, prev_key)
            run_length = tl.where(new_run, 1, run_length)
            have_prev = tl.where(
                active, tl.full((), 1, dtype=tl.int1), have_prev
            )

        tl.store(agg_keys + unique_count, prev_key, mask=have_prev)
        tl.store(agg_counts + unique_count, run_length.to(tl.int64), mask=have_prev)
        unique_count = tl.where(have_prev, unique_count + 1, unique_count)

        base = tl.atomic_add(total_length_ptr, unique_count)
        limit = capacity - base
        overflow = limit < unique_count
        tl.atomic_max(error_flag_ptr, overflow.to(tl.int32))

        store_mask = tl.logical_not(overflow)
        for idx in range(PAD_WIDTH):
            write_mask = store_mask & (idx < unique_count)
            key = tl.load(agg_keys + idx)
            count = tl.load(agg_counts + idx)
            tl.store(packed_keys_ptr + base + idx, key, mask=write_mask)
            tl.store(packed_counts_ptr + base + idx, count, mask=write_mask)


def can_use_triton_count_pairs(
    seqs: torch.Tensor,
    valid: torch.Tensor,
    pair_keys_buffer: torch.Tensor,
    pair_counts_buffer: torch.Tensor,
    pair_count_length: torch.Tensor,
) -> bool:
    if triton is None:
        return False
    if seqs.dim() != 2 or valid.dim() != 2:
        return False
    if seqs.shape != valid.shape:
        return False
    if not (seqs.is_cuda and valid.is_cuda):
        return False
    if not pair_keys_buffer.is_cuda or not pair_counts_buffer.is_cuda:
        return False
    if not pair_count_length.is_cuda:
        return False
    if not seqs.is_contiguous() or not valid.is_contiguous():
        return False
    if not pair_keys_buffer.is_contiguous() or not pair_counts_buffer.is_contiguous():
        return False
    if pair_count_length.numel() != 1:
        return False
    if pair_keys_buffer.dim() != 2 or pair_keys_buffer.shape[1] != 2:
        return False
    if pair_counts_buffer.dim() != 1:
        return False
    if pair_keys_buffer.size(0) != pair_counts_buffer.size(0):
        return False
    if pair_keys_buffer.dtype not in (torch.int32, torch.int64):
        return False
    if pair_counts_buffer.dtype != torch.int64:
        return False
    if seqs.dtype not in (torch.int32, torch.int64):
        return False
    if valid.dtype not in (torch.uint8, torch.bool):
        return False
    B, L = seqs.shape
    if B == 0 or L <= 1:
        return True
    width = L - 1
    max_width = 2048
    return width <= max_width


def count_pairs_histogram(
    seqs: torch.Tensor,
    valid: torch.Tensor,
    pair_keys_buffer: torch.Tensor,
    pair_counts_buffer: torch.Tensor,
    pair_count_length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not can_use_triton_count_pairs(seqs, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length):
        raise RuntimeError("Triton count_pairs kernel cannot be used for the provided tensors")

    B, L = seqs.shape
    width = max(L - 1, 0)
    capacity = pair_counts_buffer.numel()

    if B == 0 or width == 0 or capacity == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=seqs.device),
            torch.empty((0,), dtype=torch.int64, device=seqs.device),
            0,
        )

    pad_width = 1 << (width - 1).bit_length()
    pad_width = max(pad_width, 1)

    packed_keys = torch.empty((capacity,), dtype=torch.int64, device=seqs.device)
    packed_counts = torch.empty_like(pair_counts_buffer)
    error_flag = torch.zeros((1,), dtype=torch.int32, device=seqs.device)

    pair_count_length.zero_()

    grid = (B,)
    _count_pairs_histogram_kernel[grid](
        seqs,
        valid,
        packed_keys,
        packed_counts,
        pair_count_length,
        error_flag,
        seqs.stride(0),
        valid.stride(0),
        capacity,
        width,
        pad_width,
        seqs.dtype == torch.int64,
        valid.dtype == torch.bool,
        num_warps=4,
        num_stages=2,
    )

    if int(error_flag.item()) != 0:
        raise RuntimeError("pair workspace capacity exceeded")

    total = int(pair_count_length.item())
    return packed_keys, packed_counts, total


__all__ = [
    "is_triton_available",
    "can_use_triton_apply_merge",
    "apply_merge_and_compact",
    "can_use_triton_count_pairs",
    "count_pairs_histogram",
]

