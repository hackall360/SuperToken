"""Shared dtype helpers for tokenizer length bookkeeping."""

from __future__ import annotations

import torch

if not hasattr(torch, "uint16"):
    torch.uint16 = torch.int32  # type: ignore[attr-defined]

UINT16_MAX: int = (1 << 16) - 1
UINT16_DTYPE: torch.dtype = torch.uint16
INT32_MAX: int = int(torch.iinfo(torch.int32).max)


def length_storage_dtype(storage_width: int) -> torch.dtype:
    """Return the storage dtype for sequence lengths given ``storage_width``.

    The helper prefers ``torch.uint16`` when the backing storage can never exceed
    ``UINT16_MAX`` to reduce memory pressure, otherwise ``torch.int32`` is used.
    """

    if storage_width <= UINT16_MAX:
        return UINT16_DTYPE
    return torch.int32


def promote_length_sum_dtype(length_dtype: torch.dtype) -> torch.dtype:
    """Return a dtype suitable for summing valid-token masks safely."""

    if length_dtype == UINT16_DTYPE:
        return torch.int32
    if length_dtype == torch.int16:
        return torch.int32
    if length_dtype == torch.int8:
        return torch.int32
    if length_dtype == torch.uint8:
        return torch.int32
    return length_dtype


def clamp_lengths_to_dtype(
    values: torch.Tensor, target_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Clamp ``values`` to ``target_dtype`` while reporting any overflow.

    The returned tensor resides on the same device as ``values``.  If
    ``target_dtype`` does not require clipping ``None`` is returned for the
    overflow mask.
    """

    working = values.to(torch.int64)
    overflow: torch.Tensor | None = None

    if target_dtype == UINT16_DTYPE:
        if working.numel() > 0:
            overflow = working > UINT16_MAX
            working = working.clamp(max=UINT16_MAX)
        result = working.to(UINT16_DTYPE)
        return result, overflow

    if target_dtype == torch.int32:
        if working.numel() > 0:
            # Lengths are non-negative so we only need to guard the upper bound.
            overflow = working > INT32_MAX
            working = working.clamp(max=INT32_MAX)
        result = working.to(torch.int32)
        return result, overflow

    result = working.to(target_dtype)
    return result, None
