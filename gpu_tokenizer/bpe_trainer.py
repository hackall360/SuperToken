"""GPU-accelerated BPE trainer with optional autoscaling."""

from __future__ import annotations

import heapq
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from .autoscaler import AutoScaler, ScaleState
from .cuda_kernels import apply_merge_and_compact as cuda_apply_merge_and_compact
from .cpu_fastpath import (
    FastPathWorkspaces,
    apply_merge_fastpath,
    count_pairs_fastpath,
    should_route_to_cpu,
)
from .dtypes import (
    clamp_lengths_to_dtype,
    length_storage_dtype,
    promote_length_sum_dtype,
)
from .utils import (
    aggregate_pair_keys,
    apply_merge_once,
    count_pairs,
    peer_copy_tensor,
    reduce_pair_histograms,
)


def _aggregate_pair_keys(
    keys: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for :func:`gpu_tokenizer.utils.aggregate_pair_keys`.

    The public trainer used to expose a private helper that forwarded to the
    shared aggregation utility.  Some downstream tests (and a few lightweight
    integration scripts) still import ``_aggregate_pair_keys`` from
    ``gpu_tokenizer.bpe_trainer``.  A previous refactor removed the shim which
    broke those imports.  Restoring the wrapper keeps the historical API
    surface while delegating the work to the canonical implementation in
    ``gpu_tokenizer.utils``.
    """

    return aggregate_pair_keys(keys, counts)

try:  # pragma: no cover - optional dependency
    from tokenizers import Tokenizer as _HFTokenizer
except Exception:  # pragma: no cover - optional dependency in CI
    _HFTokenizer = None


UINT32_MAX = (1 << 32) - 1


@dataclass
class TrainerMetricsEWMA:
    """Maintain exponential moving averages for trainer throughput metrics."""

    alpha: float = 0.2
    window_size: int = 16
    enabled: bool = False
    overlap_enabled: bool = True
    _tokens_per_s: float | None = None
    _lease_per_s: float | None = None
    _stage_windows: dict[str, deque[float]] = field(default_factory=dict)
    _copy_window: deque[float] = field(init=False, repr=False)
    _compute_window: deque[float] = field(init=False, repr=False)

    _COPY_STAGES = frozenset({"h2d", "d2h"})
    _COMPUTE_STAGES = frozenset({"kernel", "reduction"})

    def __post_init__(self) -> None:
        # Clamp configuration to sensible defaults while tolerating bad inputs.
        try:
            alpha = float(self.alpha)
        except (TypeError, ValueError):
            alpha = 0.2
        if alpha <= 0.0:
            alpha = 0.01
        if alpha > 1.0:
            alpha = 1.0
        self.alpha = alpha
        try:
            window = int(self.window_size)
        except (TypeError, ValueError):
            window = 16
        if window <= 0:
            window = 1
        self.window_size = window
        self.enabled = bool(self.enabled)
        self.overlap_enabled = bool(self.overlap_enabled)
        self._copy_window = deque(maxlen=self.window_size)
        self._compute_window = deque(maxlen=self.window_size)

    def reset(self) -> None:
        """Clear accumulated metrics."""

        self._tokens_per_s = None
        self._lease_per_s = None
        self._stage_windows.clear()
        self._copy_window = deque(maxlen=self.window_size)
        self._compute_window = deque(maxlen=self.window_size)

    @property
    def tokens_per_s(self) -> float | None:
        return self._tokens_per_s

    @property
    def lease_per_s(self) -> float | None:
        return self._lease_per_s

    def record_stage(self, stage: str, duration_s: float) -> str:
        if not self.enabled or duration_s < 0:
            return "other"
        window = self._stage_windows.setdefault(stage, deque(maxlen=self.window_size))
        window.append(float(duration_s))
        stage_key = stage.lower()
        kind = "other"
        if stage_key in self._COPY_STAGES:
            self._copy_window.append(float(duration_s))
            kind = "copy"
        elif stage_key in self._COMPUTE_STAGES:
            self._compute_window.append(float(duration_s))
            kind = "compute"
        return kind

    def record_tokens(self, tokens: int, duration_s: float, *, leases: int | None = None) -> None:
        if not self.enabled or duration_s <= 0:
            return
        rate = float(tokens) / float(duration_s) if tokens > 0 else 0.0
        if self._tokens_per_s is None:
            self._tokens_per_s = rate
        else:
            self._tokens_per_s = (self.alpha * rate) + ((1.0 - self.alpha) * self._tokens_per_s)
        if leases is not None:
            lease_rate = float(leases) / float(duration_s) if leases > 0 else 0.0
            if self._lease_per_s is None:
                self._lease_per_s = lease_rate
            else:
                self._lease_per_s = (
                    self.alpha * lease_rate
                    + (1.0 - self.alpha) * self._lease_per_s
                )

    def summaries(self) -> dict[str, object]:
        stage_summary: dict[str, dict[str, object]] = {}
        for name, window in self._stage_windows.items():
            samples = list(window)
            count = len(samples)
            avg = sum(samples) / count if count else 0.0
            stage_summary[name] = {
                "samples": count,
                "avg_s": avg,
                "latest_s": samples[-1] if samples else 0.0,
                "window": samples,
            }

        def _window_stats(window: deque[float]) -> dict[str, object]:
            values = list(window)
            count = len(values)
            avg = sum(values) / count if count else 0.0
            latest = values[-1] if values else 0.0
            return {
                "samples": count,
                "avg_s": avg,
                "latest_s": latest,
                "window": values,
            }

        return {
            "enabled": self.enabled,
            "alpha": self.alpha,
            "window": self.window_size,
            "tokens_per_s": self._tokens_per_s,
            "lease_per_s": self._lease_per_s,
            "stages": stage_summary,
            "copy": _window_stats(self._copy_window),
            "compute": _window_stats(self._compute_window),
            "overlap_enabled": self.overlap_enabled,
        }


@dataclass
class StageTiming:
    """Accumulates timing samples for a named stage."""

    name: str
    samples: list[float] = field(default_factory=list)

    def record(self, duration_s: float) -> None:
        if duration_s < 0:
            return
        self.samples.append(float(duration_s))

    def summary(self) -> dict[str, object]:
        total = float(sum(self.samples))
        count = len(self.samples)
        avg = total / count if count else 0.0
        return {
            "stage": self.name,
            "count": count,
            "total_s": total,
            "avg_s": avg,
            "samples": list(self.samples),
        }


@dataclass
class GPUBatchRecord:
    """Container for GPU-resident batch state with optional host mirrors."""

    tokens: torch.Tensor
    valid: torch.Tensor
    lengths: torch.Tensor
    host_tokens: Optional[torch.Tensor] = None
    host_valid: Optional[torch.Tensor] = None
    host_lengths: Optional[torch.Tensor] = None
    host_dirty: bool = False
    device_event: Optional[torch.cuda.Event] = None
    host_event: Optional[torch.cuda.Event] = None
    pair_workspace: Optional[torch.Tensor] = None
    prefix_workspace: Optional[torch.Tensor] = None
    span_workspace: Optional[torch.Tensor] = None
    span_history: list[torch.Tensor] = field(default_factory=list)
    pair_keys: Optional[torch.Tensor] = None
    pair_counts: Optional[torch.Tensor] = None
    pair_keys_buffer: Optional[torch.Tensor] = None
    pair_counts_buffer: Optional[torch.Tensor] = None
    pair_count_length: Optional[torch.Tensor] = None
    merge_kernel_warm: bool = False
    length_overflow: Optional[torch.Tensor] = None
    host_length_overflow: Optional[torch.Tensor] = None


    @classmethod
    def from_cpu(
        cls,
        tokens_cpu: torch.Tensor,
        valid_cpu: torch.Tensor,
        lengths_cpu: torch.Tensor,
        device: torch.device,
        *,
        ctx: Optional["DeviceContext"] = None,
    ) -> "GPUBatchRecord":
        storage_width = int(tokens_cpu.shape[1])
        length_dtype = length_storage_dtype(storage_width)
        tokens_int = tokens_cpu.to(torch.int32)
        valid_uint = valid_cpu.to(torch.uint8)
        coerced_lengths, overflow = clamp_lengths_to_dtype(lengths_cpu, length_dtype)
        overflow_dev: Optional[torch.Tensor] = None
        host_tokens: Optional[torch.Tensor]
        host_valid: Optional[torch.Tensor]
        host_lengths: Optional[torch.Tensor]
        host_length_overflow: Optional[torch.Tensor]
        transfer_event: Optional[torch.cuda.Event] = None
        overlap_enabled = bool(getattr(ctx, "overlap_enabled", True)) if ctx else True
        if ctx is not None and ctx.device.type == "cuda":
            tokens_host, valid_host, lengths_host = ctx.prepare_staging_buffers(
                tokens_int.shape, coerced_lengths.dtype
            )
            tokens_host.copy_(tokens_int)
            valid_host.copy_(valid_uint)
            lengths_host.copy_(coerced_lengths)
            overflow_host: Optional[torch.Tensor] = None
            if overflow is not None:
                overflow_host = ctx.prepare_overflow_buffer(tokens_int.shape[0])
                overflow_host.copy_(overflow)
            stream = ctx.h2d_stream if overlap_enabled else ctx.compute_stream
            with torch.cuda.device(device), torch.cuda.stream(stream):
                tokens_dev = tokens_host.to(
                    device=device, non_blocking=overlap_enabled
                )
                valid_dev = valid_host.to(device=device, non_blocking=overlap_enabled)
                lengths_dev = lengths_host.to(device=device, non_blocking=overlap_enabled)
                if overflow is not None and overflow_host is not None:
                    overflow_dev = overflow_host.to(
                        device=device, non_blocking=overlap_enabled
                    )
                if overlap_enabled:
                    transfer_event = torch.cuda.Event(blocking=False)
                    transfer_event.record(stream)
                else:
                    transfer_event = None
            ctx.pending_h2d_event = transfer_event
            if overlap_enabled:
                host_tokens = None
                host_valid = None
                host_lengths = None
                host_length_overflow = None
            else:
                host_tokens = tokens_host
                host_valid = valid_host
                host_lengths = lengths_host
                host_length_overflow = overflow_host if overflow is not None else None
        else:
            tokens_host = tokens_int.pin_memory()
            valid_host = valid_uint.pin_memory()
            lengths_host = coerced_lengths.pin_memory()
            tokens_dev = tokens_host.to(device=device, non_blocking=True)
            valid_dev = valid_host.to(device=device, non_blocking=True)
            lengths_dev = lengths_host.to(device=device, non_blocking=True)
            if overflow is not None:
                host_length_overflow = overflow.pin_memory()
                overflow_dev = host_length_overflow.to(device=device, non_blocking=True)
            else:
                host_length_overflow = None
            host_tokens = tokens_host
            host_valid = valid_host
            host_lengths = lengths_host
        record = cls(
            tokens=tokens_dev,
            valid=valid_dev,
            lengths=lengths_dev,
            host_tokens=host_tokens,
            host_valid=host_valid,
            host_lengths=host_lengths,
            host_dirty=False,
            length_overflow=overflow_dev,
            host_length_overflow=host_length_overflow,
        )
        if transfer_event is not None:
            record.device_event = transfer_event
        record.ensure_workspaces()
        return record

    def ensure_workspaces(self) -> None:
        """Allocate or resize device workspaces for merges."""

        B, L = self.tokens.shape
        device = self.tokens.device
        width = max(L - 1, 0)
        if width == 0:
            if self.pair_workspace is None or self.pair_workspace.shape != (B, 0):
                self.pair_workspace = torch.empty((B, 0), dtype=torch.bool, device=device)
                self.merge_kernel_warm = False
            if self.span_workspace is None or self.span_workspace.shape != (B, 0):
                self.span_workspace = torch.empty((B, 0), dtype=torch.bool, device=device)
        elif self.pair_workspace is None or self.pair_workspace.shape != (B, width):
            self.pair_workspace = torch.zeros((B, width), dtype=torch.bool, device=device)
            self.merge_kernel_warm = False
        if width > 0 and (self.span_workspace is None or self.span_workspace.shape != (B, width)):
            self.span_workspace = torch.zeros((B, width), dtype=torch.bool, device=device)
        prefix_dtype = self.lengths.dtype
        if (
            self.prefix_workspace is None
            or self.prefix_workspace.shape[0] != B
            or self.prefix_workspace.dtype != prefix_dtype
        ):
            self.prefix_workspace = torch.zeros((B,), dtype=prefix_dtype, device=device)
            self.merge_kernel_warm = False
        if (
            self.length_overflow is None
            or self.length_overflow.shape != (B,)
            or self.length_overflow.device != device
        ):
            self.length_overflow = torch.zeros((B,), dtype=torch.bool, device=device)
        else:
            self.length_overflow.zero_()
        capacity = B * width
        if capacity == 0:
            if self.pair_keys_buffer is None or self.pair_keys_buffer.shape != (0, 2):
                self.pair_keys_buffer = torch.empty((0, 2), dtype=self.tokens.dtype, device=device)
            if self.pair_counts_buffer is None or self.pair_counts_buffer.shape != (0,):
                self.pair_counts_buffer = torch.empty(
                    (0,), dtype=torch.int64, device=device
                )
        else:
            if (
                self.pair_keys_buffer is None
                or self.pair_keys_buffer.shape[0] != capacity
                or self.pair_keys_buffer.shape[1] != 2
            ):
                self.pair_keys_buffer = torch.empty(
                    (capacity, 2), dtype=self.tokens.dtype, device=device
                )
            if (
                self.pair_counts_buffer is None
                or self.pair_counts_buffer.shape[0] != capacity
                or self.pair_counts_buffer.dtype != torch.int64
            ):
                self.pair_counts_buffer = torch.empty(
                    (capacity,), dtype=torch.int64, device=device
                )
        if self.pair_count_length is None or self.pair_count_length.shape != (1,):
            self.pair_count_length = torch.zeros((1,), dtype=torch.long, device=device)

    def ensure_host_buffers(self) -> None:
        """Allocate pinned host mirrors if absent or stale."""

        if self.host_tokens is None or self.host_tokens.shape != self.tokens.shape:
            self.host_tokens = torch.empty_like(self.tokens, device="cpu").pin_memory()
        if self.host_valid is None or self.host_valid.shape != self.valid.shape:
            self.host_valid = torch.empty_like(self.valid, device="cpu").pin_memory()
        if (
            self.host_lengths is None
            or self.host_lengths.shape != self.lengths.shape
            or self.host_lengths.dtype != self.lengths.dtype
        ):
            self.host_lengths = torch.empty_like(self.lengths, device="cpu").pin_memory()
        if (
            self.host_length_overflow is None
            or self.host_length_overflow.shape != self.lengths.shape
        ):
            self.host_length_overflow = torch.zeros(
                self.lengths.shape, dtype=torch.bool, device="cpu"
            ).pin_memory()

    def wait_for_device(self, stream: torch.cuda.Stream) -> None:
        """Ensure device computations affecting this batch are completed."""

        if self.host_event is not None:
            stream.wait_event(self.host_event)
        if self.device_event is not None:
            stream.wait_event(self.device_event)
            self.device_event = None

    def schedule_host_sync(
        self, copy_stream: torch.cuda.Stream, *, overlap: bool = True
    ) -> None:
        """Schedule a copy of device data back to host memory."""

        self.ensure_host_buffers()
        assert self.host_tokens is not None
        assert self.host_valid is not None
        if not overlap:
            with torch.cuda.stream(copy_stream):
                self.wait_for_device(copy_stream)
                self.host_tokens.copy_(self.tokens, non_blocking=False)
                self.host_valid.copy_(self.valid, non_blocking=False)
            self.host_event = None
            return

        event = torch.cuda.Event(blocking=False, enable_timing=True)
        with torch.cuda.stream(copy_stream):
            self.wait_for_device(copy_stream)
            self.host_tokens.copy_(self.tokens, non_blocking=True)
            self.host_valid.copy_(self.valid, non_blocking=True)
            event.record(copy_stream)
        self.host_event = event

    def wait_for_host(self) -> None:
        """Block until any pending host copy for this batch completes."""

        if self.host_event is not None:
            self.host_event.synchronize()
            self.host_event = None

    def resolve_host(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return up-to-date host buffers for this batch."""

        assert self.host_tokens is not None
        assert self.host_valid is not None
        self.wait_for_host()
        sum_dtype = promote_length_sum_dtype(self.lengths.dtype)
        lengths_cpu = self.host_valid.sum(dim=-1, dtype=sum_dtype)
        coerced, overflow = clamp_lengths_to_dtype(lengths_cpu, self.lengths.dtype)
        if self.host_lengths is None or self.host_lengths.shape != coerced.shape:
            self.host_lengths = coerced.pin_memory()
        else:
            self.host_lengths.copy_(coerced)
        if overflow is not None:
            if self.host_length_overflow is None or self.host_length_overflow.shape != overflow.shape:
                self.host_length_overflow = overflow.pin_memory()
            else:
                self.host_length_overflow.copy_(overflow)
            if self.length_overflow is not None and self.length_overflow.shape == overflow.shape:
                self.length_overflow.copy_(overflow.to(self.length_overflow.device))
        elif self.host_length_overflow is not None:
            self.host_length_overflow.zero_()
            if self.length_overflow is not None:
                self.length_overflow.zero_()
        self.host_dirty = False
        return self.host_tokens, self.host_valid, self.host_lengths

    def mark_device_event(self, event: torch.cuda.Event) -> None:
        """Record the latest device-side event for downstream synchronization."""

        self.device_event = event
        self.host_dirty = True


@dataclass
class DeviceContext:
    """Execution context for a single CUDA device."""

    device: torch.device
    compute_stream: torch.cuda.Stream
    h2d_stream: torch.cuda.Stream
    d2h_stream: torch.cuda.Stream
    overlap_enabled: bool = True
    bytes_h2d: int = 0
    bytes_d2h: int = 0
    h2d_events: int = 0
    d2h_events: int = 0
    active_batches: list[GPUBatchRecord] = field(default_factory=list)
    stage_transfers: dict[str, dict[str, int]] = field(default_factory=dict)
    memory_snapshots: list[dict[str, object]] = field(default_factory=list)
    utilization_samples: list[dict[str, float]] = field(default_factory=list)
    staging_tokens: Optional[torch.Tensor] = None
    staging_valid: Optional[torch.Tensor] = None
    staging_lengths: Optional[torch.Tensor] = None
    staging_overflow: Optional[torch.Tensor] = None
    pending_h2d_event: Optional[torch.cuda.Event] = None

    def reset_activity(self) -> None:
        self.active_batches.clear()

    def _await_pending_h2d(self) -> None:
        if self.pending_h2d_event is not None:
            self.pending_h2d_event.synchronize()
            self.pending_h2d_event = None

    def prepare_staging_buffers(
        self, shape: tuple[int, int], length_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._await_pending_h2d()
        rows, width = shape
        if self.staging_tokens is None or self.staging_tokens.shape != (rows, width):
            self.staging_tokens = (
                torch.empty((rows, width), dtype=torch.int32, device="cpu").pin_memory()
            )
        if self.staging_valid is None or self.staging_valid.shape != (rows, width):
            self.staging_valid = (
                torch.empty((rows, width), dtype=torch.uint8, device="cpu").pin_memory()
            )
        if (
            self.staging_lengths is None
            or self.staging_lengths.shape != (rows,)
            or self.staging_lengths.dtype != length_dtype
        ):
            self.staging_lengths = (
                torch.empty((rows,), dtype=length_dtype, device="cpu").pin_memory()
            )
        return self.staging_tokens, self.staging_valid, self.staging_lengths

    def prepare_overflow_buffer(self, rows: int) -> torch.Tensor:
        self._await_pending_h2d()
        if self.staging_overflow is None or self.staging_overflow.shape != (rows,):
            self.staging_overflow = (
                torch.empty((rows,), dtype=torch.bool, device="cpu").pin_memory()
            )
        return self.staging_overflow

    def log_transfer(self, stage: str, direction: str, amount: int) -> None:
        stage_key = stage or "general"
        entry = self.stage_transfers.setdefault(stage_key, {"h2d": 0, "d2h": 0})
        if direction == "h2d":
            entry["h2d"] += int(amount)
        elif direction == "d2h":
            entry["d2h"] += int(amount)

    def capture_snapshot(self, stage: str) -> dict[str, object]:
        timestamp = time.time()
        snapshot: dict[str, object] = {
            "stage": stage,
            "timestamp": timestamp,
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                with torch.cuda.device(self.device):
                    free, total = torch.cuda.mem_get_info(self.device)
                    snapshot["memory"] = {
                        "free_bytes": int(free),
                        "total_bytes": int(total),
                        "used_bytes": int(max(0, total - free)),
                    }
                    snapshot["allocated_bytes"] = int(
                        torch.cuda.memory_allocated(self.device)
                    )
                    snapshot["reserved_bytes"] = int(
                        torch.cuda.memory_reserved(self.device)
                    )
                    util_fn = getattr(torch.cuda, "utilization", None)
                    utilization: float | None = None
                    if callable(util_fn):
                        try:
                            utilization = float(util_fn(self.device))
                        except Exception:
                            utilization = None
                    if utilization is not None:
                        snapshot["utilization"] = utilization
                        self.utilization_samples.append(
                            {
                                "stage": stage,
                                "timestamp": timestamp,
                                "utilization": utilization,
                            }
                        )
            except RuntimeError:
                pass
        self.memory_snapshots.append(snapshot)
        return snapshot


@dataclass
class MultiDeviceBatch:
    """Wrapper holding per-device GPU batch shards."""

    shards: Dict[torch.device, GPUBatchRecord] = field(default_factory=dict)

    def total_rows(self) -> int:
        return sum(int(record.tokens.shape[0]) for record in self.shards.values())

    def iter_shards(self):
        return self.shards.items()

    def sequences(self) -> list[list[int]]:
        sequences: list[list[int]] = []
        for record in self.shards.values():
            tokens_cpu = record.tokens.detach().to("cpu")
            lengths_cpu = record.lengths.detach().to("cpu")
            rows = int(tokens_cpu.shape[0])
            for row in range(rows):
                length = int(lengths_cpu[row].item())
                if length <= 0:
                    sequences.append([])
                    continue
                seq = tokens_cpu[row, :length].to(torch.long).tolist()
                sequences.append(seq)
        return sequences


@dataclass
class CPUFallbackBatch:
    """Container representing a CPU-resident batch handled by the fast path."""

    tokens: torch.Tensor
    valid: torch.Tensor
    lengths: torch.Tensor
    spans: list[torch.Tensor] = field(default_factory=list)
    pair_keys: Optional[torch.Tensor] = None
    pair_counts: Optional[torch.Tensor] = None
    workspaces: FastPathWorkspaces = field(default_factory=FastPathWorkspaces)

    def clone(self) -> "CPUFallbackBatch":
        return CPUFallbackBatch(
            tokens=self.tokens.clone(),
            valid=self.valid.clone(),
            lengths=self.lengths.clone(),
            spans=list(self.spans),
            pair_keys=None if self.pair_keys is None else self.pair_keys.clone(),
            pair_counts=None if self.pair_counts is None else self.pair_counts.clone(),
            workspaces=self.workspaces,
        )

    @classmethod
    def from_cpu_batch(
        cls,
        tokens_cpu: torch.Tensor,
        valid_cpu: torch.Tensor,
        lengths_cpu: torch.Tensor,
        contexts: Dict[torch.device, DeviceContext],
        record_transfer: Callable[[DeviceContext, GPUBatchRecord], None],
    ) -> "MultiDeviceBatch":
        if not contexts:
            raise RuntimeError("No CUDA contexts available for multi-device batch")
        total_rows = int(tokens_cpu.shape[0])
        if total_rows == 0:
            return cls({})
        shards: Dict[torch.device, GPUBatchRecord] = {}
        devices = list(contexts.keys())
        num_devices = len(devices)
        rows_per_device = [total_rows // num_devices] * num_devices
        remainder = total_rows % num_devices
        for idx in range(remainder):
            rows_per_device[idx] += 1
        start = 0
        for device, rows in zip(devices, rows_per_device):
            if rows == 0:
                continue
            end = start + rows
            shard_tokens = tokens_cpu[start:end].clone()
            shard_valid = valid_cpu[start:end].clone()
            shard_lengths = lengths_cpu[start:end].clone()
            ctx = contexts[device]
            record = GPUBatchRecord.from_cpu(
                shard_tokens, shard_valid, shard_lengths, device, ctx=ctx
            )
            record_transfer(ctx, record)
            shards[device] = record
            start = end
        return cls(shards)

class GPUBPETokenizer:
    """Runtime tokenizer compatible with Hugging Face `tokenizers.Tokenizer`."""

    def __init__(
        self,
        tokenizer: "_HFTokenizer",
        *,
        config: dict[str, object],
        vocab: dict[str, int],
        merges: list[str],
        artifact_path: str | None = None,
    ) -> None:
        if tokenizer is None:
            raise RuntimeError(
                "The `tokenizers` library is required to construct GPUBPETokenizer"
            )
        self._tokenizer = tokenizer
        # Store deep copies to avoid accidental mutation from callers.
        self.config = json.loads(json.dumps(config))
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.artifact_path = artifact_path

    @staticmethod
    def _require_tokenizers() -> None:
        if _HFTokenizer is None:
            raise RuntimeError(
                "The `tokenizers` library is required to use GPUBPETokenizer"
            )

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "GPUBPETokenizer":
        """Instantiate a tokenizer directly from an in-memory config."""

        cls._require_tokenizers()
        tokenizer = _HFTokenizer.from_str(json.dumps(config))  # type: ignore[union-attr]
        model_cfg = dict(config.get("model", {}))
        vocab = dict(model_cfg.get("vocab", {}))
        merges = list(model_cfg.get("merges", []))
        return cls(
            tokenizer,
            config=config,
            vocab=vocab,
            merges=merges,
        )

    @classmethod
    def from_file(cls, path: str) -> "GPUBPETokenizer":
        """Load tokenizer artifacts saved via :class:`GPUBPETrainer`."""

        cls._require_tokenizers()
        tokenizer = _HFTokenizer.from_file(path)  # type: ignore[union-attr]
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        model_cfg = dict(config.get("model", {}))
        vocab = dict(model_cfg.get("vocab", {}))
        merges = list(model_cfg.get("merges", []))
        return cls(
            tokenizer,
            config=config,
            vocab=vocab,
            merges=merges,
            artifact_path=os.fspath(path),
        )

    def encode(self, *args, **kwargs):
        """Encode text using the wrapped Hugging Face tokenizer."""

        self._require_tokenizers()
        return self._tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token ids back into text."""

        self._require_tokenizers()
        return self._tokenizer.decode(*args, **kwargs)

    @property
    def tokenizer(self) -> "_HFTokenizer":
        """Expose the underlying Hugging Face tokenizer instance."""

        return self._tokenizer

    def __getattr__(self, name: str):  # pragma: no cover - passthrough to HF tokenizer
        return getattr(self._tokenizer, name)


class GPUBPETrainer:
    """Train byte-pair encodings on the GPU."""

    def __init__(
        self,
        base_vocab: int = 256,
        merges: int = 50_000,
        device: str | None = None,
        devices: Optional[Sequence[str]] = None,
        autoscaler: Optional[AutoScaler] = None,
        sync_every: int = 1,
        warm_start_merges: Optional[Sequence[tuple[int, int]]] = None,
        freeze_warm_start: bool = False,
    ) -> None:
        self.base_vocab = base_vocab
        self.target_merges = merges
        resolved_devices: list[torch.device] = []
        if devices is not None:
            if not devices:
                raise ValueError("devices must be a non-empty sequence when provided")
            resolved_devices = [torch.device(d) for d in devices]
        if device is not None:
            primary = torch.device(device)
            if resolved_devices and primary not in resolved_devices:
                raise ValueError("primary device must be included in devices sequence")
            if not resolved_devices:
                resolved_devices = [primary]
        if not resolved_devices:
            default = "cuda" if torch.cuda.is_available() else "cpu"
            resolved_devices = [torch.device(default)]
        self.devices: list[torch.device] = resolved_devices
        self.device = str(self.devices[0])
        self.vocab_size = base_vocab
        self.merges: List[Tuple[int, int]] = []
        self._seed_warm_start_merges: list[tuple[int, int]] = (
            list(warm_start_merges) if warm_start_merges is not None else []
        )
        self.freeze_warm_start = freeze_warm_start
        self._warm_start_plan: Optional[dict[str, object]] = (
            {
                "merges": list(self._seed_warm_start_merges),
                "counts": None,
                "source": "constructor" if self._seed_warm_start_merges else None,
            }
            if self._seed_warm_start_merges
            else None
        )
        self._warm_start_applied: bool = False
        self._frozen_pair_keys: set[int] = set()
        self.autoscaler = autoscaler or AutoScaler()
        self.sync_every = max(sync_every, 1)
        metrics_flag = os.getenv("SUPERTOKEN_ENABLE_TRAINER_METRICS", "")

        def _parse_bool(raw: str) -> bool:
            lowered = raw.strip().lower()
            return lowered in {"1", "true", "yes", "on"}

        def _parse_float_env(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        def _parse_int_env(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        metrics_enabled = _parse_bool(metrics_flag)
        metrics_alpha = _parse_float_env("SUPERTOKEN_TRAINER_METRICS_ALPHA", 0.2)
        metrics_window = max(1, _parse_int_env("SUPERTOKEN_TRAINER_METRICS_WINDOW", 16))
        self._metrics = TrainerMetricsEWMA(
            alpha=metrics_alpha,
            window_size=metrics_window,
            enabled=metrics_enabled,
        )
        self._metrics_iteration_summary: dict[str, float] | None = None
        # Host↔device transfer accounting, populated during ``fit``.
        self.bytes_h2d: int = 0
        self.bytes_d2h: int = 0
        self.h2d_events: int = 0
        self.d2h_events: int = 0
        self.merge_transfer_log: list[dict[str, object]] = []
        self.sync_intervals: list[dict[str, object]] = []
        self._interval_merges: int = 0
        self._active_batch_size: int | None = None
        self._device_contexts: Dict[torch.device, DeviceContext] = {}
        self._transfer_stage_totals: dict[str, dict[str, int]] = {}
        self._enable_histogram_cache: bool = True
        self._cached_pair_keys: torch.Tensor = torch.empty((0,), dtype=torch.long)
        self._cached_pair_counts: torch.Tensor = torch.empty((0,), dtype=torch.int64)
        self._hist_cache_valid: bool = False
        self._force_recount: bool = True
        self._top_pairs_limit: int = 512
        self._top_pairs_heap: list[tuple[int, int]] = []
        self._top_pairs_index: dict[int, int] = {}
        self._pair_count_lookup: dict[int, int] = {}
        self._cached_pair_keys_per_device: dict[torch.device, torch.Tensor] = {}
        self._cached_pair_counts_per_device: dict[torch.device, torch.Tensor] = {}
        self._cpu_fallback_batches: int = 0
        self._last_cpu_fallback_ratio: float = 0.0
        self._merge_step: int = 0
        self._overlap_enabled: bool = True

    # ------------------------------------------------------------------
    # Transfer accounting helpers
    @property
    def metrics(self) -> TrainerMetricsEWMA:
        """Expose the EWMA tracker for external consumers (e.g. CLI tooling)."""

        return self._metrics

    def _reset_transfer_counters(self) -> None:
        self.bytes_h2d = 0
        self.bytes_d2h = 0
        self.h2d_events = 0
        self.d2h_events = 0
        self.merge_transfer_log = []
        self.sync_intervals = []
        self._interval_merges = 0
        self._cpu_fallback_batches = 0
        self._last_cpu_fallback_ratio = 0.0
        self._transfer_stage_totals = {}
        for ctx in self._device_contexts.values():
            ctx.bytes_h2d = 0
            ctx.bytes_d2h = 0
            ctx.h2d_events = 0
            ctx.d2h_events = 0
            ctx.reset_activity()
            ctx.stage_transfers = {}
            ctx.memory_snapshots = []
            ctx.utilization_samples = []

    def _record_h2d(
        self,
        *tensors: torch.Tensor,
        device: torch.device | None = None,
        stage: str = "general",
    ) -> None:
        if not tensors:
            return
        metrics = self._metrics
        start = time.perf_counter() if metrics.enabled else None
        total = sum(int(t.nbytes) for t in tensors)
        self.bytes_h2d += total
        self.h2d_events += 1
        stage_totals = self._transfer_stage_totals.setdefault(
            stage or "general", {"h2d": 0, "d2h": 0}
        )
        stage_totals["h2d"] += total
        if device is not None:
            ctx = self._device_contexts.get(device)
            if ctx is not None:
                ctx.bytes_h2d += total
                ctx.h2d_events += 1
                ctx.log_transfer(stage, "h2d", total)
        if start is not None:
            duration = time.perf_counter() - start
            stage_kind = metrics.record_stage("h2d", duration)
            if self._metrics_iteration_summary is not None:
                self._accumulate_iteration_stage(
                    self._metrics_iteration_summary,
                    "h2d",
                    duration,
                    stage_kind,
                )

    def _record_d2h(
        self,
        *tensors: torch.Tensor,
        device: torch.device | None = None,
        stage: str = "general",
    ) -> int:
        if not tensors:
            return 0
        metrics = self._metrics
        start = time.perf_counter() if metrics.enabled else None
        total = sum(int(t.nbytes) for t in tensors)
        self.bytes_d2h += total
        self.d2h_events += 1
        stage_totals = self._transfer_stage_totals.setdefault(
            stage or "general", {"h2d": 0, "d2h": 0}
        )
        stage_totals["d2h"] += total
        if device is not None:
            ctx = self._device_contexts.get(device)
            if ctx is not None:
                ctx.bytes_d2h += total
                ctx.d2h_events += 1
                ctx.log_transfer(stage, "d2h", total)
        if start is not None:
            duration = time.perf_counter() - start
            stage_kind = metrics.record_stage("d2h", duration)
            if self._metrics_iteration_summary is not None:
                self._accumulate_iteration_stage(
                    self._metrics_iteration_summary,
                    "d2h",
                    duration,
                    stage_kind,
                )
        return total

    def _accumulate_iteration_stage(
        self,
        summary: dict[str, object],
        stage: str,
        duration_s: float,
        stage_kind: str,
    ) -> None:
        key = f"{stage}_s"
        summary[key] = float(summary.get(key, 0.0)) + float(duration_s)
        if stage_kind == "copy":
            summary["copy_s"] = float(summary.get("copy_s", 0.0)) + float(duration_s)
        elif stage_kind == "compute":
            summary["compute_s"] = float(summary.get("compute_s", 0.0)) + float(duration_s)
        overlap_flag = bool(summary.get("overlap", self._overlap_enabled))
        compute_total = float(summary.get("compute_s", 0.0))
        copy_total = float(summary.get("copy_s", 0.0))
        summary["token_time_s"] = (
            max(compute_total, copy_total)
            if overlap_flag
            else compute_total + copy_total
        )

    # ------------------------------------------------------------------
    # Batch serialization helpers
    def _unpack_cpu_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
        | Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            list[torch.Tensor],
            Optional[tuple[torch.Tensor, torch.Tensor]],
        ]
        | CPUFallbackBatch,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if isinstance(batch, CPUFallbackBatch):
            return (
                batch.tokens,
                batch.valid,
                batch.lengths,
                list(batch.spans),
                batch.pair_keys,
                batch.pair_counts,
            )
        tokens_cpu = batch[0]
        valid_cpu = batch[1]
        lengths_cpu = batch[2]
        spans = batch[3] if len(batch) > 3 else []
        if not isinstance(spans, list):
            spans = list(spans)
        pair_keys: Optional[torch.Tensor] = None
        pair_counts: Optional[torch.Tensor] = None
        if len(batch) > 4 and batch[4] is not None:
            pair_keys, pair_counts = batch[4]
        return tokens_cpu, valid_cpu, lengths_cpu, spans, pair_keys, pair_counts

    def _pack_cpu_batch(
        self,
        tokens_cpu: torch.Tensor,
        valid_cpu: torch.Tensor,
        lengths_cpu: torch.Tensor,
        spans: list[torch.Tensor] | None = None,
        pair_keys: Optional[torch.Tensor] = None,
        pair_counts: Optional[torch.Tensor] = None,
        *,
        as_fallback: bool = False,
        workspaces: Optional[FastPathWorkspaces] = None,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            list[torch.Tensor],
            Optional[tuple[torch.Tensor, torch.Tensor]],
        ]
        | CPUFallbackBatch
    ):
        pair_tuple: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        if pair_keys is not None and pair_counts is not None:
            pair_tuple = (pair_keys, pair_counts)
        if as_fallback:
            return CPUFallbackBatch(
                tokens=tokens_cpu,
                valid=valid_cpu,
                lengths=lengths_cpu,
                spans=[] if spans is None else list(spans),
                pair_keys=None if pair_keys is None else pair_keys,
                pair_counts=None if pair_counts is None else pair_counts,
                workspaces=workspaces or FastPathWorkspaces(),
            )
        return (
            tokens_cpu,
            valid_cpu,
            lengths_cpu,
            [] if spans is None else spans,
            pair_tuple,
        )

    def _extract_sequences_from_cpu_batches(
        self,
        cpu_batches: Iterable[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
            | Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                list[torch.Tensor],
                Optional[tuple[torch.Tensor, torch.Tensor]],
            ]
            | CPUFallbackBatch,
        ],
    ) -> list[list[int]]:
        sequences: list[list[int]] = []
        for batch in cpu_batches:
            tokens_cpu, _valid_cpu, lengths_cpu, _spans, _pair_keys, _pair_counts = (
                self._unpack_cpu_batch(batch)
            )
            rows = int(tokens_cpu.shape[0])
            for row in range(rows):
                length = int(lengths_cpu[row].item())
                if length <= 0:
                    sequences.append([])
                    continue
                seq = tokens_cpu[row, :length].to(torch.long).tolist()
                sequences.append(seq)
        return sequences

    def _extract_sequences_from_gpu_records(
        self, records: Iterable[MultiDeviceBatch]
    ) -> list[list[int]]:
        sequences: list[list[int]] = []
        for record in records:
            sequences.extend(record.sequences())
        return sequences

    def _iter_cpu_batches_from_sequences(
        self, sequences: list[list[int]], batch_size: int, pin: bool
    ) -> Iterator[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            list[torch.Tensor],
            Optional[tuple[torch.Tensor, torch.Tensor]],
        ]
    ]:
        if batch_size <= 0:
            return
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            if not chunk:
                continue
            rows = len(chunk)
            max_len = max((len(seq) for seq in chunk), default=0)
            width = max(1, max_len)
            tokens = torch.full((rows, width), -1, dtype=torch.int32)
            valid = torch.zeros((rows, width), dtype=torch.uint8)
            length_dtype = length_storage_dtype(width)
            lengths = torch.zeros((rows,), dtype=length_dtype)
            for row, seq in enumerate(chunk):
                if not seq:
                    continue
                L = len(seq)
                tokens[row, :L] = torch.tensor(seq, dtype=torch.int32)
                valid[row, :L] = 1
                lengths[row] = L
            if pin:
                tokens = tokens.pin_memory()
                valid = valid.pin_memory()
            yield tokens, valid, lengths, [], None

    def _collect_sequences_from_batches(
        self,
        batches_iter: Iterable[
            Union[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]],
                MultiDeviceBatch,
                CPUFallbackBatch,
            ]
        ],
    ) -> list[list[int]]:
        sequences: list[list[int]] = []
        cpu_batches: list[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
            | Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                list[torch.Tensor],
                Optional[tuple[torch.Tensor, torch.Tensor]],
            ]
            | CPUFallbackBatch
        ] = []
        gpu_records: list[MultiDeviceBatch] = []
        for batch in batches_iter:
            if isinstance(batch, MultiDeviceBatch):
                gpu_records.append(batch)
            else:
                cpu_batches.append(batch)
        if gpu_records:
            sequences.extend(self._extract_sequences_from_gpu_records(gpu_records))
        if cpu_batches:
            sequences.extend(self._extract_sequences_from_cpu_batches(cpu_batches))
        return sequences

    def _build_gpu_batches_from_sequences(
        self,
        sequences: list[list[int]],
        batch_size: int,
        device_contexts: Dict[torch.device, DeviceContext],
        record_new_batch: Callable[[DeviceContext, GPUBatchRecord], None],
    ) -> tuple[list[MultiDeviceBatch], list[CPUFallbackBatch]]:
        new_records: list[MultiDeviceBatch] = []
        cpu_fallbacks: list[CPUFallbackBatch] = []
        if batch_size <= 0 or not sequences or not device_contexts:
            return new_records, cpu_fallbacks
        for cpu_batch in self._iter_cpu_batches_from_sequences(sequences, batch_size, pin=True):
            (
                tokens_cpu,
                valid_cpu,
                lengths_cpu,
                spans,
                pair_keys,
                pair_counts,
            ) = self._unpack_cpu_batch(cpu_batch)
            multi_batch = MultiDeviceBatch.from_cpu_batch(
                tokens_cpu, valid_cpu, lengths_cpu, device_contexts, record_new_batch
            )
            B, L = tokens_cpu.shape
            width = max(L - 1, 0)
            if should_route_to_cpu(int(B), int(width)):
                fallback = self._pack_cpu_batch(
                    tokens_cpu.clone().to("cpu"),
                    valid_cpu.clone().to("cpu"),
                    lengths_cpu.clone().to("cpu"),
                    spans=list(spans),
                    pair_keys=None if pair_keys is None else pair_keys.clone(),
                    pair_counts=None if pair_counts is None else pair_counts.clone(),
                    as_fallback=True,
                )
                if isinstance(fallback, CPUFallbackBatch):
                    cpu_fallbacks.append(fallback)
                continue
            new_records.append(multi_batch)
        self._mark_active_batches(new_records, device_contexts)
        return new_records, cpu_fallbacks

    def _mark_active_batches(
        self,
        batches_to_mark: Iterable[MultiDeviceBatch],
        device_contexts: Dict[torch.device, DeviceContext],
    ) -> None:
        for ctx in device_contexts.values():
            ctx.reset_activity()
        for batch in batches_to_mark:
            for dev, record in batch.iter_shards():
                ctx = device_contexts.get(dev)
                if ctx is not None:
                    ctx.active_batches.append(record)

    def _serialize_batches(
        self,
        current_batches: Optional[
            Iterable[
                Union[
                    MultiDeviceBatch,
                    CPUFallbackBatch,
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]],
                    Tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        list[torch.Tensor],
                        Optional[tuple[torch.Tensor, torch.Tensor]],
                    ],
                ]
            ]
        ],
    ) -> Optional[dict[str, object]]:
        if current_batches is None:
            return None
        has_gpu = False
        sequences: list[list[int]] = []
        cpu_batches: list[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
            | Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                list[torch.Tensor],
                Optional[tuple[torch.Tensor, torch.Tensor]],
            ]
            | CPUFallbackBatch
        ] = []
        for batch in current_batches:
            if isinstance(batch, MultiDeviceBatch):
                has_gpu = True
                sequences.extend(self._extract_sequences_from_gpu_records([batch]))
            else:
                cpu_batches.append(batch)
        if cpu_batches:
            sequences.extend(self._extract_sequences_from_cpu_batches(cpu_batches))
        return {
            "sequences": [[int(token) for token in seq] for seq in sequences],
            "has_gpu": has_gpu,
            "active_batch_size": self._active_batch_size,
        }

    def _deserialize_batches(
        self,
        serialized: Optional[dict[str, object]],
        *,
        device_contexts: Optional[Dict[torch.device, DeviceContext]] = None,
        use_cuda: bool = False,
    ) -> tuple[
        Optional[
            list[
                Union[
                    MultiDeviceBatch,
                    CPUFallbackBatch,
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]],
                    Tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        list[torch.Tensor],
                        Optional[tuple[torch.Tensor, torch.Tensor]],
                    ],
                ]
            ]
        ],
        Optional[list[MultiDeviceBatch]],
    ]:
        if not serialized:
            return None, None
        sequences_raw = serialized.get("sequences", [])
        sequences: list[list[int]] = []
        if isinstance(sequences_raw, list):
            for seq in sequences_raw:
                if isinstance(seq, list):
                    sequences.append([int(tok) for tok in seq])
        active_batch_size = int(serialized.get("active_batch_size") or 0)
        if active_batch_size <= 0 and sequences:
            active_batch_size = len(sequences)
        if active_batch_size <= 0:
            active_batch_size = self._active_batch_size or 0
        has_gpu = bool(serialized.get("has_gpu")) and use_cuda and device_contexts
        if has_gpu and device_contexts is not None:
            new_records, cpu_fallbacks = self._build_gpu_batches_from_sequences(
                sequences,
                active_batch_size,
                device_contexts,
                lambda _ctx, _record: None,
            )
            combined: list[
                Union[
                    MultiDeviceBatch,
                    CPUFallbackBatch,
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]],
                    Tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        list[torch.Tensor],
                        Optional[tuple[torch.Tensor, torch.Tensor]],
                    ],
                ]
            ] = []
            if cpu_fallbacks:
                combined.extend(cpu_fallbacks)
            combined.extend(new_records)
            return combined, new_records
        cpu_batches = list(
            self._iter_cpu_batches_from_sequences(
                sequences,
                active_batch_size if active_batch_size > 0 else len(sequences) or 0,
                pin=False,
            )
        )
        return cpu_batches if cpu_batches else None, None

    # ------------------------------------------------------------------
    # State serialization
    def state_dict(
        self,
        *,
        current_batches: Optional[
            Iterable[
                Union[
                    MultiDeviceBatch,
                    CPUFallbackBatch,
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]],
                    Tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        list[torch.Tensor],
                        Optional[tuple[torch.Tensor, torch.Tensor]],
                    ],
                ]
            ]
        ] = None,
        include_batches: bool = True,
        stage_timings: Optional[dict[str, StageTiming]] = None,
        stage_event_log: Optional[list[dict[str, object]]] = None,
        host_sync_events: Optional[list[dict[str, object]]] = None,
        device_snapshot_log: Optional[list[dict[str, object]]] = None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "base_vocab": int(self.base_vocab),
            "target_merges": int(self.target_merges),
            "vocab_size": int(self.vocab_size),
            "merges": [list(map(int, pair)) for pair in self.merges],
            "merge_step": int(self._merge_step),
            "warm_start_plan": (
                {k: v for k, v in self._warm_start_plan.items()}
                if self._warm_start_plan is not None
                else None
            ),
            "warm_start_applied": bool(self._warm_start_applied),
            "freeze_warm_start": bool(self.freeze_warm_start),
            "seed_warm_start_merges": [
                list(map(int, pair)) for pair in self._seed_warm_start_merges
            ],
            "frozen_pair_keys": [int(key) for key in sorted(self._frozen_pair_keys)],
            "histogram_cache": {
                "enable": bool(self._enable_histogram_cache),
                "valid": bool(self._hist_cache_valid),
                "force_recount": bool(self._force_recount),
            },
            "cached_histogram_sizes": {
                "keys": int(self._cached_pair_keys.numel()),
                "counts": int(self._cached_pair_counts.numel()),
            },
            "bytes_h2d": int(self.bytes_h2d),
            "bytes_d2h": int(self.bytes_d2h),
            "h2d_events": int(self.h2d_events),
            "d2h_events": int(self.d2h_events),
            "merge_transfer_log": [dict(entry) for entry in self.merge_transfer_log],
            "sync_intervals": [dict(entry) for entry in self.sync_intervals],
            "transfer_stage_totals": {
                key: {"h2d": int(val.get("h2d", 0)), "d2h": int(val.get("d2h", 0))}
                for key, val in self._transfer_stage_totals.items()
            },
            "cpu_fallback_batches": int(self._cpu_fallback_batches),
            "last_cpu_fallback_ratio": float(self._last_cpu_fallback_ratio),
            "active_batch_size": (
                int(self._active_batch_size)
                if self._active_batch_size is not None
                else None
            ),
            "autoscaler": self.autoscaler.state_dict(),
            "autoscaler_metrics": self.autoscaler.snapshot_metrics(),
        }
        if include_batches and current_batches is not None:
            serialized_batches = self._serialize_batches(current_batches)
            metadata["batches"] = serialized_batches
        telemetry_progress: dict[str, object] = {}
        if stage_timings is not None:
            telemetry_progress["stage_timings"] = {
                name: list(timing.samples) for name, timing in stage_timings.items()
            }
        if stage_event_log is not None:
            telemetry_progress["stage_event_log"] = [dict(evt) for evt in stage_event_log]
        if host_sync_events is not None:
            telemetry_progress["host_sync_events"] = [dict(evt) for evt in host_sync_events]
        if device_snapshot_log is not None:
            telemetry_progress["device_snapshot_log"] = [dict(evt) for evt in device_snapshot_log]
        if telemetry_progress:
            metadata["telemetry_progress"] = telemetry_progress
        tensors: dict[str, torch.Tensor] = {}
        tensors["cached_pair_keys"] = self._cached_pair_keys.detach().clone().cpu()
        tensors["cached_pair_counts"] = self._cached_pair_counts.detach().clone().cpu()
        return {"metadata": metadata, "tensors": tensors}

    def load_state_dict(
        self,
        state_dict: dict[str, object],
        *,
        device_contexts: Optional[Dict[torch.device, DeviceContext]] = None,
        use_cuda: bool = False,
    ) -> dict[str, object]:
        if not state_dict:
            return {
                "current_batches": None,
                "gpu_batches": None,
                "scale_state": self.autoscaler.state,
                "step": len(self.merges),
                "stage_timings": None,
                "stage_event_log": [],
                "host_sync_events": [],
                "device_snapshot_log": [],
            }
        metadata = dict(state_dict.get("metadata", {}))
        tensors = state_dict.get("tensors", {})
        self.base_vocab = int(metadata.get("base_vocab", self.base_vocab))
        self.target_merges = int(metadata.get("target_merges", self.target_merges))
        merges_raw = metadata.get("merges", [])
        self.merges = [tuple(map(int, pair)) for pair in merges_raw]
        self.vocab_size = int(metadata.get("vocab_size", self.vocab_size))
        self._merge_step = int(metadata.get("merge_step", len(self.merges)))
        warm_plan = metadata.get("warm_start_plan")
        self._warm_start_plan = warm_plan
        self._warm_start_applied = bool(metadata.get("warm_start_applied", self._warm_start_applied))
        self.freeze_warm_start = bool(metadata.get("freeze_warm_start", self.freeze_warm_start))
        self._seed_warm_start_merges = [
            tuple(map(int, pair))
            for pair in metadata.get(
                "seed_warm_start_merges", list(self._seed_warm_start_merges)
            )
        ]
        self._frozen_pair_keys = set(
            int(key) for key in metadata.get("frozen_pair_keys", list(self._frozen_pair_keys))
        )
        hist_meta = metadata.get("histogram_cache", {})
        self._enable_histogram_cache = bool(
            hist_meta.get("enable", self._enable_histogram_cache)
        )
        self._hist_cache_valid = bool(hist_meta.get("valid", self._hist_cache_valid))
        self._force_recount = bool(hist_meta.get("force_recount", self._force_recount))
        cached_keys = tensors.get("cached_pair_keys")
        cached_counts = tensors.get("cached_pair_counts")
        if cached_keys is not None and cached_counts is not None:
            self._cached_pair_keys = cached_keys.detach().clone().to(torch.long)
            self._cached_pair_counts = cached_counts.detach().clone().to(torch.int64)
        else:
            self._cached_pair_keys = torch.empty((0,), dtype=torch.long)
            self._cached_pair_counts = torch.empty((0,), dtype=torch.int64)
            self._hist_cache_valid = False
        if self._hist_cache_valid:
            self._refresh_top_pairs_from_cache()
        else:
            self._reset_top_pairs()
        if use_cuda:
            self._materialize_histogram_cache_on_devices(device_contexts)
        self.bytes_h2d = int(metadata.get("bytes_h2d", self.bytes_h2d))
        self.bytes_d2h = int(metadata.get("bytes_d2h", self.bytes_d2h))
        self.h2d_events = int(metadata.get("h2d_events", self.h2d_events))
        self.d2h_events = int(metadata.get("d2h_events", self.d2h_events))
        self.merge_transfer_log = [dict(entry) for entry in metadata.get("merge_transfer_log", self.merge_transfer_log)]
        self.sync_intervals = [dict(entry) for entry in metadata.get("sync_intervals", self.sync_intervals)]
        stage_totals_raw = metadata.get("transfer_stage_totals", {})
        self._transfer_stage_totals = {
            key: {"h2d": int(val.get("h2d", 0)), "d2h": int(val.get("d2h", 0))}
            for key, val in stage_totals_raw.items()
        }
        self._cpu_fallback_batches = int(
            metadata.get("cpu_fallback_batches", self._cpu_fallback_batches)
        )
        self._last_cpu_fallback_ratio = float(
            metadata.get("last_cpu_fallback_ratio", self._last_cpu_fallback_ratio)
        )
        active_batch_size = metadata.get("active_batch_size")
        self._active_batch_size = int(active_batch_size) if active_batch_size is not None else None
        autoscaler_meta = metadata.get("autoscaler") or {}
        scale_state = self.autoscaler.state
        if autoscaler_meta:
            if "state" in autoscaler_meta or "h2d_mb" in autoscaler_meta:
                self.autoscaler.load_state_dict(autoscaler_meta)
                scale_state = self.autoscaler.state
            else:
                # Backwards compatibility for checkpoints created before autoscaler
                # serialization helpers were introduced.
                self.autoscaler.device = autoscaler_meta.get("device", self.autoscaler.device)
                self.autoscaler.tu = float(
                    autoscaler_meta.get("target_util", self.autoscaler.tu)
                )
                window_size = int(
                    autoscaler_meta.get("window_size", self.autoscaler._window_size)
                )
                self.autoscaler._window_size = window_size
                step_times = autoscaler_meta.get("step_times", [])
                vram_util = autoscaler_meta.get("vram_utilization", [])
                self.autoscaler._step_times = deque(step_times, maxlen=window_size)
                self.autoscaler._vram_fracs = deque(vram_util, maxlen=window_size)
                state_payload = autoscaler_meta.get("state")
                if state_payload is not None:
                    scale_state = ScaleState(
                        batch_size=int(state_payload.get("batch_size", 0)),
                        cpu_workers=int(state_payload.get("cpu_workers", 0)),
                        h2d_mb=int(state_payload.get("h2d_mb", 0)),
                        cpu_fallback_rate=float(
                            state_payload.get("cpu_fallback_rate", 0.0)
                        ),
                    )
                    self.autoscaler.state = scale_state
                else:
                    self.autoscaler.state = None
        batches_payload = metadata.get("batches")
        current_batches, gpu_batches = self._deserialize_batches(
            batches_payload,
            device_contexts=device_contexts,
            use_cuda=use_cuda,
        )
        telemetry_progress = metadata.get("telemetry_progress") or {}
        stage_timings_data = telemetry_progress.get("stage_timings")
        stage_event_log = telemetry_progress.get("stage_event_log", [])
        host_sync_events = telemetry_progress.get("host_sync_events", [])
        device_snapshot_log = telemetry_progress.get("device_snapshot_log", [])
        return {
            "current_batches": current_batches,
            "gpu_batches": gpu_batches,
            "scale_state": scale_state,
            "step": self._merge_step,
            "stage_timings": stage_timings_data,
            "stage_event_log": stage_event_log,
            "host_sync_events": host_sync_events,
            "device_snapshot_log": device_snapshot_log,
        }

    def save_checkpoint(
        self,
        path: str,
        include_batches: bool = True,
        *,
        current_batches: Optional[
            Iterable[
                Union[
                    MultiDeviceBatch,
                    CPUFallbackBatch,
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]],
                    Tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        list[torch.Tensor],
                        Optional[tuple[torch.Tensor, torch.Tensor]],
                    ],
                ]
            ]
        ] = None,
        stage_timings: Optional[dict[str, StageTiming]] = None,
        stage_event_log: Optional[list[dict[str, object]]] = None,
        host_sync_events: Optional[list[dict[str, object]]] = None,
        device_snapshot_log: Optional[list[dict[str, object]]] = None,
    ) -> dict[str, object]:
        state = self.state_dict(
            current_batches=current_batches,
            include_batches=include_batches,
            stage_timings=stage_timings,
            stage_event_log=stage_event_log,
            host_sync_events=host_sync_events,
            device_snapshot_log=device_snapshot_log,
        )
        os.makedirs(path, exist_ok=True)
        meta_path = os.path.join(path, "state.json")
        tensor_path = os.path.join(path, "tensors.pt")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(state["metadata"], f, indent=2, sort_keys=True)
        tensor_payload = {
            name: tensor.detach().clone().cpu() for name, tensor in state["tensors"].items()
        }
        torch.save(tensor_payload, tensor_path)
        return state

    def load_checkpoint(self, path: str) -> dict[str, object]:
        meta_path = os.path.join(path, "state.json")
        tensor_path = os.path.join(path, "tensors.pt")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        tensors: dict[str, torch.Tensor] = {}
        if os.path.exists(tensor_path):
            loaded = torch.load(tensor_path, map_location="cpu")
            if isinstance(loaded, dict):
                tensors = loaded
        return {"metadata": metadata, "tensors": tensors}

    def _record_merge_snapshot(self, merge_idx: int) -> None:
        self.merge_transfer_log.append(
            {
                "merge": merge_idx,
                "bytes_h2d_cumulative": self.bytes_h2d,
                "bytes_d2h_cumulative": self.bytes_d2h,
            }
        )

    def _close_sync_interval(self, merge_idx: int, copied_bytes: int) -> None:
        merges_in_interval = self._interval_merges
        avg = copied_bytes / merges_in_interval if merges_in_interval else 0.0
        self.sync_intervals.append(
            {
                "completed_merge": merge_idx,
                "merges_in_interval": merges_in_interval,
                "bytes_d2h": copied_bytes,
                "avg_bytes_per_merge": avg,
            }
        )
        self._interval_merges = 0

    def _reset_histogram_cache(self) -> None:
        self._cached_pair_keys = torch.empty((0,), dtype=torch.long)
        self._cached_pair_counts = torch.empty((0,), dtype=torch.int64)
        self._hist_cache_valid = False
        self._force_recount = True
        self._reset_top_pairs()
        self._cached_pair_keys_per_device.clear()
        self._cached_pair_counts_per_device.clear()

    def _invalidate_hist_cache(self) -> None:
        self._hist_cache_valid = False
        self._force_recount = True
        self._reset_top_pairs()
        self._cached_pair_keys_per_device.clear()
        self._cached_pair_counts_per_device.clear()

    def _materialize_histogram_cache_on_devices(
        self,
        device_contexts: Optional[Dict[torch.device, "DeviceContext"]],
        *,
        source: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        if (
            device_contexts is None
            or not device_contexts
            or not self._enable_histogram_cache
            or not self._hist_cache_valid
        ):
            self._cached_pair_keys_per_device.clear()
            self._cached_pair_counts_per_device.clear()
            return

        if self._cached_pair_keys.numel() == 0:
            self._cached_pair_keys_per_device.clear()
            self._cached_pair_counts_per_device.clear()
            return

        if not torch.cuda.is_available():
            return

        primary: Optional[torch.device] = None
        for dev in device_contexts:
            if dev.type == "cuda":
                primary = dev
                break

        if primary is None:
            return

        cached_keys = self._cached_pair_keys_per_device.get(primary)
        cached_counts = self._cached_pair_counts_per_device.get(primary)
        if cached_keys is None or cached_counts is None or cached_keys.device != primary:
            if source is not None:
                cached_keys = source[0].to(device=primary, dtype=torch.long, non_blocking=True)
                cached_counts = source[1].to(
                    device=primary, dtype=torch.int64, non_blocking=True
                )
            else:
                cached_keys = self._cached_pair_keys.to(
                    device=primary, dtype=torch.long, non_blocking=True
                )
                cached_counts = self._cached_pair_counts.to(
                    device=primary, dtype=torch.int64, non_blocking=True
                )
            self._cached_pair_keys_per_device[primary] = cached_keys
            self._cached_pair_counts_per_device[primary] = cached_counts

        for dev, ctx in device_contexts.items():
            if dev == primary or dev.type != "cuda":
                continue

            dst_keys = torch.empty_like(cached_keys, device=dev)
            dst_counts = torch.empty_like(cached_counts, device=dev)
            stream = getattr(ctx, "h2d_stream", None)

            peer_keys = peer_copy_tensor(dst_keys, cached_keys, stream=stream)
            peer_counts = peer_copy_tensor(dst_counts, cached_counts, stream=stream)

            if not (peer_keys and peer_counts):
                if stream is not None:
                    with torch.cuda.stream(stream):
                        dst_keys.copy_(cached_keys)
                        dst_counts.copy_(cached_counts)
                else:
                    dst_keys.copy_(cached_keys)
                    dst_counts.copy_(cached_counts)

            self._cached_pair_keys_per_device[dev] = dst_keys
            self._cached_pair_counts_per_device[dev] = dst_counts

    def _expand_recount_spans(self, span_mask: torch.Tensor) -> torch.Tensor:
        if span_mask.numel() == 0:
            return span_mask
        prev = torch.zeros_like(span_mask, dtype=torch.bool)
        if span_mask.shape[-1] > 0:
            prev[:, :-1] = span_mask[:, 1:].to(torch.bool)
        return span_mask.to(torch.bool) | prev

    def _compute_histogram_deltas(
        self,
        span_mask: torch.Tensor,
        pre_lhs: torch.Tensor,
        pre_rhs: torch.Tensor,
        pre_mask: torch.Tensor,
        post_lhs: torch.Tensor,
        post_rhs: torch.Tensor,
        post_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if span_mask.numel() == 0:
            empty_keys = torch.empty((0,), dtype=torch.long)
            empty_counts = torch.empty((0,), dtype=torch.int64)
            return (
                empty_keys,
                empty_counts,
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
            )

        affected = self._expand_recount_spans(span_mask)
        pre_mask_bool = pre_mask.to(torch.bool)
        post_mask_bool = post_mask.to(torch.bool)

        remove_mask = affected & pre_mask_bool
        add_mask = affected & post_mask_bool

        device = torch.device("cpu")

        if remove_mask.any():
            remove_keys = (pre_lhs[remove_mask].to(torch.long) << 32) | pre_rhs[remove_mask].to(torch.long)
            remove_keys = remove_keys.to(device=device)
            remove_counts = torch.ones(
                (remove_keys.numel(),), dtype=torch.int64, device=device
            )
            remove_keys, remove_counts = aggregate_pair_keys(remove_keys, remove_counts)
        else:
            remove_keys = torch.empty((0,), dtype=torch.long, device=device)
            remove_counts = torch.empty((0,), dtype=torch.int64, device=device)

        if add_mask.any():
            add_keys = (post_lhs[add_mask].to(torch.long) << 32) | post_rhs[add_mask].to(torch.long)
            add_keys = add_keys.to(device=device)
            add_counts = torch.ones((add_keys.numel(),), dtype=torch.int64, device=device)
            add_keys, add_counts = aggregate_pair_keys(add_keys, add_counts)
        else:
            add_keys = torch.empty((0,), dtype=torch.long, device=device)
            add_counts = torch.empty((0,), dtype=torch.int64, device=device)

        return remove_keys, remove_counts, add_keys, add_counts

    @staticmethod
    def _merge_histogram(
        base_keys: torch.Tensor,
        base_counts: torch.Tensor,
        delta_keys: torch.Tensor,
        delta_counts: torch.Tensor,
        sign: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if delta_keys.numel() == 0:
            return base_keys, base_counts
        if base_keys.numel() == 0 and sign < 0:
            return base_keys, base_counts
        if base_keys.numel() == 0:
            return delta_keys.clone(), delta_counts.clone()
        signed_delta = delta_counts.clone()
        if sign < 0:
            signed_delta = -signed_delta
        combined_keys = torch.cat([base_keys, delta_keys], dim=0)
        combined_counts = torch.cat([base_counts, signed_delta], dim=0)
        merged_keys, merged_counts = aggregate_pair_keys(combined_keys, combined_counts)
        if merged_keys.numel() == 0:
            return merged_keys, merged_counts
        mask = merged_counts > 0
        if mask.all():
            return merged_keys, merged_counts
        return merged_keys[mask], merged_counts[mask]

    @classmethod
    def _apply_histogram_delta(
        cls,
        base_keys: torch.Tensor,
        base_counts: torch.Tensor,
        remove_keys: torch.Tensor,
        remove_counts: torch.Tensor,
        add_keys: torch.Tensor,
        add_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, counts = cls._merge_histogram(base_keys, base_counts, remove_keys, remove_counts, -1)
        keys, counts = cls._merge_histogram(keys, counts, add_keys, add_counts, 1)
        return keys, counts

    @classmethod
    def precompute_warm_start_plan(
        cls,
        batches: Iterable[
            Tuple[torch.Tensor, torch.Tensor]
            | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            | GPUBatchRecord
        ],
        top_k: int,
        device: Optional[torch.device] = None,
    ) -> dict[str, object]:
        """Select warm-start merges from n-gram histograms.

        Args:
            batches: Iterable of batch-like inputs compatible with
                :func:`compute_ngram_histograms`.
            top_k: Number of bigrams to seed.
            device: Optional device override for histogram computation.

        Returns:
            Dictionary describing the warm-start plan, including the selected
            merges and their counts.
        """

        if top_k is None or top_k <= 0:
            return {
                "merges": [],
                "counts": [],
                "top_k": 0,
                "requested_top_k": int(top_k or 0),
                "order": 2,
                "histogram_size": 0,
                "source": "ngram",
            }

        from .ngram_stats import compute_ngram_histograms

        histograms = compute_ngram_histograms(batches, max_order=2, device=device)
        bigram_keys, bigram_counts = histograms.get(2, (None, None))
        if bigram_keys is None or bigram_counts is None or bigram_keys.numel() == 0:
            return {
                "merges": [],
                "counts": [],
                "top_k": 0,
                "requested_top_k": int(top_k),
                "order": 2,
                "histogram_size": 0,
                "source": "ngram",
            }

        limit = min(int(top_k), int(bigram_keys.numel()))
        if limit <= 0:
            return {
                "merges": [],
                "counts": [],
                "top_k": 0,
                "requested_top_k": int(top_k),
                "order": 2,
                "histogram_size": int(bigram_keys.numel()),
                "source": "ngram",
            }

        values, indices = torch.topk(bigram_counts, k=limit, largest=True, sorted=True)
        seen: set[tuple[int, int]] = set()
        merges: list[tuple[int, int]] = []
        counts: list[int] = []
        for idx, count_val in zip(indices.tolist(), values.tolist()):
            key_val = int(bigram_keys[idx].item())
            a_id = int(key_val >> 32)
            b_id = int(key_val & ((1 << 32) - 1))
            pair = (a_id, b_id)
            if pair in seen:
                continue
            seen.add(pair)
            merges.append(pair)
            counts.append(int(count_val))
            if len(merges) >= limit:
                break

        return {
            "merges": merges,
            "counts": counts,
            "top_k": len(merges),
            "requested_top_k": int(top_k),
            "order": 2,
            "histogram_size": int(bigram_keys.numel()),
            "source": "ngram",
        }

    def _reset_top_pairs(self) -> None:
        self._top_pairs_heap = []
        self._top_pairs_index = {}
        self._pair_count_lookup = {}

    def _cleanup_top_heap(self) -> None:
        while self._top_pairs_heap:
            neg_count, key = self._top_pairs_heap[0]
            lookup = self._top_pairs_index.get(key)
            if lookup is not None and lookup == -neg_count:
                break
            heapq.heappop(self._top_pairs_heap)

    def _enforce_top_limit(self) -> None:
        limit = self._top_pairs_limit
        if limit <= 0:
            return
        while len(self._top_pairs_index) > limit:
            worst_key, _ = min(self._top_pairs_index.items(), key=lambda kv: (kv[1], kv[0]))
            self._top_pairs_index.pop(worst_key, None)
        self._cleanup_top_heap()

    def _insert_top_pair(self, key: int, count: int) -> None:
        if self.freeze_warm_start and key in self._frozen_pair_keys:
            self._top_pairs_index.pop(key, None)
            return
        if self._top_pairs_limit <= 0 or count <= 0:
            if count <= 0:
                self._top_pairs_index.pop(key, None)
            return
        self._top_pairs_index[key] = count
        heapq.heappush(self._top_pairs_heap, (-count, key))
        self._enforce_top_limit()

    def _select_top_candidate(self) -> tuple[Optional[int], int, int]:
        if not self._enable_histogram_cache or self._top_pairs_limit <= 0:
            return None, 0, 0
        best_seen = 0
        while self._top_pairs_heap:
            neg_count, key = self._top_pairs_heap[0]
            lookup = self._top_pairs_index.get(key)
            if lookup is not None:
                best_seen = max(best_seen, lookup)
            if lookup is None or lookup != -neg_count:
                heapq.heappop(self._top_pairs_heap)
                continue
            return key, lookup, best_seen
        return None, 0, best_seen

    def _refresh_top_pairs_from_cache(self) -> None:
        self._reset_top_pairs()
        if not self._enable_histogram_cache:
            return
        keys = self._cached_pair_keys
        counts = self._cached_pair_counts
        if keys.numel() == 0:
            return
        keys_list = keys.cpu().tolist()
        counts_list = counts.cpu().tolist()
        for key_val, count_val in zip(keys_list, counts_list):
            count_int = int(count_val)
            if count_int <= 0:
                continue
            key_int = int(key_val)
            self._pair_count_lookup[key_int] = count_int
            self._insert_top_pair(key_int, count_int)
        self._cleanup_top_heap()

    def _update_global_histogram(
        self,
        remove_keys: torch.Tensor,
        remove_counts: torch.Tensor,
        add_keys: torch.Tensor,
        add_counts: torch.Tensor,
    ) -> None:
        self._cached_pair_keys, self._cached_pair_counts = self._apply_histogram_delta(
            self._cached_pair_keys,
            self._cached_pair_counts,
            remove_keys,
            remove_counts,
            add_keys,
            add_counts,
        )
        if not self._enable_histogram_cache:
            return
        deltas: dict[int, int] = {}
        if remove_keys.numel() > 0:
            for key, count in zip(remove_keys.cpu().tolist(), remove_counts.cpu().tolist()):
                deltas[int(key)] = deltas.get(int(key), 0) - int(count)
        if add_keys.numel() > 0:
            for key, count in zip(add_keys.cpu().tolist(), add_counts.cpu().tolist()):
                deltas[int(key)] = deltas.get(int(key), 0) + int(count)
        for key, delta in deltas.items():
            new_total = self._pair_count_lookup.get(key, 0) + delta
            if new_total <= 0 or (
                self.freeze_warm_start and key in self._frozen_pair_keys
            ):
                self._pair_count_lookup.pop(key, None)
                self._top_pairs_index.pop(key, None)
            else:
                self._pair_count_lookup[key] = new_total
                self._insert_top_pair(key, new_total)
        if deltas:
            self._cleanup_top_heap()

    def _invoke_count_pairs_gpu(
        self,
        batch_iter: Iterable[
            Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], MultiDeviceBatch]
        ],
        impl: Callable[
            [
                Iterable[
                    Union[
                        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        MultiDeviceBatch,
                    ]
                ]
            ],
            tuple[Optional[torch.Tensor], Optional[torch.Tensor], list[MultiDeviceBatch]],
        ],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], list[MultiDeviceBatch]]:
        return impl(batch_iter)

    def _invoke_count_pairs_cpu(
        self,
        batch_iter: Iterable[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
            | Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                list[torch.Tensor],
                Optional[tuple[torch.Tensor, torch.Tensor]],
            ],
        ],
        impl: Callable[
            [
                Iterable[
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                    | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
                    | Tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        list[torch.Tensor],
                        Optional[tuple[torch.Tensor, torch.Tensor]],
                    ]
                ]
            ],
            tuple[
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                list[
                    tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        list[torch.Tensor],
                        Optional[tuple[torch.Tensor, torch.Tensor]],
                    ]
                ],
            ],
        ],
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                list[torch.Tensor],
                Optional[tuple[torch.Tensor, torch.Tensor]],
            ]
        ],
    ]:
        return impl(batch_iter)

    def fit(
        self,
        batches: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        log_every: int = 100,
        profile_streams: bool = False,
        on_batch_size_change: Optional[Callable[[int], None]] = None,
        warm_start_merges: Optional[Sequence[tuple[int, int]]] = None,
        freeze_warm_start: Optional[bool] = None,
        overlap_transfers: bool = True,
        warm_start_plan: Optional[dict[str, object]] = None,
        warm_start_ngrams: Optional[int] = None,
        warm_start_device: Optional[torch.device] = None,
        checkpoint_interval: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        resume_state: Optional[dict[str, object]] = None,
        on_iteration_summary: Optional[Callable[[dict[str, object]], None]] = None,
    ) -> dict[str, object]:
        """Train merges using a pipelined GPU workflow when possible."""

        if checkpoint_interval is not None:
            checkpoint_interval = int(checkpoint_interval)
            if checkpoint_interval <= 0:
                checkpoint_interval = None
        if checkpoint_interval is not None and checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be provided when using checkpoint_interval")

        self._overlap_enabled = bool(overlap_transfers)

        self._reset_transfer_counters()
        self._reset_histogram_cache()
        self._active_batch_size = None
        if freeze_warm_start is not None:
            self.freeze_warm_start = freeze_warm_start
        self._frozen_pair_keys.clear()
        step = 0
        current_batches: Iterable[
            Union[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]],
                Tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    list[torch.Tensor],
                    Optional[tuple[torch.Tensor, torch.Tensor]],
                ],
                MultiDeviceBatch,
            ]
        ] = batches
        gpu_batches: Optional[list[MultiDeviceBatch]] = None
        scale_state = None

        self._merge_step = len(self.merges)
        checkpoint_dir_path = os.fspath(checkpoint_dir) if checkpoint_dir is not None else None

        cpu_device = torch.device("cpu")

        stage_timings: dict[str, StageTiming] = {
            "pair_count": StageTiming("pair_count"),
            "apply_merge": StageTiming("apply_merge"),
            "host_sync": StageTiming("host_sync"),
        }
        stage_event_log: list[dict[str, object]] = []
        host_sync_events: list[dict[str, object]] = []
        device_snapshot_log: list[dict[str, object]] = []
        metrics_tracker = self._metrics
        metrics_tracker.overlap_enabled = self._overlap_enabled
        metrics_tracker.reset()
        metrics_enabled = metrics_tracker.enabled
        iteration_summary: dict[str, object] | None = None
        iteration_summaries: list[dict[str, object]] = []
        self._metrics_iteration_summary = None

        def _record_iteration_summary(summary: dict[str, object]) -> None:
            snapshot = dict(summary)
            iteration_summaries.append(snapshot)
            if on_iteration_summary is not None:
                on_iteration_summary(snapshot)

        def _new_iteration_summary(merge_idx: int, kind: str) -> dict[str, object]:
            return {
                "merge": merge_idx,
                "kind": kind,
                "h2d_s": 0.0,
                "kernel_s": 0.0,
                "d2h_s": 0.0,
                "reduction_s": 0.0,
                "copy_s": 0.0,
                "compute_s": 0.0,
                "overlap": self._overlap_enabled,
                "tokens": 0,
                "leases": 0,
                "token_time_s": 0.0,
            }

        def _tally_tokens_from_batches(
            batches_iter: Iterable[
                Union[
                    MultiDeviceBatch,
                    CPUFallbackBatch,
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        list[torch.Tensor],
                        Optional[tuple[torch.Tensor, torch.Tensor]],
                    ],
                ]
            ]
        ) -> tuple[int, int]:
            total_tokens = 0
            total_sequences = 0
            for batch in batches_iter:
                if isinstance(batch, MultiDeviceBatch):
                    for _dev, shard in batch.iter_shards():
                        total_sequences += int(shard.tokens.shape[0])
                        if shard.host_lengths is not None and not shard.host_dirty:
                            total_tokens += int(
                                shard.host_lengths.to(torch.int64).sum().item()
                            )
                        else:
                            total_tokens += int(shard.lengths.to(torch.int64).sum().item())
                elif isinstance(batch, CPUFallbackBatch):
                    total_sequences += int(batch.tokens.shape[0])
                    total_tokens += int(batch.lengths.to(torch.int64).sum().item())
                else:
                    (
                        tokens_cpu,
                        _valid_cpu,
                        lengths_cpu,
                        _spans,
                        _pair_keys,
                        _pair_counts,
                    ) = self._unpack_cpu_batch(batch)
                    total_sequences += int(tokens_cpu.shape[0])
                    total_tokens += int(lengths_cpu.to(torch.int64).sum().item())
            return total_tokens, total_sequences

        def _unpack_cpu_batch(
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
            | Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                list[torch.Tensor],
                tuple[torch.Tensor, torch.Tensor] | None,
            ]
            | CPUFallbackBatch,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            list[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ]:
            return self._unpack_cpu_batch(batch)

        def _pack_cpu_batch(
            tokens_cpu: torch.Tensor,
            valid_cpu: torch.Tensor,
            lengths_cpu: torch.Tensor,
            spans: list[torch.Tensor] | None = None,
            pair_keys: Optional[torch.Tensor] = None,
            pair_counts: Optional[torch.Tensor] = None,
            *,
            as_fallback: bool = False,
            workspaces: Optional[FastPathWorkspaces] = None,
        ) -> (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                list[torch.Tensor],
                Optional[tuple[torch.Tensor, torch.Tensor]],
            ]
            | CPUFallbackBatch
        ):
            return self._pack_cpu_batch(
                tokens_cpu,
                valid_cpu,
                lengths_cpu,
                spans,
                pair_keys,
                pair_counts,
                as_fallback=as_fallback,
                workspaces=workspaces,
            )

        def _extract_sequences_from_cpu(
            cpu_batches: Iterable[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
            ]
        ) -> list[list[int]]:
            return self._extract_sequences_from_cpu_batches(cpu_batches)

        def _extract_sequences_from_gpu(
            records: Iterable[MultiDeviceBatch],
        ) -> list[list[int]]:
            return self._extract_sequences_from_gpu_records(records)

        def _iter_cpu_batches_from_sequences(
            sequences: list[list[int]],
            batch_size: int,
            pin: bool,
        ) -> Iterator[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                list[torch.Tensor],
                Optional[tuple[torch.Tensor, torch.Tensor]],
            ]
        ]:
            yield from self._iter_cpu_batches_from_sequences(sequences, batch_size, pin)

        def _collect_sequences_from_mixed(
            batches_iter: Iterable[
                Union[
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]],
                    MultiDeviceBatch,
                ]
            ]
        ) -> list[list[int]]:
            return self._collect_sequences_from_batches(batches_iter)

        gpu_devices = [dev for dev in self.devices if dev.type == "cuda"]
        if gpu_devices and not torch.cuda.is_available():
            raise RuntimeError("CUDA devices requested but CUDA is not available")
        use_cuda = bool(gpu_devices)
        device_contexts: Dict[torch.device, DeviceContext] = {}
        if use_cuda:
            for dev in gpu_devices:
                torch.cuda.set_device(dev)
                device_contexts[dev] = DeviceContext(
                    device=dev,
                    compute_stream=torch.cuda.Stream(device=dev),
                    h2d_stream=torch.cuda.Stream(device=dev),
                    d2h_stream=torch.cuda.Stream(device=dev),
                    overlap_enabled=self._overlap_enabled,
                )
            self._device_contexts = device_contexts
            for ctx in device_contexts.values():
                ctx.overlap_enabled = self._overlap_enabled
                ctx.bytes_h2d = 0
                ctx.bytes_d2h = 0
                ctx.h2d_events = 0
                ctx.d2h_events = 0
                ctx.reset_activity()
        else:
            self._device_contexts = {}

        device = torch.device(self.device)

        resume_payload: Optional[dict[str, object]] = None
        resumed = False
        if resume_state is not None:
            resume_payload = self.load_state_dict(
                resume_state,
                device_contexts=device_contexts if use_cuda else None,
                use_cuda=use_cuda,
            )
            restored_batches = resume_payload.get("current_batches")
            if restored_batches is not None:
                current_batches = restored_batches
                resumed = True
            restored_gpu = resume_payload.get("gpu_batches")
            if use_cuda and restored_gpu is not None:
                gpu_batches = restored_gpu
                resumed = True
            stage_timings_data = resume_payload.get("stage_timings")
            if isinstance(stage_timings_data, dict):
                for name, samples in stage_timings_data.items():
                    if name in stage_timings and isinstance(samples, list):
                        stage_timings[name].samples = [float(val) for val in samples]
            stage_event_log.extend(
                [dict(evt) for evt in resume_payload.get("stage_event_log", [])]
            )
            host_sync_events.extend(
                [dict(evt) for evt in resume_payload.get("host_sync_events", [])]
            )
            device_snapshot_log.extend(
                [dict(evt) for evt in resume_payload.get("device_snapshot_log", [])]
            )
            loaded_scale_state = resume_payload.get("scale_state")
            if loaded_scale_state is not None:
                scale_state = loaded_scale_state
                if self.autoscaler.state is None:
                    self.autoscaler.state = loaded_scale_state
            step = int(resume_payload.get("step", len(self.merges)))
            self._merge_step = step
            if checkpoint_interval is not None and checkpoint_dir_path is not None:
                os.makedirs(checkpoint_dir_path, exist_ok=True)
        else:
            resumed = False

        if scale_state is None and self.autoscaler.state is not None:
            scale_state = self.autoscaler.state

        def _capture_device_snapshots(stage_label: str) -> None:
            if not use_cuda:
                return
            for dev, ctx in device_contexts.items():
                snapshot = ctx.capture_snapshot(stage_label)
                snapshot["device"] = str(dev)
                device_snapshot_log.append(snapshot)

        def _record_new_batch(ctx: DeviceContext, record: GPUBatchRecord) -> None:
            self._record_h2d(
                record.tokens,
                record.valid,
                record.lengths,
                device=ctx.device,
                stage="initial_load",
            )

        def _register_batch(batch: MultiDeviceBatch) -> None:
            for dev, record in batch.iter_shards():
                ctx = device_contexts.get(dev)
                if ctx is not None:
                    ctx.active_batches.append(record)

        def _mark_active_batches(batches_to_mark: Iterable[MultiDeviceBatch]) -> None:
            self._mark_active_batches(batches_to_mark, device_contexts)

        def _build_gpu_batches_from_sequences(
            sequences: list[list[int]], batch_size: int
        ) -> tuple[list[MultiDeviceBatch], list[CPUFallbackBatch]]:
            if not use_cuda:
                return [], []
            return self._build_gpu_batches_from_sequences(
                sequences,
                batch_size,
                device_contexts,
                _record_new_batch,
            )

        def _reconfigure_batches(batch_size: int) -> None:
            nonlocal current_batches, gpu_batches
            if batch_size <= 0:
                return
            if use_cuda:
                source_batches: Iterable[
                    Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], MultiDeviceBatch]
                ]
                if gpu_batches is not None:
                    source_batches = gpu_batches
                else:
                    source_batches = current_batches
                sequences = _collect_sequences_from_mixed(source_batches)
                new_records, cpu_fallbacks = _build_gpu_batches_from_sequences(
                    sequences, batch_size
                )
                combined: list[
                    Union[
                        MultiDeviceBatch,
                        CPUFallbackBatch,
                        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]],
                        Tuple[
                            torch.Tensor,
                            torch.Tensor,
                            torch.Tensor,
                            list[torch.Tensor],
                            Optional[tuple[torch.Tensor, torch.Tensor]],
                        ],
                    ]
                ] = []
                if cpu_fallbacks:
                    combined.extend(cpu_fallbacks)
                combined.extend(new_records)
                gpu_batches = new_records
                current_batches = combined
            else:
                cpu_batches: list[
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                    | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
                ] = []
                if not isinstance(current_batches, list):
                    current_batches = list(current_batches)
                sequences = _extract_sequences_from_cpu(current_batches)
                for cpu_batch in _iter_cpu_batches_from_sequences(
                    sequences, batch_size, pin=False
                ):
                    cpu_batches.append(cpu_batch)
                current_batches = cpu_batches
            self._invalidate_hist_cache()
            if on_batch_size_change is not None:
                on_batch_size_change(batch_size)

        def _count_pairs_cpu(
            batch_iter: Iterable[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
                | Tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    list[torch.Tensor],
                    Optional[tuple[torch.Tensor, torch.Tensor]],
                ]
            ]
        ) -> tuple[
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            list[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    list[torch.Tensor],
                    Optional[tuple[torch.Tensor, torch.Tensor]],
                ]
            ],
        ]:
            batches = list(batch_iter)
            if (
                self._enable_histogram_cache
                and self._hist_cache_valid
                and not self._force_recount
            ):
                return (
                    self._cached_pair_keys.clone(),
                    self._cached_pair_counts.clone(),
                    batches,
                )

            global_keys: list[torch.Tensor] = []
            global_counts: list[torch.Tensor] = []
            consumed: list[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    list[torch.Tensor],
                    Optional[tuple[torch.Tensor, torch.Tensor]],
                ]
            ] = []
            pair_keys_workspace: Optional[torch.Tensor] = None
            pair_counts_workspace: Optional[torch.Tensor] = None
            pair_count_length_workspace: Optional[torch.Tensor] = None
            for batch in batches:
                is_fallback = isinstance(batch, CPUFallbackBatch)
                x_cpu, v_cpu, lengths_cpu, spans, _, _ = _unpack_cpu_batch(batch)
                x_snapshot = x_cpu.clone()
                v_snapshot = v_cpu.clone()
                lengths_snapshot = lengths_cpu.clone()
                B, L = x_snapshot.shape
                width = max(L - 1, 0)
                capacity = int(B * width)
                if capacity == 0:
                    pair_keys_workspace = torch.empty(
                        (0, 2), dtype=x_snapshot.dtype, device=cpu_device
                    )
                    pair_counts_workspace = torch.empty(
                        (0,), dtype=torch.int64, device=cpu_device
                    )
                else:
                    if (
                        pair_keys_workspace is None
                        or pair_keys_workspace.shape[0] != capacity
                        or pair_keys_workspace.shape[1] != 2
                    ):
                        pair_keys_workspace = torch.empty(
                            (capacity, 2), dtype=x_snapshot.dtype, device=cpu_device
                        )
                    if (
                        pair_counts_workspace is None
                        or pair_counts_workspace.shape[0] != capacity
                        or pair_counts_workspace.dtype != torch.int64
                    ):
                        pair_counts_workspace = torch.empty(
                            (capacity,), dtype=torch.int64, device=cpu_device
                        )
                if (
                    pair_count_length_workspace is None
                    or pair_count_length_workspace.device != cpu_device
                ):
                    pair_count_length_workspace = torch.zeros(
                        (1,), dtype=torch.long, device=cpu_device
                    )
                count_pairs(
                    x_snapshot.to(cpu_device),
                    v_snapshot.to(cpu_device),
                    pair_keys_workspace,
                    pair_counts_workspace,
                    pair_count_length_workspace,
                )
                length = int(pair_count_length_workspace.item())
                if length > 0:
                    pairs_view = pair_keys_workspace[:length]
                    counts_view = pair_counts_workspace[:length]
                    a_ids = pairs_view[:, 0].to(torch.long)
                    b_ids = pairs_view[:, 1].to(torch.long)
                    keys = ((a_ids << 32) | b_ids).to(torch.long)
                    counts_cpu = counts_view.to(torch.int64)
                else:
                    keys = torch.empty((0,), dtype=torch.long, device=cpu_device)
                    counts_cpu = torch.empty((0,), dtype=torch.int64, device=cpu_device)
                keys_cpu = keys.clone().to("cpu")
                counts_cpu = counts_cpu.clone().to("cpu")
                consumed.append(
                    _pack_cpu_batch(
                        x_snapshot,
                        v_snapshot,
                        lengths_snapshot,
                        list(spans),
                        keys_cpu,
                        counts_cpu,
                        as_fallback=is_fallback,
                        workspaces=batch.workspaces if is_fallback else None,
                    )
                )
                if keys_cpu.numel() > 0:
                    global_keys.append(keys_cpu)
                    global_counts.append(counts_cpu)

            if global_keys:
                combined_keys = torch.cat(global_keys, dim=0)
                combined_counts = torch.cat(global_counts, dim=0)
                reduction_start = time.perf_counter() if metrics_enabled else None
                reduced_keys, reduced_counts = aggregate_pair_keys(
                    combined_keys, combined_counts
                )
                if reduction_start is not None:
                    reduction_duration = time.perf_counter() - reduction_start
                    stage_kind = metrics_tracker.record_stage(
                        "reduction", reduction_duration
                    )
                    if self._metrics_iteration_summary is not None:
                        self._accumulate_iteration_stage(
                            self._metrics_iteration_summary,
                            "reduction",
                            reduction_duration,
                            stage_kind,
                        )
            else:
                reduced_keys = torch.empty((0,), dtype=torch.long)
                reduced_counts = torch.empty((0,), dtype=torch.int64)

            self._cached_pair_keys = reduced_keys.clone()
            self._cached_pair_counts = reduced_counts.clone()
            self._hist_cache_valid = True
            self._force_recount = False
            self._refresh_top_pairs_from_cache()
            if not self._enable_histogram_cache:
                self._invalidate_hist_cache()
            return reduced_keys.clone(), reduced_counts.clone(), consumed

        def _apply_merge_cpu(
            batch_iter: Iterable[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
            ],
            a_id: int,
            b_id: int,
            new_id: int,
        ) -> tuple[
            list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]],
            bool,
            dict[str, object],
        ]:
            new_batches: list[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
            ] = []
            oom_seen = False
            pair_workspace: Optional[torch.Tensor] = None
            prefix_workspace: Optional[torch.Tensor] = None
            span_workspace: Optional[torch.Tensor] = None
            overflow_workspace: Optional[torch.Tensor] = None
            for batch in batch_iter:
                (
                    x_cpu,
                    v_cpu,
                    lengths_cpu,
                    spans,
                    pair_keys,
                    pair_counts,
                ) = _unpack_cpu_batch(batch)
                is_fallback = isinstance(batch, CPUFallbackBatch)
                try:
                    B, L = x_cpu.shape
                    width = max(L - 1, 0)
                    if pair_workspace is None or pair_workspace.shape != (B, width):
                        pair_workspace = torch.zeros((B, width), dtype=torch.bool, device=cpu_device)
                    prefix_dtype = lengths_cpu.dtype
                    if (
                        prefix_workspace is None
                        or prefix_workspace.shape[0] != B
                        or prefix_workspace.dtype != prefix_dtype
                    ):
                        prefix_workspace = torch.zeros((B,), dtype=prefix_dtype, device=cpu_device)
                    if span_workspace is None or span_workspace.shape != (B, width):
                        span_workspace = torch.zeros((B, width), dtype=torch.bool, device=cpu_device)
                    if overflow_workspace is None or overflow_workspace.shape != (B,):
                        overflow_workspace = torch.zeros((B,), dtype=torch.bool, device=cpu_device)
                    else:
                        overflow_workspace.zero_()
                    if (
                        self._enable_histogram_cache
                        and self._hist_cache_valid
                        and width > 0
                    ):
                        pre_lhs = x_cpu[:, :-1].clone()
                        pre_rhs = x_cpu[:, 1:].clone()
                        pre_mask = (
                            v_cpu[:, :-1].to(torch.bool).clone()
                            & v_cpu[:, 1:].to(torch.bool).clone()
                        )
                    else:
                        pre_lhs = pre_rhs = pre_mask = None
                    _, _, _, span_mask = apply_merge_once(
                        x_cpu,
                        v_cpu,
                        lengths_cpu,
                        a_id,
                        b_id,
                        new_id,
                        pair_workspace,
                        prefix_workspace,
                        span_workspace,
                        overflow_workspace,
                    )
                    span_bool = span_mask.to(torch.bool).clone()
                    updated_spans = list(spans)
                    updated_spans.append(span_bool)
                    if (
                        self._enable_histogram_cache
                        and self._hist_cache_valid
                        and width > 0
                        and pre_lhs is not None
                        and pre_rhs is not None
                        and pre_mask is not None
                    ):
                        post_lhs = x_cpu[:, :-1]
                        post_rhs = x_cpu[:, 1:]
                        post_mask = v_cpu[:, :-1].to(torch.bool) & v_cpu[:, 1:].to(torch.bool)
                        remove_keys, remove_counts, add_keys, add_counts = self._compute_histogram_deltas(
                            span_bool,
                            pre_lhs,
                            pre_rhs,
                            pre_mask,
                            post_lhs,
                            post_rhs,
                            post_mask,
                        )
                        if pair_keys is None or pair_counts is None:
                            pair_keys = torch.empty((0,), dtype=torch.long)
                            pair_counts = torch.empty((0,), dtype=torch.int64)
                        pair_keys, pair_counts = self._apply_histogram_delta(
                            pair_keys,
                            pair_counts,
                            remove_keys,
                            remove_counts,
                            add_keys,
                            add_counts,
                        )
                        self._update_global_histogram(
                            remove_keys, remove_counts, add_keys, add_counts
                        )
                    else:
                        if self._enable_histogram_cache and width > 0:
                            self._invalidate_hist_cache()
                        pair_keys = (
                            pair_keys if pair_keys is not None else torch.empty((0,), dtype=torch.long)
                        )
                        pair_counts = (
                            pair_counts
                            if pair_counts is not None
                            else torch.empty((0,), dtype=torch.int64)
                        )
                    new_batches.append(
                        _pack_cpu_batch(
                            x_cpu.clone(),
                            v_cpu.clone(),
                            lengths_cpu.clone(),
                            updated_spans,
                            pair_keys.clone(),
                            pair_counts.clone(),
                            as_fallback=is_fallback,
                            workspaces=batch.workspaces if is_fallback else None,
                        )
                    )
                except RuntimeError as exc:
                    if "CUDA out of memory" in str(exc):
                        oom_seen = True
                        torch.cuda.empty_cache()
                        new_batches.append(
                            _pack_cpu_batch(
                                x_cpu,
                                v_cpu,
                                lengths_cpu,
                                list(spans),
                                pair_keys,
                                pair_counts,
                                as_fallback=is_fallback,
                                workspaces=batch.workspaces if is_fallback else None,
                            )
                        )
                else:
                    raise
            return new_batches, oom_seen, {"sync": True, "bytes": 0}

        if use_cuda:
            def _finalize_host_sync(records: list[MultiDeviceBatch]) -> tuple[int, float]:
                sync_start = time.perf_counter()
                copied_bytes = 0
                pending_copy = False
                _capture_device_snapshots("host_sync:final:start")
                _mark_active_batches(records)
                for batch in records:
                    for dev, shard in batch.iter_shards():
                        ctx = device_contexts[dev]
                        if shard.host_dirty:
                            copied_bytes += self._record_d2h(
                                shard.tokens,
                                shard.valid,
                                device=dev,
                                stage="host_sync",
                            )
                            with torch.cuda.device(dev):
                                shard.schedule_host_sync(
                                    ctx.d2h_stream, overlap=ctx.overlap_enabled
                                )
                            pending_copy = True
                        elif shard.host_event is not None:
                            pending_copy = True
                for dev, ctx in device_contexts.items():
                    torch.cuda.current_stream(device=dev).wait_stream(ctx.d2h_stream)
                if profile_streams and pending_copy:
                    for dev, ctx in device_contexts.items():
                        torch.cuda.set_device(dev)
                        torch.cuda.synchronize(dev)
                        assert ctx.d2h_stream.query(), "copy stream did not drain"
                for batch in records:
                    for _dev, shard in batch.iter_shards():
                        shard.resolve_host()
                elapsed = time.perf_counter() - sync_start if pending_copy else 0.0
                if pending_copy:
                    _capture_device_snapshots("host_sync:final:end")
                    if metrics_enabled and elapsed > 0.0:
                        stage_kind = metrics_tracker.record_stage("d2h", elapsed)
                        if self._metrics_iteration_summary is not None:
                            self._accumulate_iteration_stage(
                                self._metrics_iteration_summary,
                                "d2h",
                                elapsed,
                                stage_kind,
                            )
                return copied_bytes, elapsed

            def _count_pairs_gpu(
                batch_iter: Iterable[
                    Union[
                        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        MultiDeviceBatch,
                        CPUFallbackBatch,
                    ]
                ]
            ) -> tuple[
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                list[Union[MultiDeviceBatch, CPUFallbackBatch]],
            ]:
                nonlocal current_batches, gpu_batches, scale_state
                if (
                    self._enable_histogram_cache
                    and self._hist_cache_valid
                    and not self._force_recount
                ):
                    realized = list(batch_iter)
                    _mark_active_batches(realized)
                    self._materialize_histogram_cache_on_devices(device_contexts)
                    return (
                        self._cached_pair_keys.clone(),
                        self._cached_pair_counts.clone(),
                        realized,
                    )
                while True:
                    for ctx in device_contexts.values():
                        ctx.reset_activity()
                    local_keys: list[torch.Tensor] = []
                    local_counts: list[torch.Tensor] = []
                    consumed: list[Union[MultiDeviceBatch, CPUFallbackBatch]] = []
                    cpu_consumed: list[CPUFallbackBatch] = []
                    pair_results: Dict[torch.device, list[GPUBatchRecord]] = {
                        dev: [] for dev in device_contexts
                    }
                    pending_compute: Dict[torch.device, bool] = {
                        dev: False for dev in device_contexts
                    }
                    oom_triggered = False
                    failed_cpu_batch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
                    iterator = iter(batch_iter)
                    for batch in iterator:
                        multi_batch: Optional[MultiDeviceBatch]
                        try:
                            if isinstance(batch, CPUFallbackBatch):
                                consumed.append(batch)
                                cpu_consumed.append(batch)
                                tokens_cpu, valid_cpu, _lengths_cpu, _, _, _ = _unpack_cpu_batch(
                                    batch
                                )
                                keys_cpu, counts_cpu = count_pairs_fastpath(tokens_cpu, valid_cpu)
                                if keys_cpu.numel() > 0:
                                    local_keys.append(keys_cpu.to(torch.long))
                                    local_counts.append(counts_cpu.to(torch.int64))
                                if self._enable_histogram_cache:
                                    batch.pair_keys = keys_cpu.clone()
                                    batch.pair_counts = counts_cpu.clone()
                                continue
                            if isinstance(batch, MultiDeviceBatch):
                                multi_batch = batch
                            else:
                                tokens_cpu, valid_cpu, lengths_cpu = batch
                                multi_batch = MultiDeviceBatch.from_cpu_batch(
                                    tokens_cpu,
                                    valid_cpu,
                                    lengths_cpu,
                                    device_contexts,
                                    _record_new_batch,
                                )
                            consumed.append(multi_batch)
                            _register_batch(multi_batch)
                            for dev, shard in multi_batch.iter_shards():
                                ctx = device_contexts[dev]
                                try:
                                    with torch.cuda.device(dev), torch.cuda.stream(ctx.compute_stream):
                                        shard.wait_for_device(ctx.compute_stream)
                                        shard.ensure_workspaces()
                                        assert shard.pair_keys_buffer is not None
                                        assert shard.pair_counts_buffer is not None
                                        assert shard.pair_count_length is not None
                                        count_pairs(
                                            shard.tokens,
                                            shard.valid,
                                            shard.pair_keys_buffer,
                                            shard.pair_counts_buffer,
                                            shard.pair_count_length,
                                        )
                                    pair_results[dev].append(shard)
                                    pending_compute[dev] = True
                                    shard.pair_keys = None
                                    shard.pair_counts = None
                                except RuntimeError as exc:
                                    if "CUDA out of memory" in str(exc):
                                        oom_triggered = True
                                        if not isinstance(batch, MultiDeviceBatch):
                                            failed_cpu_batch = batch
                                        torch.cuda.empty_cache()
                                        break
                                    raise
                        except RuntimeError:
                            raise
                        if oom_triggered:
                            break
                    if oom_triggered:
                        tail_batches: list[
                            Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], MultiDeviceBatch]
                        ] = list(iterator)
                        sequences = _extract_sequences_from_gpu(
                            [b for b in consumed if isinstance(b, MultiDeviceBatch)]
                        )
                        if cpu_consumed:
                            sequences.extend(_extract_sequences_from_cpu(cpu_consumed))
                        if failed_cpu_batch is not None:
                            sequences.extend(_extract_sequences_from_cpu([failed_cpu_batch]))
                        if tail_batches:
                            sequences.extend(_collect_sequences_from_mixed(tail_batches))
                        prev_bs = scale_state.batch_size if scale_state is not None else None
                        self.autoscaler.feedback(
                            oom=True, cpu_fallback_rate=self._last_cpu_fallback_ratio
                        )
                        scale_state = self.autoscaler.state or scale_state
                        if scale_state is None:
                            raise RuntimeError("Autoscaler did not provide a state after OOM")
                        new_bs = scale_state.batch_size
                        if prev_bs is not None and new_bs >= prev_bs:
                            raise RuntimeError(
                                "CUDA OOM could not be resolved by autoscaler batch size reduction"
                            )
                        new_gpu_batches, fallback_batches = _build_gpu_batches_from_sequences(
                            sequences, new_bs
                        )
                        gpu_batches = new_gpu_batches
                        combined_batches: list[Union[MultiDeviceBatch, CPUFallbackBatch]] = []
                        if fallback_batches:
                            combined_batches.extend(fallback_batches)
                        combined_batches.extend(new_gpu_batches)
                        current_batches = combined_batches
                        batch_iter = current_batches
                        if on_batch_size_change is not None:
                            on_batch_size_change(new_bs)
                        self._active_batch_size = new_bs
                        self._invalidate_hist_cache()
                        continue
                    for dev, ctx in device_contexts.items():
                        if pending_compute.get(dev):
                            torch.cuda.current_stream(device=dev).wait_stream(ctx.compute_stream)
                    if profile_streams:
                        for dev, pending in pending_compute.items():
                            if pending:
                                torch.cuda.set_device(dev)
                                torch.cuda.synchronize(dev)
                                assert device_contexts[dev].compute_stream.query(), "compute stream did not drain"
                    for dev, per_device_records in pair_results.items():
                        for shard in per_device_records:
                            assert shard.pair_count_length is not None
                            length = int(shard.pair_count_length.item())
                            if length <= 0:
                                if self._enable_histogram_cache:
                                    shard.pair_keys = torch.empty((0,), dtype=torch.long)
                                    shard.pair_counts = torch.empty((0,), dtype=torch.int64)
                                continue
                            assert shard.pair_keys_buffer is not None
                            assert shard.pair_counts_buffer is not None
                            pairs_view = shard.pair_keys_buffer.narrow(0, 0, length)
                            counts_view = shard.pair_counts_buffer.narrow(0, 0, length)
                            a_ids = pairs_view[:, 0].to(torch.long)
                            b_ids = pairs_view[:, 1].to(torch.long)
                            keys = (a_ids << 32) | b_ids
                            local_keys.append(keys)
                            local_counts.append(counts_view.to(torch.int64))
                            if self._enable_histogram_cache:
                                shard.pair_keys = keys.to(torch.long).to("cpu")
                                shard.pair_counts = counts_view.to(torch.int64).to("cpu")
                    if local_keys and local_counts:
                        combined_keys = torch.cat(local_keys, dim=0)
                        combined_counts = torch.cat(local_counts, dim=0)
                        reduction_start = time.perf_counter() if metrics_enabled else None
                        reduced_keys, reduced_counts = reduce_pair_histograms(
                            combined_keys, combined_counts
                        )
                        if reduction_start is not None:
                            reduction_duration = time.perf_counter() - reduction_start
                            stage_kind = metrics_tracker.record_stage(
                                "reduction", reduction_duration
                            )
                            if self._metrics_iteration_summary is not None:
                                self._accumulate_iteration_stage(
                                    self._metrics_iteration_summary,
                                    "reduction",
                                    reduction_duration,
                                    stage_kind,
                                )
                    else:
                        reduced_keys = None
                        reduced_counts = None
                    gpu_batches = [
                        batch for batch in consumed if isinstance(batch, MultiDeviceBatch)
                    ]
                    if reduced_keys is None or reduced_counts is None:
                        cached_keys = torch.empty((0,), dtype=torch.long)
                        cached_counts = torch.empty((0,), dtype=torch.int64)
                        gpu_source: Optional[tuple[torch.Tensor, torch.Tensor]] = None
                    else:
                        gpu_keys = reduced_keys.to(torch.long)
                        gpu_counts = reduced_counts.to(torch.int64)
                        gpu_source = (gpu_keys, gpu_counts)
                        cached_keys = gpu_keys.to("cpu")
                        cached_counts = gpu_counts.to("cpu")
                    self._cached_pair_keys = cached_keys.clone()
                    self._cached_pair_counts = cached_counts.clone()
                    self._hist_cache_valid = True
                    self._force_recount = False
                    self._refresh_top_pairs_from_cache()
                    if not self._enable_histogram_cache:
                        self._invalidate_hist_cache()
                    else:
                        self._materialize_histogram_cache_on_devices(
                            device_contexts, source=gpu_source
                        )
                    total_batches = max(1, len(consumed))
                    self._cpu_fallback_batches += len(cpu_consumed)
                    self._last_cpu_fallback_ratio = len(cpu_consumed) / float(total_batches)
                    return cached_keys, cached_counts, consumed

            def _apply_merge_gpu(
                batch_iter: Iterable[Union[MultiDeviceBatch, CPUFallbackBatch]],
                a_id: int,
                b_id: int,
                new_id: int,
                merge_idx: int,
                force_sync: bool = False,
            ) -> tuple[
                list[Union[MultiDeviceBatch, CPUFallbackBatch]],
                bool,
                dict[str, object],
            ]:
                nonlocal current_batches, gpu_batches, scale_state
                oom_seen = False
                while True:
                    records = list(batch_iter)
                    gpu_records = [b for b in records if isinstance(b, MultiDeviceBatch)]
                    cpu_records = [b for b in records if isinstance(b, CPUFallbackBatch)]
                    _mark_active_batches(gpu_records)
                    need_sync = force_sync or (merge_idx % self.sync_every == 0)
                    pending_copy: Dict[torch.device, bool] = {
                        dev: False for dev in device_contexts
                    }
                    pending_compute: Dict[torch.device, bool] = {
                        dev: False for dev in device_contexts
                    }
                    copied_bytes = 0
                    retry_required = False
                    updated_cpu: list[Union[MultiDeviceBatch, CPUFallbackBatch]] = []
                    if cpu_records:
                        cpu_updates, _cpu_oom, _cpu_sync = _apply_merge_cpu(
                            cpu_records, a_id, b_id, new_id
                        )
                        updated_cpu.extend(cpu_updates)
                    for batch in gpu_records:
                        for dev, shard in batch.iter_shards():
                            ctx = device_contexts[dev]
                            try:
                                with torch.cuda.device(dev), torch.cuda.stream(ctx.compute_stream):
                                    shard.wait_for_device(ctx.compute_stream)
                                    shard.ensure_workspaces()
                                    width = max(shard.tokens.shape[1] - 1, 0)
                                    if shard.tokens.is_cuda and not shard.merge_kernel_warm:
                                        sentinel = torch.iinfo(shard.tokens.dtype).min
                                        cuda_apply_merge_and_compact(
                                            shard.tokens,
                                            shard.valid,
                                            shard.prefix_workspace,
                                            shard.pair_workspace,
                                            int(sentinel),
                                            int(sentinel),
                                            int(sentinel),
                                        )
                                        prefix_vals = shard.prefix_workspace.to(torch.int64)
                                        if shard.lengths.dtype == torch.uint16:
                                            max_val = 65535
                                            shard.lengths.copy_(
                                                torch.clamp_max(prefix_vals, max_val).to(torch.uint16)
                                            )
                                        else:
                                            shard.lengths.copy_(prefix_vals.to(shard.lengths.dtype))
                                        if shard.length_overflow is not None:
                                            shard.length_overflow.zero_()
                                        shard.merge_kernel_warm = True
                                    if (
                                        self._enable_histogram_cache
                                        and self._hist_cache_valid
                                        and width > 0
                                    ):
                                        pre_lhs = shard.tokens[:, :-1].clone()
                                        pre_rhs = shard.tokens[:, 1:].clone()
                                        pre_mask = (
                                            shard.valid[:, :-1].to(torch.bool).clone()
                                            & shard.valid[:, 1:].to(torch.bool).clone()
                                        )
                                    else:
                                        pre_lhs = pre_rhs = pre_mask = None
                                    _, _, _, span_mask = apply_merge_once(
                                        shard.tokens,
                                        shard.valid,
                                        shard.lengths,
                                        a_id,
                                        b_id,
                                        new_id,
                                        shard.pair_workspace,
                                        shard.prefix_workspace,
                                        shard.span_workspace,
                                        shard.length_overflow,
                                    )
                                    span_bool = span_mask.to(torch.bool).clone()
                                    shard.span_history.append(span_bool)
                                    if (
                                        self._enable_histogram_cache
                                        and self._hist_cache_valid
                                        and width > 0
                                        and pre_lhs is not None
                                        and pre_rhs is not None
                                        and pre_mask is not None
                                    ):
                                        post_lhs = shard.tokens[:, :-1]
                                        post_rhs = shard.tokens[:, 1:]
                                        post_mask = shard.valid[:, :-1].to(torch.bool) & shard.valid[:, 1:].to(torch.bool)
                                        (
                                            remove_keys,
                                            remove_counts,
                                            add_keys,
                                            add_counts,
                                        ) = self._compute_histogram_deltas(
                                            span_bool,
                                            pre_lhs,
                                            pre_rhs,
                                            pre_mask,
                                            post_lhs,
                                            post_rhs,
                                            post_mask,
                                        )
                                        if shard.pair_keys is None or shard.pair_counts is None:
                                            shard.pair_keys = torch.empty((0,), dtype=torch.long)
                                            shard.pair_counts = torch.empty((0,), dtype=torch.int64)
                                        shard.pair_keys, shard.pair_counts = self._apply_histogram_delta(
                                            shard.pair_keys,
                                            shard.pair_counts,
                                            remove_keys,
                                            remove_counts,
                                            add_keys,
                                            add_counts,
                                        )
                                        self._update_global_histogram(
                                            remove_keys, remove_counts, add_keys, add_counts
                                        )
                                    elif self._enable_histogram_cache and width > 0:
                                        self._invalidate_hist_cache()
                                        if shard.pair_keys is None:
                                            shard.pair_keys = torch.empty((0,), dtype=torch.long)
                                        if shard.pair_counts is None:
                                            shard.pair_counts = torch.empty((0,), dtype=torch.int64)
                                    compute_event = torch.cuda.Event(blocking=False)
                                    compute_event.record(ctx.compute_stream)
                                shard.mark_device_event(compute_event)
                                pending_compute[dev] = True
                                if need_sync:
                                    copied_bytes += self._record_d2h(
                                        shard.tokens,
                                        shard.valid,
                                        device=dev,
                                        stage="host_sync",
                                    )
                                    with torch.cuda.device(dev):
                                        shard.schedule_host_sync(
                                            ctx.d2h_stream, overlap=ctx.overlap_enabled
                                        )
                                    pending_copy[dev] = True
                            except RuntimeError as exc:
                                if "CUDA out of memory" in str(exc):
                                    oom_seen = True
                                    retry_required = True
                                    torch.cuda.empty_cache()
                                    break
                                raise
                    if retry_required:
                        break
                    if retry_required:
                        sequences = _extract_sequences_from_gpu(gpu_records)
                        prev_bs = scale_state.batch_size if scale_state is not None else None
                        self.autoscaler.feedback(
                            oom=True, cpu_fallback_rate=self._last_cpu_fallback_ratio
                        )
                        scale_state = self.autoscaler.state or scale_state
                        if scale_state is None:
                            raise RuntimeError("Autoscaler did not provide a state after OOM")
                        new_bs = scale_state.batch_size
                        if prev_bs is not None and new_bs >= prev_bs:
                            raise RuntimeError(
                                "CUDA OOM could not be resolved by autoscaler batch size reduction"
                            )
                        new_gpu_batches, fallback_batches = _build_gpu_batches_from_sequences(
                            sequences, new_bs
                        )
                        gpu_batches = new_gpu_batches
                        combined_batches: list[Union[MultiDeviceBatch, CPUFallbackBatch]] = []
                        if fallback_batches:
                            combined_batches.extend(fallback_batches)
                        combined_batches.extend(new_gpu_batches)
                        current_batches = combined_batches
                        batch_iter = current_batches
                        if on_batch_size_change is not None:
                            on_batch_size_change(new_bs)
                        self._active_batch_size = new_bs
                        self._invalidate_hist_cache()
                        continue
                    for dev, ctx in device_contexts.items():
                        torch.cuda.current_stream(device=dev).wait_stream(ctx.compute_stream)
                    sync_duration = 0.0
                    if need_sync:
                        _capture_device_snapshots("host_sync:merge:start")
                        sync_start = time.perf_counter()
                        for dev, ctx in device_contexts.items():
                            torch.cuda.current_stream(device=dev).wait_stream(ctx.d2h_stream)
                        for batch in gpu_records:
                            for _dev, shard in batch.iter_shards():
                                shard.resolve_host()
                        sync_duration = (
                            time.perf_counter() - sync_start if copied_bytes > 0 else 0.0
                        )
                        if copied_bytes > 0:
                            _capture_device_snapshots("host_sync:merge:end")
                    if profile_streams:
                        for dev, ctx in device_contexts.items():
                            compute_pending = pending_compute.get(dev, False)
                            copy_pending = pending_copy.get(dev, False) and need_sync
                            if compute_pending or copy_pending:
                                torch.cuda.set_device(dev)
                                torch.cuda.synchronize(dev)
                                if copy_pending:
                                    assert ctx.d2h_stream.query(), "copy stream did not drain"
                                if compute_pending:
                                    assert ctx.compute_stream.query(), "compute stream did not drain"
                    updated_records: list[Union[MultiDeviceBatch, CPUFallbackBatch]] = []
                    if updated_cpu:
                        updated_records.extend(updated_cpu)
                    updated_records.extend(gpu_records)
                    return updated_records, oom_seen, {
                        "sync": need_sync and copied_bytes > 0,
                        "bytes": copied_bytes,
                        "duration_s": sync_duration,
                    }

        warm_plan = warm_start_plan if warm_start_plan is not None else self._warm_start_plan
        if warm_plan is not None:
            warm_plan = {
                "merges": list(warm_plan.get("merges", [])),
                "counts": (
                    list(warm_plan.get("counts", []))
                    if warm_plan.get("counts") is not None
                    else None
                ),
                "source": warm_plan.get("source"),
                "order": warm_plan.get("order"),
                "top_k": warm_plan.get("top_k"),
                "requested_top_k": warm_plan.get("requested_top_k"),
                "histogram_size": warm_plan.get("histogram_size"),
            }
        else:
            warm_plan = None

        if warm_start_merges is not None:
            merges_override = [tuple(map(int, pair)) for pair in warm_start_merges]
            if warm_plan is None:
                warm_plan = {"merges": merges_override, "counts": None, "source": "fit"}
            else:
                warm_plan["merges"] = merges_override

        requested_top_k = warm_plan.get("requested_top_k") if warm_plan else None
        warm_counts = warm_plan.get("counts") if warm_plan else None
        warm_source = warm_plan.get("source") if warm_plan else None

        if (
            (warm_plan is None or not warm_plan.get("merges"))
            and warm_start_ngrams
            and not resumed
        ):
            computed_plan = self.precompute_warm_start_plan(
                batches,
                warm_start_ngrams,
                device=warm_start_device or device,
            )
            warm_plan = computed_plan
            warm_counts = computed_plan.get("counts")
            warm_source = computed_plan.get("source")
            requested_top_k = computed_plan.get("requested_top_k")

        warm_merges_to_apply: list[tuple[int, int]] = []
        if warm_plan is not None and warm_plan.get("merges"):
            seen_pairs: set[tuple[int, int]] = set()
            for pair in warm_plan.get("merges", []):
                a_id, b_id = int(pair[0]), int(pair[1])
                normalized = (a_id, b_id)
                if normalized in seen_pairs:
                    continue
                seen_pairs.add(normalized)
                warm_merges_to_apply.append(normalized)
        warm_counts_list: list[int] | None = None
        if warm_counts is not None:
            warm_counts_list = [int(val) for val in warm_counts][: len(warm_merges_to_apply)]

        warm_meta: dict[str, object] = {
            "merges": warm_merges_to_apply,
            "counts": warm_counts_list,
            "frozen": bool(self.freeze_warm_start),
            "source": warm_source,
            "requested_top_k": requested_top_k,
        }

        if warm_merges_to_apply and not self._warm_start_applied:
            warm_limit = max(self.target_merges - step, 0)
            if warm_limit <= 0:
                warm_merges_to_apply = []
            elif len(warm_merges_to_apply) > warm_limit:
                warm_merges_to_apply = warm_merges_to_apply[:warm_limit]
                if warm_counts_list is not None:
                    warm_counts_list = warm_counts_list[: len(warm_merges_to_apply)]
                warm_meta["merges"] = warm_merges_to_apply
                warm_meta["counts"] = warm_counts_list

        if warm_merges_to_apply and not self._warm_start_applied:
            applied_merges: list[tuple[int, int]] = []
            for idx, (a_id, b_id) in enumerate(warm_merges_to_apply, start=1):
                new_id = self.vocab_size
                if max(a_id, b_id, new_id) > UINT32_MAX:
                    raise OverflowError(
                        f"Token id overflow during warm start: encountered id above UINT32_MAX (limit {UINT32_MAX})."
                    )
                if metrics_enabled:
                    iteration_summary = _new_iteration_summary(step + 1, "warm_start")
                    self._metrics_iteration_summary = iteration_summary
                else:
                    iteration_summary = None
                if use_cuda:
                    _capture_device_snapshots("apply_merge:warm:start")
                    warm_merge_start = time.perf_counter()
                    current_batches, oom_seen, sync_report = _apply_merge_gpu(
                        current_batches, a_id, b_id, new_id, step + 1, force_sync=True
                    )
                    merge_duration = time.perf_counter() - warm_merge_start
                    stage_timings["apply_merge"].record(merge_duration)
                    stage_event_log.append(
                        {
                            "stage": "apply_merge",
                            "duration_s": merge_duration,
                            "mode": "gpu",
                            "merge": step,
                            "type": "warm_start",
                            "timestamp": time.time(),
                        }
                    )
                    if metrics_enabled:
                        stage_kind = metrics_tracker.record_stage("kernel", merge_duration)
                        if iteration_summary is not None:
                            self._accumulate_iteration_stage(
                                iteration_summary,
                                "kernel",
                                merge_duration,
                                stage_kind,
                            )
                    if oom_seen:
                        raise RuntimeError(
                            "CUDA out of memory encountered while applying warm-start merges"
                        )
                    gpu_batches = [
                        batch
                        for batch in current_batches
                        if isinstance(batch, MultiDeviceBatch)
                    ]
                    self._interval_merges += 1
                    if sync_report.get("sync"):
                        self._close_sync_interval(step + 1, sync_report.get("bytes", 0))
                    if sync_report.get("duration_s", 0.0) > 0:
                        sync_duration = float(sync_report.get("duration_s", 0.0))
                        stage_timings["host_sync"].record(sync_duration)
                        host_sync_events.append(
                            {
                                "merge": step,
                                "bytes": sync_report.get("bytes", 0),
                                "duration_s": sync_duration,
                                "type": "warm_start",
                                "timestamp": time.time(),
                            }
                        )
                        if metrics_enabled:
                            stage_kind = metrics_tracker.record_stage("d2h", sync_duration)
                            if iteration_summary is not None:
                                self._accumulate_iteration_stage(
                                    iteration_summary,
                                    "d2h",
                                    sync_duration,
                                    stage_kind,
                                )
                    _capture_device_snapshots("apply_merge:warm:end")
                else:
                    warm_merge_start = time.perf_counter()
                    current_batches, _oom_seen, _ = _apply_merge_cpu(
                        current_batches, a_id, b_id, new_id
                    )
                    merge_duration = time.perf_counter() - warm_merge_start
                    stage_timings["apply_merge"].record(merge_duration)
                    stage_event_log.append(
                        {
                            "stage": "apply_merge",
                            "duration_s": merge_duration,
                            "mode": "cpu",
                            "merge": step,
                            "type": "warm_start",
                            "timestamp": time.time(),
                        }
                    )
                    if metrics_enabled:
                        stage_kind = metrics_tracker.record_stage("kernel", merge_duration)
                        if iteration_summary is not None:
                            self._accumulate_iteration_stage(
                                iteration_summary,
                                "kernel",
                                merge_duration,
                                stage_kind,
                            )
                self.merges.append((a_id, b_id))
                self.vocab_size += 1
                step += 1
                applied_merges.append((a_id, b_id))
                if self.freeze_warm_start:
                    key = (int(a_id) << 32) | int(b_id)
                    self._frozen_pair_keys.add(key)
                self._record_merge_snapshot(step)
                if metrics_enabled and iteration_summary is not None:
                    iteration_summary["tokens_per_s"] = 0.0
                    iteration_summary["lease_per_s"] = 0.0
                    _record_iteration_summary(iteration_summary)
                    self._metrics_iteration_summary = None
            warm_meta["applied"] = applied_merges
            self._warm_start_applied = True
            if use_cuda:
                self._interval_merges = 0
            self._invalidate_hist_cache()
        else:
            warm_meta["applied"] = []

        if warm_plan is not None:
            warm_meta["order"] = warm_plan.get("order")
            warm_meta["top_k"] = warm_plan.get("top_k")
            warm_meta["histogram_size"] = warm_plan.get("histogram_size")
        self._warm_start_plan = warm_plan
        self._seed_warm_start_merges = warm_merges_to_apply

        while step < self.target_merges:
            scale_state = self.autoscaler.suggest(token_bytes_per_example=int(8 * 1024))
            if self._active_batch_size is None:
                self._active_batch_size = scale_state.batch_size
            elif scale_state.batch_size != self._active_batch_size:
                _reconfigure_batches(scale_state.batch_size)
                self._active_batch_size = scale_state.batch_size
            t0 = time.time()
            if metrics_enabled:
                iteration_summary = _new_iteration_summary(step + 1, "merge")
                self._metrics_iteration_summary = iteration_summary
            else:
                iteration_summary = None
            merge_applied = False
            candidate_key: Optional[int] = None
            candidate_count = 0
            best_seen = 0
            if (
                self._enable_histogram_cache
                and self._hist_cache_valid
                and not self._force_recount
            ):
                candidate_key, candidate_count, best_seen = self._select_top_candidate()
            use_fast_path = (
                candidate_key is not None and candidate_count >= best_seen and candidate_count > 0
            )
            best_key_value: Optional[int] = None
            best_count = 0
            if use_fast_path:
                best_key_value = int(candidate_key) if candidate_key is not None else None
                best_count = int(candidate_count)
            else:
                if use_cuda:
                    _capture_device_snapshots("pair_count:start")
                    count_start = time.perf_counter()
                    (
                        global_keys,
                        global_counts,
                        consumed_batches,
                    ) = self._invoke_count_pairs_gpu(current_batches, _count_pairs_gpu)
                    count_duration = time.perf_counter() - count_start
                    stage_timings["pair_count"].record(count_duration)
                    stage_event_log.append(
                        {
                            "stage": "pair_count",
                            "duration_s": count_duration,
                            "mode": "gpu",
                            "merge": step,
                            "type": "merge",
                            "timestamp": time.time(),
                        }
                    )
                    if metrics_enabled:
                        stage_kind = metrics_tracker.record_stage("kernel", count_duration)
                        if iteration_summary is not None:
                            self._accumulate_iteration_stage(
                                iteration_summary,
                                "kernel",
                                count_duration,
                                stage_kind,
                            )
                    gpu_batches = [
                        batch
                        for batch in consumed_batches
                        if isinstance(batch, MultiDeviceBatch)
                    ]
                    _capture_device_snapshots("pair_count:end")
                else:
                    count_start = time.perf_counter()
                    (
                        global_keys,
                        global_counts,
                        consumed_batches,
                    ) = self._invoke_count_pairs_cpu(current_batches, _count_pairs_cpu)
                    count_duration = time.perf_counter() - count_start
                    stage_timings["pair_count"].record(count_duration)
                    stage_event_log.append(
                        {
                            "stage": "pair_count",
                            "duration_s": count_duration,
                            "mode": "cpu",
                            "merge": step,
                            "type": "merge",
                            "timestamp": time.time(),
                        }
                    )
                    if metrics_enabled:
                        stage_kind = metrics_tracker.record_stage("kernel", count_duration)
                        if iteration_summary is not None:
                            self._accumulate_iteration_stage(
                                iteration_summary,
                                "kernel",
                                count_duration,
                                stage_kind,
                            )
                if global_keys is None or global_keys.numel() == 0:
                    print("No pairs left to merge.")
                    if metrics_enabled:
                        self._metrics_iteration_summary = None
                    break
                current_batches = consumed_batches
                if metrics_enabled and iteration_summary is not None:
                    tokens_processed, leases_processed = _tally_tokens_from_batches(
                        consumed_batches
                    )
                    iteration_summary["tokens"] = tokens_processed
                    iteration_summary["leases"] = leases_processed
                    metrics_tracker.record_tokens(
                        tokens_processed,
                        count_duration,
                        leases=leases_processed,
                    )
                reduction_start = time.perf_counter() if metrics_enabled else None
                agg_keys, agg_counts = aggregate_pair_keys(global_keys, global_counts)
                if reduction_start is not None:
                    reduction_duration = time.perf_counter() - reduction_start
                    stage_kind = metrics_tracker.record_stage(
                        "reduction", reduction_duration
                    )
                    if self._metrics_iteration_summary is not None:
                        self._accumulate_iteration_stage(
                            self._metrics_iteration_summary,
                            "reduction",
                            reduction_duration,
                            stage_kind,
                        )
                best_tensor_count = torch.max(agg_counts)
                candidate_indices = torch.nonzero(
                    agg_counts == best_tensor_count, as_tuple=False
                ).flatten()
                if candidate_indices.numel() == 1:
                    best_idx = candidate_indices[0]
                else:
                    best_idx = candidate_indices[torch.argmin(agg_keys[candidate_indices])]
                best_key = agg_keys[best_idx]
                best_key_value = int(best_key.item())
                best_count = int(best_tensor_count.item())
            if best_key_value is None:
                print("No pairs left to merge.")
                if metrics_enabled:
                    self._metrics_iteration_summary = None
                break
            a_id = int(best_key_value >> 32)
            b_id = int(best_key_value & ((1 << 32) - 1))
            if best_count <= 1:
                print("Stopping: no frequent pairs.")
                if metrics_enabled:
                    self._metrics_iteration_summary = None
                break
            new_id = self.vocab_size
            if max(a_id, b_id, new_id) > UINT32_MAX:
                raise OverflowError(
                    f"Token id overflow: encountered id above UINT32_MAX (limit {UINT32_MAX})."
                )
            if step % log_every == 0:
                print(f"merge {step:6d}: ({a_id},{b_id}) -> {new_id}  count={best_count}")
            self.merges.append((a_id, b_id))
            self.vocab_size += 1
            merge_applied = True
            if self.vocab_size > UINT32_MAX:
                raise OverflowError(
                    f"Vocabulary size exceeded UINT32_MAX (limit {UINT32_MAX})."
                )
                if use_cuda:
                    _capture_device_snapshots("apply_merge:start")
                    merge_start = time.perf_counter()
                    current_batches, oom_seen, sync_report = _apply_merge_gpu(
                        current_batches, a_id, b_id, new_id, step + 1
                    )
                    merge_duration = time.perf_counter() - merge_start
                    stage_timings["apply_merge"].record(merge_duration)
                    stage_event_log.append(
                        {
                            "stage": "apply_merge",
                        "duration_s": merge_duration,
                        "mode": "gpu",
                        "merge": step,
                        "type": "merge",
                            "timestamp": time.time(),
                        }
                    )
                    if metrics_enabled:
                        stage_kind = metrics_tracker.record_stage("kernel", merge_duration)
                        if iteration_summary is not None:
                            self._accumulate_iteration_stage(
                                iteration_summary,
                                "kernel",
                                merge_duration,
                                stage_kind,
                            )
                    gpu_batches = [
                        batch
                        for batch in current_batches
                        if isinstance(batch, MultiDeviceBatch)
                    ]
                self._interval_merges += 1
                if sync_report.get("sync"):
                    self._close_sync_interval(step + 1, sync_report.get("bytes", 0))
                if sync_report.get("duration_s", 0.0) > 0:
                    sync_duration = float(sync_report.get("duration_s", 0.0))
                    stage_timings["host_sync"].record(sync_duration)
                    host_sync_events.append(
                        {
                            "merge": step,
                            "bytes": sync_report.get("bytes", 0),
                            "duration_s": sync_duration,
                            "type": "merge",
                            "timestamp": time.time(),
                        }
                    )
                    if metrics_enabled:
                        stage_kind = metrics_tracker.record_stage("d2h", sync_duration)
                        if iteration_summary is not None:
                            self._accumulate_iteration_stage(
                                iteration_summary,
                                "d2h",
                                sync_duration,
                                stage_kind,
                            )
                _capture_device_snapshots("apply_merge:end")
            else:
                merge_start = time.perf_counter()
                current_batches, oom_seen, _ = _apply_merge_cpu(current_batches, a_id, b_id, new_id)
                merge_duration = time.perf_counter() - merge_start
                stage_timings["apply_merge"].record(merge_duration)
                stage_event_log.append(
                    {
                        "stage": "apply_merge",
                        "duration_s": merge_duration,
                        "mode": "cpu",
                        "merge": step,
                        "type": "merge",
                        "timestamp": time.time(),
                    }
                )
                if metrics_enabled:
                    stage_kind = metrics_tracker.record_stage("kernel", merge_duration)
                    if iteration_summary is not None:
                        self._accumulate_iteration_stage(
                            iteration_summary,
                            "kernel",
                            merge_duration,
                            stage_kind,
                        )
            step += 1
            self._merge_step = step
            if not use_cuda:
                self._interval_merges = 0
            self._record_merge_snapshot(step)
            self.autoscaler.feedback(
                step_time_s=time.time() - t0,
                oom=oom_seen,
                cpu_fallback_rate=self._last_cpu_fallback_ratio,
            )
            if metrics_enabled:
                if merge_applied and iteration_summary is not None:
                    token_time = float(iteration_summary.get("token_time_s", 0.0))
                    tokens_val = int(iteration_summary.get("tokens", 0))
                    leases_val = int(iteration_summary.get("leases", 0))
                    iteration_summary["tokens_per_s"] = (
                        tokens_val / token_time if token_time > 0 else 0.0
                    )
                    iteration_summary["lease_per_s"] = (
                        leases_val / token_time if token_time > 0 else 0.0
                    )
                    _record_iteration_summary(iteration_summary)
                self._metrics_iteration_summary = None
            if (
                checkpoint_interval is not None
                and checkpoint_dir_path is not None
                and step % checkpoint_interval == 0
            ):
                self.save_checkpoint(
                    checkpoint_dir_path,
                    include_batches=True,
                    current_batches=current_batches,
                    stage_timings=stage_timings,
                    stage_event_log=stage_event_log,
                    host_sync_events=host_sync_events,
                    device_snapshot_log=device_snapshot_log,
                )
        if use_cuda and gpu_batches is not None:
            final_bytes, final_duration = _finalize_host_sync(gpu_batches)
            if final_duration > 0:
                stage_timings["host_sync"].record(final_duration)
                host_sync_events.append(
                    {
                        "merge": step,
                        "bytes": final_bytes,
                        "duration_s": final_duration,
                        "timestamp": time.time(),
                        "type": "final",
                    }
                )
                if metrics_enabled:
                    stage_kind = metrics_tracker.record_stage("d2h", final_duration)
                    if self._metrics_iteration_summary is not None:
                        self._accumulate_iteration_stage(
                            self._metrics_iteration_summary,
                            "d2h",
                            final_duration,
                            stage_kind,
                        )
            if self._interval_merges > 0:
                self._close_sync_interval(step, final_bytes)
        per_device_metrics = {
            str(device): {
                "bytes_h2d": ctx.bytes_h2d,
                "bytes_d2h": ctx.bytes_d2h,
                "h2d_events": ctx.h2d_events,
                "d2h_events": ctx.d2h_events,
                "stage_breakdown": dict(ctx.stage_transfers),
                "memory_snapshots": list(ctx.memory_snapshots),
                "utilization_samples": list(ctx.utilization_samples),
            }
            for device, ctx in self._device_contexts.items()
        }
        for ctx in self._device_contexts.values():
            ctx.reset_activity()
        autoscaler_metrics = self.autoscaler.snapshot_metrics()
        timings_summary = {
            name: timing.summary() for name, timing in stage_timings.items()
        }
        telemetry_summary = {
            "timings": timings_summary,
            "events": stage_event_log,
            "host_sync_events": host_sync_events,
            "device_snapshots": device_snapshot_log,
            "autoscaler": autoscaler_metrics,
        }
        telemetry_summary["iteration_metrics"] = {
            "ewma": metrics_tracker.summaries(),
            "iterations": iteration_summaries,
        }
        return {
            "base_vocab": self.base_vocab,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "warm_start": warm_meta,
            "transfer_metrics": {
                "bytes_h2d": self.bytes_h2d,
                "bytes_d2h": self.bytes_d2h,
                "h2d_events": self.h2d_events,
                "d2h_events": self.d2h_events,
                "merge_stats": self.merge_transfer_log,
                "sync_intervals": self.sync_intervals,
                "per_device": per_device_metrics,
                "per_stage": self._transfer_stage_totals,
                "avg_d2h_bytes_per_merge": (
                    self.bytes_d2h / len(self.merge_transfer_log)
                    if self.merge_transfer_log
                    else 0.0
                ),
                "cpu_fallback": {
                    "batches": self._cpu_fallback_batches,
                    "ratio": self._last_cpu_fallback_ratio,
                },
            },
            "telemetry": telemetry_summary,
        }

    @staticmethod
    def _bytes_to_unicode() -> dict[int, str]:
        """Mirror the mapping used by Hugging Face's ByteLevel BPE."""

        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}

    def _build_tokenizer_artifacts(
        self,
    ) -> tuple[dict[str, int], list[str], dict[str, object]]:
        byte_encoder = self._bytes_to_unicode()
        token_strings: list[str] = []
        added_tokens: list[dict[str, object]] = []
        for token_id in range(self.base_vocab):
            if 0 <= token_id < 256:
                token_strings.append(byte_encoder[token_id])
            else:
                content = f"<|{token_id}|>"
                token_strings.append(content)
                added_tokens.append(
                    {
                        "id": token_id,
                        "content": content,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
                )

        for idx, (left_id, right_id) in enumerate(self.merges):
            try:
                left = token_strings[left_id]
                right = token_strings[right_id]
            except IndexError as exc:
                raise ValueError(
                    f"Invalid merge pair {(left_id, right_id)} at position {idx}"
                ) from exc
            token_strings.append(left + right)

        expected_vocab = self.base_vocab + len(self.merges)
        if self.vocab_size != expected_vocab:
            raise ValueError(
                "Mismatch between vocab_size and merges: "
                f"expected {expected_vocab}, found {self.vocab_size}"
            )

        vocab = {token: idx for idx, token in enumerate(token_strings)}
        merges = [
            f"{token_strings[left]} {token_strings[right]}"
            for left, right in self.merges
        ]

        tokenizer_config = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": None,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "post_processor": None,
            "decoder": {"type": "ByteLevel"},
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": False,
                "byte_fallback": False,
                "vocab": vocab,
                "merges": merges,
            },
        }

        return vocab, merges, tokenizer_config

    def _write_tokenizer_artifacts(
        self,
        out_dir: str,
        vocab: dict[str, int],
        merges: list[str],
        tokenizer_config: dict[str, object],
    ) -> dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)

        vocab_path = os.path.join(out_dir, "vocab.json")
        merges_path = os.path.join(out_dir, "merges.txt")
        tokenizer_path = os.path.join(out_dir, "tokenizer.json")
        meta_path = os.path.join(out_dir, "bpe_merges.json")

        with open(vocab_path, "w", encoding="utf-8") as handle:
            json.dump(vocab, handle, ensure_ascii=False)

        with open(merges_path, "w", encoding="utf-8") as handle:
            handle.write("#version: 0.2\n")
            for merge in merges:
                handle.write(f"{merge}\n")

        with open(tokenizer_path, "w", encoding="utf-8") as handle:
            json.dump(tokenizer_config, handle, ensure_ascii=False)

        meta = {
            "base_vocab": self.base_vocab,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)

        return {
            "vocab": vocab_path,
            "merges": merges_path,
            "tokenizer": tokenizer_path,
            "metadata": meta_path,
        }

    def export_tokenizer(self, out_dir: str | None = None) -> GPUBPETokenizer:
        """Create a :class:`GPUBPETokenizer` from the current trainer state."""

        vocab, merges, tokenizer_config = self._build_tokenizer_artifacts()
        if out_dir is not None:
            paths = self._write_tokenizer_artifacts(out_dir, vocab, merges, tokenizer_config)
            return GPUBPETokenizer.from_file(paths["tokenizer"])
        if _HFTokenizer is None:
            raise RuntimeError(
                "The `tokenizers` library is required for in-memory export; "
                "provide `out_dir` to persist artifacts instead."
            )
        return GPUBPETokenizer.from_config(tokenizer_config)

    def save(self, out_dir: str) -> None:
        vocab, merges, tokenizer_config = self._build_tokenizer_artifacts()
        paths = self._write_tokenizer_artifacts(out_dir, vocab, merges, tokenizer_config)
        print(
            "Saved tokenizer artifacts → "
            f"{paths['vocab']}, {paths['merges']}, {paths['tokenizer']}"
        )


__all__ = ["GPUBPETokenizer", "GPUBPETrainer"]
