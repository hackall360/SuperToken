"""GPU-accelerated BPE trainer with optional autoscaling."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from .autoscaler import AutoScaler
from .utils import apply_merge_once, count_pairs

def _aggregate_pair_keys(
    keys: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate duplicate 64‑bit pair keys by summing their counts.

    Given a 1‑D tensor of packed pair keys and a 1‑D tensor of counts of the
    same length, this helper will sort the keys, perform a run–length encoding
    on the sorted keys, and sum the corresponding counts for each unique key.
    It returns a tuple of unique keys and their aggregated counts.  Both
    tensors are on the same device as the inputs.

    The packed key format is ``(a_id << 32) | b_id``, so values should be
    64‑bit integers.  See the training loop for how keys are constructed.
    """
    # Ensure inputs are 1‑D and on the same device
    if keys.numel() == 0:
        return keys, counts
    device = keys.device
    # Sort the keys; keep corresponding counts aligned
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    sorted_counts = counts[order]
    # Identify boundaries where the key changes (i.e. new unique starts)
    diff = torch.ones_like(sorted_keys, dtype=torch.bool)
    diff[1:] = sorted_keys[1:] != sorted_keys[:-1]
    # Positions of the first occurrence of each unique key
    pos = torch.nonzero(diff, as_tuple=False).flatten()
    # Compute prefix sums of counts with a leading zero for convenience
    prefix = torch.cat(
        [torch.zeros((1,), dtype=sorted_counts.dtype, device=device), torch.cumsum(sorted_counts, dim=0)]
    )
    # Next boundaries are the subsequent start positions (or the end of the array)
    next_pos = torch.cat(
        [pos[1:], torch.tensor([sorted_keys.numel()], dtype=pos.dtype, device=device)]
    )
    aggregated_counts = prefix[next_pos] - prefix[pos]
    aggregated_keys = sorted_keys[pos]
    return aggregated_keys, aggregated_counts


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


    @classmethod
    def from_cpu(
        cls,
        tokens_cpu: torch.Tensor,
        valid_cpu: torch.Tensor,
        lengths_cpu: torch.Tensor,
        device: torch.device,
    ) -> "GPUBatchRecord":
        tokens_host = tokens_cpu.clone().pin_memory()
        valid_host = valid_cpu.clone().pin_memory()
        lengths_host = lengths_cpu.clone().pin_memory()
        tokens_dev = tokens_host.to(device=device, non_blocking=True)
        valid_dev = valid_host.to(device=device, non_blocking=True)
        lengths_dev = lengths_host.to(device=device, non_blocking=True)
        record = cls(
            tokens=tokens_dev,
            valid=valid_dev,
            lengths=lengths_dev,
            host_tokens=tokens_host,
            host_valid=valid_host,
            host_lengths=lengths_host,
            host_dirty=False,
        )
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
        elif self.pair_workspace is None or self.pair_workspace.shape != (B, width):
            self.pair_workspace = torch.zeros((B, width), dtype=torch.bool, device=device)
        if self.prefix_workspace is None or self.prefix_workspace.shape[0] != B:
            self.prefix_workspace = torch.zeros((B,), dtype=torch.long, device=device)

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

    def wait_for_device(self, stream: torch.cuda.Stream) -> None:
        """Ensure device computations affecting this batch are completed."""

        if self.device_event is not None:
            stream.wait_event(self.device_event)
            self.device_event = None

    def schedule_host_sync(self, copy_stream: torch.cuda.Stream) -> None:
        """Schedule an asynchronous copy of device data back to host."""

        self.ensure_host_buffers()
        event = torch.cuda.Event(blocking=False)
        with torch.cuda.stream(copy_stream):
            self.wait_for_device(copy_stream)
            assert self.host_tokens is not None
            assert self.host_valid is not None
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
        lengths_cpu = self.host_valid.sum(dim=-1, dtype=self.lengths.dtype)
        if self.host_lengths is None or self.host_lengths.shape != lengths_cpu.shape:
            self.host_lengths = lengths_cpu.pin_memory()
        else:
            self.host_lengths.copy_(lengths_cpu)
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
    copy_stream: torch.cuda.Stream
    bytes_h2d: int = 0
    bytes_d2h: int = 0
    h2d_events: int = 0
    d2h_events: int = 0
    active_batches: list[GPUBatchRecord] = field(default_factory=list)

    def reset_activity(self) -> None:
        self.active_batches.clear()


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
            record = GPUBatchRecord.from_cpu(shard_tokens, shard_valid, shard_lengths, device)
            record_transfer(contexts[device], record)
            shards[device] = record
            start = end
        return cls(shards)

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
        self.autoscaler = autoscaler or AutoScaler()
        self.sync_every = max(sync_every, 1)
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

    # ------------------------------------------------------------------
    # Transfer accounting helpers
    def _reset_transfer_counters(self) -> None:
        self.bytes_h2d = 0
        self.bytes_d2h = 0
        self.h2d_events = 0
        self.d2h_events = 0
        self.merge_transfer_log = []
        self.sync_intervals = []
        self._interval_merges = 0
        for ctx in self._device_contexts.values():
            ctx.bytes_h2d = 0
            ctx.bytes_d2h = 0
            ctx.h2d_events = 0
            ctx.d2h_events = 0
            ctx.reset_activity()

    def _record_h2d(
        self, *tensors: torch.Tensor, device: torch.device | None = None
    ) -> None:
        if not tensors:
            return
        total = sum(int(t.nbytes) for t in tensors)
        self.bytes_h2d += total
        self.h2d_events += 1
        if device is not None:
            ctx = self._device_contexts.get(device)
            if ctx is not None:
                ctx.bytes_h2d += total
                ctx.h2d_events += 1

    def _record_d2h(
        self, *tensors: torch.Tensor, device: torch.device | None = None
    ) -> int:
        if not tensors:
            return 0
        total = sum(int(t.nbytes) for t in tensors)
        self.bytes_d2h += total
        self.d2h_events += 1
        if device is not None:
            ctx = self._device_contexts.get(device)
            if ctx is not None:
                ctx.bytes_d2h += total
                ctx.d2h_events += 1
        return total

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

    def fit(
        self,
        batches: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        log_every: int = 100,
        profile_streams: bool = False,
        on_batch_size_change: Optional[Callable[[int], None]] = None,
    ) -> dict[str, object]:
        """Train merges using a pipelined GPU workflow when possible."""

        self._reset_transfer_counters()
        self._active_batch_size = None
        step = 0
        current_batches: Iterable[
            Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], MultiDeviceBatch]
        ] = batches
        gpu_batches: Optional[list[MultiDeviceBatch]] = None
        scale_state = None

        cpu_device = torch.device("cpu")

        def _extract_sequences_from_cpu(
            cpu_batches: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ) -> list[list[int]]:
            sequences: list[list[int]] = []
            for tokens_cpu, _valid_cpu, lengths_cpu in cpu_batches:
                rows = int(tokens_cpu.shape[0])
                for row in range(rows):
                    length = int(lengths_cpu[row].item())
                    if length <= 0:
                        sequences.append([])
                        continue
                    seq = tokens_cpu[row, :length].to(torch.long).tolist()
                    sequences.append(seq)
            return sequences

        def _extract_sequences_from_gpu(
            records: Iterable[MultiDeviceBatch],
        ) -> list[list[int]]:
            sequences: list[list[int]] = []
            for record in records:
                sequences.extend(record.sequences())
            return sequences

        def _iter_cpu_batches_from_sequences(
            sequences: list[list[int]],
            batch_size: int,
            pin: bool,
        ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            if batch_size <= 0:
                return
            for start in range(0, len(sequences), batch_size):
                chunk = sequences[start : start + batch_size]
                if not chunk:
                    continue
                rows = len(chunk)
                max_len = max((len(seq) for seq in chunk), default=0)
                width = max(1, max_len)
                tokens = torch.full((rows, width), -1, dtype=torch.long)
                valid = torch.zeros((rows, width), dtype=torch.long)
                lengths = torch.zeros((rows,), dtype=torch.long)
                for row, seq in enumerate(chunk):
                    if not seq:
                        continue
                    L = len(seq)
                    tokens[row, :L] = torch.tensor(seq, dtype=torch.long)
                    valid[row, :L] = 1
                    lengths[row] = L
                if pin:
                    tokens = tokens.pin_memory()
                    valid = valid.pin_memory()
                yield tokens, valid, lengths

        def _collect_sequences_from_mixed(
            batches_iter: Iterable[
                Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], MultiDeviceBatch]
            ]
        ) -> list[list[int]]:
            sequences: list[list[int]] = []
            cpu_batches: list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            gpu_records: list[MultiDeviceBatch] = []
            for batch in batches_iter:
                if isinstance(batch, MultiDeviceBatch):
                    gpu_records.append(batch)
                else:
                    cpu_batches.append(batch)
            if gpu_records:
                sequences.extend(_extract_sequences_from_gpu(gpu_records))
            if cpu_batches:
                sequences.extend(_extract_sequences_from_cpu(cpu_batches))
            return sequences

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
                    copy_stream=torch.cuda.Stream(device=dev),
                )
            self._device_contexts = device_contexts
            for ctx in device_contexts.values():
                ctx.bytes_h2d = 0
                ctx.bytes_d2h = 0
                ctx.h2d_events = 0
                ctx.d2h_events = 0
                ctx.reset_activity()
        else:
            self._device_contexts = {}

        device = torch.device(self.device)

        def _record_new_batch(ctx: DeviceContext, record: GPUBatchRecord) -> None:
            self._record_h2d(record.tokens, record.valid, record.lengths, device=ctx.device)

        def _register_batch(batch: MultiDeviceBatch) -> None:
            for dev, record in batch.iter_shards():
                ctx = device_contexts.get(dev)
                if ctx is not None:
                    ctx.active_batches.append(record)

        def _mark_active_batches(batches_to_mark: Iterable[MultiDeviceBatch]) -> None:
            for ctx in device_contexts.values():
                ctx.reset_activity()
            for batch in batches_to_mark:
                _register_batch(batch)

        def _build_gpu_batches_from_sequences(
            sequences: list[list[int]], batch_size: int
        ) -> list[MultiDeviceBatch]:
            new_records: list[MultiDeviceBatch] = []
            if batch_size <= 0 or not sequences or not use_cuda:
                return new_records
            for tokens_cpu, valid_cpu, lengths_cpu in _iter_cpu_batches_from_sequences(
                sequences, batch_size, pin=True
            ):
                multi_batch = MultiDeviceBatch.from_cpu_batch(
                    tokens_cpu, valid_cpu, lengths_cpu, device_contexts, _record_new_batch
                )
                new_records.append(multi_batch)
            _mark_active_batches(new_records)
            return new_records

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
                new_records = _build_gpu_batches_from_sequences(sequences, batch_size)
                gpu_batches = new_records
                current_batches = new_records
            else:
                cpu_batches: list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
                if not isinstance(current_batches, list):
                    current_batches = list(current_batches)
                sequences = _extract_sequences_from_cpu(current_batches)
                for cpu_batch in _iter_cpu_batches_from_sequences(
                    sequences, batch_size, pin=False
                ):
                    cpu_batches.append(cpu_batch)
                current_batches = cpu_batches
            if on_batch_size_change is not None:
                on_batch_size_change(batch_size)

        def _count_pairs_cpu(
            batch_iter: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
            global_keys: Optional[torch.Tensor] = None
            global_counts: Optional[torch.Tensor] = None
            consumed: list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for (x_cpu, v_cpu, lengths_cpu) in batch_iter:
                x_snapshot = x_cpu.clone()
                v_snapshot = v_cpu.clone()
                lengths_snapshot = lengths_cpu.clone()
                consumed.append((x_snapshot, v_snapshot, lengths_snapshot))
                pairs, counts = count_pairs(x_snapshot.to(cpu_device), v_snapshot.to(cpu_device))
                if pairs.numel() == 0:
                    continue
                a_ids = pairs[:, 0].to(torch.long)
                b_ids = pairs[:, 1].to(torch.long)
                keys = (a_ids << 32) | b_ids
                if global_keys is None:
                    global_keys = keys
                    global_counts = counts.to(torch.long)
                else:
                    global_keys = torch.cat([global_keys, keys], dim=0)
                    global_counts = torch.cat([global_counts, counts.to(torch.long)], dim=0)
            return global_keys, global_counts, consumed

        def _apply_merge_cpu(
            batch_iter: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            a_id: int,
            b_id: int,
            new_id: int,
        ) -> tuple[list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], bool, dict[str, object]]:
            new_batches: list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            oom_seen = False
            pair_workspace: Optional[torch.Tensor] = None
            prefix_workspace: Optional[torch.Tensor] = None
            for (x_cpu, v_cpu, lengths_cpu) in batch_iter:
                try:
                    B, L = x_cpu.shape
                    width = max(L - 1, 0)
                    if pair_workspace is None or pair_workspace.shape != (B, width):
                        pair_workspace = torch.zeros((B, width), dtype=torch.bool, device=cpu_device)
                    if prefix_workspace is None or prefix_workspace.shape[0] != B:
                        prefix_workspace = torch.zeros((B,), dtype=torch.long, device=cpu_device)
                    apply_merge_once(
                        x_cpu,
                        v_cpu,
                        lengths_cpu,
                        a_id,
                        b_id,
                        new_id,
                        pair_workspace,
                        prefix_workspace,
                    )
                    new_batches.append((x_cpu.clone(), v_cpu.clone(), lengths_cpu.clone()))
                except RuntimeError as exc:
                    if "CUDA out of memory" in str(exc):
                        oom_seen = True
                        torch.cuda.empty_cache()
                        new_batches.append((x_cpu, v_cpu, lengths_cpu))
                    else:
                        raise
            return new_batches, oom_seen, {"sync": True, "bytes": 0}

        if use_cuda:
            def _finalize_host_sync(records: list[MultiDeviceBatch]) -> int:
                copied_bytes = 0
                pending_copy = False
                _mark_active_batches(records)
                for batch in records:
                    for dev, shard in batch.iter_shards():
                        ctx = device_contexts[dev]
                        if shard.host_dirty:
                            copied_bytes += self._record_d2h(
                                shard.tokens, shard.valid, device=dev
                            )
                            with torch.cuda.device(dev):
                                shard.schedule_host_sync(ctx.copy_stream)
                            pending_copy = True
                        elif shard.host_event is not None:
                            pending_copy = True
                for dev, ctx in device_contexts.items():
                    torch.cuda.current_stream(device=dev).wait_stream(ctx.copy_stream)
                if profile_streams and pending_copy:
                    for dev, ctx in device_contexts.items():
                        torch.cuda.set_device(dev)
                        torch.cuda.synchronize(dev)
                        assert ctx.copy_stream.query(), "copy stream did not drain"
                for batch in records:
                    for _dev, shard in batch.iter_shards():
                        shard.resolve_host()
                return copied_bytes

            def _count_pairs_gpu(
                batch_iter: Iterable[
                    Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], MultiDeviceBatch]
                ]
            ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], list[MultiDeviceBatch]]:
                nonlocal current_batches, gpu_batches, scale_state
                while True:
                    for ctx in device_contexts.values():
                        ctx.reset_activity()
                    global_keys: Optional[torch.Tensor] = None
                    global_counts: Optional[torch.Tensor] = None
                    consumed: list[MultiDeviceBatch] = []
                    pair_results: Dict[torch.device, list[tuple[torch.Tensor, torch.Tensor]]] = {
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
                                        pairs, counts = count_pairs(shard.tokens, shard.valid)
                                    pair_results[dev].append((pairs, counts))
                                    pending_compute[dev] = True
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
                        sequences = _extract_sequences_from_gpu(consumed) if consumed else []
                        if failed_cpu_batch is not None:
                            sequences.extend(_extract_sequences_from_cpu([failed_cpu_batch]))
                        if tail_batches:
                            sequences.extend(_collect_sequences_from_mixed(tail_batches))
                        prev_bs = scale_state.batch_size if scale_state is not None else None
                        self.autoscaler.feedback(oom=True)
                        scale_state = self.autoscaler.state or scale_state
                        if scale_state is None:
                            raise RuntimeError("Autoscaler did not provide a state after OOM")
                        new_bs = scale_state.batch_size
                        if prev_bs is not None and new_bs >= prev_bs:
                            raise RuntimeError(
                                "CUDA OOM could not be resolved by autoscaler batch size reduction"
                            )
                        new_gpu_batches = _build_gpu_batches_from_sequences(sequences, new_bs)
                        gpu_batches = new_gpu_batches
                        current_batches = new_gpu_batches
                        batch_iter = current_batches
                        if on_batch_size_change is not None:
                            on_batch_size_change(new_bs)
                        self._active_batch_size = new_bs
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
                    for per_device_results in pair_results.values():
                        for pairs, counts in per_device_results:
                            if pairs.numel() == 0:
                                continue
                            a_ids = pairs[:, 0].to(torch.long)
                            b_ids = pairs[:, 1].to(torch.long)
                            keys = (a_ids << 32) | b_ids
                            if global_keys is None:
                                global_keys = keys
                                global_counts = counts.to(torch.long)
                            else:
                                global_keys = torch.cat([global_keys, keys], dim=0)
                                global_counts = torch.cat([global_counts, counts.to(torch.long)], dim=0)
                    gpu_batches = consumed
                    return global_keys, global_counts, consumed

            def _apply_merge_gpu(
                batch_iter: Iterable[MultiDeviceBatch],
                a_id: int,
                b_id: int,
                new_id: int,
                merge_idx: int,
                force_sync: bool = False,
            ) -> tuple[list[MultiDeviceBatch], bool, dict[str, object]]:
                nonlocal current_batches, gpu_batches, scale_state
                oom_seen = False
                while True:
                    records = list(batch_iter)
                    _mark_active_batches(records)
                    need_sync = force_sync or (merge_idx % self.sync_every == 0)
                    pending_copy: Dict[torch.device, bool] = {
                        dev: False for dev in device_contexts
                    }
                    pending_compute: Dict[torch.device, bool] = {
                        dev: False for dev in device_contexts
                    }
                    copied_bytes = 0
                    retry_required = False
                    for batch in records:
                        for dev, shard in batch.iter_shards():
                            ctx = device_contexts[dev]
                            try:
                                with torch.cuda.device(dev), torch.cuda.stream(ctx.compute_stream):
                                    shard.wait_for_device(ctx.compute_stream)
                                    shard.ensure_workspaces()
                                    apply_merge_once(
                                        shard.tokens,
                                        shard.valid,
                                        shard.lengths,
                                        a_id,
                                        b_id,
                                        new_id,
                                        shard.pair_workspace,
                                        shard.prefix_workspace,
                                    )
                                    compute_event = torch.cuda.Event(blocking=False)
                                    compute_event.record(ctx.compute_stream)
                                shard.mark_device_event(compute_event)
                                pending_compute[dev] = True
                                if need_sync:
                                    copied_bytes += self._record_d2h(
                                        shard.tokens, shard.valid, device=dev
                                    )
                                    with torch.cuda.device(dev):
                                        shard.schedule_host_sync(ctx.copy_stream)
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
                        sequences = _extract_sequences_from_gpu(records)
                        prev_bs = scale_state.batch_size if scale_state is not None else None
                        self.autoscaler.feedback(oom=True)
                        scale_state = self.autoscaler.state or scale_state
                        if scale_state is None:
                            raise RuntimeError("Autoscaler did not provide a state after OOM")
                        new_bs = scale_state.batch_size
                        if prev_bs is not None and new_bs >= prev_bs:
                            raise RuntimeError(
                                "CUDA OOM could not be resolved by autoscaler batch size reduction"
                            )
                        new_gpu_batches = _build_gpu_batches_from_sequences(sequences, new_bs)
                        gpu_batches = new_gpu_batches
                        current_batches = new_gpu_batches
                        batch_iter = current_batches
                        if on_batch_size_change is not None:
                            on_batch_size_change(new_bs)
                        self._active_batch_size = new_bs
                        continue
                    for dev, ctx in device_contexts.items():
                        torch.cuda.current_stream(device=dev).wait_stream(ctx.compute_stream)
                    if need_sync:
                        for dev, ctx in device_contexts.items():
                            torch.cuda.current_stream(device=dev).wait_stream(ctx.copy_stream)
                        for batch in records:
                            for _dev, shard in batch.iter_shards():
                                shard.resolve_host()
                    if profile_streams:
                        for dev, ctx in device_contexts.items():
                            compute_pending = pending_compute.get(dev, False)
                            copy_pending = pending_copy.get(dev, False) and need_sync
                            if compute_pending or copy_pending:
                                torch.cuda.set_device(dev)
                                torch.cuda.synchronize(dev)
                                if copy_pending:
                                    assert ctx.copy_stream.query(), "copy stream did not drain"
                                if compute_pending:
                                    assert ctx.compute_stream.query(), "compute stream did not drain"
                    return records, oom_seen, {"sync": need_sync and copied_bytes > 0, "bytes": copied_bytes}

        while step < self.target_merges:
            scale_state = self.autoscaler.suggest(token_bytes_per_example=int(8 * 1024))
            if self._active_batch_size is None:
                self._active_batch_size = scale_state.batch_size
            elif scale_state.batch_size != self._active_batch_size:
                _reconfigure_batches(scale_state.batch_size)
                self._active_batch_size = scale_state.batch_size
            t0 = time.time()
            if use_cuda:
                global_keys, global_counts, consumed_batches = _count_pairs_gpu(current_batches)
                gpu_batches = consumed_batches
            else:
                global_keys, global_counts, consumed_batches = _count_pairs_cpu(current_batches)
            if global_keys is None or global_keys.numel() == 0:
                print("No pairs left to merge.")
                break
            current_batches = consumed_batches
            agg_keys, agg_counts = _aggregate_pair_keys(global_keys, global_counts)
            best_idx = torch.argmax(agg_counts)
            best_key = agg_keys[best_idx]
            a_id = int((best_key >> 32).item())
            b_id = int((best_key & ((1 << 32) - 1)).item())
            best_count = int(agg_counts[best_idx].item())
            if best_count <= 1:
                print("Stopping: no frequent pairs.")
                break
            new_id = self.vocab_size
            if step % log_every == 0:
                print(f"merge {step:6d}: ({a_id},{b_id}) -> {new_id}  count={best_count}")
            self.merges.append((a_id, b_id))
            self.vocab_size += 1
            if use_cuda:
                current_batches, oom_seen, sync_report = _apply_merge_gpu(
                    current_batches, a_id, b_id, new_id, step + 1
                )
                gpu_batches = list(current_batches)
                self._interval_merges += 1
                if sync_report.get("sync"):
                    self._close_sync_interval(step + 1, sync_report.get("bytes", 0))
            else:
                current_batches, oom_seen, _ = _apply_merge_cpu(current_batches, a_id, b_id, new_id)
            step += 1
            if not use_cuda:
                self._interval_merges = 0
            self._record_merge_snapshot(step)
            self.autoscaler.feedback(step_time_s=time.time() - t0, oom=oom_seen)
        if use_cuda and gpu_batches is not None:
            final_bytes = _finalize_host_sync(gpu_batches)
            if self._interval_merges > 0:
                self._close_sync_interval(step, final_bytes)
        per_device_metrics = {
            str(device): {
                "bytes_h2d": ctx.bytes_h2d,
                "bytes_d2h": ctx.bytes_d2h,
                "h2d_events": ctx.h2d_events,
                "d2h_events": ctx.d2h_events,
            }
            for device, ctx in self._device_contexts.items()
        }
        for ctx in self._device_contexts.values():
            ctx.reset_activity()
        return {
            "base_vocab": self.base_vocab,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "transfer_metrics": {
                "bytes_h2d": self.bytes_h2d,
                "bytes_d2h": self.bytes_d2h,
                "h2d_events": self.h2d_events,
                "d2h_events": self.d2h_events,
                "merge_stats": self.merge_transfer_log,
                "sync_intervals": self.sync_intervals,
                "per_device": per_device_metrics,
                "avg_d2h_bytes_per_merge": (
                    self.bytes_d2h / len(self.merge_transfer_log)
                    if self.merge_transfer_log
                    else 0.0
                ),
            },
        }

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "base_vocab": self.base_vocab,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }
        with open(os.path.join(out_dir, "bpe_merges.json"), "w") as f:
            json.dump(meta, f)
        print(f"Saved merges → {out_dir}/bpe_merges.json")


__all__ = ["GPUBPETrainer"]
