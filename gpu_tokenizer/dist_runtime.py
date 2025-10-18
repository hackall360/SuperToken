"""Distributed runtime configuration and helpers for GPU tokenization workers."""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import logging
import math
import os
import queue
import signal
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from types import FrameType
from collections import deque
from typing import (
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

import time

try:  # pragma: no cover - torch is optional in some environments
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
except Exception:  # pragma: no cover - fallback when torch missing
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    mp = None  # type: ignore[assignment]

try:  # pragma: no cover - optional type for static analysis
    from torch.multiprocessing.spawn import ProcessContext
except Exception:  # pragma: no cover - fallback when torch missing
    ProcessContext = object  # type: ignore[misc,assignment]

from . import utils
from .io import make_chunker
from .lease_queue import LeaseNotary

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .trainers.metrics import TrainerMetricsEWMA

# Configure a module level logger. Downstream applications can adjust the
# configuration to taste but this ensures we at least emit something when the
# module is used in isolation.
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:  # pragma: no cover - configuration guard
    logging.basicConfig(level=logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


SignalHandler = Callable[[int, Optional[FrameType]], None]
_SHUTDOWN_SIGNALS: Tuple[int, int] = (signal.SIGINT, signal.SIGTERM)


@dataclass
class _LeaseHostState:
    """Container storing the shared :class:`LeaseNotary` for a job."""

    notary: LeaseNotary
    lock: threading.Lock

    @property
    def total_chunks(self) -> int:
        return self.notary.total_chunks


_LEASE_REGISTRY_LOCK = threading.Lock()
_LEASE_REGISTRY: Dict[str, _LeaseHostState] = {}

_STARTUP_SAMPLE_LEASES = 2
_REBALANCE_BLEND = 0.5


def _normalize_job_id(job_id: str | None) -> str:
    if not job_id:
        return "default"
    return job_id


def _get_or_create_host_state(
    job_id: str, total_chunks: int, *, max_active_leases: int | None = None
) -> _LeaseHostState:
    if total_chunks < 0:
        raise ValueError("total_chunks must be non-negative")
    if max_active_leases is not None and max_active_leases <= 0:
        raise ValueError("max_active_leases must be positive when provided")
    with _LEASE_REGISTRY_LOCK:
        state = _LEASE_REGISTRY.get(job_id)
        if state is None:
            kwargs: dict[str, object] = {"total_chunks": total_chunks}
            if max_active_leases is not None:
                kwargs["max_active_leases"] = max_active_leases
            notary = LeaseNotary(**kwargs)
            state = _LeaseHostState(notary=notary, lock=threading.Lock())
            _LEASE_REGISTRY[job_id] = state
        elif state.total_chunks != total_chunks:
            raise ValueError(
                f"Existing lease registry for job {job_id!r} was initialised with "
                f"{state.total_chunks} chunks but {total_chunks} were requested"
            )
        elif max_active_leases is not None:
            state.notary.update_max_active_leases(max_active_leases)
    return state

@dataclass(frozen=True)
class ChunkLeaseInfo:
    """Describe a leased chunk handed to a worker."""

    start: int
    end: int
    chunk_id: int
    attempts: int
    completed: bool

    @property
    def width(self) -> int:
        return max(0, self.end - self.start)

    @property
    def reprocessed(self) -> bool:
        return self.attempts > 1


class DistributedLeaseClient:
    """Thin wrapper around :class:`LeaseNotary` shared across distributed ranks."""

    def __init__(
        self,
        *,
        job_id: str,
        rank: int,
        world_size: int,
        host_state: _LeaseHostState,
        root: int = 0,
    ) -> None:
        self.job_id = job_id
        self.rank = rank
        self.world_size = world_size
        self._host_state = host_state
        self._root = root
        self._idle_accumulator = 0.0

    @property
    def total_chunks(self) -> int:
        return self._host_state.total_chunks

    def _dist_ready(self) -> bool:
        return (
            dist is not None
            and hasattr(dist, "broadcast_object_list")
            and dist.is_available()
            and dist.is_initialized()
            and self.world_size > 1
        )

    def _broadcast_event(self, event: str, **payload: object) -> None:
        if not self._dist_ready():
            return
        message = [{"event": event, "rank": self.rank, **payload}]
        try:
            dist.broadcast_object_list(message, src=self.rank)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - best effort notification
            logger.debug(
                "lease_client_broadcast_failed",
                extra={"event": event, "rank": self.rank},
                exc_info=True,
            )

    def request_lease(self, preferred_size: int) -> Optional[Tuple[int, int, int]]:
        if preferred_size <= 0:
            raise ValueError("preferred_size must be positive")
        with self._host_state.lock:
            return self._host_state.notary.grant_lease(self.rank, preferred_size)

    def complete_lease(self, start: int, end: int, chunk_id: int) -> None:
        with self._host_state.lock:
            self._host_state.notary.complete_lease(self.rank, start, end, chunk_id)
        self._broadcast_event("complete", lease=(int(start), int(end), int(chunk_id)))

    def requeue_lease(self, start: int, end: int, chunk_id: int) -> None:
        with self._host_state.lock:
            self._host_state.notary.requeue_lease(self.rank, start, end, chunk_id)
        self._broadcast_event("requeue", lease=(int(start), int(end), int(chunk_id)))

    def heartbeat(self) -> float:
        with self._host_state.lock:
            return self._host_state.notary.heartbeat(self.rank)

    def requeue_outstanding(self) -> Optional[Tuple[int, int, int]]:
        """Best-effort requeue of the inflight lease for this rank."""

        with self._host_state.lock:
            snapshot = self._host_state.notary.state_dict()
        inflight_entry = snapshot["inflight"].get(self.rank)
        if not inflight_entry:
            return None
        raw_records: List[Tuple[int, int, int]] = []
        if isinstance(inflight_entry, dict):
            if "records" in inflight_entry and isinstance(inflight_entry["records"], list):
                for entry in inflight_entry["records"]:
                    if isinstance(entry, dict) and "lease" in entry:
                        lease = entry["lease"]
                        if isinstance(lease, (list, tuple)):
                            if len(lease) >= 3:
                                start, end, chunk_id = lease[:3]
                            else:
                                start, end = lease[:2]
                                chunk_id = entry.get("chunk_id", 0)
                            raw_records.append((int(start), int(end), int(chunk_id)))
            elif "lease" in inflight_entry:
                lease = inflight_entry["lease"]
                if isinstance(lease, (list, tuple)):
                    if len(lease) >= 3:
                        start, end, chunk_id = lease[:3]
                    else:
                        start, end = lease[:2]
                        chunk_id = inflight_entry.get("chunk_id", 0)
                    raw_records.append((int(start), int(end), int(chunk_id)))
        elif isinstance(inflight_entry, list):
            for entry in inflight_entry:
                if isinstance(entry, dict) and "lease" in entry:
                    lease = entry["lease"]
                    if isinstance(lease, (list, tuple)):
                        if len(lease) >= 3:
                            start, end, chunk_id = lease[:3]
                        else:
                            start, end = lease[:2]
                            chunk_id = entry.get("chunk_id", 0)
                        raw_records.append((int(start), int(end), int(chunk_id)))
        if not raw_records:
            return None
        leases: List[Tuple[int, int, int]] = list(raw_records)
        try:
            with self._host_state.lock:
                for start, end, chunk_id in leases:
                    self._host_state.notary.requeue_lease(
                        self.rank, int(start), int(end), int(chunk_id)
                    )
        except Exception:
            logger.debug(
                "lease_requeue_on_shutdown_failed",
                extra={"rank": self.rank, "leases": leases},
                exc_info=True,
            )
            return None
        for start, end, chunk_id in leases:
            self._broadcast_event("requeue", lease=(int(start), int(end), int(chunk_id)))
        return leases[0]

    def describe_chunk(self, chunk_id: int) -> Optional[Dict[str, object]]:
        """Return metadata describing ``chunk_id`` if known."""

        return self._host_state.notary.chunk_status(int(chunk_id))

    def update_rank_weights(self, weights: Dict[int, float]) -> None:
        """Persist *weights* into the shared :class:`LeaseNotary`."""

        with self._host_state.lock:
            self._host_state.notary.update_rank_weights(weights)

    def rank_weights(self) -> Dict[int, float]:
        """Return the cached per-rank weights from the notary."""

        with self._host_state.lock:
            return self._host_state.notary.rank_weights()

    def record_idle(self, duration_s: float) -> None:
        """Record *duration_s* seconds of idle time for this rank."""

        try:
            idle = float(duration_s)
        except (TypeError, ValueError):
            return
        if idle <= 0.0 or not math.isfinite(idle):
            return
        self._idle_accumulator += idle
        with self._host_state.lock:
            self._host_state.notary.record_idle(self.rank, idle)

    def drain_idle_time(self) -> float:
        """Return and reset the accumulated idle time since last feedback."""

        idle = self._idle_accumulator
        self._idle_accumulator = 0.0
        return idle

    def feedback(
        self,
        completed_tokens: int,
        *,
        duration_s: float | None = None,
        idle_duration_s: float | None = None,
    ) -> Dict[str, float]:
        """Propagate throughput feedback to the shared :class:`LeaseNotary`."""

        tokens = max(0, int(completed_tokens))
        idle = idle_duration_s if idle_duration_s is not None else self.drain_idle_time()
        self._idle_accumulator = 0.0
        with self._host_state.lock:
            return self._host_state.notary.apply_feedback(
                self.rank,
                completed_tokens=tokens,
                duration_s=duration_s,
                idle_duration_s=idle if idle is not None else 0.0,
            )

    def lease_status(self) -> Dict[int, Dict[str, float]]:
        """Return the current adaptive lease telemetry for all ranks."""

        with self._host_state.lock:
            return self._host_state.notary.rank_status()

    def iter_leases(self, preferred_size: int) -> Iterator[Tuple[int, int, int]]:
        while True:
            lease = self.request_lease(preferred_size)
            if lease is None:
                break
            yield lease


def register_lease_client(
    *,
    job_id: str | None,
    total_chunks: int,
    rank: int,
    world_size: int,
    root: int = 0,
    max_active_leases: int | None = None,
) -> DistributedLeaseClient:
    """Return a :class:`DistributedLeaseClient` registered with rank ``root``."""

    normalized_job = _normalize_job_id(job_id)

    should_broadcast = (
        dist is not None
        and hasattr(dist, "broadcast_object_list")
        and dist.is_available()
        and dist.is_initialized()
        and world_size > 1
    )

    if should_broadcast:
        payload: List[object] = [
            {"job_id": normalized_job, "total_chunks": int(total_chunks)}
            if rank == root
            else None
        ]
        dist.broadcast_object_list(payload, src=root)  # type: ignore[arg-type]
        info = payload[0]
        if isinstance(info, dict):
            normalized_job = _normalize_job_id(str(info.get("job_id")))
            total_chunks = int(info.get("total_chunks", total_chunks))

    host_state = _get_or_create_host_state(
        normalized_job, int(total_chunks), max_active_leases=max_active_leases
    )
    return DistributedLeaseClient(
        job_id=normalized_job,
        rank=rank,
        world_size=world_size,
        host_state=host_state,
        root=root,
    )


def _measure_startup_throughput(
    lease_client: Optional[DistributedLeaseClient],
    metrics: "TrainerMetricsEWMA | None",
    sample_leases: int = _STARTUP_SAMPLE_LEASES,
) -> Optional[float]:
    """Return a throughput estimate derived from ``metrics``."""

    if metrics is None or not getattr(metrics, "enabled", False):
        return None

    processed = 0
    start_ts = time.perf_counter()
    if lease_client is not None and sample_leases > 0:
        sample_budget = min(sample_leases, max(0, lease_client.total_chunks))
        for _ in range(sample_budget):
            lease = lease_client.request_lease(1)
            if lease is None:
                break
            start, end, _chunk_id = lease
            width = max(0, int(end) - int(start))
            processed += width
            try:
                lease_client.requeue_lease(*lease)
            except Exception:
                logger.debug("startup_lease_requeue_failed", exc_info=True)
                break

    elapsed = time.perf_counter() - start_ts
    if processed > 0 and elapsed > 0.0:
        try:
            metrics.record_tokens(tokens=processed, duration_s=elapsed, leases=processed)
        except Exception:
            logger.debug("startup_metrics_record_failed", exc_info=True)

    sample = getattr(metrics, "tokens_per_s", None)
    if sample is None:
        return None
    try:
        return float(sample)
    except (TypeError, ValueError):
        return None


def _normalise_throughput_samples(samples: Sequence[Optional[float]]) -> Dict[int, float]:
    """Convert raw throughput measurements into normalised weights."""

    cleaned: List[float] = []
    for value in samples:
        if value is None:
            cleaned.append(0.0)
            continue
        try:
            cleaned.append(max(0.0, float(value)))
        except (TypeError, ValueError):
            cleaned.append(0.0)

    world_size = len(cleaned)
    if world_size == 0:
        return {}

    total = sum(cleaned)
    if total <= 0.0:
        uniform = 1.0 / float(world_size)
        return {idx: uniform for idx in range(world_size)}

    return {idx: (value / total) for idx, value in enumerate(cleaned)}


def _compute_rebalance_weights(
    snapshots: Sequence[object],
    *,
    metrics: "TrainerMetricsEWMA | None",
    world_size: int,
) -> Dict[int, float]:
    """Derive normalised rank weights from gathered EWMA snapshots."""

    values: List[Optional[float]] = []
    for idx in range(world_size):
        entry: object | None = snapshots[idx] if idx < len(snapshots) else None
        snapshot: Mapping[str, object] | None = entry if isinstance(entry, Mapping) else None
        if snapshot is not None and metrics is not None and hasattr(metrics, "update_rank_snapshot"):
            try:
                metrics.update_rank_snapshot(snapshot)
            except Exception:
                logger.debug("rebalance_snapshot_merge_failed", exc_info=True)
        sample_val: Optional[float]
        if snapshot is not None:
            token_obj = snapshot.get("tokens_per_s")
            try:
                sample_val = float(token_obj) if token_obj is not None else 0.0
            except (TypeError, ValueError):
                sample_val = 0.0
        else:
            sample_val = 0.0
        values.append(sample_val)
    return _normalise_throughput_samples(values)


def _blend_rank_weights(
    existing: Mapping[int, float],
    updated: Mapping[int, float],
    *,
    new_fraction: float,
    world_size: int,
) -> Dict[int, float]:
    """Blend previous rank weights with freshly computed updates."""

    fraction = max(0.0, min(1.0, float(new_fraction)))
    keys = {int(key) for key in existing.keys()} | {int(key) for key in updated.keys()}
    keys |= {int(idx) for idx in range(world_size)}
    blended: Dict[int, float] = {}
    for key in sorted(keys):
        new_val = float(updated.get(key, 0.0))
        if not existing:
            old_val = new_val
        else:
            old_val = float(existing.get(key, new_val))
        blended_val = ((1.0 - fraction) * old_val) + (fraction * new_val)
        blended[key] = blended_val
    total = sum(blended.values())
    if total > 0.0:
        return {key: value / total for key, value in blended.items()}
    if world_size <= 0:
        return dict(blended)
    uniform = 1.0 / float(world_size)
    return {key: uniform for key in blended.keys()}


def _rebalance_once(
    *,
    rank: int,
    world_size: int,
    metrics: "TrainerMetricsEWMA | None",
    lease_client: Optional[DistributedLeaseClient],
    blend: float = _REBALANCE_BLEND,
) -> None:
    """Execute a single rebalance round collecting EWMA throughput samples."""

    if (
        dist is None
        or not hasattr(dist, "gather_object")
        or not hasattr(dist, "broadcast_object_list")
        or not dist.is_available()
        or not dist.is_initialized()
        or world_size <= 1
    ):
        return

    sample: object
    if metrics is not None and hasattr(metrics, "snapshot"):
        try:
            sample = metrics.snapshot()
        except Exception:
            logger.debug("rebalance_snapshot_failed", exc_info=True)
            sample = {"rank": rank}
    else:
        sample = {"rank": rank}

    gather_list: List[object] | None = [None] * world_size if rank == 0 else None
    try:
        dist.gather_object(sample, gather_list, dst=0)  # type: ignore[arg-type]
    except Exception:
        logger.debug("rebalance_gather_failed", exc_info=True)
        gather_list = None

    weights: Dict[int, float] | None = None
    if rank == 0 and gather_list is not None:
        weights = _compute_rebalance_weights(
            gather_list,
            metrics=metrics,
            world_size=world_size,
        )
        if weights and lease_client is not None:
            try:
                existing = lease_client.rank_weights()
            except Exception:
                logger.debug("rebalance_weight_fetch_failed", exc_info=True)
                existing = {}
            blended = _blend_rank_weights(
                existing,
                weights,
                new_fraction=blend,
                world_size=world_size,
            )
            try:
                lease_client.update_rank_weights(blended)
            except Exception:
                logger.debug("rebalance_weight_update_failed", exc_info=True)
    payload: List[object] = [weights]
    try:
        dist.broadcast_object_list(payload, src=0)  # type: ignore[arg-type]
    except Exception:
        logger.debug("rebalance_broadcast_failed", exc_info=True)
        return

    broadcast_weights = payload[0]
    if (
        lease_client is not None
        and isinstance(broadcast_weights, dict)
    ):
        try:
            lease_client.update_rank_weights({int(k): float(v) for k, v in broadcast_weights.items()})
        except Exception:
            logger.debug("rebalance_weight_apply_failed", exc_info=True)


def _rebalance_loop(
    rank: int,
    world_size: int,
    metrics: "TrainerMetricsEWMA | None",
    lease_client: Optional[DistributedLeaseClient],
    rebalance_secs: float,
    stop_event: threading.Event,
    *,
    blend: float = _REBALANCE_BLEND,
) -> None:
    """Background loop that periodically rebalances lease weights."""

    if rebalance_secs <= 0.0:
        return

    interval = float(rebalance_secs)
    while not stop_event.is_set():
        start = time.monotonic()
        try:
            _rebalance_once(
                rank=rank,
                world_size=world_size,
                metrics=metrics,
                lease_client=lease_client,
                blend=blend,
            )
        except Exception:
            logger.debug("rebalance_iteration_failed", exc_info=True)
        elapsed = time.monotonic() - start
        remaining = max(0.0, interval - elapsed)
        if stop_event.wait(remaining):
            break


def _collect_startup_throughput_samples(
    *,
    rank: int,
    world_size: int,
    lease_client: Optional[DistributedLeaseClient],
    metrics: "TrainerMetricsEWMA | None",
) -> None:
    """Gather per-rank throughput samples and persist normalised weights."""

    if (
        dist is None
        or not hasattr(dist, "gather_object")
        or not hasattr(dist, "broadcast_object_list")
        or not dist.is_available()
        or not dist.is_initialized()
        or world_size <= 1
    ):
        return

    sample = _measure_startup_throughput(lease_client, metrics)
    payload = float(sample) if sample is not None else 0.0

    gather_list: List[Optional[float]] | None = None
    if rank == 0:
        gather_list = [None] * world_size

    try:
        dist.gather_object(payload, gather_list, dst=0)  # type: ignore[arg-type]
    except Exception:
        logger.debug("startup_throughput_gather_failed", exc_info=True)
        return

    weights: Dict[int, float] | None = None
    if rank == 0 and gather_list is not None:
        weights = _normalise_throughput_samples(gather_list)

    message: List[object] = [weights]
    try:
        dist.broadcast_object_list(message, src=0)  # type: ignore[arg-type]
    except Exception:
        logger.debug("startup_throughput_broadcast_failed", exc_info=True)
        return

    payload_weights = message[0]
    if isinstance(payload_weights, dict) and lease_client is not None:
        try:
            lease_client.update_rank_weights({int(k): float(v) for k, v in payload_weights.items()})
        except Exception:
            logger.debug("startup_rank_weight_update_failed", exc_info=True)


def plan_chunk_slices(
    total_items: int,
    *,
    target_ms: float,
    batch_tokens: int,
    ewma: "TrainerMetricsEWMA | None" = None,
) -> List[Tuple[int, int]]:
    """Return contiguous slices describing how ``total_items`` should be chunked."""

    if total_items < 0:
        raise ValueError("total_items must be non-negative")
    chunker = make_chunker(target_ms=target_ms, batch_tokens=batch_tokens, ewma=ewma)
    slices: List[Tuple[int, int]] = []
    cursor = 0
    for spec in chunker:
        if cursor >= total_items:
            break
        width = max(1, int(getattr(spec, "batches", 1)))
        end = min(total_items, cursor + width)
        slices.append((cursor, end))
        cursor = end
    if total_items > 0 and not slices:
        raise RuntimeError("make_chunker did not yield any slices for the corpus")
    if slices and slices[-1][1] < total_items:
        slices.append((slices[-1][1], total_items))
    return slices


def compute_lease_job_id(paths: Sequence[str]) -> str:
    """Return a stable job identifier derived from *paths*."""

    digest = hashlib.blake2s(digest_size=8)
    for path in paths:
        digest.update(path.encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
    return f"lease:{digest.hexdigest()}"


def iterate_leased_shards(
    shard_paths: Sequence[str],
    chunk_slices: Sequence[Tuple[int, int]],
    *,
    lease_client: DistributedLeaseClient,
    encode_shard: Callable[[object], Iterator[int]],
    shard_opener: Callable[[str], contextlib.AbstractContextManager],
    preferred_lease_size: int = 1,
    prefetch_threshold: int = 0,
    min_inflight: int = 1,
    prefetch_slack_ms: float = 50.0,
    on_chunk_start: Optional[Callable[[ChunkLeaseInfo], None]] = None,
) -> Iterator[Iterator[int]]:
    """Yield shard iterators governed by leases from ``lease_client``."""

    if preferred_lease_size <= 0:
        raise ValueError("preferred_lease_size must be positive")
    if prefetch_threshold < 0:
        raise ValueError("prefetch_threshold must be non-negative")
    if min_inflight <= 0:
        raise ValueError("min_inflight must be positive")
    if prefetch_slack_ms < 0:
        raise ValueError("prefetch_slack_ms must be non-negative")
    total_chunks = len(chunk_slices)
    outstanding_chunks = 0
    active: Deque[Dict[str, int]] = deque()
    base_target = max(min_inflight, prefetch_threshold + 1)
    idle_threshold_s = prefetch_slack_ms / 1000.0
    idle_start: float | None = None

    def _record_idle(duration_s: float) -> None:
        if duration_s <= 0.0 or not math.isfinite(duration_s):
            return
        try:
            lease_client.record_idle(duration_s)
        except Exception:
            logger.debug("lease_idle_record_failed", exc_info=True)

    def _request_next() -> bool:
        nonlocal outstanding_chunks
        try:
            lease = lease_client.request_lease(preferred_lease_size)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "active lease" in message:
                return False
            raise
        if lease is None:
            return False
        start, end, chunk_id = lease
        if not (0 <= start <= end <= total_chunks):
            lease_client.requeue_lease(start, end, chunk_id)
            raise ValueError(
                f"Lease [{start}, {end}) lies outside the planned chunk range"
            )
        width = max(0, end - start)
        if width == 0:
            lease_client.complete_lease(start, end, chunk_id)
            return True
        outstanding_chunks += width
        attempts = 0
        completed = False
        if on_chunk_start is not None:
            status = lease_client.describe_chunk(chunk_id)
            if status is not None:
                attempts = int(status.get("attempts", 0))
                completed = bool(status.get("completed", False))
        info = ChunkLeaseInfo(
            start=start,
            end=end,
            chunk_id=chunk_id,
            attempts=max(1, attempts or 1),
            completed=completed,
        )
        if on_chunk_start is not None:
            try:
                on_chunk_start(info)
            except Exception:
                logger.debug("chunk_start_callback_failed", exc_info=True)
        active.append({
            "start": start,
            "end": end,
            "cursor": start,
            "chunk_id": chunk_id,
        })
        return True

    def _ensure_minimum(target: int) -> None:
        while outstanding_chunks < target:
            if not _request_next():
                break

    def _requeue_all() -> None:
        while active:
            info = active.popleft()
            lease_client.requeue_lease(info["start"], info["end"], info["chunk_id"])

    if not _request_next():
        return
    _ensure_minimum(base_target)

    try:
        while True:
            if not active:
                if idle_start is not None:
                    _record_idle(time.monotonic() - idle_start)
                    idle_start = None
                _ensure_minimum(base_target)
                if not active:
                    break
            current = active[0]
            start, end, chunk_id = (
                current["start"],
                current["end"],
                current["chunk_id"],
            )
            cursor = current["cursor"]
            if cursor >= end:
                lease_client.complete_lease(start, end, chunk_id)
                active.popleft()
                continue
            chunk_idx = cursor
            current["cursor"] = cursor + 1
            try:
                slice_start, slice_end = chunk_slices[chunk_idx]
            except Exception:
                _requeue_all()
                raise
            try:
                if slice_start < slice_end:
                    stack = contextlib.ExitStack()
                    shards: List[object] = []
                    try:
                        for path in shard_paths[slice_start:slice_end]:
                            shard = stack.enter_context(shard_opener(path))
                            shards.append(shard)
                        for shard in shards:
                            yield encode_shard(shard)
                    finally:
                        stack.close()
            except GeneratorExit:
                _requeue_all()
                raise
            except Exception:
                _requeue_all()
                raise
            outstanding_chunks = max(0, outstanding_chunks - 1)
            if current["cursor"] >= end:
                lease_client.complete_lease(start, end, chunk_id)
                active.popleft()
            if idle_start is None and outstanding_chunks <= prefetch_threshold:
                idle_start = time.monotonic()
            _ensure_minimum(base_target)
            if idle_start is not None and outstanding_chunks > prefetch_threshold:
                idle_duration = time.monotonic() - idle_start
                _record_idle(idle_duration)
                if idle_duration >= idle_threshold_s:
                    extra_target = max(base_target, outstanding_chunks + 1, min_inflight)
                    _ensure_minimum(extra_target)
                idle_start = None
    except GeneratorExit:
        _requeue_all()
        raise
    except Exception:
        _requeue_all()
        raise


def _reset_lease_registry(job_id: str | None = None) -> None:
    """Test helper that clears cached lease state."""

    with _LEASE_REGISTRY_LOCK:
        if job_id is None:
            _LEASE_REGISTRY.clear()
        else:
            _LEASE_REGISTRY.pop(_normalize_job_id(job_id), None)


def _destroy_process_group_if_available() -> None:
    """Best-effort destruction of an initialized torch.distributed group."""

    if dist is None:  # pragma: no cover - guarded by optional dependency
        return
    with contextlib.suppress(Exception):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _terminate_outstanding_leases(scope: str) -> None:
    """Log a placeholder shutdown event for pending work leases."""

    logger.info("lease_shutdown", extra={"scope": scope})


def _install_signal_handlers(handler: SignalHandler) -> Dict[int, SignalHandler | int | None]:
    """Register *handler* for shutdown signals, returning previous handlers."""

    previous: Dict[int, SignalHandler | int | None] = {}
    for sig in _SHUTDOWN_SIGNALS:
        try:
            prior = signal.getsignal(sig)
        except Exception:  # pragma: no cover - defensive fallback
            prior = None
        previous[sig] = prior
        signal.signal(sig, handler)
    return previous


def _restore_signal_handlers(previous: Dict[int, SignalHandler | int | None]) -> None:
    """Restore handlers captured by :func:`_install_signal_handlers`."""

    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - defensive fallback
            logger.debug("failed_restoring_signal_handler", extra={"signal": sig})


def _summarize_exception(payload: object) -> str:
    """Return a concise human readable description of *payload*."""

    if payload is None:
        return ""
    if isinstance(payload, BaseException):
        return f"{payload.__class__.__name__}: {payload}"
    for attr in ("message", "msg", "exc_msg", "reason", "detail"):
        value = getattr(payload, attr, None)
        if value:
            return str(value)
    if hasattr(payload, "exc_type") and hasattr(payload, "exc_value"):
        return f"{getattr(payload, 'exc_type')}: {getattr(payload, 'exc_value')}"
    return str(payload)


def _drain_error_queue(queue_obj: object) -> Optional[str]:
    """Fetch and format the first exception payload from *queue_obj*."""

    if queue_obj is None:
        return None
    getter = getattr(queue_obj, "get_nowait", None)
    if getter is None:
        getter = getattr(queue_obj, "get", None)
        if getter is None:
            return None
        try:
            item = getter(False)
        except TypeError:
            with contextlib.suppress(Exception):
                item = getter(timeout=0.0)
        except Exception:
            return None
        else:
            return _summarize_exception(item)
    try:
        item = getter()
    except queue.Empty:
        return None
    except Exception:
        return None
    return _summarize_exception(item)


def _collect_process_failures(context: ProcessContext) -> List[str]:
    """Return human readable failure descriptions for *context* workers."""

    failures: List[str] = []

    processes = list(getattr(context, "processes", []) or [])
    error_queues = list(getattr(context, "error_queues", []) or [])

    for rank, proc in enumerate(processes):
        exitcode = getattr(proc, "exitcode", None)
        errors: List[str] = []
        if rank < len(error_queues):
            summary = _drain_error_queue(error_queues[rank])
            if summary:
                errors.append(summary)
        if exitcode not in (None, 0):
            errors.insert(0, f"exit code {exitcode}")
        if errors:
            message = f"Rank {rank} failed: {'; '.join(errors)}"
            logger.error(
                "worker_exit_nonzero",
                extra={"worker_rank": rank, "details": errors},
            )
            failures.append(message)

    # Capture any additional error queues beyond len(processes)
    for offset, queue_obj in enumerate(error_queues[len(processes):], start=len(processes)):
        summary = _drain_error_queue(queue_obj)
        if summary:
            message = f"Rank {offset} reported error: {summary}"
            logger.error(
                "worker_error_only",
                extra={"worker_rank": offset, "details": summary},
            )
            failures.append(message)

    return failures


@dataclass(frozen=True)
class DeviceRequest:
    """Describe a type of accelerator device that should be provisioned."""

    kind: str
    count: int
    capability: Optional[str] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.count <= 0:
            raise ValueError("DeviceRequest.count must be positive")
        if not self.kind:
            raise ValueError("DeviceRequest.kind must be non-empty")


@dataclass(frozen=True)
class ChunkTarget:
    """Declare how many samples each worker should accumulate per chunk."""

    tokens: int
    sequences: int

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.tokens <= 0:
            raise ValueError("ChunkTarget.tokens must be positive")
        if self.sequences <= 0:
            raise ValueError("ChunkTarget.sequences must be positive")


@dataclass(frozen=True)
class LaunchConfig:
    """Aggregate the full configuration for launching distributed workers."""

    devices: List[DeviceRequest] = field(default_factory=list)
    chunk_target: Optional[ChunkTarget] = None
    rebalance_cadence: Optional[timedelta] = None
    environment: dict[str, str] = field(default_factory=dict)

    def iter_env(self) -> Iterable[tuple[str, str]]:
        """Iterate over environment variables with defaults applied."""

        for key, value in self.environment.items():
            yield key, value


@dataclass(frozen=True)
class RendezvousSettings:
    """Describe how distributed workers rendezvous for process group creation."""

    init_method: str = "env://"
    timeout_seconds: float | None = 300.0

    def __post_init__(self) -> None:  # pragma: no cover - basic validation
        if not self.init_method:
            raise ValueError("init_method must be a non-empty string")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")

    def resolve_timeout(self) -> Optional[timedelta]:
        """Return a :class:`~datetime.timedelta` suitable for torch.distributed."""

        if self.timeout_seconds is None:
            return None
        return timedelta(seconds=float(self.timeout_seconds))


@dataclass(frozen=True)
class DistributedLaunchConfig:
    """Configuration payload for launching distributed BPE training workers."""

    device_ids: Sequence[int]
    world_size: int
    log_level: int | str = logging.INFO
    rendezvous: RendezvousSettings = field(default_factory=RendezvousSettings)

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not self.device_ids:
            raise ValueError("device_ids must be a non-empty sequence")
        canonical: Tuple[int, ...] = tuple(int(idx) for idx in self.device_ids)
        if len(canonical) < self.world_size:
            raise ValueError("device_ids must cover all ranks in world_size")
        object.__setattr__(self, "device_ids", canonical)

        level = self.log_level
        if isinstance(level, str):
            numeric = logging.getLevelName(level.upper())
            if isinstance(numeric, str):
                raise ValueError(f"Unknown log level {level!r}")
            level = int(numeric)
        object.__setattr__(self, "log_level", int(level))


def get_env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean from an environment variable."""

    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw in {"1", "true", "yes", "on"}


def get_env_int(name: str, default: int) -> int:
    """Return an integer from an environment variable, with a default."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Environment variable {name!r} must be an integer") from exc


def launch_training(config: DistributedLaunchConfig, cli_args: Sequence[str]) -> None:
    """Launch distributed tokenizer training workers using torch.multiprocessing."""

    if mp is None or torch is None:
        raise RuntimeError("PyTorch is required for distributed training")
    if not torch.cuda.is_available():  # pragma: no cover - requires CUDA hardware
        raise RuntimeError("CUDA must be available to launch distributed training")

    if not hasattr(mp, "start_processes"):
        raise RuntimeError("torch.multiprocessing.start_processes is required for distributed launch")

    logger.info(
        "spawning_workers",
        extra={
            "world_size": config.world_size,
            "devices": list(config.device_ids),
            "rendezvous": {
                "init_method": config.rendezvous.init_method,
                "timeout_seconds": config.rendezvous.timeout_seconds,
            },
        },
    )

    args = (config, tuple(cli_args))
    context_holder: Dict[str, Optional[ProcessContext]] = {"context": None}

    def _parent_signal_handler(signum: int, _frame: Optional[FrameType]) -> None:
        logger.warning(
            "launcher_signal_received",
            extra={"signal": signum, "world_size": config.world_size},
        )
        _terminate_outstanding_leases("launcher")
        _destroy_process_group_if_available()
        ctx = context_holder.get("context")
        if ctx is not None:
            with contextlib.suppress(Exception):
                ctx.terminate()

    previous_handlers = _install_signal_handlers(_parent_signal_handler)

    try:
        start_processes = getattr(mp, "start_processes")
        context = start_processes(  # type: ignore[operator]
            _worker_entry,
            args=args,
            nprocs=config.world_size,
            join=False,
            daemon=False,
            start_method="spawn",
        )
        context_holder["context"] = context

        join_exception: Optional[BaseException] = None
        try:
            context.join()
        except BaseException as exc:  # pragma: no cover - join only fails on error
            join_exception = exc

        failures = _collect_process_failures(context)
        if failures:
            message = "One or more distributed workers exited non-zero:\n" + "\n".join(failures)
            aggregated_error = RuntimeError(message)
            if join_exception is not None:
                aggregated_error.__cause__ = join_exception
            raise aggregated_error
        if join_exception is not None:
            raise join_exception
    finally:
        _restore_signal_handlers(previous_handlers)


def _worker_entry(rank: int, config: DistributedLaunchConfig, cli_args: Sequence[str]) -> None:
    """Worker entry point invoked by :func:`launch_training`."""

    if torch is None or dist is None:
        raise RuntimeError("PyTorch distributed runtime is unavailable")

    device_index = config.device_ids[rank % len(config.device_ids)]
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)

    logging.getLogger().setLevel(config.log_level)
    worker_logger = logging.getLogger(f"{__name__}.worker")

    _enable_peer_access_for_devices(config.device_ids, logger=worker_logger)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(config.world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    timeout = config.rendezvous.resolve_timeout()
    dist.init_process_group(
        backend="nccl",
        init_method=config.rendezvous.init_method,
        world_size=config.world_size,
        rank=rank,
        timeout=timeout,
    )

    lease_client: Optional[DistributedLeaseClient]
    lease_client = None
    lease_metadata: Dict[str, object] | None = None
    total_chunks_env = os.getenv("SUPERTOKEN_LEASE_TOTAL_CHUNKS")
    if total_chunks_env is not None:
        try:
            total_chunks_value = int(total_chunks_env)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError("SUPERTOKEN_LEASE_TOTAL_CHUNKS must be an integer") from exc
        job_id_env = os.getenv("SUPERTOKEN_LEASE_JOB")
        lease_client = register_lease_client(
            job_id=job_id_env,
            total_chunks=total_chunks_value,
            rank=rank,
            world_size=config.world_size,
        )
        lease_metadata = {
            "job_id": lease_client.job_id,
            "total_chunks": lease_client.total_chunks,
        }
        if lease_client.total_chunks > 0:
            initial = lease_client.request_lease(1)
            if initial is not None:
                lease_client.requeue_lease(*initial)

    shutdown_state = {"triggered": False}
    rebalance_thread: Optional[threading.Thread] = None
    rebalance_stop: Optional[threading.Event] = None
    heartbeat_thread: Optional[threading.Thread] = None
    heartbeat_stop = threading.Event()

    def _initiate_shutdown(reason: str) -> None:
        if shutdown_state["triggered"]:
            return
        shutdown_state["triggered"] = True
        if rebalance_stop is not None:
            rebalance_stop.set()
        if rebalance_thread is not None and rebalance_thread.is_alive():
            rebalance_thread.join(timeout=5.0)
        heartbeat_stop.set()
        if heartbeat_thread is not None and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=5.0)
        if lease_client is not None:
            with contextlib.suppress(Exception):
                lease_client.requeue_outstanding()
        worker_logger.info(
            "worker_shutdown",
            extra={
                "worker_rank": rank,
                "world_size": config.world_size,
                "device": str(device),
                "reason": reason,
            },
        )
        _terminate_outstanding_leases("worker")
        _destroy_process_group_if_available()

    def _worker_signal_handler(signum: int, _frame: Optional[FrameType]) -> None:
        worker_logger.warning(
            "worker_signal_received",
            extra={"worker_rank": rank, "signal": signum},
        )
        _initiate_shutdown(f"signal:{signum}")

    atexit.register(lambda: _initiate_shutdown("atexit"))
    worker_previous_handlers = _install_signal_handlers(_worker_signal_handler)

    try:
        from .autoscaler import AutoScaler
        from .bpe_trainer import GPUBPETrainer

        autoscaler = AutoScaler(device=str(device))
        trainer = GPUBPETrainer(devices=[str(device)], autoscaler=autoscaler)
        metrics_tracker = getattr(trainer, "metrics", None)
        if metrics_tracker is not None and hasattr(metrics_tracker, "set_rank"):
            try:
                metrics_tracker.set_rank(rank)
            except Exception:
                logger.debug("metrics_rank_assignment_failed", exc_info=True)

        rebalance_secs_raw = os.getenv("SUPERTOKEN_REBALANCE_SECS", "10")
        try:
            rebalance_secs = float(rebalance_secs_raw)
        except ValueError:
            rebalance_secs = 0.0
        if rebalance_secs < 0.0:
            rebalance_secs = 0.0
        if (
            rebalance_secs > 0.0
            and metrics_tracker is not None
            and getattr(metrics_tracker, "enabled", False)
            and lease_client is not None
            and dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and config.world_size > 1
        ):
            rebalance_stop = threading.Event()
            rebalance_thread = threading.Thread(
                name=f"rebalance-rank{rank}",
                target=_rebalance_loop,
                args=(
                    rank,
                    config.world_size,
                    metrics_tracker,
                    lease_client,
                    float(rebalance_secs),
                    rebalance_stop,
                ),
                kwargs={"blend": _REBALANCE_BLEND},
                daemon=True,
            )
            rebalance_thread.start()

        def _heartbeat_loop() -> None:
            interval = 2.0
            tensor: Optional["torch.Tensor"] = None
            while not heartbeat_stop.is_set():
                start = time.monotonic()
                try:
                    if lease_client is not None:
                        try:
                            lease_client.heartbeat()
                        except KeyError:
                            pass
                    if (
                        dist is not None
                        and dist.is_available()
                        and dist.is_initialized()
                        and config.world_size > 1
                    ):
                        if torch is not None:
                            device_for_ping = device
                            if tensor is None:
                                with torch.no_grad():
                                    tensor = torch.zeros(1, device=device_for_ping, dtype=torch.float32)
                            with torch.no_grad():
                                tensor.fill_(float(rank))
                            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                except Exception:
                    worker_logger.debug(
                        "worker_heartbeat_failed",
                        extra={"worker_rank": rank},
                        exc_info=True,
                    )
                elapsed = time.monotonic() - start
                remaining = max(0.0, interval - elapsed)
                if heartbeat_stop.wait(remaining):
                    break

        should_start_heartbeat = (
            lease_client is not None
            or (
                dist is not None
                and dist.is_available()
                and dist.is_initialized()
                and config.world_size > 1
            )
        )
        if should_start_heartbeat:
            heartbeat_thread = threading.Thread(
                name=f"heartbeat-rank{rank}",
                target=_heartbeat_loop,
                daemon=True,
            )
            heartbeat_thread.start()

        worker_logger.info(
            "worker_ready",
            extra={
                "worker_rank": rank,
                "world_size": config.world_size,
                "device": str(device),
                "cli_args": list(cli_args),
                "autoscaler": autoscaler.state_dict(),
                "trainer_device_count": len(trainer.devices),
                "lease_metadata": lease_metadata,
            },
        )

        _collect_startup_throughput_samples(
            rank=rank,
            world_size=config.world_size,
            lease_client=lease_client,
            metrics=getattr(trainer, "metrics", None),
        )
    finally:
        if rebalance_stop is not None:
            rebalance_stop.set()
        if rebalance_thread is not None and rebalance_thread.is_alive():
            rebalance_thread.join(timeout=5.0)
        heartbeat_stop.set()
        if heartbeat_thread is not None and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=5.0)
        _restore_signal_handlers(worker_previous_handlers)


def _enable_peer_access_for_devices(
    device_ids: Sequence[int], *, logger: Optional[logging.Logger] = None
) -> None:
    """Probe peer access between ``device_ids`` and enable it when possible."""

    if torch is None or not hasattr(torch, "cuda"):
        return

    cuda_mod = torch.cuda
    if not hasattr(cuda_mod, "is_available") or not cuda_mod.is_available():
        return

    enable_fn = getattr(cuda_mod, "device_enable_peer_access", None)
    device_ctx = getattr(cuda_mod, "device", None)
    if enable_fn is None or device_ctx is None:
        return

    try:
        canonical = sorted({int(idx) for idx in device_ids})
    except Exception:
        return

    if len(canonical) <= 1:
        return

    for src in canonical:
        try:
            with device_ctx(src):
                for dst in canonical:
                    if dst == src:
                        continue
                    try:
                        if not utils.can_peer(src, dst):
                            continue
                    except Exception:
                        if logger is not None:
                            logger.debug(
                                "peer_access_probe_failed",
                                extra={"src_device": src, "dst_device": dst},
                                exc_info=True,
                            )
                        continue
                    try:
                        enable_fn(dst)
                    except RuntimeError as exc:
                        message = str(exc).lower()
                        if "already enabled" in message:
                            continue
                        if logger is not None:
                            logger.debug(
                                "peer_access_enable_failed",
                                extra={"src_device": src, "dst_device": dst},
                                exc_info=True,
                            )
                    except Exception:
                        if logger is not None:
                            logger.debug(
                                "peer_access_enable_failed",
                                extra={"src_device": src, "dst_device": dst},
                                exc_info=True,
                            )
        except Exception:
            if logger is not None:
                logger.debug(
                    "peer_access_context_failed",
                    extra={"src_device": src},
                    exc_info=True,
                )


def launch_workers(config: LaunchConfig) -> None:
    """Spawn worker processes as described by *config*.

    Future iterations of the distributed runtime will implement orchestration
    logic here. For now this placeholder makes the API explicit while clearly
    signalling that the functionality is not yet implemented.
    """

    raise NotImplementedError("Distributed worker launch is not implemented yet")


def worker_main() -> None:
    """Entry point for a tokenizer worker process."""

    raise NotImplementedError("Distributed worker execution is not implemented yet")
