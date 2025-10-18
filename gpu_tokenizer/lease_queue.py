"""Lease management primitives for coordinating distributed workers."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, MutableMapping, Optional, Tuple


@dataclass(frozen=True)
class Lease:
    """Represents a half-open interval ``[start, end)`` of work."""

    start: int
    end: int
    chunk_id: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.start, self.end, self.chunk_id)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        if self.start < 0:
            raise ValueError("Lease start must be non-negative")
        if self.end < self.start:
            raise ValueError("Lease end must be greater or equal to start")
        if self.chunk_id < 0:
            raise ValueError("chunk_id must be non-negative")


@dataclass
class _LeaseRecord:
    lease: Lease
    last_heartbeat: float


class LeaseNotary:
    """Thread-safe coordinator that hands out contiguous work intervals.

    ``LeaseNotary`` dispenses leases covering ``total_chunks`` units of work.
    Each rank may hold a bounded number of inflight leases at a time. Ranks can complete
    leases, requeue them for reassignment, and periodically record heartbeats
    to indicate they are still making progress.
    """

    def __init__(
        self,
        total_chunks: int,
        *,
        lease_ttl: float = 30.0,
        min_lease_size: int = 1,
        max_lease_size: Optional[int] = None,
        max_active_leases: int = 2,
    ) -> None:
        if total_chunks < 0:
            raise ValueError("total_chunks must be non-negative")
        if lease_ttl <= 0:
            raise ValueError("lease_ttl must be positive")
        if min_lease_size <= 0:
            raise ValueError("min_lease_size must be positive")
        if max_lease_size is not None:
            if max_lease_size <= 0:
                raise ValueError("max_lease_size must be positive when provided")
            if max_lease_size < min_lease_size:
                raise ValueError("max_lease_size must be greater or equal to min_lease_size")
        if max_active_leases <= 0:
            raise ValueError("max_active_leases must be positive")

        self._lock = threading.Lock()
        self._total_chunks = total_chunks
        self._next_idx = 0
        self._next_chunk_id = 0
        self._inflight: Dict[int, Deque[_LeaseRecord]] = {}
        self._pending_requeue: Deque[Lease] = deque()
        self._lease_ttl = float(lease_ttl)
        self._rank_weights: Dict[int, float] = {}
        self._rank_lease_scale: Dict[int, float] = {}
        self._rank_max_active: Dict[int, int] = {}
        self._rank_throughput: Dict[int, float] = {}
        self._rank_last_lease: Dict[int, int] = {}
        self._min_lease_size = int(min_lease_size)
        self._max_lease_size = int(max_lease_size) if max_lease_size is not None else None
        self._max_active_leases = int(max_active_leases)
        self._idle_metrics: Dict[int, Dict[str, float]] = {}
        self._rank_heartbeats: Dict[int, float] = {}
        self._chunk_records: Dict[int, Dict[str, object]] = {}

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    @property
    def total_chunks(self) -> int:
        return self._total_chunks

    @property
    def lease_ttl(self) -> float:
        """Return the configured lease time-to-live in seconds."""

        return self._lease_ttl

    @property
    def max_active_leases(self) -> int:
        """Return the configured per-rank inflight lease limit."""

        return self._max_active_leases

    def _record_idle_locked(self, rank: int, duration_s: float) -> None:
        if duration_s <= 0.0 or not math.isfinite(duration_s):
            return
        stats = self._idle_metrics.setdefault(
            int(rank),
            {"ewma_s": float(duration_s), "last_s": float(duration_s), "samples": 0.0},
        )
        samples = int(stats.get("samples", 0) or 0)
        if samples <= 0:
            ewma = float(duration_s)
        else:
            ewma = (0.2 * float(duration_s)) + (0.8 * float(stats.get("ewma_s", duration_s)))
        stats["ewma_s"] = ewma
        stats["last_s"] = float(duration_s)
        stats["samples"] = float(samples + 1)

    def record_idle(self, rank: int, duration_s: float) -> None:
        """Record *duration_s* seconds of idle time for ``rank``."""

        with self._lock:
            self._record_idle_locked(rank, duration_s)

    def apply_feedback(
        self,
        rank: int,
        *,
        completed_tokens: int,
        duration_s: Optional[float] = None,
        idle_duration_s: float = 0.0,
    ) -> Dict[str, float]:
        """Update adaptive lease parameters using throughput feedback."""

        tokens = max(0, int(completed_tokens))
        try:
            elapsed = float(duration_s) if duration_s is not None else 0.0
        except (TypeError, ValueError):
            elapsed = 0.0
        idle = float(idle_duration_s) if idle_duration_s is not None else 0.0

        with self._lock:
            if idle > 0.0 and math.isfinite(idle):
                self._record_idle_locked(rank, idle)

            if elapsed > 0.0 and math.isfinite(elapsed):
                sample = float(tokens) / elapsed if tokens > 0 else 0.0
            else:
                sample = float(tokens)

            if not math.isfinite(sample):
                sample = 0.0

            prev = self._rank_throughput.get(rank)
            if prev is None or not math.isfinite(prev):
                ewma = sample
            else:
                alpha = 0.2
                ewma = (alpha * sample) + ((1.0 - alpha) * prev)

            if not math.isfinite(ewma):
                ewma = 0.0

            self._rank_throughput[int(rank)] = ewma

            values = [
                value
                for value in self._rank_throughput.values()
                if value > 0.0 and math.isfinite(value)
            ]
            average = sum(values) / len(values) if values else (ewma if ewma > 0.0 else 1.0)
            if not math.isfinite(average) or average <= 0.0:
                average = 1.0

            raw_scale = ewma / average if average > 0.0 else 1.0
            if not math.isfinite(raw_scale) or raw_scale <= 0.0:
                raw_scale = self._rank_lease_scale.get(rank, 1.0)
            min_scale = 0.25
            max_scale = 4.0
            raw_scale = max(min_scale, min(max_scale, raw_scale))

            prev_scale = self._rank_lease_scale.get(rank)
            if prev_scale is None or prev_scale <= 0.0 or not math.isfinite(prev_scale):
                scale = raw_scale
            else:
                lower = max(min_scale, prev_scale * 0.5)
                upper = min(max_scale, prev_scale * 1.5)
                scale = max(lower, min(raw_scale, upper))
            self._rank_lease_scale[int(rank)] = scale

            target_limit = int(round(scale * float(self._max_active_leases)))
            target_limit = max(1, min(self._max_active_leases, target_limit))
            prev_limit = self._rank_max_active.get(rank)
            if prev_limit is None or prev_limit <= 0:
                limit = target_limit
            elif target_limit > prev_limit:
                limit = min(self._max_active_leases, prev_limit + 1)
            elif target_limit < prev_limit:
                limit = max(1, prev_limit - 1)
            else:
                limit = prev_limit
            self._rank_max_active[int(rank)] = limit

            idle_stats = self._idle_metrics.get(int(rank), {})
            idle_ewma_s = float(idle_stats.get("ewma_s", 0.0)) if idle_stats else 0.0
            last_width = int(self._rank_last_lease.get(int(rank), self._min_lease_size))

            return {
                "throughput": ewma,
                "scale": scale,
                "max_active": limit,
                "idle_ewma_s": idle_ewma_s,
                "idle_ewma_ms": idle_ewma_s * 1000.0,
                "lease_width": last_width,
            }

    def idle_metrics(self) -> Dict[int, Dict[str, float]]:
        """Return a snapshot of the accumulated idle telemetry."""

        with self._lock:
            snapshot: Dict[int, Dict[str, float]] = {}
            for rank, stats in self._idle_metrics.items():
                snapshot[int(rank)] = {
                    "ewma_ms": float(stats.get("ewma_s", 0.0)) * 1000.0,
                    "last_ms": float(stats.get("last_s", 0.0)) * 1000.0,
                    "samples": int(stats.get("samples", 0) or 0),
                }
            return snapshot

    def update_max_active_leases(self, max_active_leases: int) -> None:
        """Increase :attr:`max_active_leases` when *max_active_leases* grows."""

        if max_active_leases <= 0:
            raise ValueError("max_active_leases must be positive")
        with self._lock:
            if max_active_leases > self._max_active_leases:
                self._max_active_leases = int(max_active_leases)
                for rank, limit in list(self._rank_max_active.items()):
                    if limit > self._max_active_leases:
                        self._rank_max_active[rank] = self._max_active_leases

    def _record_new_chunk(self, lease: Lease, *, attempts: int = 1) -> None:
        record = self._chunk_records.setdefault(
            lease.chunk_id,
            {
                "lease": (lease.start, lease.end),
                "completed": False,
                "attempts": 0,
            },
        )
        record["lease"] = (lease.start, lease.end)
        record["attempts"] = int(record.get("attempts", 0)) + attempts

    def _mark_chunk_complete(self, lease: Lease) -> None:
        if lease.chunk_id in self._chunk_records:
            self._chunk_records[lease.chunk_id]["completed"] = True

    def chunk_status(self, chunk_id: int) -> Optional[Dict[str, object]]:
        with self._lock:
            entry = self._chunk_records.get(int(chunk_id))
            if entry is None:
                return None
            return {
                "lease": tuple(entry.get("lease", (0, 0))),
                "completed": bool(entry.get("completed", False)),
                "attempts": int(entry.get("attempts", 0)),
            }

    def grant_lease(self, rank: int, preferred_size: int) -> Optional[Tuple[int, int, int]]:
        """Return the next available lease for ``rank``.

        Leases are granted from the ``pending_requeue`` queue first and then
        from the monotonic ``next_idx`` counter. Returns ``None`` when no more
        work remains.
        """

        if preferred_size <= 0:
            raise ValueError("preferred_size must be positive")

        with self._lock:
            queue = self._inflight.setdefault(rank, deque())
            limit = int(self._rank_max_active.get(rank, self._max_active_leases))
            if limit <= 0:
                limit = 1
            if len(queue) >= limit:
                raise RuntimeError("rank already holds an active lease")

            attempts_increment = 1
            if self._pending_requeue:
                lease = self._pending_requeue.popleft()
                attempts_increment = 1
            else:
                if self._next_idx >= self._total_chunks:
                    if not queue:
                        self._inflight.pop(rank, None)
                    return None
                start = self._next_idx
                scale = self._rank_lease_scale.get(rank)
                if scale is None:
                    scale = self._rank_weights.get(rank, 1.0)
                scale = max(0.0, float(scale))
                weighted = math.ceil(scale * float(preferred_size))
                size = max(self._min_lease_size, int(weighted))
                if self._max_lease_size is not None:
                    size = min(size, self._max_lease_size)
                end = min(self._total_chunks, start + size)
                chunk_id = self._next_chunk_id
                self._next_chunk_id += 1
                lease = Lease(start, end, chunk_id)
                self._next_idx = end
            self._rank_last_lease[rank] = max(0, lease.end - lease.start)
            self._record_new_chunk(lease, attempts=attempts_increment)

            now = self._now()
            record = _LeaseRecord(lease=lease, last_heartbeat=now)
            queue.append(record)
            self._rank_heartbeats[rank] = now
            return lease.as_tuple()

    def complete_lease(self, rank: int, start: int, end: int, chunk_id: int) -> None:
        """Mark the specified lease as completed by ``rank``."""

        lease = Lease(start, end, chunk_id)
        with self._lock:
            queue = self._inflight.get(rank)
            if not queue:
                raise KeyError(f"rank {rank} does not hold a lease")
            record = queue[0]
            if record.lease != lease:
                raise ValueError("completed lease does not match inflight record")
            queue.popleft()
            if not queue:
                self._inflight.pop(rank, None)
                self._rank_heartbeats.pop(rank, None)
            self._mark_chunk_complete(lease)

    def requeue_lease(self, rank: int, start: int, end: int, chunk_id: int) -> None:
        """Return an inflight lease to the queue for reassignment."""

        lease = Lease(start, end, chunk_id)
        with self._lock:
            queue = self._inflight.get(rank)
            if not queue:
                raise KeyError(f"rank {rank} does not hold a lease")
            record = queue[0]
            if record.lease != lease:
                raise ValueError("requeued lease does not match inflight record")
            queue.popleft()
            if not queue:
                self._inflight.pop(rank, None)
                self._rank_heartbeats.pop(rank, None)
            self._pending_requeue.appendleft(lease)

    def heartbeat(self, rank: int) -> float:
        """Record a progress heartbeat for ``rank`` and return the timestamp."""

        with self._lock:
            queue = self._inflight.get(rank)
            if not queue:
                raise KeyError(f"rank {rank} does not hold a lease")
            new_ts = self._now()
            for record in queue:
                record.last_heartbeat = new_ts
            self._rank_heartbeats[rank] = new_ts
            return new_ts

    def check_timeouts(self, now: Optional[float] = None) -> Dict[int, Tuple[int, int]]:
        """Requeue leases whose heartbeats have exceeded :attr:`lease_ttl`.

        Parameters
        ----------
        now:
            Optional wall clock timestamp (``time.monotonic`` seconds). When not
            provided the current monotonic time is used.

        Returns
        -------
        dict
            Mapping of timed-out ranks to the lease that was requeued.
        """

        deadline = self._now() if now is None else float(now)
        timed_out: Dict[int, Tuple[int, int]] = {}

        with self._lock:
            expired_ranks = []
            for rank, queue in self._inflight.items():
                if not queue:
                    continue
                last_seen = self._rank_heartbeats.get(rank)
                if last_seen is None:
                    last_seen = queue[0].last_heartbeat
                if deadline - last_seen >= self._lease_ttl:
                    expired_ranks.append(rank)

            for rank in expired_ranks:
                queue = self._inflight.pop(rank, deque())
                self._rank_heartbeats.pop(rank, None)
                if not queue:
                    continue
                first = queue[0]
                timed_out[rank] = first.lease.as_tuple()
                while queue:
                    record = queue.popleft()
                    self._pending_requeue.appendleft(record.lease)

        return timed_out

    def state_dict(self) -> Dict[str, object]:
        """Return a serialisable snapshot of the notary state."""

        with self._lock:
            inflight: MutableMapping[int, Dict[str, object]] = {}
            for rank, queue in self._inflight.items():
                records = [
                    {
                        "lease": record.lease.as_tuple(),
                        "last_heartbeat": record.last_heartbeat,
                    }
                    for record in queue
                ]
                entry: Dict[str, object] = {"records": records}
                if records:
                    entry["lease"] = records[0]["lease"]
                    entry["last_heartbeat"] = records[0]["last_heartbeat"]
                inflight[rank] = entry

            pending: Iterable[Tuple[int, int, int]] = (
                lease.as_tuple() for lease in self._pending_requeue
            )

            chunk_records = {
                chunk_id: {
                    "lease": tuple(entry.get("lease", (0, 0))),
                    "completed": bool(entry.get("completed", False)),
                    "attempts": int(entry.get("attempts", 0)),
                }
                for chunk_id, entry in self._chunk_records.items()
            }

            idle_metrics = {
                rank: {
                    "ewma_ms": float(stats.get("ewma_s", 0.0)) * 1000.0,
                    "last_ms": float(stats.get("last_s", 0.0)) * 1000.0,
                    "samples": int(stats.get("samples", 0) or 0),
                }
                for rank, stats in self._idle_metrics.items()
            }

            return {
                "total_chunks": self._total_chunks,
                "next_idx": self._next_idx,
                "next_chunk_id": self._next_chunk_id,
                "inflight": dict(inflight),
                "pending_requeue": list(pending),
                "lease_ttl": self._lease_ttl,
                "rank_weights": dict(self._rank_weights),
                "rank_lease_scale": dict(self._rank_lease_scale),
                "rank_max_active": dict(self._rank_max_active),
                "rank_throughput": dict(self._rank_throughput),
                "rank_last_lease": dict(self._rank_last_lease),
                "min_lease_size": self._min_lease_size,
                "max_lease_size": self._max_lease_size,
                "max_active_leases": self._max_active_leases,
                "idle_metrics": idle_metrics,
                "rank_heartbeats": dict(self._rank_heartbeats),
                "chunk_records": chunk_records,
            }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`."""

        required_keys = {
            "total_chunks",
            "next_idx",
            "inflight",
            "pending_requeue",
        }
        if not required_keys.issubset(state):
            missing = required_keys - set(state)
            raise KeyError(f"state missing keys: {sorted(missing)}")

        total_chunks = int(state["total_chunks"])
        next_idx = int(state["next_idx"])
        inflight_raw = state["inflight"]
        pending_raw = state["pending_requeue"]
        next_chunk_id = int(state.get("next_chunk_id", 0))
        lease_ttl = float(state.get("lease_ttl", self._lease_ttl))
        rank_weights_raw = state.get("rank_weights", {})
        rank_lease_scale_raw = state.get("rank_lease_scale", {})
        rank_max_active_raw = state.get("rank_max_active", {})
        rank_throughput_raw = state.get("rank_throughput", {})
        rank_last_lease_raw = state.get("rank_last_lease", {})
        min_lease_size = int(state.get("min_lease_size", self._min_lease_size))
        raw_max_lease = state.get("max_lease_size", self._max_lease_size)
        max_active_leases = int(state.get("max_active_leases", self._max_active_leases))
        rank_heartbeats_raw = state.get("rank_heartbeats", {})

        if total_chunks < 0:
            raise ValueError("total_chunks must be non-negative")
        if not 0 <= next_idx <= total_chunks:
            raise ValueError("next_idx must be between 0 and total_chunks")
        if not isinstance(inflight_raw, dict):
            raise TypeError("inflight must be a mapping")
        if not isinstance(pending_raw, list):
            raise TypeError("pending_requeue must be a list")
        if lease_ttl <= 0:
            raise ValueError("lease_ttl must be positive")
        if min_lease_size <= 0:
            raise ValueError("min_lease_size must be positive")
        max_lease_size: Optional[int]
        if raw_max_lease is None:
            max_lease_size = None
        else:
            max_lease_size = int(raw_max_lease)
            if max_lease_size <= 0:
                raise ValueError("max_lease_size must be positive when provided")
            if max_lease_size < min_lease_size:
                raise ValueError("max_lease_size must be greater or equal to min_lease_size")
        if max_active_leases <= 0:
            raise ValueError("max_active_leases must be positive")
        if rank_weights_raw is None:
            rank_weights_raw = {}
        if not isinstance(rank_weights_raw, dict):
            raise TypeError("rank_weights must be a mapping")

        inflight: Dict[int, Deque[_LeaseRecord]] = {}
        for raw_rank, raw_entry in inflight_raw.items():
            rank = int(raw_rank)
            queue: Deque[_LeaseRecord] = deque()
            if isinstance(raw_entry, dict) and "records" in raw_entry:
                raw_records = raw_entry["records"]
            elif isinstance(raw_entry, list):
                raw_records = raw_entry
            elif isinstance(raw_entry, dict) and "lease" in raw_entry and "last_heartbeat" in raw_entry:
                raw_records = [raw_entry]
            else:
                raise TypeError("inflight entry must be a mapping or list of records")
            if not isinstance(raw_records, list):
                raise TypeError("inflight records must be a list")
            for record_entry in raw_records:
                if not isinstance(record_entry, dict):
                    raise TypeError("inflight record must be a mapping")
                if "lease" not in record_entry or "last_heartbeat" not in record_entry:
                    raise KeyError("inflight record missing required fields")
                raw_lease = record_entry["lease"]
                if not isinstance(raw_lease, (tuple, list)) or len(raw_lease) < 3:
                    start, end = raw_lease[:2]
                    chunk_id = record_entry.get("chunk_id", 0)
                else:
                    start, end, chunk_id = raw_lease[:3]
                lease = Lease(int(start), int(end), int(chunk_id))
                last_hb = float(record_entry["last_heartbeat"])
                queue.append(_LeaseRecord(lease=lease, last_heartbeat=last_hb))
            if queue:
                inflight[rank] = queue

        pending_queue: Deque[Lease] = deque()
        for entry in pending_raw:
            if isinstance(entry, (tuple, list)) and len(entry) >= 3:
                start, end, chunk_id = entry[:3]
            else:
                start, end = entry[:2]
                chunk_id = 0
            pending_queue.append(Lease(int(start), int(end), int(chunk_id)))

        rank_weights: Dict[int, float] = {}
        for raw_rank, raw_weight in rank_weights_raw.items():
            rank = int(raw_rank)
            weight = float(raw_weight)
            if weight < 0:
                raise ValueError("rank weight must be non-negative")
            rank_weights[rank] = weight

        rank_lease_scale: Dict[int, float] = {}
        if rank_lease_scale_raw is None:
            rank_lease_scale_raw = {}
        if isinstance(rank_lease_scale_raw, dict):
            for raw_rank, raw_scale in rank_lease_scale_raw.items():
                rank = int(raw_rank)
                scale = float(raw_scale)
                if not math.isfinite(scale) or scale < 0.0:
                    continue
                rank_lease_scale[rank] = scale

        rank_max_active: Dict[int, int] = {}
        if rank_max_active_raw is None:
            rank_max_active_raw = {}
        if isinstance(rank_max_active_raw, dict):
            for raw_rank, raw_limit in rank_max_active_raw.items():
                rank = int(raw_rank)
                limit = int(raw_limit)
                if limit <= 0:
                    continue
                rank_max_active[rank] = limit

        rank_throughput: Dict[int, float] = {}
        if isinstance(rank_throughput_raw, dict):
            for raw_rank, raw_value in rank_throughput_raw.items():
                try:
                    rank = int(raw_rank)
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                rank_throughput[rank] = value

        rank_last_lease: Dict[int, int] = {}
        if isinstance(rank_last_lease_raw, dict):
            for raw_rank, raw_value in rank_last_lease_raw.items():
                try:
                    rank = int(raw_rank)
                    width = int(raw_value)
                except (TypeError, ValueError):
                    continue
                if width <= 0:
                    continue
                rank_last_lease[rank] = width

        idle_metrics_raw = state.get("idle_metrics", {})
        idle_metrics: Dict[int, Dict[str, float]] = {}
        if isinstance(idle_metrics_raw, dict):
            for raw_rank, raw_stats in idle_metrics_raw.items():
                try:
                    rank = int(raw_rank)
                except (TypeError, ValueError):
                    continue
                if not isinstance(raw_stats, dict):
                    continue
                try:
                    ewma_ms = float(raw_stats.get("ewma_ms", 0.0))
                    last_ms = float(raw_stats.get("last_ms", 0.0))
                    samples = int(raw_stats.get("samples", 0) or 0)
                except (TypeError, ValueError):
                    continue
                idle_metrics[rank] = {
                    "ewma_s": ewma_ms / 1000.0,
                    "last_s": last_ms / 1000.0,
                    "samples": float(max(0, samples)),
                }

        rank_heartbeats: Dict[int, float] = {}
        if isinstance(rank_heartbeats_raw, dict):
            for raw_rank, raw_ts in rank_heartbeats_raw.items():
                try:
                    rank = int(raw_rank)
                    ts = float(raw_ts)
                except (TypeError, ValueError):
                    continue
                rank_heartbeats[rank] = ts

        chunk_records_raw = state.get("chunk_records", {})
        chunk_records: Dict[int, Dict[str, object]] = {}
        if isinstance(chunk_records_raw, dict):
            for raw_chunk, raw_entry in chunk_records_raw.items():
                try:
                    chunk_id = int(raw_chunk)
                except (TypeError, ValueError):
                    continue
                if not isinstance(raw_entry, dict):
                    continue
                lease_entry = raw_entry.get("lease", (0, 0))
                if isinstance(lease_entry, (tuple, list)):
                    lease_tuple = tuple(int(x) for x in lease_entry[:2])
                else:
                    lease_tuple = (0, 0)
                chunk_records[chunk_id] = {
                    "lease": lease_tuple,
                    "completed": bool(raw_entry.get("completed", False)),
                    "attempts": int(raw_entry.get("attempts", 0)),
                }

        with self._lock:
            self._total_chunks = total_chunks
            self._next_idx = next_idx
            self._next_chunk_id = max(0, next_chunk_id)
            self._inflight = inflight
            self._pending_requeue = pending_queue
            self._lease_ttl = lease_ttl
            self._rank_weights = rank_weights
            self._rank_lease_scale = rank_lease_scale
            self._rank_max_active = {
                rank: min(limit, self._max_active_leases)
                for rank, limit in rank_max_active.items()
            }
            self._rank_throughput = rank_throughput
            self._rank_last_lease = rank_last_lease
            self._min_lease_size = min_lease_size
            self._max_lease_size = max_lease_size
            self._max_active_leases = max_active_leases
            self._idle_metrics = idle_metrics
            self._rank_heartbeats = rank_heartbeats
            self._chunk_records = chunk_records

    def update_rank_weights(self, weights: Dict[int, float]) -> None:
        """Persist normalised per-rank weights for adaptive lease sizing."""

        with self._lock:
            cleaned: Dict[int, float] = {}
            for raw_rank, raw_weight in weights.items():
                rank = int(raw_rank)
                weight = float(raw_weight)
                if weight < 0:
                    raise ValueError("rank weight must be non-negative")
                if not math.isfinite(weight):
                    continue
                cleaned[rank] = weight

            existing_keys = (
                set(self._rank_weights.keys())
                | set(self._rank_lease_scale.keys())
                | set(self._rank_max_active.keys())
            )
            keys = existing_keys | set(cleaned.keys())
            if not keys:
                self._rank_weights = {}
                self._rank_lease_scale = {}
                self._rank_max_active = {}
                return

            if cleaned:
                total = sum(cleaned.values())
                count = len(cleaned)
            else:
                total = sum(self._rank_weights.values())
                count = len(self._rank_weights) if self._rank_weights else 0
            average = float(total) / float(count) if count > 0 else 0.0
            if average <= 0.0 or not math.isfinite(average):
                average = 1.0

            min_scale = 0.25
            max_scale = 4.0

            new_weights: Dict[int, float] = {}
            new_scales: Dict[int, float] = {}
            new_limits: Dict[int, int] = {}

            for rank in sorted(keys):
                if rank in cleaned:
                    weight = cleaned[rank]
                else:
                    weight = float(self._rank_weights.get(rank, average))
                if weight < 0 or not math.isfinite(weight):
                    weight = average
                new_weights[rank] = weight

                target_scale = weight / average if average > 0.0 else 1.0
                target_scale = max(min_scale, min(max_scale, target_scale))

                prev_scale = self._rank_lease_scale.get(rank)
                if prev_scale is None or prev_scale <= 0.0 or not math.isfinite(prev_scale):
                    scale = target_scale
                else:
                    lower = max(min_scale, prev_scale * 0.5)
                    upper = min(max_scale, prev_scale * 1.5)
                    scale = max(lower, min(target_scale, upper))
                new_scales[rank] = scale

                target_limit = int(round(scale * float(self._max_active_leases)))
                target_limit = max(1, min(self._max_active_leases, target_limit))
                prev_limit = self._rank_max_active.get(rank)
                if prev_limit is None or prev_limit <= 0:
                    limit = target_limit
                elif target_limit > prev_limit:
                    limit = min(self._max_active_leases, prev_limit + 1)
                elif target_limit < prev_limit:
                    limit = max(1, prev_limit - 1)
                else:
                    limit = prev_limit
                new_limits[rank] = limit

            self._rank_weights = new_weights
            self._rank_lease_scale = new_scales
            self._rank_max_active = new_limits

    def rank_weights(self) -> Dict[int, float]:
        """Return a copy of the stored per-rank weights."""

        with self._lock:
            return dict(self._rank_weights)

    def rank_status(self) -> Dict[int, Dict[str, float]]:
        """Return a snapshot of adaptive lease parameters per rank."""

        with self._lock:
            ranks = (
                set(self._rank_weights)
                | set(self._rank_lease_scale)
                | set(self._rank_max_active)
                | set(self._rank_throughput)
                | set(self._rank_last_lease)
                | set(self._idle_metrics)
            )
            status: Dict[int, Dict[str, float]] = {}
            for rank in sorted(ranks):
                idle_stats = self._idle_metrics.get(rank, {})
                status[rank] = {
                    "scale": float(self._rank_lease_scale.get(rank, 1.0)),
                    "max_active": float(self._rank_max_active.get(rank, self._max_active_leases)),
                    "throughput": float(self._rank_throughput.get(rank, 0.0)),
                    "lease_width": float(self._rank_last_lease.get(rank, self._min_lease_size)),
                    "idle_ewma_ms": float(idle_stats.get("ewma_s", 0.0)) * 1000.0,
                }
            return status

