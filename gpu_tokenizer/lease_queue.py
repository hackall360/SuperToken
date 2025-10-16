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

    def as_tuple(self) -> Tuple[int, int]:
        return (self.start, self.end)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        if self.start < 0:
            raise ValueError("Lease start must be non-negative")
        if self.end < self.start:
            raise ValueError("Lease end must be greater or equal to start")


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
        max_active_leases: int = 1,
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
        self._inflight: Dict[int, Deque[_LeaseRecord]] = {}
        self._pending_requeue: Deque[Lease] = deque()
        self._lease_ttl = float(lease_ttl)
        self._rank_weights: Dict[int, float] = {}
        self._min_lease_size = int(min_lease_size)
        self._max_lease_size = int(max_lease_size) if max_lease_size is not None else None
        self._max_active_leases = int(max_active_leases)

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

    def grant_lease(self, rank: int, preferred_size: int) -> Optional[Tuple[int, int]]:
        """Return the next available lease for ``rank``.

        Leases are granted from the ``pending_requeue`` queue first and then
        from the monotonic ``next_idx`` counter. Returns ``None`` when no more
        work remains.
        """

        if preferred_size <= 0:
            raise ValueError("preferred_size must be positive")

        with self._lock:
            queue = self._inflight.setdefault(rank, deque())
            if len(queue) >= self._max_active_leases:
                raise RuntimeError("rank already holds an active lease")

            if self._pending_requeue:
                lease = self._pending_requeue.popleft()
            else:
                if self._next_idx >= self._total_chunks:
                    if not queue:
                        self._inflight.pop(rank, None)
                    return None
                start = self._next_idx
                weight = self._rank_weights.get(rank)
                if weight is None:
                    weight = 1.0
                else:
                    weight = max(0.0, float(weight))
                weighted = math.ceil(weight * float(preferred_size))
                size = max(self._min_lease_size, int(weighted))
                if self._max_lease_size is not None:
                    size = min(size, self._max_lease_size)
                end = min(self._total_chunks, start + size)
                lease = Lease(start, end)
                self._next_idx = end

            record = _LeaseRecord(lease=lease, last_heartbeat=self._now())
            queue.append(record)
            return lease.as_tuple()

    def complete_lease(self, rank: int, start: int, end: int) -> None:
        """Mark the specified lease as completed by ``rank``."""

        lease = Lease(start, end)
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

    def requeue_lease(self, rank: int, start: int, end: int) -> None:
        """Return an inflight lease to the queue for reassignment."""

        lease = Lease(start, end)
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
                oldest = queue[0]
                if deadline - oldest.last_heartbeat >= self._lease_ttl:
                    expired_ranks.append(rank)

            for rank in expired_ranks:
                queue = self._inflight.pop(rank, deque())
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

            pending: Iterable[Tuple[int, int]] = (
                lease.as_tuple() for lease in self._pending_requeue
            )

            return {
                "total_chunks": self._total_chunks,
                "next_idx": self._next_idx,
                "inflight": dict(inflight),
                "pending_requeue": list(pending),
                "lease_ttl": self._lease_ttl,
                "rank_weights": dict(self._rank_weights),
                "min_lease_size": self._min_lease_size,
                "max_lease_size": self._max_lease_size,
                "max_active_leases": self._max_active_leases,
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
        lease_ttl = float(state.get("lease_ttl", self._lease_ttl))
        rank_weights_raw = state.get("rank_weights", {})
        min_lease_size = int(state.get("min_lease_size", self._min_lease_size))
        raw_max_lease = state.get("max_lease_size", self._max_lease_size)
        max_active_leases = int(state.get("max_active_leases", self._max_active_leases))

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
                start, end = record_entry["lease"]
                lease = Lease(int(start), int(end))
                last_hb = float(record_entry["last_heartbeat"])
                queue.append(_LeaseRecord(lease=lease, last_heartbeat=last_hb))
            if queue:
                inflight[rank] = queue

        pending_queue: Deque[Lease] = deque()
        for entry in pending_raw:
            start, end = entry
            pending_queue.append(Lease(int(start), int(end)))

        rank_weights: Dict[int, float] = {}
        for raw_rank, raw_weight in rank_weights_raw.items():
            rank = int(raw_rank)
            weight = float(raw_weight)
            if weight < 0:
                raise ValueError("rank weight must be non-negative")
            rank_weights[rank] = weight

        with self._lock:
            self._total_chunks = total_chunks
            self._next_idx = next_idx
            self._inflight = inflight
            self._pending_requeue = pending_queue
            self._lease_ttl = lease_ttl
            self._rank_weights = rank_weights
            self._min_lease_size = min_lease_size
            self._max_lease_size = max_lease_size
            self._max_active_leases = max_active_leases

    def update_rank_weights(self, weights: Dict[int, float]) -> None:
        """Persist normalised per-rank weights for adaptive lease sizing."""

        with self._lock:
            cleaned: Dict[int, float] = {}
            for raw_rank, raw_weight in weights.items():
                rank = int(raw_rank)
                weight = float(raw_weight)
                if weight < 0:
                    raise ValueError("rank weight must be non-negative")
                cleaned[rank] = weight
            self._rank_weights = cleaned

    def rank_weights(self) -> Dict[int, float]:
        """Return a copy of the stored per-rank weights."""

        with self._lock:
            return dict(self._rank_weights)

