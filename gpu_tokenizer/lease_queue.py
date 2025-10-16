"""Lease management primitives for coordinating distributed workers."""

from __future__ import annotations

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
    Each rank may hold at most one inflight lease at a time. Ranks can complete
    leases, requeue them for reassignment, and periodically record heartbeats
    to indicate they are still making progress.
    """

    def __init__(self, total_chunks: int) -> None:
        if total_chunks < 0:
            raise ValueError("total_chunks must be non-negative")

        self._lock = threading.Lock()
        self._total_chunks = total_chunks
        self._next_idx = 0
        self._inflight: Dict[int, _LeaseRecord] = {}
        self._pending_requeue: Deque[Lease] = deque()

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    @property
    def total_chunks(self) -> int:
        return self._total_chunks

    def grant_lease(self, rank: int, preferred_size: int) -> Optional[Tuple[int, int]]:
        """Return the next available lease for ``rank``.

        Leases are granted from the ``pending_requeue`` queue first and then
        from the monotonic ``next_idx`` counter. Returns ``None`` when no more
        work remains.
        """

        if preferred_size <= 0:
            raise ValueError("preferred_size must be positive")

        with self._lock:
            if rank in self._inflight:
                raise RuntimeError("rank already holds an active lease")

            if self._pending_requeue:
                lease = self._pending_requeue.popleft()
            else:
                if self._next_idx >= self._total_chunks:
                    return None
                start = self._next_idx
                end = min(self._total_chunks, start + preferred_size)
                lease = Lease(start, end)
                self._next_idx = end

            record = _LeaseRecord(lease=lease, last_heartbeat=self._now())
            self._inflight[rank] = record
            return lease.as_tuple()

    def complete_lease(self, rank: int, start: int, end: int) -> None:
        """Mark the specified lease as completed by ``rank``."""

        lease = Lease(start, end)
        with self._lock:
            record = self._inflight.pop(rank, None)
            if record is None:
                raise KeyError(f"rank {rank} does not hold a lease")
            if record.lease != lease:
                self._inflight[rank] = record  # restore for debuggability
                raise ValueError("completed lease does not match inflight record")

    def requeue_lease(self, rank: int, start: int, end: int) -> None:
        """Return an inflight lease to the queue for reassignment."""

        lease = Lease(start, end)
        with self._lock:
            record = self._inflight.pop(rank, None)
            if record is None:
                raise KeyError(f"rank {rank} does not hold a lease")
            if record.lease != lease:
                self._inflight[rank] = record
                raise ValueError("requeued lease does not match inflight record")
            self._pending_requeue.appendleft(lease)

    def heartbeat(self, rank: int) -> float:
        """Record a progress heartbeat for ``rank`` and return the timestamp."""

        with self._lock:
            record = self._inflight.get(rank)
            if record is None:
                raise KeyError(f"rank {rank} does not hold a lease")
            new_ts = self._now()
            record.last_heartbeat = new_ts
            return new_ts

    def state_dict(self) -> Dict[str, object]:
        """Return a serialisable snapshot of the notary state."""

        with self._lock:
            inflight: MutableMapping[int, Dict[str, float | Tuple[int, int]]] = {}
            for rank, record in self._inflight.items():
                inflight[rank] = {
                    "lease": record.lease.as_tuple(),
                    "last_heartbeat": record.last_heartbeat,
                }

            pending: Iterable[Tuple[int, int]] = (
                lease.as_tuple() for lease in self._pending_requeue
            )

            return {
                "total_chunks": self._total_chunks,
                "next_idx": self._next_idx,
                "inflight": dict(inflight),
                "pending_requeue": list(pending),
            }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`."""

        required_keys = {"total_chunks", "next_idx", "inflight", "pending_requeue"}
        if not required_keys.issubset(state):
            missing = required_keys - set(state)
            raise KeyError(f"state missing keys: {sorted(missing)}")

        total_chunks = int(state["total_chunks"])
        next_idx = int(state["next_idx"])
        inflight_raw = state["inflight"]
        pending_raw = state["pending_requeue"]

        if total_chunks < 0:
            raise ValueError("total_chunks must be non-negative")
        if not 0 <= next_idx <= total_chunks:
            raise ValueError("next_idx must be between 0 and total_chunks")
        if not isinstance(inflight_raw, dict):
            raise TypeError("inflight must be a mapping")
        if not isinstance(pending_raw, list):
            raise TypeError("pending_requeue must be a list")

        inflight: Dict[int, _LeaseRecord] = {}
        for raw_rank, raw_entry in inflight_raw.items():
            rank = int(raw_rank)
            if not isinstance(raw_entry, dict):
                raise TypeError("inflight entries must be mappings")
            if "lease" not in raw_entry or "last_heartbeat" not in raw_entry:
                raise KeyError("inflight entry missing required fields")
            start, end = raw_entry["lease"]
            lease = Lease(int(start), int(end))
            last_hb = float(raw_entry["last_heartbeat"])
            inflight[rank] = _LeaseRecord(lease=lease, last_heartbeat=last_hb)

        pending_queue: Deque[Lease] = deque()
        for entry in pending_raw:
            start, end = entry
            pending_queue.append(Lease(int(start), int(end)))

        with self._lock:
            self._total_chunks = total_chunks
            self._next_idx = next_idx
            self._inflight = inflight
            self._pending_requeue = pending_queue

