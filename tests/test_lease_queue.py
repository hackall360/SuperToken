"""Tests for :mod:`gpu_tokenizer.lease_queue`."""

from __future__ import annotations

import importlib.util
import pathlib
import random
import sys
import threading
import time
import types
from collections import Counter
from typing import List, Tuple

import pytest

_LEASE_QUEUE_PATH = pathlib.Path(__file__).resolve().parents[1] / "gpu_tokenizer" / "lease_queue.py"
_SPEC = importlib.util.spec_from_file_location("gpu_tokenizer.lease_queue", _LEASE_QUEUE_PATH)
assert _SPEC and _SPEC.loader

if "gpu_tokenizer" not in sys.modules:
    pkg = types.ModuleType("gpu_tokenizer")
    pkg.__path__ = [str(_LEASE_QUEUE_PATH.parent)]
    sys.modules["gpu_tokenizer"] = pkg

_LEASE_QUEUE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEASE_QUEUE
_SPEC.loader.exec_module(_LEASE_QUEUE)
LeaseNotary = _LEASE_QUEUE.LeaseNotary


def _sorted_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return sorted(intervals, key=lambda pair: (pair[0], pair[1]))


def test_basic_grant_and_complete() -> None:
    notary = LeaseNotary(total_chunks=10, lease_ttl=5.0)

    lease = notary.grant_lease(rank=0, preferred_size=4)
    assert lease == (0, 4)

    heartbeat_ts = notary.heartbeat(rank=0)
    assert isinstance(heartbeat_ts, float)

    notary.complete_lease(rank=0, start=0, end=4)
    assert notary.state_dict()["inflight"] == {}

    lease = notary.grant_lease(rank=1, preferred_size=8)
    assert lease == (4, 10)

    notary.complete_lease(rank=1, start=4, end=10)
    assert notary.grant_lease(rank=1, preferred_size=1) is None


def test_requeue_prioritised_before_new_work() -> None:
    notary = LeaseNotary(total_chunks=6, lease_ttl=5.0)

    lease = notary.grant_lease(rank=0, preferred_size=6)
    assert lease == (0, 6)

    notary.requeue_lease(rank=0, start=0, end=6)

    lease = notary.grant_lease(rank=1, preferred_size=2)
    assert lease == (0, 6)

    notary.complete_lease(rank=1, start=0, end=6)
    assert notary.grant_lease(rank=2, preferred_size=3) is None


def test_state_dict_roundtrip() -> None:
    notary = LeaseNotary(total_chunks=12, lease_ttl=7.5)

    lease_a = notary.grant_lease(rank=0, preferred_size=5)
    lease_b = notary.grant_lease(rank=1, preferred_size=5)

    assert lease_a == (0, 5)
    assert lease_b == (5, 10)

    notary.heartbeat(rank=1)
    notary.requeue_lease(rank=0, start=0, end=5)

    snap = notary.state_dict()
    assert "rank_weights" in snap
    assert snap["rank_weights"] == {}

    restored = LeaseNotary(total_chunks=0, lease_ttl=3.0)
    restored.load_state_dict(snap)

    assert restored.state_dict() == snap

    lease = restored.grant_lease(rank=2, preferred_size=5)
    assert lease == (0, 5)

    restored.complete_lease(rank=2, start=0, end=5)
    restored.complete_lease(rank=1, start=5, end=10)
    assert restored.grant_lease(rank=3, preferred_size=5) == (10, 12)


def test_concurrent_grants_produce_unique_intervals() -> None:
    total_chunks = 317
    notary = LeaseNotary(total_chunks=total_chunks, lease_ttl=1.0)

    completed: List[Tuple[int, int]] = []
    completion_lock = threading.Lock()

    def worker(rank: int) -> None:
        rng = random.Random(rank)
        while True:
            size = rng.randint(1, 11)
            lease = notary.grant_lease(rank=rank, preferred_size=size)
            if lease is None:
                break

            start, end = lease

            if rng.random() < 0.15:
                time.sleep(rng.random() * 0.002)
                notary.requeue_lease(rank=rank, start=start, end=end)
                continue

            notary.heartbeat(rank=rank)
            time.sleep(rng.random() * 0.002)
            notary.complete_lease(rank=rank, start=start, end=end)

            with completion_lock:
                completed.append((start, end))

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    state = notary.state_dict()
    assert state["inflight"] == {}
    assert state["pending_requeue"] == []

    assert completed, "workers must complete at least one lease"

    completed_sorted = _sorted_intervals(completed)

    # Ensure the intervals cover the full range without overlaps.
    expected_start = 0
    for start, end in completed_sorted:
        assert start == expected_start
        assert end > start
        expected_start = end

    assert expected_start == total_chunks

    # A Counter-based check guards against duplicate (start, end) pairs.
    counts = Counter(completed)
    assert all(count == 1 for count in counts.values())


def test_reject_invalid_progressions() -> None:
    notary = LeaseNotary(total_chunks=4, lease_ttl=2.0)

    notary.grant_lease(rank=0, preferred_size=2)

    with pytest.raises(ValueError):
        notary.complete_lease(rank=0, start=0, end=1)

    with pytest.raises(ValueError):
        notary.requeue_lease(rank=0, start=1, end=3)

    notary.requeue_lease(rank=0, start=0, end=2)

    notary.grant_lease(rank=1, preferred_size=2)

    with pytest.raises(RuntimeError):
        notary.grant_lease(rank=1, preferred_size=1)


def test_timed_out_leases_requeue_and_are_reattributed() -> None:
    ttl = 0.5
    notary = LeaseNotary(total_chunks=6, lease_ttl=ttl)

    lease_a = notary.grant_lease(rank=0, preferred_size=3)
    lease_b = notary.grant_lease(rank=1, preferred_size=3)
    assert lease_a == (0, 3)
    assert lease_b == (3, 6)

    # Advance time beyond the TTL for rank 0 but keep rank 1 active.
    state = notary.state_dict()
    heartbeat_0 = state["inflight"][0]["last_heartbeat"]

    # Ensure rank 1 remains within the TTL window by nudging its heartbeat forward.
    with notary._lock:  # type: ignore[attr-defined]
        record = notary._inflight[1]  # type: ignore[attr-defined]
        record.last_heartbeat = heartbeat_0 + ttl / 2

    timed_out = notary.check_timeouts(now=heartbeat_0 + ttl + 0.01)
    assert timed_out == {0: (0, 3)}

    state_after = notary.state_dict()
    assert 0 not in state_after["inflight"]
    assert state_after["pending_requeue"] and state_after["pending_requeue"][0] == (0, 3)
    assert 1 in state_after["inflight"]

    reassigned = notary.grant_lease(rank=2, preferred_size=3)
    assert reassigned == (0, 3)
    notary.complete_lease(rank=2, start=0, end=3)
    notary.complete_lease(rank=1, start=3, end=6)

    # Ensure no extra work becomes available.
    assert notary.grant_lease(rank=3, preferred_size=1) is None


def test_rank_weights_roundtrip() -> None:
    notary = LeaseNotary(total_chunks=4, lease_ttl=5.0)
    notary.update_rank_weights({0: 0.7, 1: 0.3})

    weights = notary.rank_weights()
    assert weights == {0: 0.7, 1: 0.3}

    snap = notary.state_dict()
    assert snap["rank_weights"] == {0: 0.7, 1: 0.3}

    restored = LeaseNotary(total_chunks=0, lease_ttl=1.0)
    restored.load_state_dict(snap)

    assert restored.rank_weights() == {0: 0.7, 1: 0.3}

