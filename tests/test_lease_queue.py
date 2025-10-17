"""Tests for :mod:`gpu_tokenizer.lease_queue`."""

from __future__ import annotations

import heapq
import importlib.util
import pathlib
import random
import sys
import threading
import time
import types
from collections import Counter
from typing import Dict, List, Tuple

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
    assert lease is not None
    start, end, chunk_id = lease
    assert (start, end) == (0, 4)
    assert chunk_id == 0

    heartbeat_ts = notary.heartbeat(rank=0)
    assert isinstance(heartbeat_ts, float)

    notary.complete_lease(rank=0, start=start, end=end, chunk_id=chunk_id)
    assert notary.state_dict()["inflight"] == {}

    lease = notary.grant_lease(rank=1, preferred_size=8)
    assert lease is not None
    start, end, chunk_id = lease
    assert (start, end) == (4, 10)
    assert chunk_id == 1

    notary.complete_lease(rank=1, start=start, end=end, chunk_id=chunk_id)
    assert notary.grant_lease(rank=1, preferred_size=1) is None


def test_requeue_prioritised_before_new_work() -> None:
    notary = LeaseNotary(total_chunks=6, lease_ttl=5.0)

    lease = notary.grant_lease(rank=0, preferred_size=6)
    assert lease is not None
    start, end, chunk_id = lease
    assert (start, end) == (0, 6)

    notary.requeue_lease(rank=0, start=start, end=end, chunk_id=chunk_id)

    lease = notary.grant_lease(rank=1, preferred_size=2)
    assert lease is not None
    start, end, chunk_id = lease
    assert (start, end) == (0, 6)

    notary.complete_lease(rank=1, start=start, end=end, chunk_id=chunk_id)
    assert notary.grant_lease(rank=2, preferred_size=3) is None


def test_state_dict_roundtrip() -> None:
    notary = LeaseNotary(total_chunks=12, lease_ttl=7.5)

    lease_a = notary.grant_lease(rank=0, preferred_size=5)
    lease_b = notary.grant_lease(rank=1, preferred_size=5)

    assert lease_a is not None and lease_b is not None
    assert lease_a[:2] == (0, 5)
    assert lease_b[:2] == (5, 10)

    notary.heartbeat(rank=1)
    notary.requeue_lease(rank=0, start=lease_a[0], end=lease_a[1], chunk_id=lease_a[2])

    snap = notary.state_dict()
    assert "rank_weights" in snap
    assert snap["rank_weights"] == {}
    assert snap["rank_lease_scale"] == {}
    assert snap["rank_max_active"] == {}
    assert snap["min_lease_size"] == 1
    assert snap["max_active_leases"] == 2
    assert snap.get("idle_metrics") == {}
    assert snap["rank_heartbeats"] and set(snap["rank_heartbeats"]) == {1}
    assert "records" in snap["inflight"][1]
    assert isinstance(snap["inflight"][1]["records"], list)

    restored = LeaseNotary(total_chunks=0, lease_ttl=3.0)
    restored.load_state_dict(snap)

    assert restored.state_dict() == snap

    lease = restored.grant_lease(rank=2, preferred_size=5)
    assert lease is not None
    start, end, chunk_id = lease
    assert (start, end) == (0, 5)

    restored.complete_lease(rank=2, start=start, end=end, chunk_id=chunk_id)
    restored.complete_lease(rank=1, start=lease_b[0], end=lease_b[1], chunk_id=lease_b[2])
    follow_up = restored.grant_lease(rank=3, preferred_size=5)
    assert follow_up is not None
    assert follow_up[:2] == (10, 12)


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

            start, end, chunk_id = lease

            if rng.random() < 0.15:
                time.sleep(rng.random() * 0.002)
                notary.requeue_lease(rank=rank, start=start, end=end, chunk_id=chunk_id)
                continue

            notary.heartbeat(rank=rank)
            time.sleep(rng.random() * 0.002)
            notary.complete_lease(rank=rank, start=start, end=end, chunk_id=chunk_id)

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
    notary = LeaseNotary(total_chunks=4, lease_ttl=2.0, max_active_leases=1)

    lease = notary.grant_lease(rank=0, preferred_size=2)
    assert lease is not None

    with pytest.raises(ValueError):
        notary.complete_lease(rank=0, start=0, end=1, chunk_id=lease[2])

    with pytest.raises(ValueError):
        notary.requeue_lease(rank=0, start=1, end=3, chunk_id=lease[2])

    notary.requeue_lease(rank=0, start=lease[0], end=lease[1], chunk_id=lease[2])

    lease_other = notary.grant_lease(rank=1, preferred_size=2)
    assert lease_other is not None

    with pytest.raises(RuntimeError):
        notary.grant_lease(rank=1, preferred_size=1)


def test_timed_out_leases_requeue_and_are_reattributed() -> None:
    ttl = 0.5
    notary = LeaseNotary(total_chunks=6, lease_ttl=ttl)

    lease_a = notary.grant_lease(rank=0, preferred_size=3)
    lease_b = notary.grant_lease(rank=1, preferred_size=3)
    assert lease_a is not None and lease_b is not None
    assert lease_a[:2] == (0, 3)
    assert lease_b[:2] == (3, 6)

    # Advance time beyond the TTL for rank 0 but keep rank 1 active.
    state = notary.state_dict()
    heartbeat_0 = state["inflight"][0]["last_heartbeat"]

    # Ensure rank 1 remains within the TTL window by nudging its heartbeat forward.
    with notary._lock:  # type: ignore[attr-defined]
        queue = notary._inflight[1]  # type: ignore[attr-defined]
        queue[0].last_heartbeat = heartbeat_0 + ttl / 2
        notary._rank_heartbeats[1] = heartbeat_0 + ttl / 2  # type: ignore[attr-defined]

    timed_out = notary.check_timeouts(now=heartbeat_0 + ttl + 0.01)
    assert timed_out == {0: lease_a}

    state_after = notary.state_dict()
    assert 0 not in state_after["inflight"]
    assert state_after["pending_requeue"] and state_after["pending_requeue"][0][:2] == (0, 3)
    assert 1 in state_after["inflight"]
    assert 1 in state_after["rank_heartbeats"]


def test_rank_heartbeat_timeout_requeues_all_active_leases() -> None:
    ttl = 0.25
    notary = LeaseNotary(total_chunks=6, lease_ttl=ttl, max_active_leases=2)

    lease_a = notary.grant_lease(rank=0, preferred_size=2)
    lease_b = notary.grant_lease(rank=0, preferred_size=2)
    assert lease_a is not None and lease_b is not None
    assert lease_a[:2] == (0, 2)
    assert lease_b[:2] == (2, 4)

    with notary._lock:  # type: ignore[attr-defined]
        stale = notary._rank_heartbeats[0] - (ttl + 0.1)  # type: ignore[attr-defined]
        notary._rank_heartbeats[0] = stale  # type: ignore[attr-defined]

    timed_out = notary.check_timeouts(now=stale + ttl + 0.05)
    assert timed_out == {0: lease_a}

    state = notary.state_dict()
    assert 0 not in state["inflight"]
    assert 0 not in state["rank_heartbeats"]
    assert sorted(state["pending_requeue"]) == sorted([lease_a, lease_b])


def test_grant_lease_respects_weights_and_bounds() -> None:
    notary = LeaseNotary(
        total_chunks=32,
        lease_ttl=5.0,
        min_lease_size=2,
        max_lease_size=7,
    )
    notary.update_rank_weights({0: 0.2, 1: 1.6})

    lease_fast = notary.grant_lease(rank=1, preferred_size=5)
    assert lease_fast is not None
    assert lease_fast[:2] == (0, 7)

    lease_slow = notary.grant_lease(rank=0, preferred_size=5)
    assert lease_slow is not None
    assert lease_slow[:2] == (7, 9)

    notary.complete_lease(
        rank=1, start=lease_fast[0], end=lease_fast[1], chunk_id=lease_fast[2]
    )
    notary.complete_lease(
        rank=0, start=lease_slow[0], end=lease_slow[1], chunk_id=lease_slow[2]
    )


def test_max_active_leases_enforced() -> None:
    notary = LeaseNotary(total_chunks=12, lease_ttl=5.0, max_active_leases=2)

    first = notary.grant_lease(rank=0, preferred_size=3)
    assert first is not None
    assert first[:2] == (0, 3)

    second = notary.grant_lease(rank=0, preferred_size=3)
    assert second is not None
    assert second[:2] == (3, 6)

    with pytest.raises(RuntimeError):
        notary.grant_lease(rank=0, preferred_size=3)

    notary.complete_lease(rank=0, start=first[0], end=first[1], chunk_id=first[2])

    third = notary.grant_lease(rank=0, preferred_size=3)
    assert third is not None
    assert third[:2] == (6, 9)


def test_weighted_prefetch_simulation_hits_target() -> None:
    total_chunks = 90
    base_lease = 12
    prefetch_threshold = 5
    throughputs = {0: 6.0, 1: 3.5, 2: 2.0}
    weights = {
        rank: value / sum(throughputs.values()) for rank, value in throughputs.items()
    }

    notary = LeaseNotary(
        total_chunks=total_chunks,
        lease_ttl=30.0,
        min_lease_size=2,
        max_lease_size=16,
        max_active_leases=3,
    )
    notary.update_rank_weights(weights)

    outstanding_chunks: Dict[int, int] = {rank: 0 for rank in throughputs}
    available_at: Dict[int, float] = {rank: 0.0 for rank in throughputs}
    assigned: Dict[int, int] = {rank: 0 for rank in throughputs}
    event_queue: List[Tuple[float, int, Tuple[int, int, int, int, int]]] = []
    counter = 0

    def _prefetch(rank: int) -> None:
        nonlocal counter
        while outstanding_chunks[rank] <= prefetch_threshold:
            lease = notary.grant_lease(rank=rank, preferred_size=base_lease)
            if lease is None:
                break
            start, end, chunk_id = lease
            width = max(0, end - start)
            if width == 0:
                notary.complete_lease(rank=rank, start=start, end=end, chunk_id=chunk_id)
                continue
            outstanding_chunks[rank] += width
            assigned[rank] += width
            counter += 1
            start_time = available_at[rank]
            finish = start_time + width / throughputs[rank]
            available_at[rank] = finish
            heapq.heappush(
                event_queue, (finish, counter, (rank, start, end, width, chunk_id))
            )

    for rank in throughputs:
        _prefetch(rank)

    wall_time = 0.0
    while event_queue:
        finish_time, _idx, payload = heapq.heappop(event_queue)
        rank, start, end, width, chunk_id = payload
        wall_time = finish_time
        outstanding_chunks[rank] -= width
        notary.complete_lease(rank=rank, start=start, end=end, chunk_id=chunk_id)
        _prefetch(rank)

    state = notary.state_dict()
    assert state["inflight"] == {}
    assert state["pending_requeue"] == []

    total_assigned = sum(assigned.values())
    assert total_assigned == total_chunks

    assert assigned[0] > assigned[1] > assigned[2]

    aggregate_throughput = sum(throughputs.values())
    target_time = total_chunks / (0.88 * aggregate_throughput)
    assert wall_time <= target_time * 1.05


def test_rank_weights_roundtrip() -> None:
    notary = LeaseNotary(total_chunks=4, lease_ttl=5.0)
    notary.update_rank_weights({0: 0.7, 1: 0.3})

    weights = notary.rank_weights()
    assert weights == {0: 0.7, 1: 0.3}

    snap = notary.state_dict()
    assert snap["rank_weights"] == {0: 0.7, 1: 0.3}
    assert 0 in snap["rank_lease_scale"]
    assert 1 in snap["rank_max_active"]

    restored = LeaseNotary(total_chunks=0, lease_ttl=1.0)
    restored.load_state_dict(snap)

    assert restored.rank_weights() == {0: 0.7, 1: 0.3}


def test_record_idle_tracks_metrics() -> None:
    notary = LeaseNotary(total_chunks=0, lease_ttl=5.0)

    notary.record_idle(rank=0, duration_s=0.120)
    notary.record_idle(rank=0, duration_s=0.020)

    metrics = notary.state_dict()["idle_metrics"]
    assert 0 in metrics
    entry = metrics[0]
    assert entry["samples"] == 2
    assert entry["last_ms"] == pytest.approx(20.0)
    assert entry["ewma_ms"] == pytest.approx((0.2 * 20.0) + (0.8 * 120.0))


def test_chunk_status_tracks_attempts_and_completion() -> None:
    notary = LeaseNotary(total_chunks=3, lease_ttl=5.0)

    lease = notary.grant_lease(rank=0, preferred_size=2)
    assert lease is not None
    chunk_id = lease[2]
    status = notary.chunk_status(chunk_id)
    assert status == {"lease": (0, 2), "completed": False, "attempts": 1}

    notary.requeue_lease(rank=0, start=lease[0], end=lease[1], chunk_id=chunk_id)
    retry = notary.grant_lease(rank=1, preferred_size=1)
    assert retry is not None
    assert retry[2] == chunk_id
    status_retry = notary.chunk_status(chunk_id)
    assert status_retry == {"lease": (0, 2), "completed": False, "attempts": 2}

    notary.complete_lease(rank=1, start=retry[0], end=retry[1], chunk_id=chunk_id)
    final_status = notary.chunk_status(chunk_id)
    assert final_status == {"lease": (0, 2), "completed": True, "attempts": 2}

    snap = notary.state_dict()
    restored = LeaseNotary(total_chunks=0, lease_ttl=5.0)
    restored.load_state_dict(snap)
    assert restored.chunk_status(chunk_id) == final_status


def test_per_rank_quota_adjusts_smoothly() -> None:
    notary = LeaseNotary(total_chunks=200, lease_ttl=10.0, max_active_leases=4)

    notary.update_rank_weights({0: 1.0, 1: 1.0})
    baseline = notary.state_dict()
    assert baseline["rank_max_active"] == {0: 4, 1: 4}
    assert baseline["rank_lease_scale"] == {0: 1.0, 1: 1.0}

    slow_limits: List[int] = []
    fast_scales: List[float] = []
    target_weights = {0: 0.25, 1: 1.75}

    for expected_limit in (3, 2, 1):
        notary.update_rank_weights(target_weights)
        state = notary.state_dict()
        slow_limits.append(state["rank_max_active"][0])
        fast_scales.append(state["rank_lease_scale"][1])
        assert state["rank_max_active"][0] == expected_limit
        assert state["rank_max_active"][1] == 4
        assert state["rank_lease_scale"][0] <= baseline["rank_lease_scale"][0]
        held: List[Tuple[int, int, int]] = []
        for _ in range(expected_limit):
            lease = notary.grant_lease(rank=0, preferred_size=2)
            assert lease is not None
            held.append(lease)
        with pytest.raises(RuntimeError):
            notary.grant_lease(rank=0, preferred_size=2)
        for start, end, chunk_id in held:
            notary.complete_lease(rank=0, start=start, end=end, chunk_id=chunk_id)

    assert slow_limits == [3, 2, 1]
    assert fast_scales[0] >= 1.0
    assert fast_scales[-1] >= fast_scales[0]

