"""Unit tests for lease scheduling primitives."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

from ._stubs import install_package_stub, install_torch_stub


install_package_stub()
install_torch_stub()

_LEASE_MODULE_PATH = Path(__file__).resolve().parents[1] / "gpu_tokenizer" / "lease_queue.py"
_LEASE_SPEC = importlib.util.spec_from_file_location("gpu_tokenizer.lease_queue", _LEASE_MODULE_PATH)
assert _LEASE_SPEC and _LEASE_SPEC.loader

_LEASE_MODULE = importlib.util.module_from_spec(_LEASE_SPEC)
sys.modules[_LEASE_SPEC.name] = _LEASE_MODULE
_LEASE_SPEC.loader.exec_module(_LEASE_MODULE)

LeaseNotary = _LEASE_MODULE.LeaseNotary


def _sorted_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return sorted(intervals, key=lambda pair: (pair[0], pair[1]))


def test_unique_lease_assignment_without_overlap() -> None:
    """Each lease should cover a unique interval until all chunks are consumed."""

    notary = LeaseNotary(total_chunks=18, lease_ttl=10.0, min_lease_size=2)

    completed: List[Tuple[int, int]] = []
    chunk_ids: List[int] = []

    # Alternate between two ranks to exercise the inflight bookkeeping.
    rank = 0
    while True:
        lease = notary.grant_lease(rank=rank, preferred_size=3)
        if lease is None:
            break
        start, end, chunk_id = lease
        notary.heartbeat(rank)
        notary.complete_lease(rank=rank, start=start, end=end, chunk_id=chunk_id)
        completed.append((start, end))
        chunk_ids.append(chunk_id)
        rank = 1 - rank

    assert completed, "at least one lease should be issued"

    # Check that the chunk identifiers are unique and monotonic.
    assert chunk_ids == sorted(chunk_ids)
    assert len(chunk_ids) == len(set(chunk_ids))

    # The granted intervals must tile the full [0, total_chunks) range without gaps.
    expected_start = 0
    for start, end in _sorted_intervals(completed):
        assert start == expected_start
        assert end > start
        expected_start = end
    assert expected_start == 18


def test_timeout_requeues_and_regen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expired leases are requeued and handed out to other ranks."""

    notary = LeaseNotary(total_chunks=6, lease_ttl=1.0)
    lease = notary.grant_lease(rank=0, preferred_size=4)
    assert lease is not None

    # Pretend time has advanced beyond the TTL so the inflight lease expires.
    base = notary._now()  # noqa: SLF001 - accessing helper for deterministic tests
    monkeypatch.setattr(notary, "_now", lambda: base + 5.0)

    timed_out = notary.check_timeouts()
    assert timed_out == {0: lease}

    # The requeued lease should be the next one granted, preserving the chunk id.
    reissued = notary.grant_lease(rank=1, preferred_size=2)
    assert reissued == lease

    # After completion the queue should advance to the remaining work.
    notary.complete_lease(rank=1, start=lease[0], end=lease[1], chunk_id=lease[2])
    follow_up = notary.grant_lease(rank=1, preferred_size=2)
    assert follow_up is not None
    assert follow_up[0] == lease[1]


def test_state_serialisation_round_trip() -> None:
    """Serialised notary state must faithfully restore inflight and pending leases."""

    notary = LeaseNotary(total_chunks=10, lease_ttl=3.0, max_active_leases=3)

    active = notary.grant_lease(rank=0, preferred_size=3)
    pending = notary.grant_lease(rank=1, preferred_size=4)
    assert active is not None and pending is not None

    notary.record_idle(rank=0, duration_s=0.25)
    notary.heartbeat(rank=1)
    notary.requeue_lease(rank=0, start=active[0], end=active[1], chunk_id=active[2])

    snapshot = notary.state_dict()
    restored = LeaseNotary(total_chunks=0, lease_ttl=1.0)
    restored.load_state_dict(snapshot)

    assert restored.state_dict() == snapshot

    # Confirm that restored instance can continue operating correctly.
    reassigned = restored.grant_lease(rank=2, preferred_size=2)
    assert reassigned == active
    restored.complete_lease(rank=2, start=reassigned[0], end=reassigned[1], chunk_id=reassigned[2])

    final = restored.grant_lease(rank=1, preferred_size=2)
    assert final is not None
    assert final[0] == pending[1]
