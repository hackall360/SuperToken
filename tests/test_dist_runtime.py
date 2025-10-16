"""Integration-style tests for the distributed runtime helpers."""

from __future__ import annotations

import importlib
import pathlib
import queue
import sys
import types
import threading

import pytest


def _load_dist_runtime(monkeypatch: pytest.MonkeyPatch):
    package = types.ModuleType("gpu_tokenizer")
    package.__path__ = [str(pathlib.Path(__file__).resolve().parents[1] / "gpu_tokenizer")]
    monkeypatch.setitem(sys.modules, "gpu_tokenizer", package)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        set_device=lambda *_args, **_kwargs: None,
    )
    fake_torch.device = lambda kind, index: f"{kind}:{index}"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_dist = types.ModuleType("torch.distributed")
    fake_dist.is_available = lambda: True
    fake_dist.is_initialized = lambda: False
    fake_dist.destroy_process_group = lambda: None
    fake_dist.init_process_group = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)

    fake_mp = types.ModuleType("torch.multiprocessing")
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", fake_mp)

    return importlib.import_module("gpu_tokenizer.dist_runtime")


class _FakeQueue:
    def __init__(self, payload: object | None = None) -> None:
        self._payload = payload

    def get_nowait(self) -> object:
        if self._payload is None:
            raise queue.Empty()
        payload = self._payload
        self._payload = None
        return payload


class _FakeProcessContext:
    def __init__(self, exitcodes: list[int], errors: list[object | None]) -> None:
        self.processes = [types.SimpleNamespace(exitcode=code) for code in exitcodes]
        self.error_queues = [_FakeQueue(payload) for payload in errors]
        self.join_called = False
        self.terminated = False

    def join(self) -> None:
        self.join_called = True

    def terminate(self) -> None:
        self.terminated = True


def _patch_common_runtime(monkeypatch: pytest.MonkeyPatch):
    dist_runtime = _load_dist_runtime(monkeypatch)

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            set_device=lambda *_args, **_kwargs: None,
        ),
        device=lambda kind, index: f"{kind}:{index}",
    )
    monkeypatch.setattr(dist_runtime, "torch", fake_torch, raising=False)

    destroy_calls: list[int] = []

    def _destroy() -> None:
        destroy_calls.append(1)

    broadcast_payload: list[object] = []
    broadcast_log: list[tuple[int, object]] = []

    def _broadcast(obj_list, *, src=0, group=None):
        if obj_list:
            broadcast_log.append((src, obj_list[0]))
        if src == 0 and obj_list:
            broadcast_payload[:] = [obj_list[0]]
        elif obj_list and broadcast_payload:
            obj_list[0] = broadcast_payload[0]

    fake_dist = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        init_process_group=lambda **_kwargs: None,
        destroy_process_group=_destroy,
        broadcast_object_list=_broadcast,
    )
    monkeypatch.setattr(dist_runtime, "dist", fake_dist, raising=False)

    handler_map: dict[int, object] = {}
    installed_handlers: dict[int, list[object]] = {}

    def fake_signal(sig: int, handler):
        previous = handler_map.get(sig)
        handler_map[sig] = handler
        if handler is not None:
            installed_handlers.setdefault(sig, []).append(handler)
        return previous

    monkeypatch.setattr(dist_runtime.signal, "signal", fake_signal)
    monkeypatch.setattr(dist_runtime.signal, "getsignal", lambda sig: handler_map.get(sig))

    registered_cleanup: list[object] = []
    monkeypatch.setattr(dist_runtime.atexit, "register", lambda fn: registered_cleanup.append(fn))

    return dist_runtime, {
        "destroy_calls": destroy_calls,
        "handler_map": handler_map,
        "installed_handlers": installed_handlers,
        "registered_cleanup": registered_cleanup,
        "broadcast_log": broadcast_log,
    }


def test_launch_training_surfaces_worker_exception(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    dist_runtime, artifacts = _patch_common_runtime(monkeypatch)

    context = _FakeProcessContext(exitcodes=[0, 1], errors=[None, RuntimeError("boom")])

    class _FakeMP:
        def __init__(self) -> None:
            self.context = context

        def start_processes(self, *_args, **_kwargs):
            return self.context

    monkeypatch.setattr(dist_runtime, "mp", _FakeMP(), raising=False)

    config = dist_runtime.DistributedLaunchConfig(device_ids=(0, 1), world_size=2)

    caplog.set_level("INFO", logger="gpu_tokenizer.dist_runtime")

    with pytest.raises(RuntimeError) as excinfo:
        dist_runtime.launch_training(config, tuple())

    assert "Rank 1" in str(excinfo.value)
    assert context.join_called
    assert any(getattr(record, "worker_rank", None) == 1 for record in caplog.records)

    assert not artifacts["destroy_calls"]


def test_launch_training_parent_signal_handler_terminates_children(monkeypatch: pytest.MonkeyPatch) -> None:
    dist_runtime, artifacts = _patch_common_runtime(monkeypatch)

    context = _FakeProcessContext(exitcodes=[0], errors=[None])

    class _FakeMP:
        def start_processes(self, *_args, **_kwargs):
            return context

    monkeypatch.setattr(dist_runtime, "mp", _FakeMP(), raising=False)

    config = dist_runtime.DistributedLaunchConfig(device_ids=(0,), world_size=1)

    dist_runtime.launch_training(config, tuple())

    handler = artifacts["installed_handlers"][dist_runtime.signal.SIGTERM][-1]
    handler(dist_runtime.signal.SIGTERM, None)

    assert context.terminated
    assert artifacts["destroy_calls"]


def test_worker_entry_signal_triggers_process_group_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    dist_runtime, artifacts = _patch_common_runtime(monkeypatch)

    class DummyAutoScaler:
        def __init__(self, *, device: str) -> None:
            self._device = device

        def state_dict(self) -> dict[str, object]:
            return {"device": self._device}

    class DummyTrainer:
        def __init__(self, *, devices: list[str], autoscaler: DummyAutoScaler) -> None:
            self.devices = devices
            self.autoscaler = autoscaler

    autoscaler_module = types.ModuleType("gpu_tokenizer.autoscaler")
    autoscaler_module.AutoScaler = DummyAutoScaler
    monkeypatch.setitem(sys.modules, "gpu_tokenizer.autoscaler", autoscaler_module)

    trainer_module = types.ModuleType("gpu_tokenizer.bpe_trainer")
    trainer_module.GPUBPETrainer = DummyTrainer
    monkeypatch.setitem(sys.modules, "gpu_tokenizer.bpe_trainer", trainer_module)

    config = dist_runtime.DistributedLaunchConfig(device_ids=(0,), world_size=1)

    dist_runtime._worker_entry(0, config, tuple())


def test_distributed_client_broadcasts_and_requeues(monkeypatch: pytest.MonkeyPatch) -> None:
    dist_runtime, artifacts = _patch_common_runtime(monkeypatch)

    host_state = dist_runtime._LeaseHostState(
        notary=dist_runtime.LeaseNotary(total_chunks=2, lease_ttl=0.25),
        lock=threading.Lock(),
    )

    worker = dist_runtime.DistributedLeaseClient(
        job_id="job",
        rank=1,
        world_size=2,
        host_state=host_state,
    )

    lease = worker.request_lease(1)
    assert lease == (0, 1)

    worker.complete_lease(*lease)
    state = host_state.notary.state_dict()
    assert state["inflight"] == {}

    lease2 = worker.request_lease(1)
    assert lease2 == (1, 2)

    reclaimed = worker.requeue_outstanding()
    assert reclaimed == lease2

    state_after = host_state.notary.state_dict()
    assert 1 not in state_after["inflight"]
    assert state_after["pending_requeue"] == [lease2]

    events = [payload for _src, payload in artifacts["broadcast_log"] if isinstance(payload, dict)]
    complete_events = [event for event in events if event.get("event") == "complete"]
    requeue_events = [event for event in events if event.get("event") == "requeue"]

    assert complete_events and complete_events[-1]["lease"] == (0, 1)
    assert requeue_events and requeue_events[-1]["lease"] == (1, 2)

def test_distributed_lease_client_assigns_disjoint_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    dist_runtime, _ = _patch_common_runtime(monkeypatch)
    dist_runtime._reset_lease_registry()

    chunk_slices = dist_runtime.plan_chunk_slices(
        6, target_ms=50.0, batch_tokens=1, ewma=None
    )
    assert chunk_slices, "chunk planning should produce slices"

    client0 = dist_runtime.register_lease_client(
        job_id="test-job", total_chunks=len(chunk_slices), rank=0, world_size=2
    )
    client1 = dist_runtime.register_lease_client(
        job_id="test-job", total_chunks=len(chunk_slices), rank=1, world_size=2
    )

    assigned: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    while True:
        lease0 = client0.request_lease(1)
        lease1 = client1.request_lease(1)
        if lease0 is None and lease1 is None:
            break
        if lease0 is not None:
            assigned[0].append(lease0)
            client0.complete_lease(*lease0)
        if lease1 is not None:
            assigned[1].append(lease1)
            client1.complete_lease(*lease1)

    all_leases = assigned[0] + assigned[1]
    assert all_leases, "at least one lease should be granted"
    seen: set[int] = set()
    for start, end in all_leases:
        for idx in range(start, end):
            assert idx not in seen, "leases must be disjoint across ranks"
            seen.add(idx)

    assert seen == set(range(len(chunk_slices)))


def test_requeued_leases_are_reassigned(monkeypatch: pytest.MonkeyPatch) -> None:
    dist_runtime, _ = _patch_common_runtime(monkeypatch)
    dist_runtime._reset_lease_registry()

    chunk_slices = dist_runtime.plan_chunk_slices(
        4, target_ms=50.0, batch_tokens=1, ewma=None
    )
    client0 = dist_runtime.register_lease_client(
        job_id="requeue-job", total_chunks=len(chunk_slices), rank=0, world_size=2
    )
    client1 = dist_runtime.register_lease_client(
        job_id="requeue-job", total_chunks=len(chunk_slices), rank=1, world_size=2
    )

    first = client0.request_lease(2)
    assert first is not None
    client0.requeue_lease(*first)

    reassigned = client1.request_lease(1)
    assert reassigned is not None
    assert reassigned[0] == first[0], "requeue should return the same starting chunk"
    client1.complete_lease(*reassigned)

    next_lease = client0.request_lease(1)
    assert next_lease is not None
    assert next_lease[0] >= reassigned[1]
    client0.complete_lease(*next_lease)

    remaining: list[tuple[int, int]] = []
    while True:
        l0 = client0.request_lease(1)
        l1 = client1.request_lease(1)
        if l0 is None and l1 is None:
            break
        if l0 is not None:
            remaining.append(l0)
            client0.complete_lease(*l0)
        if l1 is not None:
            remaining.append(l1)
            client1.complete_lease(*l1)

    covered = set()
    for start, end in remaining:
        covered.update(range(start, end))

    assert set(range(len(chunk_slices))).issubset(
        covered | set(range(reassigned[0], reassigned[1])) | set(range(next_lease[0], next_lease[1]))
    ), "all chunk indices should eventually be processed"
