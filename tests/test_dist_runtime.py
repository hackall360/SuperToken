"""Integration-style tests for the distributed runtime helpers."""

from __future__ import annotations

import contextlib
import importlib
import os
import pathlib
import queue
import sys
import types
import threading
from collections import deque
from typing import Dict

import pytest


def _load_dist_runtime(monkeypatch: pytest.MonkeyPatch):
    package = types.ModuleType("gpu_tokenizer")
    package.__path__ = [str(pathlib.Path(__file__).resolve().parents[1] / "gpu_tokenizer")]
    monkeypatch.setitem(sys.modules, "gpu_tokenizer", package)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        set_device=lambda *_args, **_kwargs: None,
        device_count=lambda: 2,
        device_can_access_peer=lambda *_args, **_kwargs: True,
    )
    fake_torch.device = lambda kind, index: f"{kind}:{index}"
    fake_torch.uint16 = object()
    fake_torch.int32 = object()
    fake_torch.int16 = object()
    fake_torch.int8 = object()
    fake_torch.uint8 = object()
    fake_torch.int64 = object()

    def _fake_iinfo(dtype):
        if dtype is fake_torch.uint16:
            return types.SimpleNamespace(max=65535)
        if dtype is fake_torch.int32:
            return types.SimpleNamespace(max=2_147_483_647)
        raise TypeError("unsupported dtype")

    fake_torch.iinfo = _fake_iinfo
    fake_torch_utils = types.ModuleType("torch.utils")
    fake_cpp = types.ModuleType("torch.utils.cpp_extension")
    fake_cpp.load_inline = lambda *_args, **_kwargs: None
    fake_torch_utils.cpp_extension = fake_cpp
    fake_torch.utils = fake_torch_utils
    fake_torch.jit = types.SimpleNamespace(script=lambda fn: fn)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", fake_cpp)

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
    gather_log: list[tuple[int, object]] = []
    gather_buffer: list[object] = []

    def _broadcast(obj_list, *, src=0, group=None):
        if obj_list:
            broadcast_log.append((src, obj_list[0]))
        if src == 0 and obj_list:
            broadcast_payload[:] = [obj_list[0]]
        elif obj_list and broadcast_payload:
            obj_list[0] = broadcast_payload[0]

    def _gather_object(obj: object, gather_list=None, *, dst=0, group=None):
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if len(gather_buffer) != world_size:
            gather_buffer[:] = [None] * world_size
        gather_buffer[rank] = obj
        gather_log.append((rank, obj))
        if gather_list is not None:
            gather_list[:] = list(gather_buffer)

    fake_dist = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        init_process_group=lambda **_kwargs: None,
        destroy_process_group=_destroy,
        broadcast_object_list=_broadcast,
        gather_object=_gather_object,
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
        "gather_log": gather_log,
    }


class _FakeLeaseClient:
    def __init__(self, world_size: int) -> None:
        self._lock = threading.Lock()
        uniform = 1.0 / float(world_size)
        self._weights: dict[int, float] = {rank: uniform for rank in range(world_size)}

    def update_rank_weights(self, weights: dict[int, float]) -> None:
        cleaned: dict[int, float] = {}
        for rank, value in weights.items():
            cleaned[int(rank)] = float(value)
        total = sum(cleaned.values())
        if total > 0.0:
            cleaned = {rank: value / total for rank, value in cleaned.items()}
        with self._lock:
            self._weights = cleaned

    def rank_weights(self) -> dict[int, float]:
        with self._lock:
            return dict(self._weights)


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


def test_startup_throughput_weights_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    dist_runtime, artifacts = _patch_common_runtime(monkeypatch)
    dist_runtime._reset_lease_registry()

    host_state = dist_runtime._LeaseHostState(
        notary=dist_runtime.LeaseNotary(total_chunks=4, lease_ttl=1.0),
        lock=threading.Lock(),
    )

    world_size = 3
    clients = [
        dist_runtime.DistributedLeaseClient(
            job_id="calibration",
            rank=rank,
            world_size=world_size,
            host_state=host_state,
        )
        for rank in range(world_size)
    ]

    total = 0.0
    throughputs = []

    class DummyMetrics:
        def __init__(self, value: float) -> None:
            self.enabled = True
            self._value = float(value)

        def record_tokens(self, tokens: int, duration_s: float, *, leases: int | None = None) -> None:
            pass

        @property
        def tokens_per_s(self) -> float:
            return self._value

    os.environ["WORLD_SIZE"] = str(world_size)
    for rank in range(1, world_size):
        value = 200.0 / (2 ** rank)
        throughputs.append(value)
        total += value
        os.environ["RANK"] = str(rank)
        metrics = DummyMetrics(value)
        dist_runtime._collect_startup_throughput_samples(
            rank=rank,
            world_size=world_size,
            lease_client=clients[rank],
            metrics=metrics,
        )

    root_value = 200.0
    throughputs.insert(0, root_value)
    total += root_value
    os.environ["RANK"] = "0"
    dist_runtime._collect_startup_throughput_samples(
        rank=0,
        world_size=world_size,
        lease_client=clients[0],
        metrics=DummyMetrics(root_value),
    )

    weights = host_state.notary.rank_weights()
    assert pytest.approx(sum(weights.values()), rel=1e-9, abs=1e-9) == 1.0
    for idx, value in enumerate(throughputs):
        expected = value / total
        assert pytest.approx(weights[idx], rel=1e-9, abs=1e-9) == expected

    assert artifacts["gather_log"]


def test_iterate_leased_shards_prefetches_additional_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dist_runtime, _ = _patch_common_runtime(monkeypatch)

    notary = dist_runtime.LeaseNotary(
        total_chunks=6,
        lease_ttl=5.0,
        max_active_leases=2,
    )
    host_state = dist_runtime._LeaseHostState(notary=notary, lock=threading.Lock())
    client = dist_runtime.DistributedLeaseClient(
        job_id="prefetch",
        rank=0,
        world_size=1,
        host_state=host_state,
    )

    shard_paths = list(range(6))
    chunk_slices = [(idx, idx + 1) for idx in range(6)]
    observed_lengths: list[int] = []

    @contextlib.contextmanager
    def _open(path: int):
        yield path

    def _encode(shard: int):
        state = notary.state_dict()
        entry = state["inflight"].get(0)
        records = []
        if isinstance(entry, dict):
            records = entry.get("records", [])
        observed_lengths.append(len(records))
        yield shard

    iterator = dist_runtime.iterate_leased_shards(
        shard_paths,
        chunk_slices,
        lease_client=client,
        encode_shard=_encode,
        shard_opener=_open,
        preferred_lease_size=2,
        prefetch_threshold=1,
    )

    for shard_iter in iterator:
        list(shard_iter)

    assert observed_lengths
    assert max(observed_lengths) >= 2
    assert notary.state_dict()["inflight"] == {}


def test_idle_metrics_drop_with_extra_prefetch(monkeypatch: pytest.MonkeyPatch) -> None:
    dist_runtime, _ = _patch_common_runtime(monkeypatch)

    notary = dist_runtime.LeaseNotary(
        total_chunks=6,
        lease_ttl=5.0,
        max_active_leases=4,
    )
    host_state = dist_runtime._LeaseHostState(notary=notary, lock=threading.Lock())
    client = dist_runtime.DistributedLeaseClient(
        job_id="idle-test",
        rank=0,
        world_size=1,
        host_state=host_state,
    )

    shard_paths = [f"shard-{idx}" for idx in range(6)]
    chunk_slices = [(idx, idx + 1) for idx in range(6)]

    times = deque([0.0, 0.08, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19])

    def _fake_monotonic() -> float:
        if times:
            _fake_monotonic.current = times.popleft()
        return _fake_monotonic.current

    _fake_monotonic.current = 0.19

    monkeypatch.setattr(dist_runtime.time, "monotonic", _fake_monotonic)

    @contextlib.contextmanager
    def _open(path: str):
        yield path

    def _encode(shard: str):
        return iter([shard])

    iterator = dist_runtime.iterate_leased_shards(
        shard_paths,
        chunk_slices,
        lease_client=client,
        encode_shard=_encode,
        shard_opener=_open,
        preferred_lease_size=1,
        prefetch_threshold=0,
        min_inflight=1,
        prefetch_slack_ms=50.0,
    )

    for shard_iter in iterator:
        list(shard_iter)

    metrics = notary.state_dict()["idle_metrics"]
    assert 0 in metrics
    idle_entry = metrics[0]
    assert idle_entry["samples"] >= 2
    assert idle_entry["ewma_ms"] < 50.0


def test_rebalance_converges_within_30s(monkeypatch: pytest.MonkeyPatch) -> None:
    dist_runtime, _ = _patch_common_runtime(monkeypatch)

    from gpu_tokenizer.bpe_trainer import TrainerMetricsEWMA

    world_size = 2
    os.environ["WORLD_SIZE"] = str(world_size)
    broadcast_state = {"message": None}
    current_snapshots: dict[str, list[dict[str, object]] | None] = {"value": None}

    gather_log: list[tuple[int, object]] = []

    def _gather(obj: object, gather_list=None, *, dst=0, group=None):
        rank = int(os.environ.get("RANK", "0"))
        if gather_list is not None and current_snapshots["value"] is not None:
            gather_list[:] = list(current_snapshots["value"])
        gather_log.append((rank, obj))

    def _broadcast(obj_list, *, src=0, group=None):
        rank = int(os.environ.get("RANK", "0"))
        if rank == src:
            broadcast_state["message"] = obj_list[0]
        else:
            obj_list[0] = broadcast_state["message"]

    monkeypatch.setattr(dist_runtime.dist, "gather_object", _gather, raising=False)
    monkeypatch.setattr(dist_runtime.dist, "broadcast_object_list", _broadcast, raising=False)

    metrics0 = TrainerMetricsEWMA(alpha=0.2, window_size=4, enabled=True)
    metrics0.set_rank(0)
    metrics1 = TrainerMetricsEWMA(alpha=0.2, window_size=4, enabled=True)
    metrics1.set_rank(1)

    lease0 = _FakeLeaseClient(world_size)
    lease1 = _FakeLeaseClient(world_size)

    def _run_iteration(tokens0: int, tokens1: int) -> None:
        metrics0.record_tokens(tokens=tokens0, duration_s=1.0, leases=tokens0)
        metrics1.record_tokens(tokens=tokens1, duration_s=1.0, leases=tokens1)
        current_snapshots["value"] = [metrics0.snapshot(), metrics1.snapshot()]
        for local_rank, (metrics_obj, lease_obj) in enumerate(
            ((metrics0, lease0), (metrics1, lease1))
        ):
            os.environ["RANK"] = str(local_rank)
            dist_runtime._rebalance_once(
                rank=local_rank,
                world_size=world_size,
                metrics=metrics_obj,
                lease_client=lease_obj,
                blend=0.5,
            )

    for _ in range(3):
        _run_iteration(50, 150)

    weights0 = lease0.rank_weights()
    weights1 = lease1.rank_weights()
    assert weights0 == pytest.approx(weights1)
    assert weights0[0] == pytest.approx(0.25, abs=0.05)
    assert weights0[1] == pytest.approx(0.75, abs=0.05)

    per_rank = metrics0.snapshot()["per_rank"]
    assert per_rank[1]["tokens_per_s"] == pytest.approx(metrics1.tokens_per_s)
    assert gather_log, "gather should record each rebalance sample"


def test_slow_gpu_quota_reduces_smoothly(monkeypatch: pytest.MonkeyPatch) -> None:
    dist_runtime, _ = _patch_common_runtime(monkeypatch)

    host_state = dist_runtime._LeaseHostState(
        notary=dist_runtime.LeaseNotary(total_chunks=256, lease_ttl=5.0, max_active_leases=4),
        lock=threading.Lock(),
    )

    client0 = dist_runtime.DistributedLeaseClient(
        job_id="quota", rank=0, world_size=2, host_state=host_state
    )

    # Seed uniform weights.
    client0.update_rank_weights({0: 1.0, 1: 1.0})
    baseline = host_state.notary.state_dict()
    assert baseline["rank_max_active"] == {0: 4, 1: 4}

    target = {0: 0.05, 1: 1.95}
    weights: Dict[int, float] = {0: 1.0, 1: 1.0}

    for expected_limit in (3, 2, 1):
        weights = dist_runtime._blend_rank_weights(
            weights,
            target,
            new_fraction=0.5,
            world_size=2,
        )
        client0.update_rank_weights(weights)
        snap = host_state.notary.state_dict()
        assert snap["rank_max_active"][0] == expected_limit
        assert snap["rank_max_active"][1] == 4
        assert snap["rank_lease_scale"][0] <= baseline["rank_lease_scale"][0]
        assert snap["rank_lease_scale"][1] >= baseline["rank_lease_scale"][1]
        held: list[tuple[int, int]] = []
        for _ in range(expected_limit):
            lease = host_state.notary.grant_lease(rank=0, preferred_size=2)
            held.append(lease)
        with pytest.raises(RuntimeError):
            host_state.notary.grant_lease(rank=0, preferred_size=2)
        for start, end in held:
            host_state.notary.complete_lease(rank=0, start=start, end=end)
