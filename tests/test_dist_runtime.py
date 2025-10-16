"""Integration-style tests for the distributed runtime helpers."""

from __future__ import annotations

import importlib
import pathlib
import queue
import sys
import types

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

    fake_dist = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        init_process_group=lambda **_kwargs: None,
        destroy_process_group=_destroy,
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

    handler = artifacts["installed_handlers"][dist_runtime.signal.SIGINT][-1]
    handler(dist_runtime.signal.SIGINT, None)

    assert artifacts["destroy_calls"]
    assert artifacts["registered_cleanup"], "cleanup should be registered"
