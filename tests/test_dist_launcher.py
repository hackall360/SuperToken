import importlib
import sys
import types

import pytest


@pytest.fixture
def launcher_env(monkeypatch):
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch = types.SimpleNamespace(device=lambda spec: spec, cuda=fake_cuda)
    fake_dist = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    base_module = importlib.import_module("gpu_tokenizer.trainers.base")
    base_module = importlib.reload(base_module)

    class _DummyTrainer(base_module.BaseTrainer):
        def fit(self, *args, **kwargs):
            return {}

        def state_dict(self, *args, **kwargs):  # pragma: no cover - simple stub
            return {}

        def load_state_dict(self, *args, **kwargs):  # pragma: no cover - simple stub
            return {}

        def save_checkpoint(self, *args, **kwargs):  # pragma: no cover - simple stub
            return {}

        def load_checkpoint(self, *args, **kwargs):  # pragma: no cover - simple stub
            return {}

        def save_artifacts(self, *args, **kwargs):  # pragma: no cover - simple stub
            return {}

        def metrics(self):  # pragma: no cover - simple stub
            return {}

    launcher_module = importlib.import_module("gpu_tokenizer.dist.launcher")
    launcher_module = importlib.reload(launcher_module)
    return launcher_module, _DummyTrainer


@pytest.fixture
def stub_torch(monkeypatch, launcher_env):
    launcher, _ = launcher_env
    cuda_calls: list[object] = []

    class _FakeDevice:
        def __init__(self, spec: object) -> None:
            text = str(spec)
            if ":" in text:
                dev_type, _, remainder = text.partition(":")
                self.type = dev_type
                self.index = remainder
            else:
                self.type = text
                self.index = None
            self._repr = text

        def __str__(self) -> str:  # pragma: no cover - trivial
            return self._repr

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        set_device=lambda device: cuda_calls.append(device),
    )
    fake_torch = types.SimpleNamespace(device=lambda spec: _FakeDevice(spec), cuda=fake_cuda)
    monkeypatch.setattr(launcher, "torch", fake_torch)
    return cuda_calls


def test_launch_rank_initialises_nccl_and_registers_lease(monkeypatch, launcher_env, stub_torch):
    launcher, DummyTrainer = launcher_env

    class _FakeDist:
        def __init__(self) -> None:
            self.initialized = False
            self.init_calls: list[dict[str, object]] = []
            self.destroy_calls = 0

        def is_available(self) -> bool:
            return True

        def is_initialized(self) -> bool:
            return self.initialized

        def init_process_group(self, **kwargs):
            self.init_calls.append(dict(kwargs))
            self.initialized = True

        def destroy_process_group(self):
            self.destroy_calls += 1
            self.initialized = False

    fake_dist = _FakeDist()
    monkeypatch.setattr(launcher, "dist", fake_dist)

    def _default_reducer(keys, counts):
        return ("default_keys", "default_counts")

    monkeypatch.setattr(launcher.utils, "reduce_pair_histograms", _default_reducer)

    lease_calls: list[dict[str, object]] = []

    def _register_lease(**kwargs):
        lease_calls.append(dict(kwargs))
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(launcher, "register_lease_client", _register_lease)

    contexts: list[launcher.LaunchContext] = []

    def _factory(context: launcher.LaunchContext) -> DummyTrainer:
        contexts.append(context)
        return DummyTrainer()

    config = launcher.RankLaunchConfig(
        rank=0,
        world_size=2,
        init_method="env://",
        timeout_seconds=5.0,
        lease_job_id="job-123",
        lease_total_chunks=8,
        lease_max_active_leases=3,
        device="cuda:0",
    )

    with launcher.launch_rank(_factory, config=config) as handle:
        assert isinstance(handle.trainer, DummyTrainer)
        assert contexts and contexts[0] is handle.context
        assert handle.context.lease_client is not None
        assert fake_dist.init_calls and fake_dist.init_calls[0]["backend"] == "nccl"
        assert lease_calls and lease_calls[0]["rank"] == 0
        reducer = launcher.get_histogram_reducer()
        assert reducer("a", "b") == ("default_keys", "default_counts")

    assert fake_dist.destroy_calls == 1
    assert launcher.get_histogram_reducer()("x", "y") == ("default_keys", "default_counts")
    assert stub_torch


def test_launch_rank_custom_histogram_reducer(monkeypatch, launcher_env):
    launcher, DummyTrainer = launcher_env
    default_marker = object()

    def _default_reducer(keys, counts):  # pragma: no cover - trivial stub
        return default_marker

    monkeypatch.setattr(launcher.utils, "reduce_pair_histograms", _default_reducer)

    contexts: list[launcher.LaunchContext] = []

    def _factory(context: launcher.LaunchContext) -> DummyTrainer:
        contexts.append(context)
        return DummyTrainer()

    custom_marker = object()

    def _custom_reducer(keys, counts):
        return custom_marker

    config = launcher.RankLaunchConfig(
        rank=0,
        world_size=1,
        histogram_reducer=_custom_reducer,
    )

    with launcher.launch_rank(_factory, config=config):
        assert launcher.get_histogram_reducer()(None, None) is custom_marker

    assert launcher.get_histogram_reducer()(None, None) is default_marker
    assert contexts and contexts[0].world_size == 1
