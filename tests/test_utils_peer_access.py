"""Tests for CUDA peer-to-peer helper utilities."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import pytest


def _snapshot_torch_modules() -> dict[str, object]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "torch" or name.startswith("torch.")
    }


def _restore_torch_modules(snapshot: dict[str, object]) -> None:
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            sys.modules.pop(name, None)
    sys.modules.update(snapshot)


def _install_package_stub() -> None:
    if "gpu_tokenizer" in sys.modules:
        return

    package = types.ModuleType("gpu_tokenizer")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "gpu_tokenizer")]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "gpu_tokenizer", loader=None, is_package=True
    )
    sys.modules["gpu_tokenizer"] = package


def _install_metrics_stub() -> None:
    if "gpu_tokenizer.trainers.metrics" in sys.modules:
        return

    module = types.ModuleType("gpu_tokenizer.trainers.metrics")

    class TrainerMetricsEWMA:
        def __init__(
            self,
            alpha: float = 0.2,
            window_size: int = 16,
            enabled: bool = False,
            overlap_enabled: bool = True,
        ) -> None:
            self.alpha = float(alpha)
            self.window_size = int(window_size)
            self.enabled = bool(enabled)
            self.overlap_enabled = bool(overlap_enabled)
            self._tokens_per_s: float | None = None
            self._lease_per_s: float | None = None

        @property
        def tokens_per_s(self) -> float | None:
            return self._tokens_per_s

        @property
        def lease_per_s(self) -> float | None:
            return self._lease_per_s

        def reset(self) -> None:
            self._tokens_per_s = None
            self._lease_per_s = None

        def record_stage(self, stage: str, duration_s: float) -> str:
            if stage in {"h2d", "d2h"}:
                return "copy"
            if stage == "kernel":
                return "compute"
            return "other"

        def record_tokens(
            self,
            tokens: int,
            duration_s: float,
            *,
            leases: int | None = None,
        ) -> None:
            if not self.enabled or duration_s <= 0:
                return
            rate = float(tokens) / float(duration_s) if tokens > 0 else 0.0
            if self._tokens_per_s is None:
                self._tokens_per_s = rate
            else:
                self._tokens_per_s = (
                    self.alpha * rate + (1.0 - self.alpha) * self._tokens_per_s
                )
            if leases is not None:
                lease_rate = float(leases) / float(duration_s) if leases > 0 else 0.0
                if self._lease_per_s is None:
                    self._lease_per_s = lease_rate
                else:
                    self._lease_per_s = (
                        self.alpha * lease_rate
                        + (1.0 - self.alpha) * self._lease_per_s
                    )

    module.TrainerMetricsEWMA = TrainerMetricsEWMA
    trainers_pkg = sys.modules.setdefault(
        "gpu_tokenizer.trainers", types.ModuleType("gpu_tokenizer.trainers")
    )
    setattr(trainers_pkg, "metrics", module)
    sys.modules["gpu_tokenizer.trainers.metrics"] = module


def _install_torch_stub() -> None:
    for name in [
        "torch",
        "torch.distributed",
        "torch.utils",
        "torch.utils.cpp_extension",
    ]:
        sys.modules.pop(name, None)

    torch_stub = types.ModuleType("torch")
    torch_stub._SUPERTOKEN_TORCH_STUB = True

    class _DeviceContext:
        def __init__(self, cuda_mod: "_CudaStub", index: int) -> None:
            self._cuda = cuda_mod
            self._index = index

        def __enter__(self) -> "_DeviceContext":
            self._cuda.set_device(self._index)
            return self

        def __exit__(self, *_: object) -> bool:
            return False

    class _StreamStub:
        def __init__(self) -> None:
            self.contexts: int = 0

    class _StreamContext:
        def __init__(self, stream: _StreamStub) -> None:
            self._stream = stream

        def __enter__(self) -> _StreamStub:
            self._stream.contexts += 1
            return self._stream

        def __exit__(self, *_: object) -> bool:
            return False

    class _CudaStub:
        def __init__(self) -> None:
            self._available = False
            self._count = 0
            self._current = 0
            self.peer_calls: list[tuple[object, int, object, int]] = []
            self.enabled_pairs: list[tuple[int, int]] = []

        def is_available(self) -> bool:
            return self._available

        def device_count(self) -> int:
            return self._count

        def current_device(self) -> int:
            return self._current

        def set_device(self, index: int) -> None:
            self._current = int(index)

        def device(self, identifier: int) -> _DeviceContext:
            return _DeviceContext(self, int(identifier))

        def mem_get_info(self, *_, **__) -> tuple[int, int]:
            return (0, 1)

        def device_can_access_peer(self, *_: int) -> bool:
            return False

        def device_enable_peer_access(self, *peers: int) -> None:
            peer = int(peers[0]) if peers else self._current
            self.enabled_pairs.append((self._current, peer))
            return None

        def Stream(self, *_: object, **__: object) -> _StreamStub:  # pragma: no cover - simple stub
            return _StreamStub()

        def stream(self, stream_obj: _StreamStub) -> _StreamContext:
            return _StreamContext(stream_obj)

        def memcpy_peer_async(
            self,
            dst: object,
            dst_device: int,
            src: object,
            src_device: int,
        ) -> None:
            self.peer_calls.append((dst, int(dst_device), src, int(src_device)))

    torch_stub.cuda = _CudaStub()

    dist_stub = types.ModuleType("torch.distributed")
    dist_stub.is_available = lambda: False
    dist_stub.is_initialized = lambda: False
    torch_stub.distributed = dist_stub

    utils_stub = types.ModuleType("torch.utils")
    cpp_stub = types.ModuleType("torch.utils.cpp_extension")
    cpp_stub.load_inline = lambda *args, **kwargs: None
    utils_stub.cpp_extension = cpp_stub
    torch_stub.utils = utils_stub
    torch_stub.jit = types.SimpleNamespace(script=lambda fn: fn)

    class _DType(str):
        pass

    torch_stub.uint16 = _DType("uint16")
    torch_stub.int32 = _DType("int32")
    torch_stub.int16 = _DType("int16")
    torch_stub.int8 = _DType("int8")
    torch_stub.uint8 = _DType("uint8")

    def iinfo(dtype: _DType) -> types.SimpleNamespace:
        max_val = (1 << 16) - 1 if dtype == torch_stub.uint16 else (1 << 31) - 1
        return types.SimpleNamespace(max=max_val)

    torch_stub.iinfo = iinfo

    def device(type_name: str, index: int | None = None) -> types.SimpleNamespace:
        return types.SimpleNamespace(type=type_name, index=index)

    torch_stub.device = device

    sys.modules["torch"] = torch_stub
    sys.modules["torch.distributed"] = dist_stub
    sys.modules["torch.utils"] = utils_stub
    sys.modules["torch.utils.cpp_extension"] = cpp_stub


_TORCH_SNAPSHOT = _snapshot_torch_modules()
utils: types.ModuleType | None = None


@pytest.fixture(scope="module", autouse=True)
def _torch_stub_env():
    global utils
    _install_package_stub()
    _install_metrics_stub()
    _install_torch_stub()
    sys.modules.pop("gpu_tokenizer.utils", None)
    utils = importlib.import_module("gpu_tokenizer.utils")
    try:
        yield
    finally:
        utils = None
        sys.modules.pop("gpu_tokenizer.utils", None)
        _restore_torch_modules(_TORCH_SNAPSHOT)


@pytest.fixture(autouse=True)
def _clear_peer_cache() -> None:
    assert utils is not None
    utils._clear_cached_peer_access()


def test_can_peer_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        utils.can_peer(0, 1)


def test_can_peer_validates_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="out of range"):
        utils.can_peer(0, 1)


def test_can_peer_caches_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils.torch.cuda, "device_count", lambda: 2)

    calls = {"count": 0}

    def fake_peer(src: int, dst: int) -> bool:
        calls["count"] += 1
        assert src == 0 and dst == 1
        return True

    monkeypatch.setattr(utils.torch.cuda, "device_can_access_peer", fake_peer)

    assert utils.can_peer(0, 1) is True
    assert utils.can_peer(0, 1) is True
    assert calls["count"] == 1


def test_can_peer_same_device_short_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils.torch.cuda, "device_count", lambda: 2)

    sentinel = types.SimpleNamespace(called=False)

    def fake_peer(src: int, dst: int) -> bool:  # pragma: no cover - defensive
        sentinel.called = True
        return False

    monkeypatch.setattr(utils.torch.cuda, "device_can_access_peer", fake_peer)

    assert utils.can_peer(1, 1) is True
    assert sentinel.called is False


def test_peer_copy_tensor_prefers_peer_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda_stub = utils.torch.cuda
    cuda_stub._available = True
    cuda_stub._count = 2

    src_tensor = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda", index=0))
    dst_tensor = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda", index=1))

    monkeypatch.setattr(utils, "can_peer", lambda src, dst: True)

    stream = cuda_stub.Stream()
    assert utils.peer_copy_tensor(dst_tensor, src_tensor, stream=stream) is True
    assert cuda_stub.peer_calls == [(dst_tensor, 1, src_tensor, 0)]


def test_peer_copy_tensor_falls_back_without_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda_stub = utils.torch.cuda
    cuda_stub.peer_calls.clear()
    cuda_stub._available = True
    cuda_stub._count = 2

    src_tensor = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda", index=0))
    dst_tensor = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda", index=1))

    monkeypatch.setattr(utils, "can_peer", lambda *_: False)

    stream = cuda_stub.Stream()
    assert utils.peer_copy_tensor(dst_tensor, src_tensor, stream=stream) is False
    assert cuda_stub.peer_calls == []
