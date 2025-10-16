"""Tests for CUDA peer-to-peer helper utilities."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

import pytest


def _install_package_stub() -> None:
    if "gpu_tokenizer" in sys.modules:
        return

    package = types.ModuleType("gpu_tokenizer")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "gpu_tokenizer")]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "gpu_tokenizer", loader=None, is_package=True
    )
    sys.modules["gpu_tokenizer"] = package


def _install_bpe_trainer_stub() -> None:
    if "gpu_tokenizer.bpe_trainer" in sys.modules:
        return

    module = types.ModuleType("gpu_tokenizer.bpe_trainer")

    class TrainerMetricsEWMA:
        def __init__(
            self,
            alpha: float = 0.2,
            window_size: int = 16,
            enabled: bool = False,
        ) -> None:
            self.alpha = float(alpha)
            self.window_size = int(window_size)
            self.enabled = bool(enabled)
            self._tokens_per_s: float | None = None
            self._lease_per_s: float | None = None

        @property
        def tokens_per_s(self) -> float | None:
            return self._tokens_per_s

        @property
        def lease_per_s(self) -> float | None:
            return self._lease_per_s

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
    sys.modules["gpu_tokenizer.bpe_trainer"] = module


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return

    torch_stub = types.ModuleType("torch")

    class _CudaStub:
        def __init__(self) -> None:
            self._available = False
            self._count = 0

        def is_available(self) -> bool:
            return self._available

        def device_count(self) -> int:
            return self._count

        def current_device(self) -> int:
            return 0

        def mem_get_info(self, *_, **__) -> tuple[int, int]:
            return (0, 1)

        def device_can_access_peer(self, *_: int) -> bool:
            return False

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


_install_package_stub()
_install_bpe_trainer_stub()
_install_torch_stub()

import gpu_tokenizer.utils as utils


@pytest.fixture(autouse=True)
def _clear_peer_cache() -> None:
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
