"""Tests for adaptive chunk generation helpers."""

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
    sys.modules["gpu_tokenizer.bpe_trainer"] = module


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

from gpu_tokenizer.bpe_trainer import TrainerMetricsEWMA
from gpu_tokenizer.io import ChunkSpec, make_chunker


def test_make_chunker_validates_inputs() -> None:
    with pytest.raises(ValueError):
        make_chunker(0.0, 128, None)
    with pytest.raises(ValueError):
        make_chunker(10.0, 0, None)


def test_make_chunker_without_metrics() -> None:
    chunker = make_chunker(50.0, 256, None)
    spec = next(chunker)
    assert isinstance(spec, ChunkSpec)
    assert spec.batches == 1
    assert spec.tokens == 256
    assert spec.expected_ms is None
    assert spec.tokens_per_s_hint is None
    assert spec.leases_per_s_hint is None


def test_make_chunker_with_metrics_updates() -> None:
    metrics = TrainerMetricsEWMA(alpha=1.0, enabled=True)
    metrics.record_tokens(tokens=1_000, duration_s=1.0)

    chunker = make_chunker(50.0, 128, metrics)

    first = next(chunker)
    assert first.tokens == 128
    assert pytest.approx(first.expected_ms, rel=1e-6) == 128.0
    assert first.tokens_per_s_hint == pytest.approx(1_000.0)

    metrics.record_tokens(tokens=8_000, duration_s=1.0)

    second = next(chunker)
    assert second.tokens == 384
    assert pytest.approx(second.expected_ms, rel=1e-6) == 48.0
    assert second.tokens_per_s_hint == pytest.approx(8_000.0)
