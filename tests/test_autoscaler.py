"""Unit tests for the adaptive autoscaler."""

from __future__ import annotations

import json
import logging
import sys
import types
from dataclasses import asdict

import pytest

if "torch" not in sys.modules:  # pragma: no cover - testing convenience
    cuda_stub = types.SimpleNamespace(
        is_available=lambda: False,
        mem_get_info=lambda: (0, 0),
        empty_cache=lambda: None,
        max_memory_allocated=lambda *args, **kwargs: 0,
    )
    jit_stub = types.SimpleNamespace(script=lambda fn: fn)
    uint16 = object()
    int32 = object()
    int16 = object()
    int8 = object()
    uint8 = object()
    int64 = object()

    def _iinfo(dtype):
        max_map = {
            uint16: (1 << 16) - 1,
            int32: (1 << 31) - 1,
        }
        return types.SimpleNamespace(max=max_map.get(dtype, (1 << 63) - 1))

    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = cuda_stub
    torch_stub.jit = jit_stub
    torch_stub.uint16 = uint16
    torch_stub.int32 = int32
    torch_stub.int16 = int16
    torch_stub.int8 = int8
    torch_stub.uint8 = uint8
    torch_stub.int64 = int64
    torch_stub.device = lambda name="cpu": name
    torch_stub.iinfo = _iinfo
    torch_stub.Tensor = object  # pragma: no cover - structural typing only

    sys.modules["torch"] = torch_stub
    torch_utils = types.ModuleType("torch.utils")
    torch_cpp_ext = types.ModuleType("torch.utils.cpp_extension")
    torch_cpp_ext.load_inline = lambda *args, **kwargs: None
    torch_utils.cpp_extension = torch_cpp_ext
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.cpp_extension"] = torch_cpp_ext
    torch_distributed = types.ModuleType("torch.distributed")
    torch_distributed.is_available = lambda: False
    sys.modules["torch.distributed"] = torch_distributed

from gpu_tokenizer.autoscaler import AutoScaler


class StubAutoScaler(AutoScaler):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("device", "cuda")
        super().__init__(*args, **kwargs)
        self._sim_free = 0
        self._sim_total = 0

    def set_gpu(self, free: int, total: int) -> None:
        self._sim_free = free
        self._sim_total = total

    def _gpu_caps(self) -> tuple[int, int]:  # pragma: no cover - simple override
        return int(self._sim_free), int(self._sim_total)

    def _cpu_caps(self) -> tuple[int, int, int, float]:  # pragma: no cover - deterministic override
        return 8, 1 << 30, 1 << 30, 0.0


def drain_feedback(scaler: StubAutoScaler, samples: int, free: int, total: int, step_time: float) -> None:
    for _ in range(samples):
        scaler.set_gpu(free, total)
        scaler.feedback(step_time_s=step_time)


def test_autoscaler_increases_batch_size_with_low_utilization(caplog: pytest.LogCaptureFixture) -> None:
    scaler = StubAutoScaler(window_size=4, min_bs=64, max_bs=2048, init_h2d_mb=512, target_util=0.85)
    scaler.set_gpu(900_000_000, 1_000_000_000)
    initial_state = scaler.suggest(token_bytes_per_example=5_000_000)
    caplog.set_level(logging.INFO)
    drain_feedback(scaler, samples=4, free=900_000_000, total=1_000_000_000, step_time=0.5)
    assert scaler.state.batch_size > initial_state.batch_size
    logs = [json.loads(record.message.split(" ", 1)[1]) for record in caplog.records if record.message.startswith("autoscale.adjust")]
    assert logs, "expected structured adjustment logs"
    assert logs[-1]["batch_size"] == scaler.state.batch_size


def test_autoscaler_reduces_batch_size_with_high_utilization(caplog: pytest.LogCaptureFixture) -> None:
    scaler = StubAutoScaler(window_size=4, min_bs=64, max_bs=2048, init_h2d_mb=1024, target_util=0.82)
    scaler.set_gpu(800_000_000, 1_000_000_000)
    initial_state = scaler.suggest(token_bytes_per_example=5_000_000)
    caplog.set_level(logging.INFO)
    for _ in range(4):
        scaler.set_gpu(40_000_000, 1_000_000_000)
        scaler.feedback(step_time_s=0.6)
    assert scaler.state.batch_size < initial_state.batch_size
    logs = [json.loads(record.message.split(" ", 1)[1]) for record in caplog.records if record.message.startswith("autoscale.adjust")]
    assert any(entry["batch_size"] < entry["prev_batch_size"] for entry in logs)


def test_autoscaler_dampens_growth_when_vram_variance_high() -> None:
    scaler_low_var = StubAutoScaler(window_size=6, min_bs=64, max_bs=2048, init_h2d_mb=512, target_util=0.8)
    scaler_low_var.set_gpu(900_000_000, 1_000_000_000)
    base_state_low = scaler_low_var.suggest(token_bytes_per_example=5_000_000)
    drain_feedback(scaler_low_var, samples=6, free=900_000_000, total=1_000_000_000, step_time=0.4)
    growth_low = scaler_low_var.state.batch_size - base_state_low.batch_size

    scaler_var = StubAutoScaler(window_size=6, min_bs=64, max_bs=2048, init_h2d_mb=512, target_util=0.8)
    scaler_var.set_gpu(900_000_000, 1_000_000_000)
    base_state_var = scaler_var.suggest(token_bytes_per_example=5_000_000)
    for free in (950_000_000, 50_000_000, 950_000_000, 50_000_000, 950_000_000, 50_000_000):
        scaler_var.set_gpu(free, 1_000_000_000)
        scaler_var.feedback(step_time_s=0.4)
    growth_var = scaler_var.state.batch_size - base_state_var.batch_size

    assert growth_low > 0
    assert growth_var >= 0
    assert growth_low >= growth_var


def test_autoscaler_state_roundtrip_preserves_feedback_loop() -> None:
    scaler = StubAutoScaler(window_size=5, min_bs=128, max_bs=4096, init_h2d_mb=768, target_util=0.83)
    scaler.set_gpu(850_000_000, 1_000_000_000)
    initial_state = scaler.suggest(token_bytes_per_example=4_000_000)
    drain_feedback(
        scaler,
        samples=5,
        free=850_000_000,
        total=1_000_000_000,
        step_time=0.42,
    )
    assert scaler.state is not None
    assert scaler.state.batch_size >= initial_state.batch_size

    saved = scaler.state_dict()
    # Ensure payload is JSON-serializable for checkpointing scenarios.
    json.dumps(saved)

    restored = StubAutoScaler(window_size=3, min_bs=64, max_bs=8192, init_h2d_mb=256, target_util=0.5)
    restored.load_state_dict(saved)

    assert restored.device == scaler.device
    assert restored.tu == scaler.tu
    assert list(restored._step_times) == list(scaler._step_times)
    assert list(restored._vram_fracs) == list(scaler._vram_fracs)
    assert restored._h2d_mb == scaler._h2d_mb
    assert restored._window_size == scaler._window_size
    assert restored.state is not None
    assert asdict(restored.state) == asdict(scaler.state)

    # Subsequent suggestions/feedback should produce identical results.
    scaler.set_gpu(750_000_000, 1_000_000_000)
    restored.set_gpu(750_000_000, 1_000_000_000)
    next_state_original = scaler.suggest(token_bytes_per_example=4_000_000)
    next_state_restored = restored.suggest(token_bytes_per_example=4_000_000)
    assert asdict(next_state_original) == asdict(next_state_restored)

    scaler.feedback(step_time_s=0.5, cpu_fallback_rate=0.15)
    restored.feedback(step_time_s=0.5, cpu_fallback_rate=0.15)
    assert restored.state is not None and scaler.state is not None
    assert asdict(restored.state) == asdict(scaler.state)
    assert list(restored._step_times) == list(scaler._step_times)
    assert list(restored._vram_fracs) == list(scaler._vram_fracs)
