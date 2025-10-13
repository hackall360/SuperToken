"""Unit tests for the adaptive autoscaler."""

from __future__ import annotations

import json
import logging
import sys
import types

import pytest

if "torch" not in sys.modules:  # pragma: no cover - testing convenience
    cuda_stub = types.SimpleNamespace(
        is_available=lambda: False,
        mem_get_info=lambda: (0, 0),
        empty_cache=lambda: None,
        max_memory_allocated=lambda *args, **kwargs: 0,
    )
    jit_stub = types.SimpleNamespace(script=lambda fn: fn)
    sys.modules["torch"] = types.SimpleNamespace(cuda=cuda_stub, jit=jit_stub)

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
