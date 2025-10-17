"""Tests covering distributed EWMA helpers and rank weight blending."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from ._stubs import install_package_stub, install_torch_stub


install_package_stub()
install_torch_stub()

_BPE_PATH = Path(__file__).resolve().parents[1] / "gpu_tokenizer" / "bpe_trainer.py"
_BPE_SPEC = importlib.util.spec_from_file_location("gpu_tokenizer.bpe_trainer", _BPE_PATH)
assert _BPE_SPEC and _BPE_SPEC.loader
_BPE_MODULE = importlib.util.module_from_spec(_BPE_SPEC)
sys.modules[_BPE_SPEC.name] = _BPE_MODULE
_BPE_SPEC.loader.exec_module(_BPE_MODULE)

_DIST_PATH = Path(__file__).resolve().parents[1] / "gpu_tokenizer" / "dist_runtime.py"
_DIST_SPEC = importlib.util.spec_from_file_location("gpu_tokenizer.dist_runtime", _DIST_PATH)
assert _DIST_SPEC and _DIST_SPEC.loader
_DIST_MODULE = importlib.util.module_from_spec(_DIST_SPEC)
sys.modules[_DIST_SPEC.name] = _DIST_MODULE
_DIST_SPEC.loader.exec_module(_DIST_MODULE)

TrainerMetricsEWMA = _BPE_MODULE.TrainerMetricsEWMA
dist_runtime = _DIST_MODULE


def test_ewma_smoothing_tracks_tokens_and_leases() -> None:
    metrics = TrainerMetricsEWMA(alpha=0.5, enabled=True)
    metrics.set_rank(0)

    metrics.record_tokens(tokens=100, duration_s=1.0, leases=10)
    assert metrics.tokens_per_s == pytest.approx(100.0)
    assert metrics.lease_per_s == pytest.approx(10.0)

    metrics.record_tokens(tokens=200, duration_s=1.0, leases=20)
    # EWMA with alpha=0.5 should average the latest sample with the prior EWMA.
    assert metrics.tokens_per_s == pytest.approx(150.0)
    assert metrics.lease_per_s == pytest.approx(15.0)

    snapshot = metrics.snapshot()
    assert snapshot["rank"] == 0
    assert snapshot["tokens_per_s"] == pytest.approx(150.0)
    assert snapshot["lease_per_s"] == pytest.approx(15.0)
    assert snapshot["samples"] == pytest.approx(2.0)


def test_blend_rank_weights_normalises_and_includes_all_ranks() -> None:
    existing = {0: 0.6, 1: 0.4}
    updated = {0: 0.2, 1: 0.8}

    blended = dist_runtime._blend_rank_weights(
        existing,
        updated,
        new_fraction=0.5,
        world_size=2,
    )

    assert set(blended) == {0, 1}
    assert blended[0] == pytest.approx(0.4)
    assert blended[1] == pytest.approx(0.6)
    assert math.isclose(sum(blended.values()), 1.0)


def test_blend_rank_weights_uniform_when_no_signal() -> None:
    blended = dist_runtime._blend_rank_weights(
        {},
        {0: 0.0},
        new_fraction=1.0,
        world_size=3,
    )

    assert set(blended) == {0, 1, 2}
    assert all(value == pytest.approx(1.0 / 3.0) for value in blended.values())
