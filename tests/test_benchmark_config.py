from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from benchmarks import load_bpe_run_config


def test_load_bpe_run_config_parses_weights() -> None:
    config_path = Path("benchmarks/configs/heterogeneous_example.json")
    specs = load_bpe_run_config(config_path)
    assert len(specs) == 2
    assert specs[0].name == "single_cuda0"
    assert specs[1].normalized_weights() == [1.0, 0.92]
    assert specs[1].scaling_reference == "single_cuda0"


def test_sample_benchmark_scaling_meets_threshold() -> None:
    sample_path = Path("benchmarks/samples/heterogeneous_benchmark_sample.json")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    runs = payload.get("bpe_runs", [])
    assert runs, "expected sample to contain at least one run"
    hetero = next(run for run in runs if run.get("name") == "cuda0_cuda1_hetero")
    scaling = hetero.get("scaling")
    assert scaling is not None
    assert scaling.get("meets_target") is True
    efficiency = float(scaling.get("efficiency"))
    threshold = float(scaling.get("target_efficiency"))
    assert efficiency >= threshold >= 0.88
