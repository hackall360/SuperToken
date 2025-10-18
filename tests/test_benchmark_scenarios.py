from dataclasses import asdict
from pathlib import Path
import json

import pytest

from benchmarks import (
    BENCHMARK_OUTPUT_SCHEMA,
    CorpusSummary,
    SchemaValidationError,
    generate_hybrid_runs,
    generate_multi_gpu_runs,
    generate_streaming_compression_runs,
    serialize_run,
    validate_benchmark_output,
)


def _build_stub_results() -> tuple[dict[str, object], dict[str, object]]:
    bpe = {
        "config": {"batch_size": 32},
        "wall_time_s": 1.25,
        "result": {"vocab_size": 128},
        "overlap_enabled": True,
        "tokens_processed": 1024,
        "tokens_per_s": 819.2,
        "autoscaler_window": [],
    }
    unigram = {
        "config": {"batch_size": 32},
        "wall_time_s": 2.5,
        "epochs": [{"epoch": 1, "loss": 0.1}],
    }
    return bpe, unigram


def test_streaming_compression_generates_scaling_reference() -> None:
    runs = generate_streaming_compression_runs(batch_size=64, device="cuda:0")
    assert [run.name for run in runs] == ["streaming_baseline", "streaming_overlap"]
    baseline, streaming = runs
    assert baseline.overlap is False
    assert streaming.overlap is True
    assert streaming.scaling_reference == baseline.name
    assert streaming.target_efficiency == pytest.approx(0.9)


def test_multi_gpu_generates_weights() -> None:
    runs = generate_multi_gpu_runs(
        batch_size=128,
        baseline_device="cuda:0",
        data_parallel_devices=["cuda:0", "cuda:1"],
    )
    assert [run.name for run in runs] == ["single_gpu", "multi_gpu"]
    baseline, multi = runs
    assert baseline.devices is None
    assert multi.devices == ["cuda:0", "cuda:1"]
    assert multi.scaling_reference == baseline.name
    assert multi.device_weights == [1.0, 1.0]


@pytest.mark.parametrize("helper_weight", [0.5, 0.75])
def test_hybrid_runs_assign_helper_weights(helper_weight: float) -> None:
    runs = generate_hybrid_runs(
        batch_size=96,
        fast_device="cuda:0",
        helper_devices=["cuda:1", "cuda:2"],
        helper_weight=helper_weight,
    )
    baseline, hybrid = runs
    assert hybrid.devices == ["cuda:0", "cuda:1", "cuda:2"]
    assert hybrid.device_weights == [1.0, helper_weight, helper_weight]
    assert hybrid.scaling_reference == baseline.name


def test_multi_gpu_requires_devices() -> None:
    with pytest.raises(ValueError):
        generate_multi_gpu_runs(
            batch_size=32,
            baseline_device="cuda:0",
            data_parallel_devices=[],
        )


def test_hybrid_requires_helpers() -> None:
    with pytest.raises(ValueError):
        generate_hybrid_runs(
            batch_size=32,
            fast_device="cuda:0",
            helper_devices=[],
        )


def test_serialize_run_validates_against_schema(tmp_path: Path) -> None:
    corpus = CorpusSummary(sequences=4, tokens=1024, max_length=512, sources=[])
    bpe, unigram = _build_stub_results()
    suite = {
        "runs": [
            {
                "name": "streaming_overlap",
                "wall_time_s": 1.25,
                "tokens_per_s": 819.2,
                "scaling": {
                    "reference": "streaming_baseline",
                    "device_weights": [1.0],
                    "expected_tokens_per_s": 819.2,
                    "efficiency": 1.0,
                    "meets_target": True,
                    "target_efficiency": 0.9,
                },
            }
        ]
    }
    path = serialize_run(
        tmp_path,
        corpus=corpus,
        config={"synthetic": True},
        bpe=bpe,
        unigram=unigram,
        bpe_runs=suite,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_benchmark_output(payload)
    assert payload["timestamp"]
    assert payload["bpe"]["tokens_per_s"] == pytest.approx(819.2)


def test_schema_validation_detects_missing_section() -> None:
    bpe, unigram = _build_stub_results()
    payload = {
        "timestamp": "20240101T000000Z",
        "config": {},
        "corpus": {
            "sequences": 1,
            "tokens": 10,
            "max_length": 10,
            "sources": [],
        },
        "bpe": bpe,
        # "unigram" missing on purpose
    }
    with pytest.raises(SchemaValidationError):
        validate_benchmark_output(payload, BENCHMARK_OUTPUT_SCHEMA)
