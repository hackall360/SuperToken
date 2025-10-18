from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

PACKAGE_NAME = "benchmarks"
MODULE_NAME = "benchmarks.embedding_pruning_benchmark"
MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "embedding_pruning_benchmark.py"

if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(MODULE_PATH.parent)]
    sys.modules[PACKAGE_NAME] = package

spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = module
spec.loader.exec_module(module)

EmbeddingBenchmarkConfig = module.EmbeddingBenchmarkConfig
run_benchmark = module.run_benchmark


def test_embedding_benchmark_generates_payload(tmp_path: Path) -> None:
    config = EmbeddingBenchmarkConfig(
        train_samples=256,
        eval_samples=64,
        epochs=4,
        batch_size=32,
        learning_rate=0.2,
        prune_frequency_threshold=0.02,
        seed=123,
    )

    artifacts = run_benchmark(config, output_dir=tmp_path)
    payload = artifacts.payload

    assert "baseline" in payload
    assert "pruned" in payload
    assert payload["baseline"]["evaluation"]["accuracy"] >= 0.0
    assert payload["pruned"]["evaluation"]["accuracy"] >= 0.0
    assert payload["pruned"]["dataset"]["vocab_size"] <= config.vocab_size

    # Pruning should not catastrophically reduce evaluation accuracy with the
    # default configuration.
    baseline_acc = payload["baseline"]["evaluation"]["accuracy"]
    pruned_acc = payload["pruned"]["evaluation"]["accuracy"]
    assert pruned_acc >= baseline_acc - 0.15

    assert artifacts.output_path is not None
    written = artifacts.output_path
    assert written.exists()
    contents = written.read_text(encoding="utf-8")
    assert payload["timestamp"] in contents
