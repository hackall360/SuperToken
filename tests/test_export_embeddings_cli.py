from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tests._stubs import install_torch_stub

install_torch_stub()
main = importlib.import_module("main")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_cli(
    tmp_path: Path,
    *,
    min_frequency: float,
    dedupe_similarity: float | None = None,
    vocab: dict[str, int] | None = None,
    stats: dict[str, object] | None = None,
) -> Path:
    vocab_path = tmp_path / "vocab.json"
    stats_path = tmp_path / "stats.json"
    output_dir = tmp_path / "artifacts"

    vocab_payload = vocab or {"<pad>": 0, "hello": 1, "world": 2, "rare": 3}
    stats_payload = stats or {
        "hello": {"count": 10, "vector": [0.1, 0.2]},
        "world": {"count": 4},
        "rare": {"count": 0.1},
    }

    _write_json(vocab_path, vocab_payload)
    _write_json(stats_path, stats_payload)

    argv = [
        "export-embeddings",
        "--vocab",
        str(vocab_path),
        "--output-dir",
        str(output_dir),
        "--dimension",
        "2",
        "--dtype",
        "float32",
        "--seed",
        "7",
        "--stats",
        str(stats_path),
        "--min-frequency",
        str(min_frequency),
        "--keep-token",
        "<pad>",
    ]
    if dedupe_similarity is not None:
        argv.extend(["--dedupe-similarity", str(dedupe_similarity)])
    main.main(argv)

    return output_dir


def test_export_embeddings_cli_writes_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = _run_cli(tmp_path, min_frequency=0.5)

    captured = capsys.readouterr()
    assert "export-embeddings" in captured.out

    vocab_path = out_dir / "vocab.json"
    embeddings_path = out_dir / "embeddings.json"
    manifest_path = out_dir / "manifest.json"
    pruning_path = out_dir / "pruning.json"

    assert vocab_path.exists()
    assert embeddings_path.exists()
    assert manifest_path.exists()
    assert pruning_path.exists()

    exported_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pruned = json.loads(pruning_path.read_text(encoding="utf-8"))
    embeddings = json.loads(embeddings_path.read_text(encoding="utf-8"))

    assert set(exported_vocab) == {"<pad>", "hello", "world"}
    assert "rare" not in exported_vocab
    assert manifest["exported_token_count"] == len(exported_vocab)
    assert manifest["original_token_count"] == 4
    assert pytest.approx(manifest["min_frequency"], rel=1e-6) == 0.5
    assert any(item["token"] == "rare" for item in pruned)

    hello_index = exported_vocab["hello"]
    assert embeddings[hello_index] == pytest.approx([0.1, 0.2])
    assert len(embeddings) == 3
    assert all(len(row) == 2 for row in embeddings)


class SimpleEmbeddingTrainer:
    def __init__(self, weights: list[list[float]]) -> None:
        self.weights = [row[:] for row in weights]

    def train(self, batches: list[list[int]], lr: float = 0.1) -> list[list[float]]:
        for batch in batches:
            for index in batch:
                self.weights[index] = [value + lr for value in self.weights[index]]
        return self.weights


def test_exported_embeddings_load_into_simple_trainer(tmp_path: Path) -> None:
    out_dir = _run_cli(tmp_path, min_frequency=0.0)

    vocab = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))
    embeddings = json.loads((out_dir / "embeddings.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

    trainer = SimpleEmbeddingTrainer(embeddings)
    batches = [[vocab["<pad>"], vocab["hello"]], [vocab["world"]]]
    updated = trainer.train(batches, lr=0.5)

    assert len(updated) == len(embeddings)
    pad_index = vocab["<pad>"]
    assert updated[pad_index] == pytest.approx([value + 0.5 for value in embeddings[pad_index]])
    assert manifest["exported_token_count"] == len(embeddings)


def test_export_embeddings_cli_deduplicates_tokens(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vocab = {"<pad>": 0, "alpha": 1, "beta": 2}
    stats = {
        "alpha": {"count": 5, "vector": [0.5, 0.5]},
        "beta": {"count": 4, "vector": [0.5, 0.5]},
    }

    out_dir = _run_cli(
        tmp_path,
        min_frequency=0.0,
        dedupe_similarity=0.95,
        vocab=vocab,
        stats=stats,
    )

    captured = capsys.readouterr()
    summary_line = next(
        line for line in captured.out.splitlines() if line.startswith("[export][export-embeddings]")
    )
    summary = json.loads(summary_line.split(" ", 1)[1])
    assert summary["deduped_tokens"] == 1
    assert summary["pruned_tokens"] == 0
    assert summary["pruning_log_entries"] == 1

    exported_vocab = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    pruning_log = json.loads((out_dir / "pruning.json").read_text(encoding="utf-8"))

    assert "beta" not in exported_vocab
    assert manifest["original_token_count"] == 3
    assert manifest["exported_token_count"] == len(exported_vocab)

    deduped_entries = [entry for entry in pruning_log if entry.get("action") == "deduped"]
    assert len(deduped_entries) == 1
    entry = deduped_entries[0]
    assert entry["token"] == "beta"
    assert entry["merged_into"] == "alpha"
    assert entry["similarity"] == pytest.approx(1.0, rel=1e-6)
