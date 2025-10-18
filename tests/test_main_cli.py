from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tests._stubs import install_torch_stub

install_torch_stub()
main = importlib.import_module("main")


def test_train_bpe_dry_run_logs_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit(self, *args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("fit should not run during dry-run")

        def load_checkpoint(self, *args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("load_checkpoint should not run during dry-run")

        def save(self, *args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("save should not run during dry-run")

    class FailPacker:
        def __init__(self, *args, **kwargs):
            raise AssertionError("BytePacker should not be instantiated during dry-run")

    class FailStreamer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("CorpusStreamer should not be instantiated during dry-run")

    monkeypatch.setattr(main, "GPUBPETrainer", DummyTrainer)
    monkeypatch.setattr(main, "BytePacker", FailPacker)
    monkeypatch.setattr(main, "CorpusStreamer", FailStreamer)

    shard = tmp_path / "shard.txt"
    shard.write_text("hello world\n", encoding="utf-8")

    main.main(
        [
            "train-bpe",
            "--data",
            str(shard),
            "--merges",
            "12",
            "--min-batch",
            "2",
            "--max-batch",
            "4",
            "--token-bytes",
            "32",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert "[config][train-bpe]" in captured.out
    assert '"merges": 12' in captured.out
    assert '"resolved_batch_size"' in captured.out
    assert "[dry-run] train-bpe initialization complete" in captured.out


def test_train_unigram_dry_run_logs_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit_epoch(self, *args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("fit_epoch should not run during dry-run")

        def save(self, *args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("save should not run during dry-run")

    def _fail_load_sequences(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("_load_sequences should not run during dry-run")

    def _fail_build_batches(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("_build_unigram_batches should not run during dry-run")

    monkeypatch.setattr(main, "GPUUnigramTrainer", DummyTrainer)
    monkeypatch.setattr(main, "_load_sequences", _fail_load_sequences)
    monkeypatch.setattr(main, "_build_unigram_batches", _fail_build_batches)

    shard = tmp_path / "shard.txt"
    shard.write_text("hello world\n", encoding="utf-8")

    main.main(
        [
            "train-unigram",
            "--data",
            str(shard),
            "--vocab-size",
            "128",
            "--batch-size",
            "16",
            "--epochs",
            "3",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert "[config][train-unigram]" in captured.out
    assert '"vocab_size": 128' in captured.out
    assert '"batch_size": 16' in captured.out
    assert "[dry-run] train-unigram initialization complete" in captured.out


def test_train_hybrid_dry_run_logs_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit(self, *args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("fit should not run during dry-run")

        def save(self, *args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("save should not run during dry-run")

    def _fail_load_sequences(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("_load_sequences should not run during dry-run")

    def _fail_iter_batches(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("_iter_packed_batches should not run during dry-run")

    monkeypatch.setattr(main, "HybridTrainer", DummyTrainer)
    monkeypatch.setattr(main, "_load_sequences", _fail_load_sequences)
    monkeypatch.setattr(main, "_iter_packed_batches", _fail_iter_batches)

    shard = tmp_path / "shard.txt"
    shard.write_text("hello world\n", encoding="utf-8")

    main.main(
        [
            "train-hybrid",
            "--data",
            str(shard),
            "--merges",
            "12",
            "--batch-size",
            "8",
            "--cycles",
            "2",
            "--unigram-epochs",
            "3",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert "[config][train-hybrid]" in captured.out
    assert '"merges": 12' in captured.out
    assert '"cycles": 2' in captured.out
    assert '"batch_size": 8' in captured.out
    assert "[dry-run] train-hybrid initialization complete" in captured.out
