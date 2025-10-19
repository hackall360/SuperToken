import importlib
from pathlib import Path
import types

import pytest

try:
    main = importlib.import_module("main")
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when torch is absent
    if exc.name == "torch":
        pytest.skip(
            "code-mode CLI tests require torch to import the main entry point",
            allow_module_level=True,
        )
    raise


def _stub_code_summary(samples: int = 1) -> dict[str, object]:
    return {
        "enabled": True,
        "samples": samples,
        "ast_samples": samples,
        "fallback_samples": 0,
        "languages": ["python"],
        "meta_compress": True,
        "meta_tokens": {},
        "meta_token_count": 0,
        "meta_max_length": 8,
        "average_sequence_length": 3.0,
    }


def test_unigram_code_mode_loads_alt_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {}

    def fake_loader(paths, **kwargs):
        calls["loader"] = calls.get("loader", 0) + 1
        return [[1, 2, 3]], _stub_code_summary()

    def fail_loader(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("_load_sequences should not be used in code mode")

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def fit_epoch(self, batches):
            assert list(batches) == ["batch"], "expected stubbed batches"
            self.completed_epochs = getattr(self, "completed_epochs", 0) + 1
            return {"loss": 0.0}

        def save(self, *args, **kwargs):
            return None

    monkeypatch.setattr(main, "GPUUnigramTrainer", DummyTrainer)
    monkeypatch.setattr(main, "_load_code_mode_sequences", fake_loader)
    monkeypatch.setattr(main, "_load_sequences", fail_loader)
    monkeypatch.setattr(
        main,
        "_build_unigram_batches",
        lambda seqs, batch_size, seed, augmentation=None: ["batch"],
    )

    shard = tmp_path / "code.jsonl"
    shard.write_text("{}\n", encoding="utf-8")

    main.main(
        [
            "train-unigram",
            "--data",
            str(shard),
            "--vocab-size",
            "32",
            "--batch-size",
            "4",
            "--epochs",
            "1",
            "--code-mode",
            "--meta-compress",
        ]
    )

    assert calls.get("loader") == 1


def test_hybrid_code_mode_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {}

    def fake_loader(paths, **kwargs):
        calls["loader"] = calls.get("loader", 0) + 1
        return [[7, 8]], _stub_code_summary()

    def fail_loader(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("_load_sequences should not be used in code mode")

    class DummyHybridTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, batches, **kwargs):
            assert list(batches) == ["packed"], "expected stubbed packed batches"
            return {"status": "ok"}

        def save(self, *args, **kwargs):
            return None

    monkeypatch.setattr(main, "HybridTrainer", DummyHybridTrainer)
    monkeypatch.setattr(main, "_load_code_mode_sequences", fake_loader)
    monkeypatch.setattr(main, "_load_sequences", fail_loader)
    monkeypatch.setattr(
        main,
        "_iter_packed_batches",
        lambda seqs, batch_size, seed, augmentation=None: ["packed"],
    )

    shard = tmp_path / "hybrid.json"
    shard.write_text("{}", encoding="utf-8")

    main.main(
        [
            "train-hybrid",
            "--data",
            str(shard),
            "--merges",
            "8",
            "--batch-size",
            "2",
            "--cycles",
            "1",
            "--unigram-epochs",
            "1",
            "--code-mode",
        ]
    )

    assert calls.get("loader") == 1


def test_bpe_code_mode_uses_packed_batches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {}

    def fake_loader(paths, **kwargs):
        calls["loader"] = calls.get("loader", 0) + 1
        return [[9, 10]], _stub_code_summary()

    def fail_sequence_loader(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("_load_sequences should not be reached in code mode")

    class DummyBatcher:
        def __init__(self, sequences, batch_size, seed, augmentation=None):
            calls["batcher"] = sequences

        def __iter__(self):
            yield ("tokens", "mask", "lengths")

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, batches, **kwargs):
            assert isinstance(batches, DummyBatcher)
            return {"status": "ok"}

        def save(self, *args, **kwargs):
            return None

        def save_checkpoint(self, *args, **kwargs):  # pragma: no cover - defensive
            return {}

        def load_checkpoint(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(main, "_build_bpe_trainer", lambda args: (DummyTrainer(), types.SimpleNamespace(), args.min_batch, {"trainer": {}, "autoscaler": {}}))
    monkeypatch.setattr(main, "_load_code_mode_sequences", fake_loader)
    monkeypatch.setattr(main, "_load_sequences", fail_sequence_loader)
    monkeypatch.setattr(main, "PackedBatcher", DummyBatcher)

    def fail_streamer(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("CorpusStreamer should not be constructed in code mode")

    monkeypatch.setattr(main, "CorpusStreamer", fail_streamer)

    shard = tmp_path / "bpe.json"
    shard.write_text("{}", encoding="utf-8")

    main.main(
        [
            "train-bpe",
            "--data",
            str(shard),
            "--merges",
            "16",
            "--min-batch",
            "2",
            "--max-batch",
            "4",
            "--token-bytes",
            "32",
            "--code-mode",
            "--meta-compress",
        ]
    )

    assert calls.get("loader") == 1
    assert calls.get("batcher") == [[9, 10]]
