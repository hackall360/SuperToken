from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Iterable, List

import pytest

from tests._stubs import install_torch_stub

install_torch_stub()
main = importlib.import_module("main")


class DummyUnigramTrainer:
    instances: list["DummyUnigramTrainer"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self._completed = 0
        self._history: list[dict[str, object]] = []
        self.save_calls: list[Path] = []
        DummyUnigramTrainer.instances.append(self)

    @property
    def completed_epochs(self) -> int:
        return self._completed

    def fit_epoch(self, batches: List[Iterable[int]]) -> dict[str, object]:
        self._completed += 1
        result = {
            "vocab": 32 + self._completed,
            "telemetry": {"epoch": self._completed},
        }
        self._history.append({"epoch": self._completed, **result})
        return result

    def save(self, out_dir: str) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "unigram.model").write_text("ok", encoding="utf-8")

    def save_checkpoint(self, path: str, *args: Any, **kwargs: Any) -> dict[str, object]:
        ckpt_path = Path(path)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "trainer": {
                "progress": {
                    "completed_epochs": self._completed,
                    "history": list(self._history),
                }
            }
        }
        (ckpt_path / "state.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        (ckpt_path / "tensors.json").write_text(
            json.dumps({"logp": [0.0]}), encoding="utf-8"
        )
        self.save_calls.append(ckpt_path)
        return payload

    def load_checkpoint(self, path: str, *args: Any, **kwargs: Any) -> dict[str, object]:
        ckpt_path = Path(path)
        payload = json.loads((ckpt_path / "state.json").read_text(encoding="utf-8"))
        progress = payload.get("trainer", {}).get("progress", {})
        self._completed = int(progress.get("completed_epochs", 0))
        history = progress.get("history", [])
        if isinstance(history, list):
            self._history = [dict(entry) for entry in history]
        else:
            self._history = []
        return {"payload": payload}


def _stub_sequences(*args: Any, **kwargs: Any) -> list[list[int]]:
    return [[1, 2, 3]]


def _stub_batches(sequences: Iterable[Iterable[int]], *args: Any, **kwargs: Any) -> list[list[int]]:
    return [list(next(iter(sequences)))]


@pytest.fixture(autouse=True)
def _clear_instances() -> None:
    DummyUnigramTrainer.instances.clear()


def _copy_fixture(name: str, target: Path) -> Path:
    source = Path(__file__).resolve().parents[0] / "data" / "checkpoints" / name
    for item in source.iterdir():
        if item.is_file():
            (target / item.name).write_text(item.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_train_unigram_resume_advances_epochs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(main, "GPUUnigramTrainer", DummyUnigramTrainer)
    monkeypatch.setattr(main, "_load_sequences", _stub_sequences)
    monkeypatch.setattr(main, "_build_unigram_batches", _stub_batches)

    shard = tmp_path / "shard.txt"
    shard.write_text("hello\nworld\n", encoding="utf-8")

    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    _copy_fixture("unigram", resume_dir)

    checkpoint_dir = tmp_path / "checkpoints"

    main.main(
        [
            "train-unigram",
            "--data",
            str(shard),
            "--epochs",
            "4",
            "--batch-size",
            "2",
            "--out-dir",
            str(tmp_path / "model"),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every",
            "2",
            "--resume-from",
            str(resume_dir),
        ]
    )

    captured = capsys.readouterr()
    assert "Restored checkpoint" in captured.out
    assert DummyUnigramTrainer.instances
    trainer = DummyUnigramTrainer.instances[0]
    assert trainer.completed_epochs == 4
    assert len(trainer.save_calls) >= 2
    assert (tmp_path / "model" / "unigram.model").exists()


def test_train_unigram_time_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(main, "GPUUnigramTrainer", DummyUnigramTrainer)
    monkeypatch.setattr(main, "_load_sequences", _stub_sequences)
    monkeypatch.setattr(main, "_build_unigram_batches", _stub_batches)

    shard = tmp_path / "shard.txt"
    shard.write_text("hello\nworld\n", encoding="utf-8")

    schedule = [0.0, 10.0, 10.1]

    def _fake_perf_counter() -> float:
        value = schedule.pop(0)
        schedule.append(value)
        return value

    monkeypatch.setattr(main.time, "perf_counter", _fake_perf_counter)

    main.main(
        [
            "train-unigram",
            "--data",
            str(shard),
            "--epochs",
            "3",
            "--out-dir",
            str(tmp_path / "time_out"),
            "--time-minutes",
            "0.0",
        ]
    )

    trainer = DummyUnigramTrainer.instances[0]
    assert trainer.completed_epochs == 0
    captured = capsys.readouterr()
    assert "Time limit" in captured.out
