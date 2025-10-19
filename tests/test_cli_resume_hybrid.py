from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Iterable

import pytest

from tests._stubs import install_torch_stub

install_torch_stub()
main = importlib.import_module("main")


class DummyHybridTrainer:
    instances: list["DummyHybridTrainer"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self._completed = 0
        self.save_calls: list[Path] = []
        self.fit_invocations: list[dict[str, Any]] = []
        self._stopped = False
        DummyHybridTrainer.instances.append(self)

    @property
    def completed_cycles(self) -> int:
        return self._completed

    def save(self, out_dir: str) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "hybrid_manifest.json").write_text("{}", encoding="utf-8")

    def save_checkpoint(self, path: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        ckpt_path = Path(path)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "trainer": {
                "progress": {
                    "completed_cycles": self._completed,
                    "history": [{"cycle": idx} for idx in range(1, self._completed + 1)],
                    "stopped_early": self._stopped,
                }
            }
        }
        (ckpt_path / "state.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        (ckpt_path / "tensors.json").write_text(json.dumps({}), encoding="utf-8")
        self.save_calls.append(ckpt_path)
        return payload

    def load_checkpoint(self, path: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        ckpt_path = Path(path)
        payload = json.loads((ckpt_path / "state.json").read_text(encoding="utf-8"))
        progress = payload.get("trainer", {}).get("progress", {})
        self._completed = int(progress.get("completed_cycles", 0))
        self._stopped = bool(progress.get("stopped_early", False))
        return {"payload": payload}

    def fit(
        self,
        batches: Iterable[object],
        *,
        cycles: int | None = None,
        unigram_epochs: int | None = None,
        warm_start_merges: Any | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int | None = None,
        time_limit_s: float | None = None,
        bpe_fit_kwargs: Any | None = None,
        unigram_fit_kwargs: Any | None = None,
    ) -> dict[str, Any]:
        invocation = {
            "cycles": cycles,
            "unigram_epochs": unigram_epochs,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_interval": checkpoint_interval,
            "time_limit_s": time_limit_s,
        }
        self.fit_invocations.append(invocation)
        if time_limit_s is not None and time_limit_s <= 0:
            self._stopped = True
            summary = {
                "cycles": self._completed,
                "phase_history": [
                    {"cycle": idx} for idx in range(1, self._completed + 1)
                ],
                "stopped_early": True,
            }
            if checkpoint_dir is not None:
                self.save_checkpoint(checkpoint_dir)
            return summary
        target = max(int(cycles or 0), self._completed)
        while self._completed < target:
            self._completed += 1
            if (
                checkpoint_dir is not None
                and checkpoint_interval is not None
                and checkpoint_interval > 0
                and self._completed % checkpoint_interval == 0
            ):
                self.save_checkpoint(checkpoint_dir)
        if self._completed < target:
            self._stopped = True
        summary = {
            "cycles": self._completed,
            "phase_history": [{"cycle": idx} for idx in range(1, self._completed + 1)],
            "stopped_early": self._stopped,
        }
        if checkpoint_dir is not None:
            self.save_checkpoint(checkpoint_dir)
        return summary


@pytest.fixture(autouse=True)
def _clear_instances() -> None:
    DummyHybridTrainer.instances.clear()


def _stub_sequences(*args: Any, **kwargs: Any) -> list[list[int]]:
    return [[1, 2, 3]]


def _stub_iter_batches(
    sequences: Iterable[Iterable[int]],
    batch_size: int | None = None,
    seed: int | None = None,
    augmentation=None,
    **_: Any,
) -> list[tuple[list[int], list[int], list[int]]]:
    return [([], [], [])]


def _copy_fixture(name: str, target: Path) -> Path:
    source = Path(__file__).resolve().parents[0] / "data" / "checkpoints" / name
    for item in source.iterdir():
        if item.is_file():
            (target / item.name).write_text(item.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_train_hybrid_resume_uses_checkpointing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(main, "HybridTrainer", DummyHybridTrainer)
    monkeypatch.setattr(main, "_load_sequences", _stub_sequences)
    monkeypatch.setattr(main, "_iter_packed_batches", _stub_iter_batches)

    shard = tmp_path / "shard.txt"
    shard.write_text("hello\nworld\n", encoding="utf-8")

    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    _copy_fixture("hybrid", resume_dir)

    checkpoint_dir = tmp_path / "checkpoints"

    main.main(
        [
            "train-hybrid",
            "--data",
            str(shard),
            "--merges",
            "8",
            "--cycles",
            "3",
            "--unigram-epochs",
            "1",
            "--batch-size",
            "2",
            "--bpe-log-every",
            "5",
            "--out-dir",
            str(tmp_path / "hybrid_out"),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every",
            "2",
            "--time-minutes",
            "1.5",
            "--resume-from",
            str(resume_dir),
        ]
    )

    captured = capsys.readouterr()
    assert "Restored checkpoint" in captured.out
    trainer = DummyHybridTrainer.instances[0]
    assert trainer.completed_cycles == 3
    assert trainer.fit_invocations[0]["checkpoint_interval"] == 2
    assert pytest.approx(trainer.fit_invocations[0]["time_limit_s"], rel=1e-6) == 90.0
    assert len(trainer.save_calls) >= 1


def test_train_hybrid_time_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(main, "HybridTrainer", DummyHybridTrainer)
    monkeypatch.setattr(main, "_load_sequences", _stub_sequences)
    monkeypatch.setattr(main, "_iter_packed_batches", _stub_iter_batches)

    shard = tmp_path / "shard.txt"
    shard.write_text("hello\nworld\n", encoding="utf-8")

    main.main(
        [
            "train-hybrid",
            "--data",
            str(shard),
            "--merges",
            "6",
            "--cycles",
            "2",
            "--unigram-epochs",
            "1",
            "--batch-size",
            "2",
            "--bpe-log-every",
            "5",
            "--out-dir",
            str(tmp_path / "hybrid_out"),
            "--time-minutes",
            "0.0",
        ]
    )

    trainer = DummyHybridTrainer.instances[0]
    assert trainer.completed_cycles == 0
    assert trainer._stopped is True
    captured = capsys.readouterr()
    assert "Hybrid time limit" in captured.out
