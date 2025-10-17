from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1] / "gpu_tokenizer"
package_stub = sys.modules.setdefault("gpu_tokenizer", types.ModuleType("gpu_tokenizer"))
package_stub.__path__ = [str(_PKG_ROOT)]
trainers_stub = sys.modules.setdefault(
    "gpu_tokenizer.trainers", types.ModuleType("gpu_tokenizer.trainers")
)
trainers_stub.__path__ = [str(_PKG_ROOT / "trainers")]

_BASE_TRAINER_PATH = _PKG_ROOT / "trainers" / "base.py"
_SPEC = importlib.util.spec_from_file_location(
    "gpu_tokenizer.trainers.base", _BASE_TRAINER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

BaseTrainer = _MODULE.BaseTrainer
from gpu_tokenizer.trainers.metrics import TrainerMetricsEWMA


class DummyTrainer(BaseTrainer):
    """Minimal concrete trainer used to assert the base contract."""

    def __init__(self) -> None:
        super().__init__()
        self._history: list[dict[str, Any]] = []
        self._tracker = TrainerMetricsEWMA(enabled=True)
        self.register_metrics_tracker("dummy", self._tracker)

    def fit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._history.append({"args": list(args), "kwargs": dict(kwargs)})
        return {"status": "ok", "invocations": len(self._history)}

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"metadata": {"history": list(self._history)}}

    def load_state_dict(
        self, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        metadata = state_dict.get("metadata", {}) if isinstance(state_dict, Mapping) else {}
        history = metadata.get("history", [])
        if isinstance(history, list):
            self._history = [dict(entry) for entry in history if isinstance(entry, Mapping)]
        else:
            self._history = []
        return {"restored": len(self._history)}

    def save_artifacts(
        self, output_dir: str | Path, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        artifact = output_path / "dummy.json"
        artifact.write_text(json.dumps({"history": self._history}))
        return {"dummy": str(artifact)}

    def metrics(self) -> Mapping[str, TrainerMetricsEWMA]:
        return self._metrics_mapping()


def test_base_trainer_enforces_required_methods() -> None:
    class MissingMethodsTrainer(BaseTrainer):
        def state_dict(self) -> dict[str, Any]:  # pragma: no cover - instantiation fails
            return {}

        def load_state_dict(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover - instantiation fails
            return {}

        def save_artifacts(self, output_dir: str | Path) -> dict[str, Any]:  # pragma: no cover - instantiation fails
            return {}

    with pytest.raises(TypeError):
        MissingMethodsTrainer()


def test_dummy_trainer_satisfies_base_contract(tmp_path: Path) -> None:
    trainer = DummyTrainer()
    registry = trainer.metrics()
    assert "dummy" in registry
    assert registry["dummy"].enabled is True

    summary = trainer.fit("payload", answer=42)
    assert summary == {"status": "ok", "invocations": 1}

    state = trainer.state_dict()
    assert state == {
        "metadata": {"history": [{"args": ["payload"], "kwargs": {"answer": 42}}]}
    }

    restored = trainer.load_state_dict(state)
    assert restored == {"restored": 1}

    artifacts = trainer.save_artifacts(tmp_path)
    artifact_path = Path(artifacts["dummy"])
    assert artifact_path.exists()
    stored = json.loads(artifact_path.read_text())
    assert stored == {"history": [{"args": ["payload"], "kwargs": {"answer": 42}}]}
