"""Common abstractions and serialization helpers for tokenizer trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os import PathLike
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, MutableMapping

from .metrics import TrainerMetricsEWMA


CHECKPOINT_SCHEMA_VERSION: int = 1


@dataclass
class CheckpointPayload:
    """Structured checkpoint metadata shared by trainer implementations."""

    version: int = CHECKPOINT_SCHEMA_VERSION
    trainer: dict[str, Any] = field(default_factory=dict)
    autoscaler: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    rng: dict[str, Any] = field(default_factory=dict)

    CURRENT_VERSION: ClassVar[int] = CHECKPOINT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping describing the checkpoint."""

        payload: dict[str, Any] = {"version": int(self.version)}
        if self.trainer:
            payload["trainer"] = _coerce_dict(self.trainer)
        if self.autoscaler:
            payload["autoscaler"] = _coerce_dict(self.autoscaler)
        if self.dataset:
            payload["dataset"] = _coerce_dict(self.dataset)
        if self.rng:
            payload["rng"] = _coerce_dict(self.rng)
        return payload

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "CheckpointPayload":
        """Materialise a payload from an arbitrary mapping."""

        if mapping is None:
            return cls()
        version = int(mapping.get("version", CHECKPOINT_SCHEMA_VERSION))
        trainer = _section_to_dict(mapping, "trainer")
        autoscaler = _section_to_dict(mapping, "autoscaler")
        dataset = _section_to_dict(mapping, "dataset")
        rng = _section_to_dict(mapping, "rng")
        return cls(
            version=version,
            trainer=trainer,
            autoscaler=autoscaler,
            dataset=dataset,
            rng=rng,
        )

    @classmethod
    def from_legacy_metadata(cls, metadata: Mapping[str, Any]) -> "CheckpointPayload":
        """Adapt legacy flat metadata dictionaries to the shared schema."""

        trainer: dict[str, Any] = dict(metadata)
        autoscaler: dict[str, Any] = {}
        dataset: dict[str, Any] = {}

        autoscaler_state = trainer.pop("autoscaler", None)
        if isinstance(autoscaler_state, Mapping):
            autoscaler["state"] = dict(autoscaler_state)
        elif autoscaler_state is not None:
            autoscaler["state"] = autoscaler_state

        autoscaler_metrics = trainer.pop("autoscaler_metrics", None)
        if autoscaler_metrics is not None:
            autoscaler["metrics"] = autoscaler_metrics

        autoscaler_window = trainer.pop("autoscaler_window", None)
        if autoscaler_window is not None:
            autoscaler["window"] = autoscaler_window

        batches = trainer.pop("batches", None)
        if batches is not None:
            dataset["batches"] = batches

        active_bs = trainer.pop("active_batch_size", None)
        if active_bs is not None:
            dataset["active_batch_size"] = active_bs

        cpu_fallback = trainer.pop("cpu_fallback_batches", None)
        if cpu_fallback is not None:
            dataset["cpu_fallback_batches"] = cpu_fallback

        cpu_ratio = trainer.pop("last_cpu_fallback_ratio", None)
        if cpu_ratio is not None:
            dataset["last_cpu_fallback_ratio"] = cpu_ratio

        version = int(trainer.pop("schema_version", 0) or 0)

        return cls(
            version=version,
            trainer=trainer,
            autoscaler=autoscaler,
            dataset=dataset,
            rng={},
        )


def _coerce_dict(section: Mapping[str, Any] | MutableMapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    """Return ``section`` as a plain ``dict`` without mutating the input."""

    if isinstance(section, dict):
        return dict(section)
    return dict(section.items())


def _section_to_dict(mapping: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Extract ``key`` from ``mapping`` as a shallow ``dict``."""

    section = mapping.get(key, {})
    if isinstance(section, Mapping):
        return _coerce_dict(section)
    return {}


class BaseTrainer(ABC):
    """Shared interface implemented by all trainer variants.

    The base contract intentionally keeps the method signatures lightweight so
    subclasses can expose additional keyword arguments without fighting the
    abstract method requirements.  Each method returns a ``dict`` of structured
    metadata to align with the existing trainer implementations which surface
    rich progress snapshots for downstream tooling.
    """

    def __init__(self) -> None:
        self._metrics_registry: dict[str, TrainerMetricsEWMA] = {}
        self._metrics_view: Mapping[str, TrainerMetricsEWMA] = MappingProxyType(
            self._metrics_registry
        )

    def register_metrics_tracker(
        self, name: str, tracker: TrainerMetricsEWMA | None
    ) -> None:
        """Register or remove a named metrics tracker."""

        key = str(name)
        if tracker is None:
            self._metrics_registry.pop(key, None)
            return
        self._metrics_registry[key] = tracker

    def _metrics_mapping(self) -> Mapping[str, TrainerMetricsEWMA]:
        """Return a read-only view of registered metrics trackers."""

        return self._metrics_view

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Run the training loop and return a summary of the run."""

    @abstractmethod
    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Produce a serialisable snapshot of the trainer state."""

    @abstractmethod
    def load_state_dict(
        self, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Restore a previously captured state snapshot."""

    @abstractmethod
    def save_checkpoint(
        self, path: str | PathLike[str], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Persist a checkpoint to *path* returning the captured state."""

    @abstractmethod
    def load_checkpoint(self, path: str | PathLike[str], *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Load a checkpoint from *path* returning the raw payload."""

    @abstractmethod
    def save_artifacts(
        self, output_dir: str | PathLike[str], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Persist model artifacts and return their filesystem locations."""

    @abstractmethod
    def metrics(self) -> Mapping[str, TrainerMetricsEWMA]:
        """Return the registered metrics trackers keyed by logical name."""


__all__ = ["BaseTrainer", "CheckpointPayload", "CHECKPOINT_SCHEMA_VERSION"]
