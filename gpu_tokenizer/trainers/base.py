"""Common abstractions for GPU-backed tokenizer trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike
from types import MappingProxyType
from typing import Any, Mapping

from .metrics import TrainerMetricsEWMA


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
    def save_artifacts(
        self, output_dir: str | PathLike[str], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Persist model artifacts and return their filesystem locations."""

    @abstractmethod
    def metrics(self) -> Mapping[str, TrainerMetricsEWMA]:
        """Return the registered metrics trackers keyed by logical name."""


__all__ = ["BaseTrainer"]
