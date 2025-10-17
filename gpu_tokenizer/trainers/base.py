"""Common abstractions for GPU-backed tokenizer trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, Mapping


class BaseTrainer(ABC):
    """Shared interface implemented by all trainer variants.

    The base contract intentionally keeps the method signatures lightweight so
    subclasses can expose additional keyword arguments without fighting the
    abstract method requirements.  Each method returns a ``dict`` of structured
    metadata to align with the existing trainer implementations which surface
    rich progress snapshots for downstream tooling.
    """

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


__all__ = ["BaseTrainer"]
