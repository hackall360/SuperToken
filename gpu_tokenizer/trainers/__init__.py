"""Utilities for working with trainer abstractions."""

from __future__ import annotations

from .base import BaseTrainer
from .metrics import TrainerMetricsEWMA

__all__ = ["BaseTrainer", "TrainerMetricsEWMA"]
