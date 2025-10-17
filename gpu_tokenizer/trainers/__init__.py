"""Utilities for working with trainer abstractions."""

from __future__ import annotations

from .base import BaseTrainer
from .metrics import TrainerMetricsEWMA
from .bpe_gpu import (
    PairHistogramResult,
    combine_histogram_results,
    count_pairs_on_device,
    select_best_pair,
)

__all__ = [
    "BaseTrainer",
    "TrainerMetricsEWMA",
    "PairHistogramResult",
    "combine_histogram_results",
    "count_pairs_on_device",
    "select_best_pair",
]
