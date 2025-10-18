"""GPU tokenizer/training toolkit."""

from __future__ import annotations

from .autoscaler import AutoScaler
from .cpu_packer import BytePacker
from .datasets import PackedBatcher, StreamingPackedBatcher
from .io import CorpusStreamer, MemoryMappedShard
from .trainers import BaseTrainer
from . import dist_runtime, lease_queue, ngram_stats, utils

try:  # pragma: no cover - optional torch dependency
    from .bpe_trainer import GPUBPETrainer  # type: ignore
except Exception:  # pragma: no cover - allow import without torch
    GPUBPETrainer = None  # type: ignore

try:  # pragma: no cover - optional torch dependency
    from .unigram_trainer import GPUUnigramTrainer  # type: ignore
except Exception:  # pragma: no cover - allow import without torch
    GPUUnigramTrainer = None  # type: ignore

try:  # pragma: no cover - optional torch dependency
    from .trainers.hybrid import HybridTrainer  # type: ignore
except Exception:  # pragma: no cover - allow import without torch
    HybridTrainer = None  # type: ignore

__all__ = [
    name
    for name in [
        "GPUBPETrainer",
        "GPUUnigramTrainer",
        "HybridTrainer",
        "BaseTrainer",
        "BytePacker",
        "PackedBatcher",
        "StreamingPackedBatcher",
        "MemoryMappedShard",
        "CorpusStreamer",
        "AutoScaler",
        "dist_runtime",
        "lease_queue",
        "ngram_stats",
        "utils",
    ]
]
