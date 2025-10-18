"""GPU tokenizer/training toolkit."""

from __future__ import annotations

try:
    from .autoscaler import AutoScaler
except Exception:  # pragma: no cover - optional torch dependency
    AutoScaler = None  # type: ignore
try:
    from .cpu_packer import BytePacker
except Exception:  # pragma: no cover - optional dependency
    BytePacker = None  # type: ignore
try:
    from .datasets import PackedBatcher, StreamingPackedBatcher
except Exception:  # pragma: no cover - optional dependency
    PackedBatcher = StreamingPackedBatcher = None  # type: ignore
try:
    from .io import CorpusStreamer, MemoryMappedShard
except Exception:  # pragma: no cover - optional dependency
    CorpusStreamer = MemoryMappedShard = None  # type: ignore
try:
    from .trainers import BaseTrainer
except Exception:  # pragma: no cover - optional dependency
    BaseTrainer = None  # type: ignore
try:
    from . import dist_runtime, lease_queue, ngram_stats, utils
except Exception:  # pragma: no cover - optional dependency
    dist_runtime = lease_queue = ngram_stats = utils = None  # type: ignore

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
