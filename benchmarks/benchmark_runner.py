"""Helpers for running tokenizer training benchmarks."""

from __future__ import annotations

import json
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import torch

from gpu_tokenizer import (
    AutoScaler,
    BytePacker,
    GPUBPETrainer,
    GPUUnigramTrainer,
    PackedBatcher,
)
from gpu_tokenizer.io import MemoryMappedShard


@dataclass
class CorpusSummary:
    sequences: int
    tokens: int
    max_length: int
    sources: list[dict[str, object]]


def _ensure_trainers_available() -> None:
    if GPUBPETrainer is None or GPUUnigramTrainer is None:  # type: ignore[truthy-function]
        raise RuntimeError(
            "Both GPUBPETrainer and GPUUnigramTrainer require torch with GPU support."
        )


def synthesize_corpus(
    *,
    documents: int,
    min_length: int,
    max_length: int,
    vocab_size: int,
    seed: int,
) -> list[list[int]]:
    if documents <= 0:
        return []
    rng = random.Random(seed)
    corpus: list[list[int]] = []
    for _ in range(documents):
        length = rng.randint(max(1, min_length), max(min_length, max_length))
        seq = [rng.randrange(vocab_size) for _ in range(length)]
        corpus.append(seq)
    return corpus


def _iter_shard_sequences(
    shard: MemoryMappedShard,
    packer: BytePacker,
) -> Iterator[list[int]]:
    encoded = packer.encode_shard(shard)
    seq = list(encoded)
    if seq:
        yield seq


def load_real_corpus(
    paths: Sequence[Path],
    *,
    bos: int | None,
    eos: int | None,
    limit: int | None,
) -> list[list[int]]:
    if not paths:
        return []
    packer = BytePacker(bos=bos, eos=eos)
    sequences: list[list[int]] = []
    with ExitStack() as stack:
        for path in paths:
            shard = stack.enter_context(MemoryMappedShard(path))
            for seq in _iter_shard_sequences(shard, packer):
                sequences.append(seq)
                if limit is not None and len(sequences) >= limit:
                    return sequences
    return sequences


def summarize_corpus(
    sequences: Sequence[Sequence[int]],
    *,
    sources: list[dict[str, object]] | None = None,
) -> CorpusSummary:
    total_tokens = sum(len(seq) for seq in sequences)
    max_len = max((len(seq) for seq in sequences), default=0)
    return CorpusSummary(
        sequences=len(sequences),
        tokens=total_tokens,
        max_length=max_len,
        sources=list(sources or []),
    )


def _build_bpe_batches(
    sequences: Sequence[Sequence[int]],
    *,
    batch_size: int,
    seed: int,
) -> PackedBatcher:
    return PackedBatcher(sequences, batch_size=batch_size, seed=seed)


def _build_unigram_batches(
    sequences: Sequence[Sequence[int]],
    *,
    batch_size: int,
    seed: int,
) -> list[torch.Tensor]:
    packed = PackedBatcher(sequences, batch_size=batch_size, seed=seed)
    return [tokens.clone() for tokens, _valid, _lengths in packed]


def run_bpe_benchmark(
    sequences: Sequence[Sequence[int]],
    *,
    base_vocab: int,
    merges: int,
    batch_size: int,
    device: str | None,
    seed: int,
    log_every: int,
) -> dict[str, object]:
    _ensure_trainers_available()
    autoscaler = AutoScaler(min_bs=batch_size, max_bs=batch_size, device=device)
    batches = _build_bpe_batches(sequences, batch_size=batch_size, seed=seed)
    trainer = GPUBPETrainer(
        base_vocab=base_vocab,
        merges=merges,
        device=device,
        autoscaler=autoscaler,
    )
    wall_start = time.perf_counter()
    meta = trainer.fit(batches, log_every=log_every)
    wall_time = time.perf_counter() - wall_start
    return {
        "config": {
            "base_vocab": base_vocab,
            "merges": merges,
            "batch_size": batch_size,
            "device": device,
            "log_every": log_every,
        },
        "wall_time_s": wall_time,
        "result": meta,
    }


def run_unigram_benchmark(
    sequences: Sequence[Sequence[int]],
    *,
    base_vocab: int,
    vocab_size: int,
    max_subword_len: int,
    batch_size: int,
    epochs: int,
    device: str | None,
    seed: int,
) -> dict[str, object]:
    _ensure_trainers_available()
    trainer = GPUUnigramTrainer(
        base_vocab=base_vocab,
        vocab_size=vocab_size,
        max_subword_len=max_subword_len,
        device=device,
        seed=seed,
    )
    batches = _build_unigram_batches(sequences, batch_size=batch_size, seed=seed)
    wall_start = time.perf_counter()
    epoch_metrics: list[dict[str, object]] = []
    for epoch in range(epochs):
        stats = trainer.fit_epoch(batches)
        stats["epoch"] = epoch + 1
        epoch_metrics.append(stats)
    wall_time = time.perf_counter() - wall_start
    return {
        "config": {
            "base_vocab": base_vocab,
            "vocab_size": vocab_size,
            "max_subword_len": max_subword_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "device": device,
        },
        "wall_time_s": wall_time,
        "epochs": epoch_metrics,
    }


def format_summary_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    columns = list(zip(*([headers] + list(rows))))
    widths = [max(len(cell) for cell in column) for column in columns]
    def _fmt(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    line = "-+-".join("-" * width for width in widths)
    parts = [_fmt(headers), line]
    parts.extend(_fmt(row) for row in rows)
    return "\n".join(parts)


def emit_benchmark_summary(
    corpus: CorpusSummary,
    bpe_result: dict[str, object],
    unigram_result: dict[str, object],
) -> str:
    rows = [
        [
            "GPUBPETrainer",
            f"{bpe_result['wall_time_s']:.2f}",
            f"{corpus.tokens / bpe_result['wall_time_s']:.2f}" if bpe_result["wall_time_s"] > 0 else "n/a",
            str(bpe_result["result"].get("vocab_size", "")),
        ],
        [
            "GPUUnigramTrainer",
            f"{unigram_result['wall_time_s']:.2f}",
            f"{corpus.tokens / unigram_result['wall_time_s']:.2f}" if unigram_result["wall_time_s"] > 0 else "n/a",
            str(unigram_result["epochs"][-1].get("vocab", "")) if unigram_result["epochs"] else "",
        ],
    ]
    headers = ["Trainer", "Wall time (s)", "Tokens/s", "Final vocab"]
    summary = [
        f"Corpus → {corpus.sequences} sequences, {corpus.tokens} tokens (max len {corpus.max_length})",
        format_summary_table(rows, headers),
    ]
    return "\n".join(summary)


def serialize_run(
    output_dir: Path,
    *,
    corpus: CorpusSummary,
    config: dict[str, object],
    bpe: dict[str, object],
    unigram: dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": timestamp,
        "config": config,
        "corpus": {
            "sequences": corpus.sequences,
            "tokens": corpus.tokens,
            "max_length": corpus.max_length,
            "sources": corpus.sources,
        },
        "bpe": bpe,
        "unigram": unigram,
    }
    path = output_dir / f"benchmark_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, torch.device):  # type: ignore[attr-defined]
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
