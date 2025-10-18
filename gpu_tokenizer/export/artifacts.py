"""Utilities for exporting embedding artifacts for GPU tokenizer vocabularies."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TokenStats:
    """Aggregated statistics for a token gathered during co-training."""

    count: float = 0.0
    vector: tuple[float, ...] | None = None


@dataclass(frozen=True)
class PruneResult:
    """Summary returned by :func:`prune_vocabulary`."""

    vocab: dict[str, int]
    pruned: list[dict[str, object]]
    original_size: int


@dataclass(frozen=True)
class ExportManifest:
    """Metadata that accompanies an exported embedding package."""

    dimension: int
    dtype: str
    seed: int | None
    original_token_count: int
    exported_token_count: int
    min_frequency: float
    preserved_tokens: tuple[str, ...]


def load_vocab(path: str | os.PathLike[str]) -> dict[str, int]:
    """Load a tokenizer vocabulary JSON mapping tokens to integer ids."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive
        raise TypeError("Vocabulary JSON must contain a mapping of token→id")
    vocab: dict[str, int] = {}
    for token, value in payload.items():
        try:
            index = int(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(f"Invalid id for token {token!r}: {value!r}") from exc
        vocab[str(token)] = index
    return vocab


def _coerce_stats_entry(value: object) -> TokenStats:
    if isinstance(value, TokenStats):
        return value
    if isinstance(value, Mapping):
        count = float(value.get("count") or value.get("frequency") or 0.0)
        vector = value.get("vector")
        if isinstance(vector, Sequence):
            try:
                data = tuple(float(v) for v in vector)
            except Exception:  # pragma: no cover - defensive
                data = None
        else:
            data = None
        return TokenStats(count=count, vector=data)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            data = tuple(float(v) for v in value)
        except Exception:  # pragma: no cover - defensive
            data = None
        return TokenStats(count=float(len(data or ())), vector=data)
    if isinstance(value, (int, float)):
        return TokenStats(count=float(value), vector=None)
    return TokenStats()


def load_token_stats(path: str | os.PathLike[str]) -> dict[str, TokenStats]:
    """Load optional co-training statistics describing token usage."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping) and "tokens" in payload:
        payload = payload["tokens"]
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive
        raise TypeError("Token statistics must be a mapping of token→metadata")
    stats: dict[str, TokenStats] = {}
    for token, value in payload.items():
        stats[str(token)] = _coerce_stats_entry(value)
    return stats


def resolve_dtype(name: str) -> str:
    """Normalize a dtype string for manifest reporting."""

    normalized = name.lower()
    mapping = {
        "half": "float16",
        "single": "float32",
        "double": "float64",
        "bf16": "bfloat16",
    }
    normalized = mapping.get(normalized, normalized)
    allowed = {"float16", "float32", "float64", "bfloat16"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return normalized


def prune_vocabulary(
    vocab: Mapping[str, int],
    stats: Mapping[str, TokenStats] | None,
    *,
    min_frequency: float = 0.0,
    keep_tokens: Iterable[str] | None = None,
) -> PruneResult:
    """Prune rarely used tokens and renumber the resulting vocabulary."""

    keep_set = {str(token) for token in keep_tokens or ()}
    stats = stats or {}
    ordered_tokens = sorted(vocab.items(), key=lambda item: item[1])

    new_vocab: dict[str, int] = {}
    pruned: list[dict[str, object]] = []
    for token, old_index in ordered_tokens:
        token_stats = stats.get(token)
        count = float(token_stats.count if token_stats else 0.0)
        if token in keep_set or count >= min_frequency:
            new_vocab[token] = len(new_vocab)
        else:
            pruned.append({"token": token, "id": int(old_index), "count": count})
    return PruneResult(vocab=new_vocab, pruned=pruned, original_size=len(vocab))


def generate_embedding_matrix(
    vocab: Mapping[str, int],
    stats: Mapping[str, TokenStats] | None,
    *,
    dimension: int,
    seed: int | None = None,
    dtype: str = "float32",
) -> list[list[float]]:
    """Generate an embedding matrix aligned with *vocab*."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    rng = random.Random(seed)
    stats = stats or {}
    matrix: list[list[float]] = []
    scale = 1.0 / math.sqrt(max(dimension, 1))
    reverse_vocab = {index: token for token, index in vocab.items()}
    for index in range(len(vocab)):
        token = reverse_vocab[index]
        token_stats = stats.get(token)
        if token_stats and token_stats.vector and len(token_stats.vector) == dimension:
            matrix.append([float(value) for value in token_stats.vector])
            continue
        count = float(token_stats.count if token_stats else 0.0)
        adjusted = scale / max(math.sqrt(count + 1.0), 1.0)
        row = [rng.gauss(0.0, adjusted) for _ in range(dimension)]
        matrix.append(row)
    return matrix


def build_manifest(
    *,
    dimension: int,
    dtype: str,
    seed: int | None,
    prune: PruneResult,
    min_frequency: float,
    preserved_tokens: Iterable[str],
) -> ExportManifest:
    """Construct an :class:`ExportManifest` summarising an export run."""

    preserved = tuple(dict.fromkeys(str(token) for token in preserved_tokens))
    return ExportManifest(
        dimension=dimension,
        dtype=str(dtype),
        seed=seed,
        original_token_count=prune.original_size,
        exported_token_count=len(prune.vocab),
        min_frequency=float(min_frequency),
        preserved_tokens=preserved,
    )


def _serialize_manifest(manifest: ExportManifest) -> dict[str, object]:
    return {
        "dimension": manifest.dimension,
        "dtype": manifest.dtype,
        "seed": manifest.seed,
        "original_token_count": manifest.original_token_count,
        "exported_token_count": manifest.exported_token_count,
        "min_frequency": manifest.min_frequency,
        "preserved_tokens": list(manifest.preserved_tokens),
    }


def write_export_package(
    out_dir: str | os.PathLike[str],
    *,
    embeddings: Sequence[Sequence[float]],
    vocab: Mapping[str, int],
    manifest: ExportManifest,
    pruned: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    """Persist embedding export artifacts to *out_dir*."""

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    vocab_path = path / "vocab.json"
    embeddings_path = path / "embeddings.json"
    manifest_path = path / "manifest.json"
    pruning_path = path / "pruning.json"

    ordered_vocab = sorted(vocab.items(), key=lambda item: item[1])
    with vocab_path.open("w", encoding="utf-8") as handle:
        json.dump({token: index for token, index in ordered_vocab}, handle, ensure_ascii=False)

    with embeddings_path.open("w", encoding="utf-8") as handle:
        json.dump(embeddings, handle, ensure_ascii=False)

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(_serialize_manifest(manifest), handle, ensure_ascii=False)

    with pruning_path.open("w", encoding="utf-8") as handle:
        json.dump(list(pruned), handle, ensure_ascii=False)

    return {
        "vocab": str(vocab_path),
        "embeddings": str(embeddings_path),
        "manifest": str(manifest_path),
        "pruning": str(pruning_path),
    }


__all__ = [
    "ExportManifest",
    "PruneResult",
    "TokenStats",
    "build_manifest",
    "generate_embedding_matrix",
    "load_token_stats",
    "load_vocab",
    "prune_vocabulary",
    "resolve_dtype",
    "write_export_package",
]
