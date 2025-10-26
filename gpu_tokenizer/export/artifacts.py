"""Utilities for exporting embedding artifacts for GPU tokenizer vocabularies."""

from __future__ import annotations

import base64
import hashlib
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
class DedupeResult:
    """Summary returned by :func:`dedupe_vocabulary`."""

    vocab: dict[str, int]
    stats: dict[str, TokenStats]
    deduped: list[dict[str, object]]
    original_size: int


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


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return -1.0
    return dot / (norm_a * norm_b)


def _weighted_average(
    a: Sequence[float] | None,
    b: Sequence[float] | None,
    weight_a: float,
    weight_b: float,
) -> list[float]:
    if not a and not b:
        return []
    if not a:
        return list(b or [])
    if not b:
        return list(a)
    max_len = max(len(a), len(b))
    data_a = list(a) + [0.0] * (max_len - len(a))
    data_b = list(b) + [0.0] * (max_len - len(b))
    total = weight_a + weight_b
    if total <= 0.0:
        total = 1.0
    return [
        (data_a[index] * weight_a + data_b[index] * weight_b) / total
        for index in range(max_len)
    ]


def _weight_from(count: float) -> float:
    return count if count > 0.0 else 1.0


def _synth_similarity_vector(token: str, *, dimension: int, seed: int | None) -> list[float]:
    token_seed_bytes = f"{seed or 0}:{token}".encode("utf-8")
    digest = hashlib.sha256(token_seed_bytes).digest()
    token_seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = random.Random(token_seed)
    length = max(int(dimension), 1)
    return [rng.uniform(-1.0, 1.0) for _ in range(length)]


class _Cluster:
    __slots__ = (
        "canonical",
        "canonical_id",
        "sim_vector",
        "total_count",
        "stats_vector",
        "stats_weight",
    )

    def __init__(
        self,
        canonical: str,
        canonical_id: int,
        *,
        sim_vector: Sequence[float],
        stats_vector: Sequence[float] | None,
        count: float,
    ) -> None:
        self.canonical = canonical
        self.canonical_id = canonical_id
        self.sim_vector = list(sim_vector)
        self.total_count = float(count)
        self.stats_vector = list(stats_vector) if stats_vector is not None else None
        self.stats_weight = _weight_from(count) if stats_vector is not None else 0.0

    def merge(
        self,
        *,
        sim_vector: Sequence[float],
        stats_vector: Sequence[float] | None,
        count: float,
    ) -> None:
        previous_total = self.total_count
        weight_existing = _weight_from(previous_total)
        weight_new = _weight_from(count)
        self.sim_vector = _weighted_average(
            self.sim_vector,
            sim_vector,
            weight_existing,
            weight_new,
        )
        self.total_count = previous_total + float(count)
        if stats_vector is not None:
            stats_weight_new = _weight_from(count)
            if self.stats_vector is None:
                self.stats_vector = list(stats_vector)
                self.stats_weight = stats_weight_new
            else:
                self.stats_vector = _weighted_average(
                    self.stats_vector,
                    stats_vector,
                    self.stats_weight,
                    stats_weight_new,
                )
                self.stats_weight += stats_weight_new

def dedupe_vocabulary(
    vocab: Mapping[str, int],
    stats: Mapping[str, TokenStats] | None,
    *,
    similarity_threshold: float,
    dimension: int,
    seed: int | None = None,
    keep_tokens: Iterable[str] | None = None,
) -> DedupeResult:
    """Cluster similar tokens and merge them into a deduplicated vocabulary."""

    original_size = len(vocab)
    stats_map: dict[str, TokenStats] = {}
    for token, entry in (stats or {}).items():
        stats_map[str(token)] = entry if isinstance(entry, TokenStats) else _coerce_stats_entry(entry)

    ordered_tokens = sorted(vocab.items(), key=lambda item: item[1])
    keep_set = {str(token) for token in keep_tokens or ()}

    if similarity_threshold <= 0.0 or len(ordered_tokens) <= 1:
        new_vocab = {token: index for index, (token, _) in enumerate(ordered_tokens)}
        new_stats: dict[str, TokenStats] = {}
        for token, _ in ordered_tokens:
            entry = stats_map.get(token)
            new_stats[token] = entry if entry is not None else TokenStats()
        return DedupeResult(
            vocab=new_vocab,
            stats=new_stats,
            deduped=[],
            original_size=original_size,
        )

    clusters: list[_Cluster] = []
    deduped_entries: list[dict[str, object]] = []

    for token, old_index in ordered_tokens:
        stats_entry = stats_map.get(token)
        stats_vector = list(stats_entry.vector) if stats_entry and stats_entry.vector else None
        count = float(stats_entry.count if stats_entry else 0.0)
        sim_vector = (
            stats_vector[:] if stats_vector is not None else _synth_similarity_vector(token, dimension=dimension, seed=seed)
        )

        if token in keep_set:
            cluster = _Cluster(
                token,
                int(old_index),
                sim_vector=sim_vector,
                stats_vector=stats_vector,
                count=count,
            )
            clusters.append(cluster)
            continue

        best_cluster: _Cluster | None = None
        best_similarity = similarity_threshold
        for cluster in clusters:
            similarity = _cosine_similarity(sim_vector, cluster.sim_vector)
            if similarity >= best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is None:
            cluster = _Cluster(
                token,
                int(old_index),
                sim_vector=sim_vector,
                stats_vector=stats_vector,
                count=count,
            )
            clusters.append(cluster)
        else:
            best_cluster.merge(sim_vector=sim_vector, stats_vector=stats_vector, count=count)
            deduped_entries.append(
                {
                    "token": token,
                    "id": int(old_index),
                    "merged_into": best_cluster.canonical,
                    "similarity": float(best_similarity),
                    "count": count,
                    "action": "deduped",
                }
            )

    ordered_clusters = sorted(clusters, key=lambda cluster: cluster.canonical_id)
    new_vocab: dict[str, int] = {}
    new_stats: dict[str, TokenStats] = {}
    for index, cluster in enumerate(ordered_clusters):
        token = cluster.canonical
        new_vocab[token] = index
        aggregated_count = cluster.total_count
        base_entry = stats_map.get(token)
        if aggregated_count <= 0.0 and base_entry is not None:
            aggregated_count = float(base_entry.count)
        vector = None
        if cluster.stats_vector is not None:
            vector = tuple(cluster.stats_vector)
        elif base_entry and base_entry.vector is not None:
            vector = tuple(base_entry.vector)
        new_stats[token] = TokenStats(count=float(aggregated_count), vector=vector)

    return DedupeResult(
        vocab=new_vocab,
        stats=new_stats,
        deduped=deduped_entries,
        original_size=original_size,
    )


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
    original_size: int | None = None,
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
    size = original_size if original_size is not None else len(vocab)
    return PruneResult(vocab=new_vocab, pruned=pruned, original_size=size)


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


def _build_byte_level_mappings() -> tuple[dict[int, str], dict[str, int]]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    encoder = {b: chr(c) for b, c in zip(bs, cs)}
    decoder = {v: k for k, v in encoder.items()}
    return encoder, decoder


_BYTE_ENCODER, _BYTE_DECODER = _build_byte_level_mappings()


def byte_level_encoder() -> dict[int, str]:
    """Return the byte→unicode mapping used by ByteLevel BPE tokenizers."""

    return dict(_BYTE_ENCODER)


def byte_level_decoder() -> dict[str, int]:
    """Return the unicode→byte mapping used by ByteLevel BPE tokenizers."""

    return dict(_BYTE_DECODER)


def _token_to_bytes(token: str, *, decoder: Mapping[str, int]) -> bytes:
    try:
        return bytes(decoder[char] for char in token)
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Token contains characters outside the ByteLevel alphabet: {token!r}"
        ) from exc


def vocab_to_tiktoken_mergeable_ranks(
    vocab: Mapping[str, int],
    *,
    skip_tokens: Iterable[str] | None = None,
    decoder: Mapping[str, int] | None = None,
) -> dict[bytes, int]:
    """Convert a vocab mapping into TikToken ``mergeable_ranks`` bytes."""

    decoder_map = dict(decoder or _BYTE_DECODER)
    skip = {token for token in (skip_tokens or ())}
    ordered = sorted(vocab.items(), key=lambda item: int(item[1]))
    mergeable: dict[bytes, int] = {}
    for token, rank in ordered:
        if token in skip:
            continue
        token_bytes = _token_to_bytes(token, decoder=decoder_map)
        mergeable[token_bytes] = int(rank)
    return mergeable


def serialize_tiktoken_bpe(mergeable_ranks: Mapping[bytes, int]) -> bytes:
    """Serialize TikToken ``mergeable_ranks`` to packed bytes."""

    lines: list[bytes] = []
    for token_bytes, rank in sorted(mergeable_ranks.items(), key=lambda item: int(item[1])):
        encoded = base64.b64encode(token_bytes)
        lines.append(encoded + b" " + str(int(rank)).encode("ascii") + b"\n")
    return b"".join(lines)


def write_tiktoken_bpe(
    path: str | os.PathLike[str], mergeable_ranks: Mapping[bytes, int]
) -> str:
    """Write TikToken mergeable ranks to *path* and return the string path."""

    payload = serialize_tiktoken_bpe(mergeable_ranks)
    with open(path, "wb") as handle:
        handle.write(payload)
    return os.fspath(path)


def load_tiktoken_bpe(path: str | os.PathLike[str]) -> dict[bytes, int]:
    """Load TikToken mergeable ranks without requiring optional dependencies."""

    mergeable: dict[bytes, int] = {}
    with open(path, "rb") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                token_b64, rank_str = line.split()
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Malformed TikToken line: {raw_line!r}") from exc
            token_bytes = base64.b64decode(token_b64)
            mergeable[token_bytes] = int(rank_str)
    return mergeable


def mergeable_ranks_to_merges(
    mergeable_ranks: Mapping[bytes, int],
    *,
    base_vocab: int | None = None,
    encoder: Mapping[int, str] | None = None,
) -> list[tuple[int, int]]:
    """Reconstruct merge pairs from TikToken ``mergeable_ranks``."""

    encoder_map = dict(encoder or _BYTE_ENCODER)
    ordered = sorted(mergeable_ranks.items(), key=lambda item: int(item[1]))
    token_strings: list[str] = [
        "".join(encoder_map[b] for b in token_bytes) for token_bytes, _rank in ordered
    ]

    if base_vocab is None:
        base_vocab = next(
            (idx for idx, token in enumerate(token_strings) if len(token) > 1),
            len(token_strings),
        )
    base_vocab = int(base_vocab)

    token_to_index = {token: idx for idx, token in enumerate(token_strings)}
    merges: list[tuple[int, int]] = []
    for idx in range(base_vocab, len(token_strings)):
        token = token_strings[idx]
        if len(token) <= 1:
            raise ValueError(
                "Encountered single-character token beyond the base vocabulary while reconstructing merges"
            )
        pair: tuple[int, int] | None = None
        for split in range(1, len(token)):
            left = token[:split]
            right = token[split:]
            left_id = token_to_index.get(left)
            right_id = token_to_index.get(right)
            if left_id is None or right_id is None:
                continue
            if left_id < idx and right_id < idx:
                pair = (left_id, right_id)
                break
        if pair is None:
            raise ValueError(
                f"Unable to decompose TikToken symbol {token!r} at rank {idx} into earlier tokens"
            )
        merges.append(pair)
    return merges


def load_tiktoken_merges(
    path: str | os.PathLike[str],
    *,
    base_vocab: int | None = None,
) -> list[tuple[int, int]]:
    """Convenience wrapper that loads TikToken bytes and returns merge pairs."""

    mergeable = load_tiktoken_bpe(path)
    return mergeable_ranks_to_merges(mergeable, base_vocab=base_vocab)


__all__ = [
    "byte_level_decoder",
    "byte_level_encoder",
    "DedupeResult",
    "ExportManifest",
    "PruneResult",
    "TokenStats",
    "build_manifest",
    "dedupe_vocabulary",
    "generate_embedding_matrix",
    "load_tiktoken_bpe",
    "load_tiktoken_merges",
    "load_token_stats",
    "load_vocab",
    "mergeable_ranks_to_merges",
    "prune_vocabulary",
    "resolve_dtype",
    "serialize_tiktoken_bpe",
    "vocab_to_tiktoken_mergeable_ranks",
    "write_export_package",
    "write_tiktoken_bpe",
]
