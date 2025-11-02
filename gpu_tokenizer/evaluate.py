"""Corpus evaluation utilities for trained tokenizer artifacts."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

from .cpu_packer import BytePacker
from .code_mode import prepare_corpus
from .evaluate_metrics import (
    compute_code_mode_reduction,
    compute_compression_ratio,
    compute_morphology_purity,
    compute_oov_rate,
)
from .evaluate_report import validate_evaluate_report
from .export import artifacts as export_artifacts
from .morphology import MorphologyPlugin


@dataclass(frozen=True)
class LoadedCorpus:
    """Materialised corpus payload used by :func:`evaluate`."""

    documents: list[bytes]
    tokens: list[list[int | str]]
    raw_bytes: int
    summary: dict[str, object]


@dataclass(frozen=True)
class MergeRule:
    """Typed representation of a single merge rule."""

    left: int
    right: int
    new_id: int


@dataclass(frozen=True)
class EvaluateCLIOptions:
    """Typed configuration that mirrors the CLI ``evaluate`` command."""

    data_files: Sequence[Path]
    vocab_path: Path
    merges_path: Path | None = None
    tokenizer_path: Path | None = None
    model_type: str = "auto"
    bos: int | None = None
    eos: int | None = None
    morphology: MorphologyPlugin | None = None
    morphology_config: Mapping[str, object] | None = None
    code_mode: bool = False
    code_languages: Sequence[str] | None = None
    meta_compress: bool = False
    meta_max_length: int = 8
    deterministic: bool = False


@dataclass(frozen=True)
class EvaluateCLIResult:
    """Structured response that mirrors CLI side-effects."""

    report: dict[str, object]
    summary: dict[str, object]


ModelType = Literal["bpe", "unigram"]


def _expand_data_patterns(patterns: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        for match in sorted(Path().glob(pattern)):
            if match.is_file():
                files.append(match.resolve())
    return files


def _load_plain_text(
    paths: Sequence[Path],
    *,
    bos: int | None,
    eos: int | None,
    morphology: MorphologyPlugin | None,
) -> LoadedCorpus:
    packer = BytePacker(bos=bos, eos=eos, morphology=morphology)
    documents: list[bytes] = []
    tokens: list[list[int]] = []
    total_bytes = 0
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise SystemExit(f"Failed to read corpus shard {path}: {exc}") from exc
        for line in content.splitlines():
            if not line:
                continue
            encoded = line.encode("utf-8")
            documents.append(encoded)
            total_bytes += len(encoded)
            tokens.append([int(tok) for tok in packer.encode_view(encoded)])
    summary = {
        "mode": "plain",
        "documents": len(documents),
    }
    return LoadedCorpus(documents, tokens, total_bytes, summary)


def _load_code_entries(paths: Sequence[Path]) -> list[Mapping[str, object]]:
    entries: list[Mapping[str, object]] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, Sequence):
                for item in payload:
                    if isinstance(item, Mapping):
                        entries.append(item)
            elif isinstance(payload, Mapping):
                data = payload.get("entries")
                if isinstance(data, Sequence):
                    for item in data:
                        if isinstance(item, Mapping):
                            entries.append(item)
                else:
                    raise SystemExit(
                        f"JSON manifest {path} must contain an array or an 'entries' list"
                    )
            else:
                raise SystemExit(
                    f"JSON manifest {path} must contain an array or mapping of entries"
                )
            continue
        if suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    payload = line.strip()
                    if not payload:
                        continue
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                        raise SystemExit(
                            f"Failed to parse JSON entry in {path} line {index + 1}: {exc}"
                        ) from exc
                    if isinstance(data, Mapping):
                        entries.append(data)
            continue
        raise SystemExit(
            "--code-mode expects JSON/JSONL manifests containing language+source entries"
        )
    return entries


def _load_code_mode(
    paths: Sequence[Path],
    *,
    bos: int | None,
    eos: int | None,
    morphology: MorphologyPlugin | None,
    languages: set[str] | None,
    meta_enabled: bool,
    meta_max_length: int,
) -> LoadedCorpus:
    raw_entries = _load_code_entries(paths)
    filtered_entries: list[Mapping[str, object]] = []
    language_set: set[str] = set()
    for entry in raw_entries:
        language = str(entry.get("language", "")).strip().lower()
        if languages and language and language not in languages:
            continue
        source = entry.get("source")
        if not isinstance(source, str):
            raise SystemExit("Code entries must provide a string 'source'")
        filtered_entries.append(entry)
        if language:
            language_set.add(language)
    if not filtered_entries:
        raise SystemExit("--code-mode did not receive any valid entries")

    corpus = prepare_corpus(
        filtered_entries,
        meta_enabled=meta_enabled,
        meta_max_length=max(int(meta_max_length), 1),
    )

    documents: list[bytes] = []
    tokens: list[list[int | str]] = []
    total_bytes = 0
    packer = BytePacker(bos=bos, eos=eos, morphology=morphology)
    for sample, entry in zip(corpus.samples, filtered_entries):
        source = str(entry.get("source", ""))
        encoded = source.encode("utf-8")
        documents.append(encoded)
        total_bytes += len(encoded)
        if sample.kind == "bytes":
            tokens.append([int(tok) for tok in packer.encode_view(encoded)])
        else:
            tokens.append([str(tok) for tok in sample.tokens])

    summary = {
        "mode": "code",
        "documents": len(documents),
        "ast_samples": sum(1 for sample in corpus.samples if sample.kind == "ast"),
        "fallback_samples": sum(1 for sample in corpus.samples if sample.kind == "bytes"),
        "languages": sorted(language_set) if language_set else None,
        "meta_compress": corpus.meta_tokens,
        "meta_token_count": len(corpus.meta_tokens),
        "meta_enabled": corpus.meta_enabled,
        "meta_max_length": corpus.meta_max_length,
    }
    return LoadedCorpus(documents, tokens, total_bytes, summary)


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_merge_entry(
    entry: object, vocab_ids: set[int], name_to_id: Mapping[str, int]
) -> MergeRule | None:
    left: int | None = None
    right: int | None = None
    new_id: int | None = None
    if isinstance(entry, Mapping):
        left = _coerce_int(entry.get("left"))
        right = _coerce_int(entry.get("right"))
        new_id = _coerce_int(entry.get("id") or entry.get("new_id"))
        if left is None:
            left_name = entry.get("left")
            if isinstance(left_name, str):
                left = name_to_id.get(left_name)
        if right is None:
            right_name = entry.get("right")
            if isinstance(right_name, str):
                right = name_to_id.get(right_name)
        if new_id is None:
            target_name = entry.get("token") or entry.get("name")
            if isinstance(target_name, str):
                new_id = name_to_id.get(target_name)
    elif isinstance(entry, Sequence):
        if len(entry) >= 2:
            left = _coerce_int(entry[0])
            right = _coerce_int(entry[1])
        if len(entry) >= 3:
            new_id = _coerce_int(entry[2])
    if left is None or right is None or new_id is None:
        return None
    return MergeRule(left=left, right=right, new_id=new_id)


def _load_merges(path: Path, vocab: Mapping[str, int]) -> list[MergeRule]:
    if not path.exists():
        return []
    name_to_id = {str(name): int(idx) for name, idx in vocab.items()}
    vocab_ids = set(name_to_id.values())
    with path.open("r", encoding="utf-8") as handle:
        content = handle.read()
    if not content.strip():
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(content)
    else:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        data: list[list[str]] = []
        for line in lines:
            if line.startswith("#"):
                continue
            pieces = line.split()
            data.append(pieces)
    merges: list[MergeRule] = []
    if isinstance(data, Sequence):
        for entry in data:
            rule = _parse_merge_entry(entry, vocab_ids, name_to_id)
            if rule is not None:
                merges.append(rule)
    return merges


def _load_sentencepiece_vocab(path: Path) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                token, _ = line.split("\t", 1)
            else:
                pieces = line.split()
                if not pieces:
                    continue
                token = pieces[0]
            if token not in vocab:
                vocab[token] = index
    return vocab


def resolve_model_type(
    vocab_path: Path,
    merges_path: Path | None,
    hint: str | None = None,
) -> ModelType:
    """Resolve the artifact flavour used during evaluation.

    Args:
        vocab_path: Path to the provided vocabulary artifact.
        merges_path: Optional path to the merge history.
        hint: Optional string hint supplied via the CLI.

    Returns:
        Literal indicating whether BPE or unigram exports are being evaluated.

    Raises:
        ValueError: If the combination of hint/paths cannot be reconciled.
    """

    choice = str(hint or "auto").strip().lower()
    if choice not in {"auto", "bpe", "unigram"}:
        raise ValueError(
            f"Unknown model type {hint!r}; expected 'auto', 'bpe', or 'unigram'."
        )

    suffix = vocab_path.suffix.lower()

    if choice == "bpe":
        if merges_path is None:
            raise ValueError(
                "BPE evaluation requires a merge history; provide --merges or "
                "an artifacts directory containing merges.(json|txt)."
            )
        return "bpe"

    if choice == "unigram":
        if merges_path is not None:
            raise ValueError(
                "Unigram evaluation ignores merge tables; remove --merges or set "
                "--model-type bpe if you meant to score a BPE export."
            )
        return "unigram"

    # Auto-detection path
    if suffix == ".vocab":
        if merges_path is not None:
            raise ValueError(
                "SentencePiece vocabulary detected but a merge file was also "
                "provided; drop --merges or force --model-type bpe if the "
                "artifacts are truly BPE."
            )
        return "unigram"

    if merges_path is None:
        raise ValueError(
            "Could not infer model type. Provide a merge history for BPE exports "
            "or set --model-type unigram when evaluating SentencePiece outputs."
        )

    return "bpe"


def _load_cli_vocab(path: Path, *, model_type: ModelType) -> dict[str, int]:
    if model_type == "unigram" and path.suffix.lower() == ".vocab":
        return _load_sentencepiece_vocab(path)
    return export_artifacts.load_vocab(path)


def _apply_merges(sequence: Sequence[int], rules: Sequence[MergeRule]) -> list[int]:
    if not sequence or not rules:
        return list(sequence)
    work: list[int] = list(sequence)
    for rule in rules:
        idx = 0
        while idx < len(work) - 1:
            if work[idx] == rule.left and work[idx + 1] == rule.right:
                work[idx : idx + 2] = [rule.new_id]
                if idx:
                    idx -= 1
            else:
                idx += 1
    return work


def _aggregate_morphology(
    documents: Iterable[bytes], *, plugin: MorphologyPlugin | None
) -> dict[str, object]:
    if plugin is None:
        return {"enabled": False}
    total_segments = 0
    tagged = 0
    role_counts: Counter[str] = Counter()
    doc_count = 0
    for payload in documents:
        doc_count += 1
        segments = list(plugin.presegment(payload))
        total_segments += len(segments)
        for segment in segments:
            tags = getattr(segment, "tags", ())
            if tags:
                tagged += 1
            role = getattr(segment, "role", None)
            if role:
                role_counts[str(role)] += 1
    average = total_segments / doc_count if doc_count else 0.0
    return {
        "enabled": True,
        "documents": doc_count,
        "total_segments": total_segments,
        "tagged_segments": tagged,
        "average_segments": average,
        "roles": dict(sorted(role_counts.items())) if role_counts else {},
    }


def evaluate(
    data_files: Sequence[Path],
    *,
    vocab_path: Path,
    merges_path: Path | None = None,
    tokenizer_path: Path | None = None,
    model_type: str | None = None,
    bos: int | None = None,
    eos: int | None = None,
    morphology: MorphologyPlugin | None = None,
    code_mode: bool = False,
    code_languages: set[str] | None = None,
    meta_compress: bool = False,
    meta_max_length: int = 8,
    deterministic: bool = False,
) -> dict[str, object]:
    """Evaluate trained tokenizer artifacts against a reference corpus."""

    if deterministic:
        random.seed(1337)

    resolved_model = resolve_model_type(vocab_path, merges_path, model_type)

    vocab = _load_cli_vocab(vocab_path, model_type=resolved_model)
    merges = _load_merges(merges_path, vocab) if merges_path else []

    if code_mode:
        corpus = _load_code_mode(
            data_files,
            bos=bos,
            eos=eos,
            morphology=morphology,
            languages=code_languages,
            meta_enabled=meta_compress,
            meta_max_length=meta_max_length,
        )
    else:
        corpus = _load_plain_text(data_files, bos=bos, eos=eos, morphology=morphology)

    vocab_ids = {int(idx) for idx in vocab.values()}
    vocab_tokens = set(vocab)

    evaluated_sequences: list[list[int | str]] = []
    total_tokens = 0
    oov_instances = 0
    oov_items: set[int | str] = set()

    for sequence in corpus.tokens:
        if sequence and isinstance(sequence[0], int):
            merged = _apply_merges([int(tok) for tok in sequence], merges)
            evaluated_sequences.append(merged)
        else:
            evaluated_sequences.append(list(sequence))

    for sequence in evaluated_sequences:
        length = len(sequence)
        total_tokens += length
        for token in sequence:
            if isinstance(token, int):
                if token not in vocab_ids:
                    oov_instances += 1
                    oov_items.add(token)
            else:
                if token not in vocab_tokens:
                    oov_instances += 1
                    oov_items.add(token)

    total_bytes = corpus.raw_bytes
    compression = compute_compression_ratio(total_tokens, total_bytes)
    avg_tokens = total_tokens / len(evaluated_sequences) if evaluated_sequences else 0.0
    avg_bytes = total_bytes / len(evaluated_sequences) if evaluated_sequences else 0.0
    oov_rate = compute_oov_rate(oov_instances, total_tokens)

    morphology_summary = _aggregate_morphology(corpus.documents, plugin=morphology)
    morphology_summary.setdefault("tagged_segments", 0)
    morphology_summary.setdefault("total_segments", 0)
    morphology_summary["purity"] = compute_morphology_purity(
        morphology_summary.get("tagged_segments"),
        morphology_summary.get("total_segments"),
    )

    code_mode_summary = dict(corpus.summary)
    if code_mode_summary.get("mode") == "code":
        meta = code_mode_summary.get("meta_compress")
        code_mode_summary["reduction"] = compute_code_mode_reduction(
            evaluated_sequences,
            meta if isinstance(meta, Mapping) else {},
        )
    else:
        code_mode_summary.setdefault("reduction", 0.0)

    def _posix(p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            return p.as_posix()
        except Exception:
            return str(p).replace("\\", "/")

    report = {
        "artifacts": {
            "vocab": _posix(vocab_path),
            "vocab_size": len(vocab),
            "merges": _posix(merges_path) if merges_path else None,
            "merge_rules": len(merges),
            "tokenizer": _posix(tokenizer_path) if tokenizer_path else None,
            "model_type": resolved_model,
        },
        "corpus": {
            "documents": len(evaluated_sequences),
            "total_bytes": total_bytes,
            "total_tokens": total_tokens,
            "average_bytes": avg_bytes,
            "average_tokens": avg_tokens,
        },
        "compression": compression,
        "oov": {
            "instances": oov_instances,
            "rate": oov_rate,
            "unique": sorted(oov_items, key=lambda item: (str(type(item)), item)),
        },
        "morphology": morphology_summary,
        "code_mode": code_mode_summary,
    }
    return report


def _normalise_cli_languages(languages: Sequence[str] | None) -> list[str] | None:
    if not languages:
        return None
    normalised: set[str] = set()
    for entry in languages:
        if entry is None:
            continue
        text = str(entry)
        if not text:
            continue
        for piece in text.split(","):
            norm = piece.strip().lower()
            if norm:
                normalised.add(norm)
    if not normalised:
        return None
    return sorted(normalised)


def _morphology_config_from_options(options: EvaluateCLIOptions) -> dict[str, object]:
    if options.morphology_config is not None:
        payload = dict(options.morphology_config)
        payload.setdefault("enabled", bool(options.morphology))
        return payload
    return {"enabled": bool(options.morphology)}


def evaluate_cli(options: EvaluateCLIOptions) -> EvaluateCLIResult:
    if not options.data_files:
        raise ValueError("EvaluateCLIOptions.data_files must not be empty")

    data_files = [Path(path) for path in options.data_files]
    language_filter = _normalise_cli_languages(options.code_languages)
    merges_path = Path(options.merges_path) if options.merges_path else None
    vocab_path = Path(options.vocab_path)
    resolved_model = resolve_model_type(vocab_path, merges_path, options.model_type)
    report = evaluate(
        data_files,
        vocab_path=vocab_path,
        merges_path=merges_path,
        tokenizer_path=Path(options.tokenizer_path) if options.tokenizer_path else None,
        model_type=resolved_model,
        bos=options.bos,
        eos=options.eos,
        morphology=options.morphology,
        code_mode=bool(options.code_mode),
        code_languages=set(language_filter) if language_filter else None,
        meta_compress=bool(options.meta_compress),
        meta_max_length=int(options.meta_max_length),
        deterministic=bool(options.deterministic),
    )

    report.setdefault("morphology", {})["config"] = _morphology_config_from_options(options)
    report.setdefault("code_mode", {})["config"] = {
        "enabled": bool(options.code_mode),
        "languages_filter": language_filter,
        "meta_compress": bool(options.meta_compress),
        "meta_max_length": int(options.meta_max_length),
    }

    validate_evaluate_report(report)

    summary = {
        "documents": report["corpus"].get("documents"),
        "tokens": report["corpus"].get("total_tokens"),
        "tokens_per_byte": report["compression"].get("tokens_per_byte"),
        "oov_rate": report["oov"].get("rate"),
        "model_type": resolved_model,
    }

    return EvaluateCLIResult(report=report, summary=summary)


__all__ = [
    "EvaluateCLIOptions",
    "EvaluateCLIResult",
    "evaluate",
    "evaluate_cli",
    "resolve_model_type",
]
