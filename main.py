"""Main entry-point that ties together the GPU tokenizer components."""

from __future__ import annotations

import glob
import os
import sys
import time
import types
import argparse
import importlib
import json
import textwrap
from dataclasses import asdict
from pathlib import Path
from contextlib import ExitStack
from typing import Iterable, Iterator, Mapping, Sequence

import torch

from gpu_tokenizer import (
    AutoScaler,
    BytePacker,
    GPUBPETrainer,
    GPUUnigramTrainer,
    HybridTrainer,
    PackedBatcher,
    StreamingPackedBatcher,
    utils,
)
from gpu_tokenizer.augmentation import AugmentationMode, AugmentationPipeline, build_augmentation
from gpu_tokenizer.export import artifacts as export_artifacts
from gpu_tokenizer.code_mode import prepare_corpus
from gpu_tokenizer.io import CorpusStreamer, MemoryMappedShard
from gpu_tokenizer.morphology import (
    MorphologyPlugin,
    available_plugins as available_morphology_plugins,
    create_plugin as create_morphology_plugin,
)
from gpu_tokenizer.dtypes import length_storage_dtype
from gpu_tokenizer.trainers.base import BaseTrainer, CheckpointPayload
from gpu_tokenizer.evaluate_report import (
    EvaluateReportValidationError,
    serialize_evaluate_report,
)
from benchmarks import benchmark_runner

evaluate_module = importlib.import_module("gpu_tokenizer.evaluate")

__all__ = [
    "AutoScaler",
    "BytePacker",
    "GPUBPETrainer",
    "GPUUnigramTrainer",
    "HybridTrainer",
    "PackedBatcher",
    "utils",
    "main",
]


SKIP_EVALUATION_ENV = "SUPERTOKEN_SKIP_EVALUATION"
DEFAULT_EVALUATE_OUTPUT = Path("reports/evaluate.json")
EVALUATE_SUMMARY_FORMATS = ("table", "json", "none")


class EvaluateHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Formatter that preserves manual layout while surfacing defaults."""


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    return True


def _should_skip_evaluation(force: bool = False) -> bool:
    if force:
        return False
    return _env_flag(SKIP_EVALUATION_ENV)


def _resolve_checkpoint_pause_seconds() -> float:
    """Delay after checkpoint writes so subsequent readers observe files."""

    raw = os.getenv("SUPERTOKEN_CHECKPOINT_PAUSE_MS", "100")
    try:
        pause_ms = float(raw)
    except (TypeError, ValueError):
        pause_ms = 100.0
    return max(0.0, pause_ms / 1000.0)


def _format_human_evaluate_summary(summary: Mapping[str, object]) -> str:
    """Render a compact table describing key evaluation metrics."""

    def _fmt(value: object) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.6f}"
        if isinstance(value, (int, str)):
            return str(value)
        return json.dumps(value)

    rows = [
        ("model", str(summary.get("model_type") or "?").upper()),
        ("documents", summary.get("documents")),
        ("tokens", summary.get("tokens")),
        ("tokens/byte", summary.get("tokens_per_byte")),
        ("oov rate", summary.get("oov_rate")),
    ]

    lines = ["[evaluate][summary]"]
    for label, value in rows:
        lines.append(f"  {label:<13}{_fmt(value)}")
    return "\n".join(lines)


def _emit_evaluate_summary(summary: Mapping[str, object], *, fmt: str) -> None:
    fmt = (fmt or "table").strip().lower()
    if fmt == "none":
        return
    if fmt == "json":
        print(f"[evaluate][summary] {json.dumps(summary, sort_keys=True)}")
        return
    if fmt == "table":
        print(_format_human_evaluate_summary(summary))
        return
    raise ValueError(f"Unsupported summary format: {fmt}")


def _load_sequences(
    paths: Iterable[Path],
    bos: int | None,
    eos: int | None,
    *,
    morphology: MorphologyPlugin | None = None,
) -> Iterator[Iterator[int]]:
    """Yield tokenized documents drawn from memory-mapped shard files.

    Args:
        paths: Iterable of shard file paths to open and encode.
        bos: Optional beginning-of-sequence token id to prefix onto each document.
        eos: Optional end-of-sequence token id to suffix onto each document.

    Returns:
        Iterator that lazily yields iterables of integer token ids for each
        document contained in the supplied shards. The outer iterator streams
        shards while the inner iterator walks the token sequence of each
        document.

    Side Effects:
        Opens :class:`MemoryMappedShard` handles via an :class:`ExitStack` so
        shard files stay memory-mapped for the lifetime of the generator.
    """
    packer = BytePacker(bos=bos, eos=eos, morphology=morphology)

    def _generator() -> Iterator[Iterator[int]]:
        with ExitStack() as stack:
            for path in paths:
                shard = stack.enter_context(MemoryMappedShard(path))
                yield packer.encode_shard(shard)

    return _generator()


def _normalize_code_languages(raw: Sequence[str] | None) -> set[str]:
    """Normalise values passed via ``--code-langs``."""

    if not raw:
        return set()
    values: set[str] = set()
    for item in raw:
        if not item:
            continue
        for piece in str(item).split(","):
            norm = piece.strip().lower()
            if norm:
                values.add(norm)
    return values


def _resolve_morphology(
    args: argparse.Namespace,
) -> tuple[MorphologyPlugin | None, dict[str, object]]:
    """Instantiate the requested morphology plugin described by *args*."""

    lang = getattr(args, "morphology_lang", None)
    if not lang:
        return None, {"enabled": False}
    case_markers = bool(getattr(args, "morphology_case_markers", False))
    affix_tags = bool(getattr(args, "morphology_affix_tags", False))
    try:
        plugin = create_morphology_plugin(
            lang,
            case_markers=case_markers,
            affix_tags=affix_tags,
        )
    except KeyError as exc:
        choices = ", ".join(available_morphology_plugins()) or "<none>"
        raise SystemExit(
            f"Unknown morphology language '{lang}'. Available plugins: {choices}"
        ) from exc
    return plugin, {
        "enabled": True,
        "language": lang,
        "case_markers": case_markers,
        "affix_tags": affix_tags,
    }


def _resolve_augmentation(args: argparse.Namespace) -> AugmentationPipeline:
    """Instantiate the requested augmentation pipeline from CLI options."""

    mode_value = getattr(args, "augmentation", "none") or "none"
    strength_raw = getattr(args, "aug_strength", 0.0)
    try:
        strength_value = float(strength_raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"--aug-strength must be a numeric value (got {strength_raw!r})") from exc
    seed_value = getattr(args, "seed", None)
    try:
        pipeline = build_augmentation(mode_value, strength=strength_value, seed=seed_value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return pipeline


def _detect_code_language(path: Path) -> str | None:
    """Infer a language label from *path* when possible."""

    mapping = {
        ".py": "python",
        ".pyi": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
    }
    return mapping.get(path.suffix.lower())


def _coerce_code_entry(
    payload: Mapping[str, object] | object,
    *,
    filename: str,
    language_hint: str | None,
) -> Mapping[str, object] | None:
    """Convert arbitrary JSON entries into :func:`prepare_corpus` payloads."""

    if not isinstance(payload, Mapping):
        return None
    entry = dict(payload)
    entry.setdefault("filename", filename)
    language = entry.get("language")
    if not language and language_hint:
        entry["language"] = language_hint
    source = entry.get("source")
    if not isinstance(source, str):
        return None
    return entry


def _iter_code_entries(
    paths: Iterable[Path], *, languages: set[str] | None
) -> list[Mapping[str, object]]:
    """Load structured code samples from ``paths``."""

    allowed = {lang.lower() for lang in languages} if languages else None
    entries: list[Mapping[str, object]] = []
    for path in paths:
        suffix = path.suffix.lower()
        language_hint = _detect_code_language(path)
        if suffix in {".json", ".jsonl", ".ndjson"}:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    if suffix == ".json":
                        payload = json.load(handle)
                        if isinstance(payload, Sequence):
                            iterator = enumerate(payload)
                        elif isinstance(payload, Mapping):
                            data = payload.get("entries")
                            if not isinstance(data, Sequence):
                                raise TypeError(
                                    "JSON manifests must contain an array or an 'entries' list"
                                )
                            iterator = enumerate(data)
                        else:
                            raise TypeError("JSON manifests must contain an array of entries")
                        for index, obj in iterator:
                            entry = _coerce_code_entry(
                                obj,
                                filename=f"{path.name}#{index}",
                                language_hint=language_hint,
                            )
                            if entry is None:
                                continue
                            language_value = str(entry.get("language", "")).strip().lower()
                            if allowed and language_value and language_value not in allowed:
                                continue
                            if allowed and not language_value and language_hint and language_hint not in allowed:
                                continue
                            entries.append(entry)
                    else:
                        for index, line in enumerate(handle):
                            stripped = line.strip()
                            if not stripped:
                                continue
                            try:
                                payload = json.loads(stripped)
                            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                                raise SystemExit(f"Failed to parse JSON entry in {path}: {exc}") from exc
                            entry = _coerce_code_entry(
                                payload,
                                filename=f"{path.name}#{index}",
                                language_hint=language_hint,
                            )
                            if entry is None:
                                continue
                            language_value = str(entry.get("language", "")).strip().lower()
                            if allowed and language_value and language_value not in allowed:
                                continue
                            if allowed and not language_value and language_hint and language_hint not in allowed:
                                continue
                            entries.append(entry)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse JSON manifest at {path}: {exc}") from exc
        else:
            try:
                source = path.read_text(encoding="utf-8")
            except OSError as exc:  # pragma: no cover - filesystem failure
                raise SystemExit(f"Unable to read source file {path}: {exc}") from exc
            language_value = language_hint or ""
            if allowed and language_value and language_value not in allowed:
                continue
            if allowed and not language_value:
                continue
            entries.append(
                {
                    "language": language_value or language_hint or "",
                    "source": source,
                    "filename": path.name,
                }
            )
    return entries


def _load_code_mode_sequences(
    paths: Iterable[Path],
    *,
    bos: int | None,
    eos: int | None,
    languages: set[str] | None,
    meta_enabled: bool,
    meta_max_length: int = 8,
    morphology: MorphologyPlugin | None = None,
) -> tuple[list[list[int]], dict[str, object]]:
    """Materialise integer token sequences for code-mode corpora."""

    entries = _iter_code_entries(paths, languages=languages)
    if not entries:
        raise SystemExit("--code-mode requires at least one parseable code sample")

    corpus = prepare_corpus(
        entries,
        meta_enabled=meta_enabled,
        meta_max_length=max(1, int(meta_max_length)),
    )

    packer = BytePacker(bos=bos, eos=eos, morphology=morphology)
    sequences: list[list[int]] = []
    ast_samples = 0
    fallback_samples = 0
    languages_seen: set[str] = set()
    total_tokens = 0

    for sample in corpus.samples:
        language = str(sample.metadata.get("language", "")).strip().lower()
        if language:
            languages_seen.add(language)
        if sample.kind == "ast":
            ast_samples += 1
            serialized = "\n".join(map(str, sample.tokens)).encode("utf-8")
        else:
            fallback_samples += 1
            serialized = bytes(int(b) & 0xFF for b in sample.tokens)
        seq = list(packer.encode_sequence(serialized))
        sequences.append(seq)
        total_tokens += len(seq)

    average_len = total_tokens / len(sequences) if sequences else 0.0
    summary = {
        "enabled": True,
        "samples": len(corpus.samples),
        "ast_samples": ast_samples,
        "fallback_samples": fallback_samples,
        "languages": sorted(languages_seen) if languages_seen else None,
        "meta_compress": bool(meta_enabled),
        "meta_tokens": {name: list(pattern) for name, pattern in corpus.meta_tokens.items()},
        "meta_token_count": len(corpus.meta_tokens),
        "meta_max_length": corpus.meta_max_length,
        "average_sequence_length": average_len,
    }
    return sequences, summary


def _expand_data_patterns(patterns: Sequence[str]) -> list[Path]:
    """Resolve input glob patterns into a deterministic list of shard paths.

    Args:
        patterns: Glob expressions pointing at data shards.

    Returns:
        List of concrete file paths that matched at least one pattern. The
        results are ordered by discovery so subsequent iteration is stable.

    Side Effects:
        Touches the filesystem to discover matching shard files.

    Raises:
        SystemExit: If no input files match the provided glob patterns.
    """
    files: list[Path] = []
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            p = Path(path)
            if p.is_file():
                files.append(p)
    if not files:
        raise SystemExit("No input files matched the provided --data globs")
    return files


def _resolve_hf_bpe_import(path: Path) -> export_artifacts.HFBPEImport | None:
    """Attempt to resolve Hugging Face BPE artifacts from *path*."""

    if path.is_dir():
        tokenizer = path / "tokenizer.json"
        if tokenizer.exists():
            return export_artifacts.load_hf_bpe_artifacts(tokenizer_path=tokenizer)
        vocab = path / "vocab.json"
        merges = path / "merges.txt"
        if vocab.exists() and merges.exists():
            return export_artifacts.load_hf_bpe_artifacts(vocab_path=vocab, merges_path=merges)
        return None

    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "tokenizer.json" or (suffix == ".json" and "tokenizer" in name):
        return export_artifacts.load_hf_bpe_artifacts(tokenizer_path=path)
    if name == "vocab.json" or (suffix == ".json" and name.endswith("vocab.json")):
        merges = path.with_name("merges.txt")
        if merges.exists():
            return export_artifacts.load_hf_bpe_artifacts(vocab_path=path, merges_path=merges)
    if name == "merges.txt" or name.endswith("merges.txt"):
        vocab = path.with_name("vocab.json")
        if vocab.exists():
            return export_artifacts.load_hf_bpe_artifacts(vocab_path=vocab, merges_path=path)
    return None


def _resolve_sentencepiece_import(
    path: Path,
) -> export_artifacts.SentencePieceUnigramImport | None:
    """Attempt to resolve SentencePiece unigram artifacts from *path*."""

    if path.is_dir():
        model_candidates = [
            path / "unigram.model",
            path / "sp.model",
            path / "sentencepiece.model",
        ]
        for candidate in model_candidates:
            if candidate.exists():
                return export_artifacts.load_sentencepiece_unigram(model_path=candidate)
        vocab_candidates = [
            path / "unigram.vocab",
            path / "unigram.prob",
            path / "sp.vocab",
        ]
        for candidate in vocab_candidates:
            if candidate.exists():
                return export_artifacts.load_sentencepiece_unigram(vocab_path=candidate)
        return None

    suffix = path.suffix.lower()
    if suffix in {".model", ".spm"}:
        return export_artifacts.load_sentencepiece_unigram(model_path=path)
    if suffix in {".vocab", ".prob"}:
        return export_artifacts.load_sentencepiece_unigram(vocab_path=path)
    return None


def _load_warm_start_merges(source: str | None) -> list[tuple[int, int]] | None:
    """Load warm-start merges from a JSON manifest when provided."""

    if not source:
        return None

    path = Path(source)
    if not path.exists():
        raise SystemExit(f"Warm-start plan not found at {path}")

    try:
        hf_import = _resolve_hf_bpe_import(path)
    except ValueError as exc:
        raise SystemExit(f"Failed to import Hugging Face merges from {path}: {exc}") from exc
    if hf_import is not None:
        return hf_import.merges

    if path.is_dir():
        candidates = [
            path / "merges.tiktoken",
            path / "tokenizer.tiktoken",
            path / "bpe_merges.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return _load_warm_start_merges(str(candidate))
        raise SystemExit(
            "Warm-start directories must contain merges.tiktoken or bpe_merges.json"
        )

    if path.suffix.lower() == ".tiktoken":
        try:
            return export_artifacts.load_tiktoken_merges(path)
        except Exception as exc:
            raise SystemExit(f"Failed to parse TikToken merges from {path}: {exc}") from exc

    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise SystemExit(
            "Warm-start plans must be JSON manifests or TikToken bundles containing a 'merges' list"
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    merges_raw: object | None = None
    if isinstance(payload, Mapping):
        merges_raw = payload.get("merges")
        if merges_raw is None:
            trainer_section = payload.get("trainer")
            if isinstance(trainer_section, Mapping):
                merges_raw = trainer_section.get("merges")
    elif isinstance(payload, Sequence):
        merges_raw = payload

    if not isinstance(merges_raw, Sequence):
        raise SystemExit(
            f"Warm-start manifest {path} does not contain a 'merges' sequence"
        )

    merges: list[tuple[int, int]] = []
    for entry in merges_raw:
        if not isinstance(entry, Sequence) or len(entry) != 2:
            raise SystemExit(
                "Warm-start merges must be sequences of two integer token ids"
            )
        left, right = entry
        try:
            merges.append((int(left), int(right)))
        except (TypeError, ValueError) as exc:
            raise SystemExit("Warm-start merge entries must be coercible to integers") from exc
    return merges


def _iter_packed_batches(
    sequences: Iterable[Iterable[int]],
    batch_size: int,
    seed: int,
    *,
    augmentation: AugmentationPipeline | None = None,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Construct a streaming batch iterator backed by :class:`PackedBatcher`.

    Args:
        sequences: Tokenized documents to be batched.
        batch_size: Maximum number of documents per batch.
        seed: Shuffle seed forwarded to :class:`PackedBatcher` for determinism.

    Returns:
        Iterable that yields ``(tokens, mask, lengths)`` tensors suitable for
        GPU consumption. Each iteration produces a packed token matrix, a mask
        indicating valid positions, and per-document lengths.

    Side Effects:
        None.
    """
    return PackedBatcher(
        sequences,
        batch_size=batch_size,
        seed=seed,
        augmentation=augmentation,
    )


def _build_unigram_batches(
    sequences: Iterable[Iterable[int]],
    batch_size: int,
    seed: int,
    *,
    augmentation: AugmentationPipeline | None = None,
) -> list[torch.Tensor]:
    """Materialize packed token batches for the unigram trainer.

    Args:
        sequences: Tokenized documents to feed into the unigram objective.
        batch_size: Number of documents per packed batch.
        seed: Shuffle seed used to stabilize batch ordering.

    Returns:
        List containing the token payload tensor for every packed batch. Masks
        and length tensors are intentionally dropped because the unigram
        objective only consumes token ids.

    Side Effects:
        Loads the entire packed representation into host memory so batches can
        be replayed across epochs without rebuilding.
    """
    packed = PackedBatcher(
        sequences,
        batch_size=batch_size,
        seed=seed,
        augmentation=augmentation,
    )
    return [x for (x, _mask, _lengths) in packed]


def _stringify_config_value(value: object) -> object:
    """Convert config payloads into JSON-serialisable structures."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _stringify_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_config_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_stringify_config_value(v) for v in value)
    return value


def _log_resolved_config(command: str, config: dict[str, object]) -> None:
    """Render structured configuration dictionaries prior to training."""

    payload = _stringify_config_value(config)
    try:
        rendered = json.dumps(payload, indent=2, sort_keys=True)
    except TypeError:
        rendered = json.dumps(payload, indent=2, sort_keys=True, default=str)  # pragma: no cover - defensive fallback
    print(f"[config][{command}] {rendered}")


def _build_bpe_trainer(
    args: argparse.Namespace,
) -> tuple[BaseTrainer, AutoScaler, int, dict[str, object]]:
    """Factory that materialises a :class:`GPUBPETrainer` from CLI args."""

    autoscaler = AutoScaler(
        target_util=args.target_util,
        min_bs=args.min_batch,
        max_bs=args.max_batch,
    )
    suggestion_state, suggestion_window = autoscaler.suggest(
        token_bytes_per_example=args.token_bytes
    )
    resolved_batch = min(args.max_batch, max(args.min_batch, suggestion_state.batch_size))
    privacy_label = str(getattr(args, "privacy", "none") or "none").lower()
    privacy_enabled = privacy_label != "none"
    randomize_ties = privacy_label == "tie-randomize"
    tie_seed = getattr(args, "tie_seed", None)
    privacy_salt = getattr(args, "privacy_salt", None)
    trainer = GPUBPETrainer(
        base_vocab=args.base_vocab,
        merges=args.merges,
        device=args.device,
        autoscaler=autoscaler,
        privacy_mode=privacy_enabled,
        randomize_ties=randomize_ties,
        tie_seed=tie_seed,
        privacy_salt=privacy_salt,
    )
    suggestion = asdict(suggestion_state)
    suggestion["requested_batch_size"] = suggestion_state.batch_size
    suggestion["resolved_batch_size"] = resolved_batch
    suggestion["window_snapshot"] = suggestion_window
    privacy_config = {
        "mode": (
            "tie-randomize"
            if randomize_ties
            else ("hash-merges" if privacy_enabled else "none")
        ),
        "tie_seed": int(tie_seed) if tie_seed is not None else None,
        "salt_configured": bool(privacy_salt),
    }
    config: dict[str, object] = {
        "trainer": {
            "base_vocab": args.base_vocab,
            "merges": args.merges,
            "device": args.device,
            "privacy": privacy_config,
        },
        "autoscaler": {
            "target_util": args.target_util,
            "min_batch": args.min_batch,
            "max_batch": args.max_batch,
            "token_bytes_per_example": args.token_bytes,
            "suggestion": suggestion,
        },
    }
    return trainer, autoscaler, resolved_batch, config


def _build_unigram_trainer(args: argparse.Namespace) -> tuple[BaseTrainer, dict[str, object]]:
    """Factory that materialises a :class:`GPUUnigramTrainer` from CLI args."""

    trainer = GPUUnigramTrainer(
        base_vocab=args.base_vocab,
        vocab_size=args.vocab_size,
        max_subword_len=args.max_subword_len,
        device=args.device,
    )
    config: dict[str, object] = {
        "trainer": {
            "base_vocab": args.base_vocab,
            "vocab_size": args.vocab_size,
            "max_subword_len": args.max_subword_len,
            "device": args.device,
        },
        "runtime": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
    }
    return trainer, config


def _cmd_benchmark(args: argparse.Namespace) -> None:
    """Run synthetic and real-corpus benchmarks and emit a serialized report.

    Args:
        args: Parsed CLI arguments configuring synthetic data, datasets, and
            trainer hyper-parameters.

    Returns:
        ``None``. The function prints progress summaries and writes a structured
        report for downstream inspection.

    Side Effects:
        Generates synthetic corpora when requested, reads optional datasets,
        writes benchmark summaries under ``args.output_dir``, and prints a
        human-readable summary to stdout.

    Raises:
        SystemExit: If no synthetic or real corpora are supplied for the run.
    """
    sequences: list[list[int]] = []
    sources: list[dict[str, object]] = []
    morphology_plugin, morphology_config = _resolve_morphology(args)
    augmentation_pipeline = _resolve_augmentation(args)
    warm_start_summary: dict[str, object] | None = None
    warm_start_source = getattr(args, "warm_start", None)
    if warm_start_source:
        warm_path = Path(warm_start_source)
        if not warm_path.exists():
            raise SystemExit(f"Warm-start model not found at {warm_path}")
        try:
            sp_bundle = _resolve_sentencepiece_import(warm_path)
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(
                f"Failed to import SentencePiece model from {warm_start_source}: {exc}"
            ) from exc
        if sp_bundle is None:
            raise SystemExit(
                "--warm-start must reference a SentencePiece .model or .vocab/.prob file"
            )
        trainer.import_sentencepiece(pieces=sp_bundle.pieces, scores=sp_bundle.scores)
        warm_start_summary = {
            "source": warm_start_source,
            "pieces": len(sp_bundle.pieces),
        }
        if sp_bundle.metadata:
            warm_start_summary["metadata"] = sp_bundle.metadata
        source_label = sp_bundle.sources[0] if sp_bundle.sources else warm_start_source
        print(
            f"[warm-start] Imported {len(sp_bundle.pieces)} unigram pieces from {source_label}"
        )
    augmentation_pipeline = _resolve_augmentation(args)

    if args.synthetic_docs > 0:
        synthetic = benchmark_runner.synthesize_corpus(
            documents=args.synthetic_docs,
            min_length=args.synthetic_min_len,
            max_length=args.synthetic_max_len,
            vocab_size=args.synthetic_vocab,
            seed=args.seed,
        )
        sequences.extend(synthetic)
        sources.append(
            {
                "type": "synthetic",
                "documents": args.synthetic_docs,
                "min_length": args.synthetic_min_len,
                "max_length": args.synthetic_max_len,
                "vocab_size": args.synthetic_vocab,
            }
        )

    real_paths: list[Path] = []
    if args.data:
        real_paths = _expand_data_patterns(args.data)
        real_sequences = benchmark_runner.load_real_corpus(
            real_paths,
            bos=args.bos,
            eos=args.eos,
            limit=args.max_real_docs,
            morphology=morphology_plugin,
        )
        if real_sequences:
            sequences.extend(real_sequences)
            sources.append(
                {
                    "type": "dataset",
                    "paths": [str(p) for p in real_paths],
                    "limit": args.max_real_docs,
                }
            )

    if not sequences:
        raise SystemExit(
            "Benchmark requires at least one corpus source via --synthetic-docs or --data"
        )

    corpus = benchmark_runner.summarize_corpus(sequences, sources=sources)
    # The synthetic/real corpus mixtures above feed both trainers, allowing
    # benchmarking code to compare how each algorithm responds to identical
    # token streams.
    bpe_suite: dict[str, object] | None = None
    if getattr(args, "bpe_config", None):
        config_path = Path(args.bpe_config)
        run_specs = benchmark_runner.load_bpe_run_config(config_path)
        if not run_specs:
            raise SystemExit(f"No runnable BPE entries found in {config_path}")
        bpe_suite = benchmark_runner.run_bpe_suite(
            sequences,
            base_vocab=args.bpe_base_vocab,
            merges=args.bpe_merges,
            seed=args.seed,
            log_every=args.bpe_log_every,
            run_configs=run_specs,
        )
        runs = bpe_suite.get("runs", []) if isinstance(bpe_suite, dict) else []
        if not runs:
            raise SystemExit(f"BPE suite {config_path} did not produce any runs")
        bpe = runs[0]
    else:
        bpe = benchmark_runner.run_bpe_benchmark(
            sequences,
            base_vocab=args.bpe_base_vocab,
            merges=args.bpe_merges,
            batch_size=args.bpe_batch_size,
            device=args.device,
            seed=args.seed,
            log_every=args.bpe_log_every,
        )
    unigram = benchmark_runner.run_unigram_benchmark(
        sequences,
        base_vocab=args.unigram_base_vocab,
        vocab_size=args.unigram_vocab,
        max_subword_len=args.unigram_max_subword,
        batch_size=args.unigram_batch_size,
        epochs=args.unigram_epochs,
        device=args.device,
        seed=args.seed,
    )
    baseline_results: list[dict[str, object]] | None = None
    baseline_config: dict[str, object] | None = None
    requested_baselines = [
        str(name) for name in getattr(args, "baseline_corpus", []) or []
    ]
    if requested_baselines:
        sp_model = getattr(args, "sentencepiece_model", None)
        hf_tokenizer = getattr(args, "huggingface_tokenizer", None)
        if not sp_model and not hf_tokenizer:
            raise SystemExit(
                "--baseline-corpus requires --sentencepiece-model and/or --huggingface-tokenizer"
            )
        try:
            corpora = benchmark_runner.resolve_baseline_corpora(requested_baselines)
        except KeyError as exc:
            available = ", ".join(benchmark_runner.available_baseline_corpora())
            raise SystemExit(
                f"Unknown baseline corpus '{exc.args[0]}'. Available corpora: {available}"
            ) from exc
        try:
            baseline_results = benchmark_runner.run_reference_tokenizers(
                corpora,
                sentencepiece_model=Path(sp_model).expanduser() if sp_model else None,
                huggingface_tokenizer=Path(hf_tokenizer).expanduser()
                if hf_tokenizer
                else None,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        baseline_config = {
            "corpora": [corpus.name for corpus in corpora],
            "sentencepiece_model": sp_model,
            "huggingface_tokenizer": hf_tokenizer,
        }
    print(
        benchmark_runner.emit_benchmark_summary(
            corpus,
            bpe,
            unigram,
            bpe_suite,
            baseline_results,
        )
    )
    # Persist full benchmark metadata so checkpointing infrastructure can re-use
    # the exact same corpora, hyper-parameters, and timing metrics in later
    # automation runs.
    evaluation_report: dict[str, object] | None = None
    evaluation_path_raw = getattr(args, "evaluation_report", None)
    if evaluation_path_raw:
        evaluation_path = Path(evaluation_path_raw)
        if not evaluation_path.exists():
            raise SystemExit(f"Evaluation report not found: {evaluation_path}")
        try:
            evaluation_report = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Failed to parse evaluation report {evaluation_path}: {exc}"
            ) from exc

    config_payload = {
        "seed": args.seed,
        "device": args.device,
        "synthetic": {
            "documents": args.synthetic_docs,
            "min_length": args.synthetic_min_len,
            "max_length": args.synthetic_max_len,
            "vocab_size": args.synthetic_vocab,
        }
        if args.synthetic_docs
        else None,
        "data": [str(p) for p in real_paths],
        "max_real_docs": args.max_real_docs,
        "bpe": bpe["config"],
        "unigram": unigram["config"],
        "morphology": morphology_config,
    }
    if baseline_config is not None:
        config_payload["baselines"] = baseline_config
    output_path = benchmark_runner.serialize_run(
        Path(args.output_dir),
        corpus=corpus,
        config=config_payload,
        bpe=bpe,
        unigram=unigram,
        bpe_runs=bpe_suite,
        evaluation=evaluation_report,
        baseline_tokenizers=baseline_results,
    )
    print(f"Saved benchmark metadata → {output_path}")


def _cmd_export_embeddings(args: argparse.Namespace) -> None:
    """Export embedding artifacts derived from a tokenizer vocabulary."""

    vocab = export_artifacts.load_vocab(args.vocab)
    stats = export_artifacts.load_token_stats(args.stats) if args.stats else {}
    dedupe = export_artifacts.dedupe_vocabulary(
        vocab,
        stats,
        similarity_threshold=getattr(args, "dedupe_similarity", 0.0),
        dimension=args.dimension,
        seed=args.seed,
        keep_tokens=args.keep_token,
    )
    vocab = dedupe.vocab
    stats = dedupe.stats
    prune = export_artifacts.prune_vocabulary(
        vocab,
        stats,
        min_frequency=args.min_frequency,
        keep_tokens=args.keep_token,
        original_size=dedupe.original_size,
    )
    combined_pruned = [*dedupe.deduped, *prune.pruned]
    prune = export_artifacts.PruneResult(
        vocab=prune.vocab,
        pruned=combined_pruned,
        original_size=dedupe.original_size,
    )
    dtype = export_artifacts.resolve_dtype(args.dtype)
    filtered_stats = {token: stats[token] for token in prune.vocab if token in stats}
    embeddings = export_artifacts.generate_embedding_matrix(
        prune.vocab,
        filtered_stats,
        dimension=args.dimension,
        seed=args.seed,
        dtype=dtype,
    )
    manifest = export_artifacts.build_manifest(
        dimension=args.dimension,
        dtype=dtype,
        seed=args.seed,
        prune=prune,
        min_frequency=args.min_frequency,
        preserved_tokens=args.keep_token or [],
    )
    paths = export_artifacts.write_export_package(
        args.output_dir,
        embeddings=embeddings,
        vocab=prune.vocab,
        manifest=manifest,
        pruned=prune.pruned,
    )
    deduped_count = len(dedupe.deduped)
    pruned_only_count = max(len(prune.pruned) - deduped_count, 0)
    summary = {
        "dimension": manifest.dimension,
        "dtype": manifest.dtype,
        "deduped_tokens": deduped_count,
        "exported_tokens": manifest.exported_token_count,
        "pruned_tokens": pruned_only_count,
        "pruning_log_entries": len(prune.pruned),
        "output_dir": str(args.output_dir),
    }
    print(f"[export][export-embeddings] {json.dumps(summary, sort_keys=True)}")
    print(
        "Exported embeddings → "
        f"{paths['embeddings']} ({manifest.exported_token_count} tokens)"
    )


def _cmd_evaluate(args: argparse.Namespace) -> None:
    if _should_skip_evaluation(force=bool(getattr(args, "force_evaluation", False))):
        print(
            f"[evaluate] Skipping evaluation because {SKIP_EVALUATION_ENV} is set",
            flush=True,
        )
        return

    data_patterns = getattr(args, "data", None)
    if not data_patterns:
        raise SystemExit(
            "[evaluate] provide at least one --data glob pattern describing the corpus"
        )
    data_files = _expand_data_patterns(data_patterns)

    artifacts_dir = (
        Path(getattr(args, "artifacts"))
        if getattr(args, "artifacts", None)
        else None
    )
    vocab_path = Path(args.vocab).expanduser() if getattr(args, "vocab", None) else None
    merges_path = (
        Path(args.merges).expanduser() if getattr(args, "merges", None) else None
    )
    tokenizer_path = (
        Path(args.tokenizer).expanduser()
        if getattr(args, "tokenizer", None)
        else None
    )

    if artifacts_dir and not artifacts_dir.exists():
        raise SystemExit(
            f"[evaluate] artifact directory not found: {artifacts_dir}."
            " Check the --artifacts path or point --vocab/--merges directly at"
            " the exported files."
        )

    if vocab_path is None and artifacts_dir is not None:
        for candidate_name in ("vocab.json", "unigram.vocab", "sentencepiece.vocab"):
            candidate = (artifacts_dir / candidate_name).expanduser()
            if candidate.exists():
                vocab_path = candidate
                break
    if vocab_path is None:
        raise SystemExit(
            "[evaluate] supply --artifacts or --vocab pointing at vocab.json or"
            " unigram.vocab"
        )
    if not vocab_path.exists():
        raise SystemExit(
            f"[evaluate] vocabulary file not found: {vocab_path}."
            " Double-check the export path or adjust --vocab."
        )

    if merges_path is None and artifacts_dir is not None:
        for name in ("merges.json", "merges.txt", "bpe_merges.json"):
            candidate = (artifacts_dir / name).expanduser()
            if candidate.exists():
                merges_path = candidate
                break
    if merges_path is not None and not merges_path.exists():
        raise SystemExit(
            f"[evaluate] merge history not found: {merges_path}."
            " Provide a valid --merges path or remove the flag when"
            " evaluating unigram exports."
        )

    if tokenizer_path is None and artifacts_dir is not None:
        candidate = (artifacts_dir / "tokenizer.json").expanduser()
        if candidate.exists():
            tokenizer_path = candidate

    try:
        resolved_model_type = evaluate_module.resolve_model_type(
            vocab_path,
            merges_path,
            getattr(args, "model_type", None),
        )
    except ValueError as exc:
        raise SystemExit(f"[evaluate] {exc}") from exc

    code_langs = _normalize_code_languages(getattr(args, "code_langs", None))
    morphology_plugin, morphology_config = _resolve_morphology(args)
    augmentation_pipeline = _resolve_augmentation(args)

    evaluate_options = evaluate_module.EvaluateCLIOptions(
        data_files=data_files,
        vocab_path=vocab_path,
        merges_path=merges_path,
        tokenizer_path=tokenizer_path,
        model_type=resolved_model_type,
        bos=args.bos,
        eos=args.eos,
        morphology=morphology_plugin,
        morphology_config=morphology_config,
        code_mode=bool(getattr(args, "code_mode", False)),
        code_languages=sorted(code_langs) if code_langs else None,
        meta_compress=bool(getattr(args, "meta_compress", False)),
        meta_max_length=getattr(args, "meta_max_length", 8),
        deterministic=bool(getattr(args, "deterministic", False)),
    )
    try:
        result = evaluate_module.evaluate_cli(evaluate_options)
    except EvaluateReportValidationError as exc:
        raise SystemExit(f"[evaluate] generated report failed schema validation: {exc}") from exc
    report = result.report

    try:
        serialized_report = serialize_evaluate_report(report)
    except EvaluateReportValidationError as exc:
        raise SystemExit(f"[evaluate] generated report failed schema validation: {exc}") from exc

    output_arg = getattr(args, "output", None)
    destination: Path | None
    if output_arg is None:
        destination = DEFAULT_EVALUATE_OUTPUT
    elif str(output_arg).strip() == "-":
        destination = None
    else:
        destination = Path(output_arg)

    if destination is not None:
        destination = destination.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            handle.write(serialized_report)
            handle.write("\n")
        print(f"[evaluate] wrote report → {destination}")
    else:
        print(serialized_report)

    summary = dict(result.summary)
    summary.setdefault("model_type", resolved_model_type)
    _emit_evaluate_summary(summary, fmt=getattr(args, "summary_format", "table"))


def _cmd_train_bpe(args: argparse.Namespace) -> None:
    """Train a GPU-accelerated BPE model with autoscaling batch management.

    Args:
        args: Parsed CLI arguments describing data sources, autoscaler targets,
            checkpointing configuration, and trainer hyper-parameters.

    Returns:
        ``None``. Progress and metadata are surfaced via stdout and optional
        checkpoint directories while the trained model may be exported.

    Side Effects:
        Uses :class:`AutoScaler` suggestions to resize the active batch size,
        starts a :class:`CorpusStreamer` that must be closed when training
        completes, and loads/saves checkpoints when ``--resume-from`` or
        ``--checkpoint-dir`` are provided.

    Raises:
        SystemExit: If no ``--data`` globs are supplied or none yield readable
            shard files.
    """

    class _StaticDatasetTracker:
        def __init__(self, state: Mapping[str, object]) -> None:
            self._state: dict[str, object] = dict(state)

        def stream_state_dict(self) -> dict[str, object]:
            return dict(self._state)

        def load_stream_state(self, state: Mapping[str, object]) -> None:
            if isinstance(state, Mapping):
                self._state = dict(state)

    data_patterns = getattr(args, "data", None)
    if not data_patterns:
        raise SystemExit("train-bpe requires at least one --data glob pattern")
    data_files = _expand_data_patterns(data_patterns)
    trainer, autoscaler, batch_size, trainer_config = _build_bpe_trainer(args)
    code_langs = _normalize_code_languages(getattr(args, "code_langs", None))
    code_mode_active = bool(getattr(args, "code_mode", False))
    code_mode_config: dict[str, object] = {
        "enabled": code_mode_active,
        "meta_compress": bool(getattr(args, "meta_compress", False)),
    }
    if code_langs:
        code_mode_config["languages"] = sorted(code_langs)
    morphology_plugin, morphology_config = _resolve_morphology(args)
    augmentation_pipeline = _resolve_augmentation(args)
    checkpoint_pause_s = _resolve_checkpoint_pause_seconds()
    tie_seed = getattr(args, "tie_seed", None)
    config = dict(trainer_config)
    config.update(
        {
            "data": {
                "patterns": list(data_patterns),
                "files": [str(path) for path in data_files],
                "bos": args.bos,
                "eos": args.eos,
                "compression": args.compression,
                "io_workers": args.io_workers,
                "prefetch_batches": args.prefetch_batches,
                "augmentation": augmentation_pipeline.summary(),
            },
            "checkpointing": {
                "checkpoint_dir": args.checkpoint_dir,
                "checkpoint_every": args.checkpoint_every,
                "resume_from": str(args.resume_from) if args.resume_from else None,
                "out_dir": args.out_dir,
            },
            "dry_run": bool(getattr(args, "dry_run", False)),
            "code_mode": code_mode_config,
        }
    )
    config["trainer"].setdefault("privacy", {}).update(
        {
            "mode": str(getattr(args, "privacy", "none") or "none").lower(),
            "tie_seed": int(tie_seed) if tie_seed is not None else None,
            "salt_configured": bool(getattr(args, "privacy_salt", None)),
        }
    )
    config["morphology"] = morphology_config
    _log_resolved_config("train-bpe", config)
    if getattr(args, "dry_run", False):
        print("[dry-run] train-bpe initialization complete")
        return
    code_mode_summary: dict[str, object] | None = None
    resume_state: dict[str, object] | None = None
    streamer: CorpusStreamer | None
    dataset_tracker: StreamingPackedBatcher | None
    batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    packer: BytePacker | None = None

    if code_mode_active:
        if args.resume_from:
            raise SystemExit("--resume-from is not supported when --code-mode is enabled")
        sequences, code_mode_summary = _load_code_mode_sequences(
            data_files,
            bos=args.bos,
            eos=args.eos,
            languages=code_langs if code_langs else None,
            meta_enabled=bool(getattr(args, "meta_compress", False)),
            morphology=morphology_plugin,
        )
        batches = PackedBatcher(
            sequences,
            batch_size=batch_size,
            seed=args.seed,
            augmentation=augmentation_pipeline.fork(),
        )
        streamer = None
        dataset_tracker = None
    else:
        packer = BytePacker(bos=args.bos, eos=args.eos, morphology=morphology_plugin)

        def _build_serialized_batches(
            serialized: dict[str, object],
            default_bs: int,
        ) -> tuple[int, Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None]:
            sequences_raw = serialized.get("sequences", [])
            sequences: list[list[int]] = []
            if isinstance(sequences_raw, list):
                for seq in sequences_raw:
                    if isinstance(seq, list):
                        sequences.append([int(token) for token in seq])
            if not sequences:
                return 0, None
            resume_bs = int(serialized.get("active_batch_size") or 0)
            if resume_bs <= 0:
                resume_bs = default_bs

            class _SerializedIterable:
                def __iter__(self) -> Iterator[
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ]:
                    pin_memory = torch.cuda.is_available()
                    storage_width = 1
                    length_dtype = length_storage_dtype(storage_width)
                    tokens = torch.full(
                        (resume_bs, storage_width),
                        -1,
                        dtype=torch.int32,
                        pin_memory=pin_memory,
                    )
                    valid = torch.zeros(
                        (resume_bs, storage_width),
                        dtype=torch.uint8,
                        pin_memory=pin_memory,
                    )
                    lengths = torch.zeros(
                        (resume_bs,),
                        dtype=length_dtype,
                        pin_memory=pin_memory,
                    )

                    for start in range(0, len(sequences), resume_bs):
                        chunk = sequences[start : start + resume_bs]
                        if not chunk:
                            continue
                        max_len = max((len(seq) for seq in chunk), default=0)
                        width = max(1, max_len)
                        if width > storage_width:
                            storage_width = width
                            length_dtype = length_storage_dtype(storage_width)
                            tokens = torch.full(
                                (resume_bs, storage_width),
                                -1,
                                dtype=torch.int32,
                                pin_memory=pin_memory,
                            )
                            valid = torch.zeros(
                                (resume_bs, storage_width),
                                dtype=torch.uint8,
                                pin_memory=pin_memory,
                            )
                            lengths = torch.zeros(
                                (resume_bs,),
                                dtype=length_dtype,
                                pin_memory=pin_memory,
                            )
                        count = len(chunk)
                        tokens[:count].fill_(-1)
                        valid[:count].zero_()
                        lengths[:count].zero_()
                        for row, seq in enumerate(chunk):
                            L = len(seq)
                            if L == 0:
                                continue
                            lengths[row] = L
                            vals = torch.as_tensor(seq, dtype=torch.int32)
                            tokens[row, :L] = vals
                            valid[row, :L] = 1
                        yield tokens[:count, :width], valid[:count, :width], lengths[:count]

            return resume_bs, _SerializedIterable()

        def _build_streamer(
            *, restore_offsets: Mapping[str, object] | None = None
        ) -> CorpusStreamer:
            streamer = CorpusStreamer(
                data_files,
                compression=args.compression,
                num_workers=args.io_workers,
                max_prefetch=args.prefetch_batches,
                autoscaler=autoscaler,
                prefetch_jitter=max(0.0, float(getattr(args, "prefetch_jitter", 0.0))),
            )
            if restore_offsets:
                streamer.restore_offsets(restore_offsets)
            streamer.start()
            return streamer

        resume_batches: Iterable[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] | None = None
        resume_stream_state: dict[str, object] | None = None
        if args.resume_from:
            resume_path = Path(args.resume_from)
            try:
                resume_state = trainer.load_checkpoint(str(resume_path))
            except FileNotFoundError:
                try:
                    hf_bundle = _resolve_hf_bpe_import(resume_path)
                except ValueError as exc:
                    raise SystemExit(
                        f"Failed to import tokenizer artifacts from {args.resume_from}: {exc}"
                    ) from exc
                if hf_bundle is None:
                    raise SystemExit(
                        "--resume-from must point to a checkpoint directory or a Hugging Face tokenizer bundle"
                    )
                vocab_ids = list(hf_bundle.vocab.values())
                vocab_size = (
                    max(vocab_ids) + 1
                    if vocab_ids
                    else trainer.base_vocab + len(hf_bundle.merges)
                )
                base_vocab = (
                    hf_bundle.base_vocab if hf_bundle.base_vocab > 0 else trainer.base_vocab
                )
                trainer.import_merges(
                    hf_bundle.merges,
                    vocab_size=vocab_size,
                    base_vocab=base_vocab,
                    source="huggingface",
                )
                source_label = hf_bundle.sources[0] if hf_bundle.sources else args.resume_from
                print(
                    f"[warm-start] Imported {len(hf_bundle.merges)} merges from {source_label}"
                )
                resume_state = None
            if resume_state is not None:
                payload_mapping = resume_state.get("payload")
                if isinstance(payload_mapping, Mapping):
                    checkpoint_payload = CheckpointPayload.from_mapping(payload_mapping)
                else:
                    legacy_meta = resume_state.get("metadata")
                    if isinstance(legacy_meta, Mapping):
                        checkpoint_payload = CheckpointPayload.from_legacy_metadata(
                            legacy_meta
                        )
                    else:
                        checkpoint_payload = CheckpointPayload()

                model_section = checkpoint_payload.trainer.get("model")
                if not isinstance(model_section, Mapping):
                    model_section = checkpoint_payload.trainer
                merge_step = model_section.get("merge_step")
                print(
                    f"[checkpoint] Restored checkpoint from {args.resume_from}"
                    + (f" at merge {merge_step}" if merge_step is not None else "")
                )

                if (
                    checkpoint_payload.version
                    != CheckpointPayload.CURRENT_VERSION
                ):
                    print(
                        "[checkpoint] Warning: checkpoint schema version "
                        f"{checkpoint_payload.version} differs from expected "
                        f"{CheckpointPayload.CURRENT_VERSION}",
                        file=sys.stderr,
                    )

                config_errors: list[str] = []
                config_warnings: list[str] = []
                target_merges = model_section.get("target_merges")
                try:
                    target_merges_int = (
                        int(target_merges) if target_merges is not None else None
                    )
                except (TypeError, ValueError):
                    target_merges_int = None
                if target_merges_int is None:
                    merges_list = model_section.get("merges")
                    if isinstance(merges_list, list):
                        target_merges_int = len(merges_list)
                if (
                    target_merges_int is not None
                    and target_merges_int != int(args.merges)
                ):
                    config_errors.append(
                        "CLI --merges does not match checkpoint target_merges"
                    )
                base_vocab_val = model_section.get("base_vocab")
                try:
                    base_vocab_int = (
                        int(base_vocab_val) if base_vocab_val is not None else None
                    )
                except (TypeError, ValueError):
                    base_vocab_int = None
                if base_vocab_int is not None and base_vocab_int != int(args.base_vocab):
                    config_warnings.append(
                        "CLI --base-vocab does not match checkpoint base_vocab"
                    )
                for warning in config_warnings:
                    print(f"[checkpoint] Warning: {warning}", file=sys.stderr)
                if config_errors:
                    for error in config_errors:
                        print(f"[checkpoint] Error: {error}", file=sys.stderr)
                    raise SystemExit("Checkpoint configuration mismatch")

                dataset_meta = checkpoint_payload.dataset
                serialized_batches = (
                    dataset_meta.get("batches")
                    if isinstance(dataset_meta, Mapping)
                    else None
                )
                if serialized_batches is None:
                    serialized_batches = checkpoint_payload.trainer.get("batches")
                if isinstance(serialized_batches, dict):
                    restored_bs, restored_iter = _build_serialized_batches(
                        serialized_batches,
                        batch_size,
                    )
                    if restored_bs > 0:
                        batch_size = restored_bs
                    resume_batches = restored_iter
                if isinstance(dataset_meta, Mapping):
                    stream_offsets = dataset_meta.get("stream_offsets")
                    if isinstance(stream_offsets, Mapping) and stream_offsets:
                        resume_stream_state = {"stream_offsets": dict(stream_offsets)}
                    stream_batch_size = dataset_meta.get("stream_batch_size")
                    try:
                        stream_bs_val = (
                            int(stream_batch_size)
                            if stream_batch_size is not None
                            else None
                        )
                    except (TypeError, ValueError):
                        stream_bs_val = None
                    if stream_bs_val and stream_bs_val > 0:
                        if resume_stream_state is None:
                            resume_stream_state = {}
                        resume_stream_state["batch_size"] = stream_bs_val
        
        streamer = None
        dataset_tracker = None
        restored_offset_count = 0
        if resume_stream_state and isinstance(resume_stream_state.get("stream_offsets"), Mapping):
            restored_offset_count = len(resume_stream_state["stream_offsets"])
        if resume_batches is None:
            restore_offsets = None
            if resume_stream_state and isinstance(
                resume_stream_state.get("stream_offsets"), Mapping
            ):
                restore_offsets = resume_stream_state.get("stream_offsets")
            streamer = _build_streamer(restore_offsets=restore_offsets)
            batches = StreamingPackedBatcher(
                streamer,
                packer.encode_view,
                batch_size=batch_size,
                augmentation=augmentation_pipeline.fork(),
            )
            dataset_tracker = batches if isinstance(batches, StreamingPackedBatcher) else None
            if dataset_tracker and resume_stream_state:
                dataset_tracker.load_stream_state(resume_stream_state)
                restored_offsets = resume_stream_state.get("stream_offsets")
                if isinstance(restored_offsets, Mapping) and restored_offsets:
                    print(
                        f"[checkpoint] Restored dataset cursor for {len(restored_offsets)} shard(s)"
                    )
        else:
            batches = resume_batches
            dataset_tracker = None
            if resume_stream_state:
                dataset_tracker = _StaticDatasetTracker(resume_stream_state)
            if restored_offset_count > 0:
                print(
                    f"[checkpoint] Restored dataset cursor for {restored_offset_count} shard(s)"
                )

    current_batch_size = batch_size

    autoscaler_windows: list[dict[str, object]] = []
    code_mode_resize_warned = False

    def _capture_autoscaler_window(summary: dict[str, object]) -> None:
        window = summary.get("autoscaler") if isinstance(summary, dict) else None
        if isinstance(window, dict):
            autoscaler_windows.append(dict(window))

    def _handle_batch_resize(new_bs: int) -> None:
        nonlocal batches, current_batch_size, streamer, dataset_tracker, code_mode_resize_warned
        if new_bs <= 0 or new_bs == current_batch_size:
            return
        if code_mode_active:
            if not code_mode_resize_warned:
                print(
                    "[code-mode] Autoscaler resize ignored; batches are pre-packed",
                    file=sys.stderr,
                )
                code_mode_resize_warned = True
            return
        current_batch_size = new_bs
        if streamer is None or packer is None:
            return
        # Autoscaler triggered a resize: tear down the current streamer so it
        # restarts with the new batch size and refreshed packing layout.
        streamer.close()
        streamer = _build_streamer()
        batches = StreamingPackedBatcher(
            streamer,
            packer.encode_view,
            batch_size=new_bs,
        )
        dataset_tracker = batches

    try:
        if args.checkpoint_dir:
            original_save_checkpoint = trainer.save_checkpoint

            def _save_checkpoint_with_log(
                self, path: str, *cargs, _pause_s: float = checkpoint_pause_s, **ckwargs
            ):
                state = original_save_checkpoint(path, *cargs, **ckwargs)
                print(f"[checkpoint] Saved checkpoint → {path}")
                if _pause_s > 0:
                    time.sleep(_pause_s)
                return state

            trainer.save_checkpoint = types.MethodType(_save_checkpoint_with_log, trainer)
        meta = trainer.fit(
            batches,
            log_every=args.log_every,
            on_batch_size_change=_handle_batch_resize,
            on_iteration_summary=_capture_autoscaler_window,
            checkpoint_interval=(
                args.checkpoint_every if args.checkpoint_every and args.checkpoint_every > 0 else None
            ),
            checkpoint_dir=args.checkpoint_dir,
            resume_state=resume_state,
            dataset_state=dataset_tracker,
        )
    finally:
        if streamer is not None:
            streamer.close()
    if args.out_dir:
        trainer.save(args.out_dir)
    if code_mode_summary:
        meta.setdefault("code_mode", {}).update(code_mode_summary)
    if autoscaler_windows:
        telemetry = meta.setdefault("telemetry", {})
        autoscaler_meta = telemetry.setdefault("autoscaler", {})
        autoscaler_meta.setdefault("window", autoscaler_windows)
    print(meta)


def _cmd_train_unigram(args: argparse.Namespace) -> None:
    """Train a unigram tokenizer model over prepacked batches.

    Args:
        args: Parsed CLI arguments providing data patterns, batching options,
            and model hyper-parameters.

    Returns:
        ``None``. Training progress is printed for each epoch and the trained
        state is optionally saved to disk.

    Side Effects:
        Loads all packed batches into host memory and writes the trained model
        to ``args.out_dir`` when provided. The trainer's internal state is
        mutated across epochs.

    Raises:
        SystemExit: If ``--data`` is omitted or the globs do not resolve to at
            least one shard.
    """
    data_patterns = getattr(args, "data", None)
    if not data_patterns:
        raise SystemExit("train-unigram requires at least one --data glob pattern")
    data_files = _expand_data_patterns(data_patterns)
    trainer, trainer_config = _build_unigram_trainer(args)

    # Handle warm-start from existing SentencePiece model
    warm_start_summary: dict[str, object] | None = None
    warm_start_source = getattr(args, "warm_start", None)
    if warm_start_source:
        warm_path = Path(warm_start_source)
        if not warm_path.exists():
            raise SystemExit(f"Warm-start model not found at {warm_path}")
        try:
            sp_bundle = _resolve_sentencepiece_import(warm_path)
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(
                f"Failed to import SentencePiece model from {warm_start_source}: {exc}"
            ) from exc
        if sp_bundle is None:
            raise SystemExit(
                "--warm-start must reference a SentencePiece .model or .vocab/.prob file"
            )
        trainer.import_sentencepiece(pieces=sp_bundle.pieces, scores=sp_bundle.scores)
        warm_start_summary = {
            "source": warm_start_source,
            "pieces": len(sp_bundle.pieces),
        }
        if sp_bundle.metadata:
            warm_start_summary["metadata"] = sp_bundle.metadata
        source_label = sp_bundle.sources[0] if sp_bundle.sources else warm_start_source
        print(
            f"[warm-start] Imported {len(sp_bundle.pieces)} unigram pieces from {source_label}"
        )

    code_langs = _normalize_code_languages(getattr(args, "code_langs", None))
    code_mode_active = bool(getattr(args, "code_mode", False))
    code_mode_config: dict[str, object] = {
        "enabled": code_mode_active,
        "meta_compress": bool(getattr(args, "meta_compress", False)),
    }
    if code_langs:
        code_mode_config["languages"] = sorted(code_langs)
    morphology_plugin, morphology_config = _resolve_morphology(args)
    augmentation_pipeline = _resolve_augmentation(args)
    checkpoint_pause_s = _resolve_checkpoint_pause_seconds()
    config = dict(trainer_config)
    config.update(
        {
            "data": {
                "patterns": list(data_patterns),
                "files": [str(path) for path in data_files],
                "bos": args.bos,
                "eos": args.eos,
                "augmentation": augmentation_pipeline.summary(),
            },
            "output": args.out_dir,
            "dry_run": bool(getattr(args, "dry_run", False)),
            "code_mode": code_mode_config,
            "checkpointing": {
                "checkpoint_dir": args.checkpoint_dir or args.resume_from,
                "checkpoint_every": args.checkpoint_every,
                "resume_from": str(args.resume_from) if args.resume_from else None,
                "time_minutes": args.time_minutes,
            },
        }
    )
    if warm_start_summary:
        config["warm_start"] = warm_start_summary
    config["morphology"] = morphology_config
    _log_resolved_config("train-unigram", config)
    if getattr(args, "dry_run", False):
        print("[dry-run] train-unigram initialization complete")
        return

    code_mode_summary: dict[str, object] | None = None
    if code_mode_active:
        sequences, code_mode_summary = _load_code_mode_sequences(
            data_files,
            bos=args.bos,
            eos=args.eos,
            languages=code_langs if code_langs else None,
            meta_enabled=bool(getattr(args, "meta_compress", False)),
            morphology=morphology_plugin,
        )
    else:
        sequences = _load_sequences(
            data_files,
            bos=args.bos,
            eos=args.eos,
            morphology=morphology_plugin,
        )

    resume_state: dict[str, object] | None = None
    resume_is_checkpoint = False
    if getattr(args, "resume_from", None):
        resume_path = Path(args.resume_from)
        try:
            resume_state = trainer.load_checkpoint(str(resume_path))
            resume_is_checkpoint = True
        except FileNotFoundError:
            try:
                sp_bundle = _resolve_sentencepiece_import(resume_path)
            except (ValueError, RuntimeError) as exc:
                raise SystemExit(
                    f"Failed to import SentencePiece artifacts from {args.resume_from}: {exc}"
                ) from exc
            if sp_bundle is None:
                raise SystemExit(
                    "--resume-from must point to a checkpoint directory or a SentencePiece model"
                )
            trainer.import_sentencepiece(pieces=sp_bundle.pieces, scores=sp_bundle.scores)
            source_label = sp_bundle.sources[0] if sp_bundle.sources else args.resume_from
            print(
                f"[warm-start] Imported {len(sp_bundle.pieces)} unigram pieces from {source_label}"
            )
            resume_state = None
        if resume_is_checkpoint and resume_state is not None:
            payload = resume_state.get("payload") if isinstance(resume_state, Mapping) else None
            trainer_section = payload.get("trainer") if isinstance(payload, Mapping) else None
            progress_meta = (
                trainer_section.get("progress") if isinstance(trainer_section, Mapping) else None
            )
            restored_epochs = getattr(trainer, "completed_epochs", 0)
            epoch_label = ""
            if isinstance(progress_meta, Mapping):
                completed = progress_meta.get("completed_epochs")
                try:
                    completed_int = int(completed) if completed is not None else restored_epochs
                except (TypeError, ValueError):
                    completed_int = restored_epochs
                if completed_int:
                    epoch_label = f" at epoch {completed_int}"
            elif restored_epochs:
                epoch_label = f" at epoch {restored_epochs}"
            print(f"[checkpoint] Restored checkpoint from {args.resume_from}{epoch_label}")

    batches = _build_unigram_batches(
        sequences,
        batch_size=args.batch_size,
        seed=args.seed,
        augmentation=augmentation_pipeline.fork(),
    )
    target_epochs = max(int(args.epochs), 0)
    checkpoint_root: Path | None = None
    checkpoint_dir_value = getattr(args, "checkpoint_dir", None)
    if not checkpoint_dir_value and resume_is_checkpoint:
        checkpoint_dir_value = getattr(args, "resume_from", None)
    if checkpoint_dir_value:
        checkpoint_root = Path(checkpoint_dir_value)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        original_save_checkpoint = trainer.save_checkpoint

        def _save_checkpoint_with_log(
            self, path: str, *cargs, _pause_s: float = checkpoint_pause_s, **ckwargs
        ):
            state = original_save_checkpoint(path, *cargs, **ckwargs)
            print(f"[checkpoint] Saved checkpoint → {path}")
            if _pause_s > 0:
                time.sleep(_pause_s)
            return state

        trainer.save_checkpoint = types.MethodType(_save_checkpoint_with_log, trainer)

    checkpoint_interval = (
        int(args.checkpoint_every)
        if getattr(args, "checkpoint_every", 0)
        and int(getattr(args, "checkpoint_every", 0)) > 0
        else None
    )

    time_minutes_raw = getattr(args, "time_minutes", None)
    time_limit_s: float | None = None
    if time_minutes_raw is not None:
        try:
            minutes_val = float(time_minutes_raw)
        except (TypeError, ValueError):
            minutes_val = None
        if minutes_val is not None:
            if minutes_val <= 0:
                time_limit_s = 0.0
            else:
                time_limit_s = minutes_val * 60.0

    start_time = time.perf_counter()

    def _time_exhausted() -> bool:
        if time_limit_s is None:
            return False
        elapsed = time.perf_counter() - start_time
        return elapsed >= time_limit_s

    time_truncated = False
    while getattr(trainer, "completed_epochs", 0) < target_epochs:
        if _time_exhausted():
            time_truncated = True
            break
        stats = trainer.fit_epoch(batches)
        current_epoch = getattr(trainer, "completed_epochs", 0)
        print(f"epoch {current_epoch}: {stats}")
        if checkpoint_root is not None and checkpoint_interval is not None:
            if current_epoch > 0 and current_epoch % checkpoint_interval == 0:
                trainer.save_checkpoint(str(checkpoint_root))

    if checkpoint_root is not None:
        trainer.save_checkpoint(str(checkpoint_root))

    if time_truncated and time_limit_s is not None:
        elapsed_minutes = (time.perf_counter() - start_time) / 60.0
        print(
            "[checkpoint] Time limit reached after "
            f"{elapsed_minutes:.2f} minute(s); training paused at epoch "
            f"{getattr(trainer, 'completed_epochs', 0)}"
        )

    if args.out_dir:
        trainer.save(args.out_dir)
    if code_mode_summary:
        print(
            "[code-mode] processed {samples} samples (ast={ast_samples}, fallback={fallback_samples})".format(
                **code_mode_summary
            )
        )


def _cmd_train_hybrid(args: argparse.Namespace) -> None:
    """Train alternating BPE and unigram phases with shared batches."""

    data_patterns = getattr(args, "data", None)
    if not data_patterns:
        raise SystemExit("train-hybrid requires at least one --data glob pattern")
    data_files = _expand_data_patterns(data_patterns)
    warm_start_merges = _load_warm_start_merges(getattr(args, "warm_start", None))

    code_langs = _normalize_code_languages(getattr(args, "code_langs", None))
    code_mode_active = bool(getattr(args, "code_mode", False))
    code_mode_config: dict[str, object] = {
        "enabled": code_mode_active,
        "meta_compress": bool(getattr(args, "meta_compress", False)),
    }
    if code_langs:
        code_mode_config["languages"] = sorted(code_langs)

    morphology_plugin, morphology_config = _resolve_morphology(args)
    augmentation_pipeline = _resolve_augmentation(args)

    privacy_label = str(getattr(args, "privacy", "none") or "none").lower()
    privacy_enabled = privacy_label != "none"
    randomize_ties = privacy_label == "tie-randomize"
    tie_seed = getattr(args, "tie_seed", None)
    privacy_salt = getattr(args, "privacy_salt", None)

    checkpoint_dir_value = getattr(args, "checkpoint_dir", None) or getattr(
        args, "resume_from", None
    )

    config = {
        "trainer": {
            "base_vocab": args.base_vocab,
            "merges": args.merges,
            "cycles": args.cycles,
            "unigram_epochs": args.unigram_epochs,
            "max_unigram_len": args.max_unigram_len,
            "privacy": {
                "mode": privacy_label,
                "tie_seed": int(tie_seed) if tie_seed is not None else None,
                "salt_configured": bool(privacy_salt),
            },
        },
        "data": {
            "patterns": list(data_patterns),
            "files": [str(path) for path in data_files],
            "bos": args.bos,
            "eos": args.eos,
            "augmentation": augmentation_pipeline.summary(),
        },
        "runtime": {
            "batch_size": args.batch_size,
            "seed": args.seed,
            "bpe_log_every": args.bpe_log_every,
        },
        "warm_start": {
            "source": getattr(args, "warm_start", None),
            "merges": [list(map(int, pair)) for pair in warm_start_merges]
            if warm_start_merges
            else None,
        },
        "checkpointing": {
            "checkpoint_dir": checkpoint_dir_value,
            "checkpoint_every": args.checkpoint_every,
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "time_minutes": args.time_minutes,
        },
        "output": args.out_dir,
        "dry_run": bool(getattr(args, "dry_run", False)),
        "code_mode": code_mode_config,
    }
    config["morphology"] = morphology_config

    _log_resolved_config("train-hybrid", config)
    if getattr(args, "dry_run", False):
        print("[dry-run] train-hybrid initialization complete")
        return

    code_mode_summary: dict[str, object] | None = None
    if code_mode_active:
        sequences, code_mode_summary = _load_code_mode_sequences(
            data_files,
            bos=args.bos,
            eos=args.eos,
            languages=code_langs if code_langs else None,
            meta_enabled=bool(getattr(args, "meta_compress", False)),
            morphology=morphology_plugin,
        )
    else:
        sequences = _load_sequences(
            data_files,
            bos=args.bos,
            eos=args.eos,
            morphology=morphology_plugin,
        )
    batches = list(
        _iter_packed_batches(
            sequences,
            batch_size=args.batch_size,
            seed=args.seed,
            augmentation=augmentation_pipeline.fork(),
        )
    )

    if HybridTrainer is None:  # pragma: no cover - optional torch dependency
        raise SystemExit("HybridTrainer is unavailable; install torch for train-hybrid")

    trainer = HybridTrainer(
        base_vocab=args.base_vocab,
        merges=args.merges,
        cycles=args.cycles,
        unigram_epochs=args.unigram_epochs,
        max_unigram_len=args.max_unigram_len,
        warm_start_merges=warm_start_merges,
        privacy_mode=privacy_enabled,
        randomize_ties=randomize_ties,
        tie_seed=tie_seed,
        privacy_salt=privacy_salt,
        bpe_init_kwargs={"device": args.device},
    )

    if getattr(args, "resume_from", None):
        resume_state = trainer.load_checkpoint(str(args.resume_from))
        payload = resume_state.get("payload") if isinstance(resume_state, Mapping) else None
        trainer_section = payload.get("trainer") if isinstance(payload, Mapping) else None
        progress_meta = trainer_section.get("progress") if isinstance(trainer_section, Mapping) else None
        restored_cycles = getattr(trainer, "completed_cycles", 0)
        cycle_label = ""
        if isinstance(progress_meta, Mapping):
            completed = progress_meta.get("completed_cycles")
            try:
                completed_int = int(completed) if completed is not None else restored_cycles
            except (TypeError, ValueError):
                completed_int = restored_cycles
            if completed_int:
                cycle_label = f" after {completed_int} cycle(s)"
        elif restored_cycles:
            cycle_label = f" after {restored_cycles} cycle(s)"
        print(f"[checkpoint] Restored checkpoint from {args.resume_from}{cycle_label}")

    checkpoint_interval = (
        int(args.checkpoint_every)
        if getattr(args, "checkpoint_every", 0)
        and int(getattr(args, "checkpoint_every", 0)) > 0
        else None
    )

    checkpoint_target = checkpoint_dir_value
    if checkpoint_target:
        checkpoint_path = Path(checkpoint_target)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        original_save_checkpoint = trainer.save_checkpoint

        checkpoint_pause_s: float = 0.0
        def _hybrid_checkpoint_with_log(
            self, path: str, *cargs, _pause_s: float = checkpoint_pause_s, **ckwargs
        ):
            state = original_save_checkpoint(path, *cargs, **ckwargs)
            print(f"[checkpoint] Saved checkpoint → {path}")
            if _pause_s > 0:
                time.sleep(_pause_s)
            return state

        trainer.save_checkpoint = types.MethodType(_hybrid_checkpoint_with_log, trainer)

    time_minutes_raw = getattr(args, "time_minutes", None)
    time_limit_s: float | None = None
    if time_minutes_raw is not None:
        try:
            minutes_val = float(time_minutes_raw)
        except (TypeError, ValueError):
            minutes_val = None
        if minutes_val is not None:
            if minutes_val <= 0:
                time_limit_s = 0.0
            else:
                time_limit_s = minutes_val * 60.0

    bpe_fit_kwargs = {"log_every": args.bpe_log_every}
    summary = trainer.fit(
        batches,
        cycles=args.cycles,
        unigram_epochs=args.unigram_epochs,
        warm_start_merges=warm_start_merges,
        checkpoint_dir=checkpoint_target,
        checkpoint_interval=checkpoint_interval,
        time_limit_s=time_limit_s,
        bpe_fit_kwargs=bpe_fit_kwargs,
    )
    print(summary)

    if summary.get("stopped_early") and time_limit_s is not None:
        print(
            "[checkpoint] Hybrid time limit reached; paused after "
            f"{trainer.completed_cycles} cycle(s)"
        )

    if code_mode_summary:
        print(
            "[code-mode] processed {samples} samples (ast={ast_samples}, fallback={fallback_samples})".format(
                **code_mode_summary
            )
        )

    if args.out_dir:
        trainer.save(args.out_dir)


def _cmd_stream_batches(args: argparse.Namespace) -> None:
    """Stream packed batches and report their tensor dimensions.

    Args:
        args: Parsed CLI arguments providing data globs, optional BOS/EOS
            tokens, seed, and the desired ``batch_size`` and ``max_batches``
            attributes.

    Returns:
        ``None``. Batch metadata is emitted to stdout for inspection.

    Side Effects:
        Opens shard files, materializes packed tensors on the host, and prints
        batch shapes until ``max_batches`` is reached (when provided).

    Raises:
        SystemExit: If no ``--data`` patterns are supplied or the globs resolve
            to no shard files.
    """

    data_patterns = getattr(args, "data", None)
    if not data_patterns:
        raise SystemExit("stream-batches requires at least one --data glob pattern")
    data_files = _expand_data_patterns(data_patterns)
    morphology_plugin, _ = _resolve_morphology(args)
    sequences = _load_sequences(
        data_files,
        bos=getattr(args, "bos", None),
        eos=getattr(args, "eos", None),
        morphology=morphology_plugin,
    )
    batch_size = getattr(args, "batch_size", 1024)
    seed = getattr(args, "seed", 1337)
    augmentation_pipeline = _resolve_augmentation(args)
    batches = _iter_packed_batches(
        sequences,
        batch_size=batch_size,
        seed=seed,
        augmentation=augmentation_pipeline.fork(),
    )
    max_batches = getattr(args, "max_batches", None)
    for idx, (tokens, mask, lengths) in enumerate(batches):
        print(
            f"batch {idx}: tokens={tuple(tokens.shape)} mask={tuple(mask.shape)} lengths={tuple(lengths.shape)}"
        )
        if max_batches is not None and max_batches > 0 and idx + 1 >= max_batches:
            break


def _cmd_resume_bpe(args: argparse.Namespace) -> None:
    """Resume a BPE training run from an on-disk checkpoint.

    Args:
        args: Parsed CLI arguments expected to include ``--resume-from`` and all
            parameters required by :func:`_cmd_train_bpe`.

    Returns:
        ``None``. All work is delegated to :func:`_cmd_train_bpe`.

    Side Effects:
        Loads checkpoint state via :class:`GPUBPETrainer`, potentially rebuilds
        :class:`CorpusStreamer` instances, and produces the same outputs as a
        standard ``train-bpe`` invocation.

    Raises:
        SystemExit: If ``--resume-from`` or ``--data`` are missing before
            dispatching to :func:`_cmd_train_bpe`, or if the delegated training
            invocation encounters its own fatal CLI condition.
    """

    if not getattr(args, "resume_from", None):
        raise SystemExit("--resume-from is required when invoking resume-bpe")
    if not getattr(args, "data", None):
        raise SystemExit("resume-bpe requires --data globs to stream training shards")
    _cmd_train_bpe(args)


def _parser() -> argparse.ArgumentParser:
    """Build the CLI parser that exposes training and benchmarking commands.

    Returns:
        Configured :class:`argparse.ArgumentParser` with subcommands registered
        for ``train-bpe``, ``train-unigram``, and ``benchmark``.

    Side Effects:
        None.
    """
    parser = argparse.ArgumentParser(description="GPU tokenizer toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", nargs="+", required=True, help="Input glob patterns")
    common.add_argument("--bos", type=int, default=None, help="Optional BOS token id")
    common.add_argument("--eos", type=int, default=None, help="Optional EOS token id")
    common.add_argument("--seed", type=int, default=1337, help="Shuffle seed")
    common.add_argument("--device", type=str, default=None, help="Torch device override")
    common.add_argument(
        "--code-mode",
        action="store_true",
        help="Enable AST-aware preprocessing for code corpora",
    )
    common.add_argument(
        "--code-langs",
        nargs="+",
        default=None,
        help="Restrict code-mode preprocessing to specific languages (e.g. python typescript)",
    )
    common.add_argument(
        "--meta-compress",
        action="store_true",
        help="Enable meta-token compression when running in code mode",
    )
    common.add_argument(
        "--compression",
        type=str,
        default="none",
        choices=["none", "zstd", "lz4"],
        help="Compression codec for input shards",
    )
    common.add_argument(
        "--io-workers",
        type=int,
        default=2,
        help="Number of background workers for shard decoding",
    )
    common.add_argument(
        "--prefetch-batches",
        type=int,
        default=4,
        help="Maximum number of prefetched batches before backpressure engages",
    )
    common.add_argument(
        "--prefetch-jitter",
        type=float,
        default=0.0,
        help="Random jitter applied when throttling prefetch depth (0 disables)",
    )
    morph_choices = available_morphology_plugins()
    morph_kwargs: dict[str, object] = {
        "type": str,
        "default": None,
        "help": (
            "Opt-in morphology preprocessing for the specified language; "
            "disabled by default and may affect downstream token statistics"
        ),
    }
    if morph_choices:
        morph_kwargs["choices"] = list(morph_choices)
    common.add_argument("--morphology-lang", **morph_kwargs)
    common.add_argument(
        "--morphology-case-markers",
        action="store_true",
        help=(
            "Segment case markers when supported by the selected morphology plugin "
            "(requires --morphology-lang)"
        ),
    )
    common.add_argument(
        "--morphology-affix-tags",
        action="store_true",
        help=(
            "Annotate productive affixes when supported by the selected morphology plugin "
            "(requires --morphology-lang)"
        ),
    )

    augmentation_parent = argparse.ArgumentParser(add_help=False)
    augmentation_parent.add_argument(
        "--augmentation",
        type=str,
        default="none",
        choices=[mode.value for mode in AugmentationMode],
        help="Data augmentation mode applied to training sequences",
    )
    augmentation_parent.add_argument(
        "--aug-strength",
        type=float,
        default=0.0,
        help="Strength parameter for the selected augmentation (0 disables)",
    )
    bpe_parent = argparse.ArgumentParser(add_help=False)
    bpe_parent.add_argument("--merges", type=int, default=50_000)
    bpe_parent.add_argument("--base-vocab", type=int, default=256)
    bpe_parent.add_argument("--target-util", type=float, default=0.80)
    bpe_parent.add_argument("--min-batch", type=int, default=512)
    bpe_parent.add_argument("--max-batch", type=int, default=4096)
    bpe_parent.add_argument("--token-bytes", type=int, default=8 * 1024)
    bpe_parent.add_argument("--log-every", type=int, default=100)
    bpe_parent.add_argument("--out-dir", type=str, default="./bpe_out")
    bpe_parent.add_argument(
        "--privacy",
        type=str,
        default="none",
        choices=["none", "hash-merges", "tie-randomize"],
        help=(
            "Privacy guard for exported merges. 'hash-merges' redacts merge IDs, "
            "while 'tie-randomize' also randomizes tie-breaks—breaking deterministic "
            "parity across devices unless you also provide --tie-seed."
        ),
    )
    bpe_parent.add_argument(
        "--privacy-salt",
        type=str,
        default=None,
        help=(
            "Optional hex or UTF-8 salt mixed into hashed merges when privacy is enabled. "
            "The salt itself is never written to manifests."
        ),
    )
    bpe_parent.add_argument(
        "--tie-seed",
        type=int,
        default=None,
        help=(
            "Seed controlling randomized tie-breaks. Combine with --privacy tie-randomize "
            "to make stochastic runs reproducible."
        ),
    )
    bpe_parent.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory where periodic training checkpoints are written",
    )
    bpe_parent.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Frequency (in merges) for checkpoint writes; disabled when set to 0",
    )
    bpe_parent.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint directory created by --checkpoint-dir",
    )
    bpe_parent.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate the trainer, log resolved configuration, and exit",
    )

    train_bpe = subparsers.add_parser(
        "train-bpe",
        parents=[common, bpe_parent, augmentation_parent],
        help="Train a BPE model",
    )
    train_bpe.set_defaults(func=_cmd_train_bpe)
    train_bpe.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="Optional JSON manifest or TikToken bundle containing seed merges",
    )

    resume_bpe = subparsers.add_parser(
        "resume-bpe",
        parents=[common, bpe_parent, augmentation_parent],
        help="Resume BPE training from a checkpoint directory",
    )
    resume_bpe.set_defaults(func=_cmd_resume_bpe)

    train_hybrid = subparsers.add_parser(
        "train-hybrid",
        parents=[common, augmentation_parent],
        help="Train alternating BPE→unigram cycles",
    )
    train_hybrid.set_defaults(func=_cmd_train_hybrid)
    train_hybrid.add_argument("--merges", type=int, default=50_000)
    train_hybrid.add_argument("--base-vocab", type=int, default=256)
    train_hybrid.add_argument("--batch-size", type=int, default=1024)
    train_hybrid.add_argument("--cycles", type=int, default=1)
    train_hybrid.add_argument("--unigram-epochs", type=int, default=1)
    train_hybrid.add_argument("--max-unigram-len", type=int, default=8)
    train_hybrid.add_argument("--bpe-log-every", type=int, default=100)
    train_hybrid.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="Optional JSON manifest or TikToken bundle containing seed merges",
    )
    train_hybrid.add_argument(
        "--privacy",
        type=str,
        default="none",
        choices=["none", "hash-merges", "tie-randomize"],
        help=(
            "Apply the BPE privacy guard when exporting hybrid manifests. See train-bpe "
            "for mode semantics; tie randomization sacrifices deterministic parity unless "
            "--tie-seed is provided."
        ),
    )
    train_hybrid.add_argument(
        "--privacy-salt",
        type=str,
        default=None,
        help=(
            "Optional salt used when hashing merge metadata in privacy-aware hybrid exports."
        ),
    )
    train_hybrid.add_argument(
        "--tie-seed",
        type=int,
        default=None,
        help=(
            "Seed forwarded to the BPE phase when tie randomization is enabled for privacy."
        ),
    )
    train_hybrid.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory where per-cycle checkpoints are written",
    )
    train_hybrid.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save the hybrid checkpoint after N cycles (0 disables periodic saves)",
    )
    train_hybrid.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a hybrid checkpoint directory created by --checkpoint-dir",
    )
    train_hybrid.add_argument(
        "--time-minutes",
        type=float,
        default=None,
        help="Optional wall-clock budget (in minutes) before pausing hybrid cycles",
    )
    train_hybrid.add_argument("--out-dir", type=str, default="./hybrid_out")
    train_hybrid.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate the trainer, log resolved configuration, and exit",
    )

    train_unigram = subparsers.add_parser(
        "train-unigram",
        parents=[common, augmentation_parent],
        help="Train a unigram model",
    )
    train_unigram.set_defaults(func=_cmd_train_unigram)
    train_unigram.add_argument("--vocab-size", type=int, default=50_000)
    train_unigram.add_argument("--base-vocab", type=int, default=256)
    train_unigram.add_argument("--max-subword-len", type=int, default=8)
    train_unigram.add_argument("--batch-size", type=int, default=1024)
    train_unigram.add_argument("--epochs", type=int, default=1)
    train_unigram.add_argument("--out-dir", type=str, default="./unigram_out")
    train_unigram.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="Optional SentencePiece .model or .vocab/.prob file to seed the unigram trainer",
    )
    train_unigram.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory where unigram checkpoints are written",
    )
    train_unigram.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Write a checkpoint every N epochs (0 disables periodic checkpoints)",
    )
    train_unigram.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory created by --checkpoint-dir",
    )
    train_unigram.add_argument(
        "--time-minutes",
        type=float,
        default=None,
        help="Optional wall-clock budget (in minutes) before pausing training",
    )
    train_unigram.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate the trainer, log resolved configuration, and exit",
    )

    benchmark = subparsers.add_parser("benchmark", help="Run tokenizer training benchmarks")
    benchmark.set_defaults(func=_cmd_benchmark)
    benchmark.add_argument(
        "--data",
        nargs="+",
        default=[],
        help="Optional glob patterns pointing at real datasets",
    )
    benchmark.add_argument("--bos", type=int, default=None, help="Optional BOS token id")
    benchmark.add_argument("--eos", type=int, default=None, help="Optional EOS token id")
    benchmark.add_argument("--synthetic-docs", type=int, default=0, help="Number of synthetic documents")
    benchmark.add_argument(
        "--synthetic-min-len", type=int, default=32, help="Minimum synthetic document length"
    )
    benchmark.add_argument(
        "--synthetic-max-len", type=int, default=256, help="Maximum synthetic document length"
    )
    benchmark.add_argument(
        "--synthetic-vocab", type=int, default=256, help="Vocabulary range for synthetic corpora"
    )
    benchmark.add_argument(
        "--max-real-docs",
        type=int,
        default=None,
        help="Optional cap on the number of real documents to load",
    )
    benchmark.add_argument("--seed", type=int, default=1337, help="Shuffle seed")
    benchmark.add_argument("--device", type=str, default=None, help="Torch device override")
    benchmark.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where benchmark JSON outputs are written",
    )
    benchmark.add_argument(
        "--bpe-base-vocab", type=int, default=256, help="Base vocabulary size for BPE"
    )
    benchmark.add_argument("--bpe-merges", type=int, default=10_000, help="Number of BPE merges")
    benchmark.add_argument(
        "--bpe-batch-size", type=int, default=1024, help="Batch size to use for BPE"
    )
    benchmark.add_argument(
        "--bpe-log-every", type=int, default=250, help="Logging cadence for BPE trainer"
    )
    benchmark.add_argument(
        "--bpe-config",
        type=str,
        default=None,
        help="Optional JSON config describing heterogeneous BPE benchmark runs",
    )
    benchmark.add_argument(
        "--evaluation-report",
        type=str,
        default=None,
        help=(
            "Optional evaluation JSON report to merge into the benchmark metadata"
        ),
    )
    benchmark.add_argument(
        "--baseline-corpus",
        action="append",
        default=[],
        help=(
            "Builtin baseline corpus to benchmark with reference tokenizers. "
            "Repeat to include multiple corpora."
        ),
    )
    benchmark.add_argument(
        "--sentencepiece-model",
        type=str,
        default=None,
        help=(
            "SentencePiece .model file evaluated against baseline corpora when "
            "--baseline-corpus is supplied"
        ),
    )
    benchmark.add_argument(
        "--huggingface-tokenizer",
        type=str,
        default=None,
        help=(
            "Hugging Face tokenizer.json evaluated against baseline corpora when "
            "--baseline-corpus is supplied"
        ),
    )

    export_embeddings = subparsers.add_parser(
        "export-embeddings",
        help="Export embedding matrices, manifests, and pruning metadata",
    )
    export_embeddings.set_defaults(func=_cmd_export_embeddings)
    export_embeddings.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to a vocab JSON mapping tokens to ids",
    )
    export_embeddings.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where embedding export artifacts are written",
    )
    export_embeddings.add_argument(
        "--dimension",
        type=int,
        default=256,
        help="Embedding dimensionality for randomly initialised tokens",
    )
    export_embeddings.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Floating-point dtype label (e.g. float32, float16) recorded in the manifest",
    )
    export_embeddings.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed controlling deterministic embedding initialisation",
    )
    export_embeddings.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Optional JSON file providing co-training statistics for tokens",
    )
    export_embeddings.add_argument(
        "--min-frequency",
        type=float,
        default=0.0,
        help="Minimum observed frequency required to retain a token",
    )
    export_embeddings.add_argument(
        "--dedupe-similarity",
        type=float,
        default=0.0,
        help=(
            "Merge tokens whose usage vectors or synthesized embeddings meet or exceed"
            " this cosine similarity threshold before pruning"
        ),
    )
    export_embeddings.add_argument(
        "--keep-token",
        action="append",
        default=[],
        help=(
            "Token string to always preserve during pruning; repeat the flag to"
            " keep multiple tokens"
        ),
    )

    evaluate_cmd = subparsers.add_parser(
        "evaluate",
        parents=[common],
        help="Score tokenizer artifacts against a reference corpus",
        description=textwrap.dedent(
            """
            Score exported BPE or unigram tokenizer artifacts against a held-out
            corpus and emit a schema-validated JSON report. Provide one or more
            --data globs that mirror the preprocessing flags used during
            training (--code-mode, --code-langs, --morphology-*) so compression,
            OOV, and morphology metrics remain comparable.

            Required inputs
              • Supply --data one or more times to point at UTF-8 text shards or
                structured code manifests.
              • Use --artifacts to scan an export directory or override the
                resolved files with --vocab/--merges/--tokenizer for bespoke
                layouts.

            Artifact flavours
              • BPE exports must provide a vocabulary plus a merge table.
              • SentencePiece unigram exports rely solely on a *.vocab file and
                ignore merge tables.

            Outputs
              • JSON written to --output (defaults to reports/evaluate.json) or
                streamed to stdout when --output - is selected.
              • A human-readable banner controlled via --summary-format
                (table/json/none) that mirrors key metrics from the report.
            """
        ),
        formatter_class=EvaluateHelpFormatter,
        epilog=textwrap.dedent(
            f"""
            The JSON payload captures artifact metadata, compression ratios, OOV
            counts, morphology coverage, and optional code-mode summaries. Pass
            --deterministic to stabilise ordering for reproducible diffs.
            """
        ),
    )
    evaluate_cmd.set_defaults(func=_cmd_evaluate)
    evaluate_cmd.add_argument(
        "--artifacts",
        type=str,
        default=None,
        help=(
            "Directory containing exported artifacts. For BPE expect vocab.json"
            " and merges.(json|txt); SentencePiece unigram exports typically"
            " expose unigram.vocab. Individual files may be overridden below."
        ),
    )
    evaluate_cmd.add_argument(
        "--vocab",
        type=str,
        default=None,
        help=(
            "Override the vocabulary path. Defaults to <artifacts>/vocab.json or"
            " <artifacts>/unigram.vocab when --artifacts is supplied."
        ),
    )
    evaluate_cmd.add_argument(
        "--merges",
        type=str,
        default=None,
        help=(
            "Optional merge history used by BPE exports (JSON or text). Required"
            " unless --model-type unigram is selected."
        ),
    )
    evaluate_cmd.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer.json manifest to record in the artifacts block",
    )
    evaluate_cmd.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "bpe", "unigram"],
        help=(
            "Explicitly declare the export flavour. Auto-detection inspects"
            " --merges and the vocabulary suffix; force 'unigram' for"
            " SentencePiece-style packages without merges."
        ),
    )
    evaluate_cmd.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            f"Destination for the JSON report. Defaults to {DEFAULT_EVALUATE_OUTPUT}"
            " when omitted; use '-' to print the payload to stdout."
        ),
    )
    evaluate_cmd.add_argument(
        "--meta-max-length",
        type=int,
        default=8,
        help="Maximum meta-token length when evaluating code-mode corpora",
    )
    evaluate_cmd.add_argument(
        "--deterministic",
        action="store_true",
        help="Stabilise OOV listings and summaries for reproducible reports",
    )
    evaluate_cmd.add_argument(
        "--force-evaluation",
        action="store_true",
        help=(
            "Ignore the SUPERTOKEN_SKIP_EVALUATION environment flag and run anyway"
        ),
    )
    evaluate_cmd.add_argument(
        "--summary-format",
        type=str,
        default="table",
        choices=EVALUATE_SUMMARY_FORMATS,
        help=(
            "Controls the post-run banner: 'table' renders human-readable"
            " metrics, 'json' mirrors the legacy structured output, and 'none'"
            " suppresses the summary."
        ),
    )
    benchmark.add_argument(
        "--unigram-base-vocab",
        type=int,
        default=256,
        help="Base vocabulary size for unigram trainer",
    )
    benchmark.add_argument(
        "--unigram-vocab", type=int, default=50_000, help="Target unigram vocabulary size"
    )
    benchmark.add_argument(
        "--unigram-max-subword",
        type=int,
        default=8,
        help="Maximum unigram subword length",
    )
    benchmark.add_argument(
        "--unigram-batch-size", type=int, default=1024, help="Batch size for unigram batches"
    )
    benchmark.add_argument(
        "--unigram-epochs", type=int, default=1, help="Number of unigram epochs"
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point that dispatches to tokenizer subcommands.

    Args:
        argv: Optional argument vector override, mirroring ``sys.argv[1:]``
            when ``None``.

    Returns:
        ``None``. The selected subcommand performs all work.

    Side Effects:
        Parses CLI arguments, writes help text and command output to stdout, and
        may exit the interpreter via :func:`argparse.ArgumentParser.parse_args`.

    Raises:
        SystemExit: If argument parsing fails or no subcommand is provided.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        raise SystemExit(1)
    func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
