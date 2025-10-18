"""High level helpers for preparing code corpora for trainer ingestion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional

from .linearizer import LinearizationResult
from .meta_compress import MetaTokenCompressor, encode_linearized_sequence
from .py_frontend import linearize_python_source
from .ts_frontend import linearize_typescript_source


@dataclass
class EncodedSample:
    """Container for a single sample processed in code mode."""

    kind: str
    tokens: List[str] | List[int]
    metadata: MutableMapping[str, Any]
    symbols: Mapping[str, str] | None


@dataclass
class CodeModeCorpus:
    """Result bundle produced by :func:`prepare_corpus`."""

    samples: List[EncodedSample]
    meta_tokens: Mapping[str, List[str]]
    meta_enabled: bool
    meta_max_length: int

    def ast_samples(self) -> List[EncodedSample]:
        return [sample for sample in self.samples if sample.kind == "ast"]

    def byte_fallbacks(self) -> List[EncodedSample]:
        return [sample for sample in self.samples if sample.kind == "bytes"]


def _linearize_source(language: str, source: str, filename: str) -> LinearizationResult:
    lowered = language.lower()
    if lowered in {"python", "py"}:
        return linearize_python_source(source, filename=filename)
    if lowered in {"typescript", "ts", "javascript", "js"}:
        return linearize_typescript_source(source, filename=filename)
    raise ValueError(f"Unsupported language '{language}'")


def prepare_corpus(
    entries: Iterable[Mapping[str, Any]],
    *,
    meta_enabled: bool = True,
    meta_max_length: int = 8,
    logger: Optional[logging.Logger] = None,
) -> CodeModeCorpus:
    """Process ``entries`` into sequences suitable for the trainer pipeline."""

    log = logger or logging.getLogger(__name__)
    samples: List[EncodedSample] = []
    ast_sequences: List[List[str]] = []
    ast_indices: List[int] = []

    for index, entry in enumerate(entries):
        language = str(entry.get("language", ""))
        source = entry.get("source")
        if not isinstance(source, str):
            raise TypeError("Entries must provide a string 'source'")
        filename = str(entry.get("filename", f"<input {index}>"))
        try:
            result = _linearize_source(language, source, filename)
        except Exception as exc:  # pragma: no cover - exercised in tests
            log.warning(
                "Falling back to byte-level tokenization for %s (%s): %s",
                filename,
                language or "unknown",
                exc,
            )
            metadata = {
                "language": language or "unknown",
                "filename": filename,
                "fallback": True,
                "reason": str(exc),
            }
            samples.append(
                EncodedSample(
                    kind="bytes",
                    tokens=list(source.encode("utf-8")),
                    metadata=metadata,
                    symbols=None,
                )
            )
            continue

        encoded_tokens = encode_linearized_sequence(result.tokens)
        metadata = dict(result.metadata)
        metadata["fallback"] = False
        samples.append(
            EncodedSample(
                kind="ast",
                tokens=encoded_tokens,
                metadata=metadata,
                symbols=result.symbols,
            )
        )
        ast_sequences.append(encoded_tokens)
        ast_indices.append(len(samples) - 1)

    compressor = MetaTokenCompressor(
        max_pattern_length=max(meta_max_length, 1),
        enabled=meta_enabled and meta_max_length > 1,
    )
    compression = compressor.compress(ast_sequences)

    for offset, sample_index in enumerate(ast_indices):
        samples[sample_index].tokens = compression.sequences[offset]

    return CodeModeCorpus(
        samples=samples,
        meta_tokens=compression.dictionary,
        meta_enabled=meta_enabled,
        meta_max_length=max(meta_max_length, 1),
    )
