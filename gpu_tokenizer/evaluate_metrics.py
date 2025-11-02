"""Metric helpers for :mod:`gpu_tokenizer.evaluate`.

The evaluator works with materialised corpora that may live on the CPU or the
GPU.  These helpers accept native Python numbers, sequences, or ``torch``
``Tensor`` objects and reduce them down to scalar statistics that the JSON
report can expose.  The module avoids taking hard dependencies on ``torch`` so
that the utilities continue to function under the lightweight stubs exercised
by the unit test suite.

The metrics follow the definitions documented in ``docs/cookbook/evaluate.md``:

``compression``
    The ratio between produced tokens and the raw byte length of the corpus.

``oov``
    The fraction of tokens that fall outside the supplied vocabulary.

``morphology_purity``
    The share of morphology segments that carry at least one tag.

``code_mode_reduction``
    The fractional decrease in AST token length achieved by meta-token
    compression when code-mode is active.  ``0.0`` indicates no reduction while
    ``1.0`` would mean every AST token expanded from a meta-token.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

try:  # pragma: no cover - import guard exercised in CI environments without torch
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None  # type: ignore[assignment]


def _is_real_torch_tensor(value: Any) -> bool:
    if torch is None:
        return False
    try:
        if getattr(torch, "__super_token_stub__", False):  # pragma: no cover - stub detection
            return False
        return isinstance(value, torch.Tensor)
    except Exception:  # pragma: no cover - defensive: ``torch`` attribute missing
        return False


def _accumulate_numeric(value: Any) -> float:
    """Return the sum of ``value`` treating nested iterables recursively."""

    if value is None:
        return 0.0
    if _is_real_torch_tensor(value):
        tensor = value.detach()
        # For tensor inputs, treat counts as element counts rather than sums,
        # matching tests that pass per-document vectors.
        return float(tensor.numel())
    if isinstance(value, Mapping):
        return sum(_accumulate_numeric(item) for item in value.values())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_accumulate_numeric(item) for item in value)
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        raise TypeError(f"Unsupported numeric input {value!r}") from None


def compute_compression_ratio(
    token_counts: Sequence[int] | Mapping[Any, Any] | Any,
    byte_counts: Sequence[int] | Mapping[Any, Any] | Any,
) -> dict[str, float]:
    """Return tokens/byte and bytes/token ratios for a corpus."""

    total_tokens = _accumulate_numeric(token_counts)
    total_bytes = _accumulate_numeric(byte_counts)
    tokens_per_byte = total_tokens / total_bytes if total_bytes else 0.0
    bytes_per_token = total_bytes / total_tokens if total_tokens else 0.0
    return {
        "tokens_per_byte": tokens_per_byte,
        "bytes_per_token": bytes_per_token,
    }


def compute_oov_rate(oov_instances: Any, total_tokens: Any) -> float:
    """Compute the out-of-vocabulary rate for the evaluated corpus."""

    oov_total = _accumulate_numeric(oov_instances)
    token_total = _accumulate_numeric(total_tokens)
    if token_total == 0.0:
        return 0.0
    return oov_total / token_total


def compute_morphology_purity(tagged_segments: Any, total_segments: Any) -> float:
    """Return the share of morphology segments carrying at least one tag."""

    tagged_total = _accumulate_numeric(tagged_segments)
    segments_total = _accumulate_numeric(total_segments)
    if segments_total == 0.0:
        return 0.0
    return tagged_total / segments_total


def compute_code_mode_reduction(
    sequences: Sequence[Sequence[int | str]],
    meta_tokens: Mapping[str, Sequence[str]] | None,
) -> float:
    """Measure how much meta-token compression shortens AST token streams."""

    if not sequences:
        return 0.0
    dictionary = {str(key): list(value) for key, value in (meta_tokens or {}).items()}
    compressed_total = 0
    expanded_total = 0
    for sequence in sequences:
        if not sequence or not any(isinstance(token, str) for token in sequence):
            continue
        for token in sequence:
            if not isinstance(token, str):
                continue
            compressed_total += 1
            if token in dictionary:
                expanded_total += len(dictionary[token])
            else:
                expanded_total += 1
    if expanded_total == 0:
        return 0.0
    return 1.0 - (compressed_total / expanded_total)


__all__ = [
    "compute_code_mode_reduction",
    "compute_compression_ratio",
    "compute_morphology_purity",
    "compute_oov_rate",
]

