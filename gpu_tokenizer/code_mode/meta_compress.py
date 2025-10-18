"""Meta-token discovery for code-mode linearizations."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

TokenSequence = Sequence[str]
MetaDictionary = Dict[str, List[str]]


@dataclass
class MetaCompressionResult:
    """Container for compressed sequences and the discovered meta patterns."""

    sequences: List[List[str]]
    dictionary: MetaDictionary


class MetaTokenCompressor:
    """Discover repeated subsequences and replace them with ``META<n>`` tokens."""

    def __init__(
        self,
        *,
        max_pattern_length: int = 8,
        enabled: bool = True,
        min_frequency: int = 2,
    ) -> None:
        if max_pattern_length < 1:
            raise ValueError("max_pattern_length must be at least 1")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        self.max_pattern_length = max_pattern_length
        self.enabled = enabled
        self.min_frequency = min_frequency

    def compress(self, sequences: Sequence[TokenSequence]) -> MetaCompressionResult:
        """Return compressed copies of ``sequences`` and the meta-token mapping."""

        materialized: List[List[str]] = [list(seq) for seq in sequences]
        if not materialized:
            return MetaCompressionResult(materialized, {})

        if not self.enabled or self.max_pattern_length <= 1:
            return MetaCompressionResult(materialized, {})

        pattern_counts: Counter[Tuple[str, ...]] = Counter()
        max_len = self.max_pattern_length
        for seq in materialized:
            n = len(seq)
            if n < 2:
                continue
            for length in range(2, min(max_len, n) + 1):
                for idx in range(0, n - length + 1):
                    window = tuple(seq[idx : idx + length])
                    pattern_counts[window] += 1

        candidates = [
            pattern
            for pattern, count in pattern_counts.items()
            if count >= self.min_frequency
        ]
        if not candidates:
            return MetaCompressionResult(materialized, {})

        candidates.sort(
            key=lambda pattern: (
                -len(pattern),
                -pattern_counts[pattern],
                pattern,
            )
        )
        pattern_to_meta: Dict[Tuple[str, ...], str] = {}
        for index, pattern in enumerate(candidates):
            pattern_to_meta[pattern] = f"META{index}"

        pattern_by_first: Dict[str, List[Tuple[Tuple[str, ...], str]]] = defaultdict(list)
        for pattern in candidates:
            meta = pattern_to_meta[pattern]
            pattern_by_first[pattern[0]].append((pattern, meta))
        for pattern_list in pattern_by_first.values():
            pattern_list.sort(key=lambda item: (-len(item[0]), item[1]))

        compressed_sequences: List[List[str]] = []
        used_meta: set[str] = set()
        for seq in materialized:
            if not seq:
                compressed_sequences.append([])
                continue
            compressed: List[str] = []
            index = 0
            length = len(seq)
            while index < length:
                token = seq[index]
                matched = False
                for pattern, meta in pattern_by_first.get(token, []):
                    size = len(pattern)
                    if index + size > length:
                        continue
                    if seq[index : index + size] == list(pattern):
                        compressed.append(meta)
                        used_meta.add(meta)
                        index += size
                        matched = True
                        break
                if not matched:
                    compressed.append(token)
                    index += 1
            compressed_sequences.append(compressed)

        if not used_meta:
            return MetaCompressionResult(materialized, {})

        # Ensure meta tokens are dense and ordered.
        used_ordered = sorted(used_meta, key=lambda name: int(name[4:]))
        renumber: Dict[str, str] = {
            meta: f"META{idx}" for idx, meta in enumerate(used_ordered)
        }

        if any(renumber[meta] != meta for meta in used_ordered):
            remapped_sequences: List[List[str]] = []
            for seq in compressed_sequences:
                remapped_sequences.append([renumber.get(tok, tok) for tok in seq])
            compressed_sequences = remapped_sequences

        dictionary: MetaDictionary = {}
        for pattern, meta in pattern_to_meta.items():
            if meta not in used_meta:
                continue
            final_name = renumber.get(meta, meta)
            dictionary[final_name] = list(pattern)

        return MetaCompressionResult(compressed_sequences, dictionary)


def encode_linearized_token(kind: str, value: str | None, metadata: Mapping[str, object]) -> str:
    """Return a canonical representation for a linearized token."""

    meta_repr = ""
    if metadata:
        items = sorted(metadata.items())
        meta_repr = "{" + ",".join(f"{k}:{repr(v)}" for k, v in items) + "}"
    value_repr = "" if value is None else value
    return f"{kind}|{value_repr}|{meta_repr}"


def encode_linearized_sequence(tokens: Sequence[Mapping[str, object] | object]) -> List[str]:
    """Convert an iterable of ``LinearizedToken`` objects into canonical strings."""

    encoded: List[str] = []
    for token in tokens:
        kind = getattr(token, "kind", None)
        value = getattr(token, "value", None)
        metadata = getattr(token, "metadata", {})
        if kind is None:
            raise ValueError("Linearized tokens must provide a 'kind' attribute")
        if not isinstance(metadata, Mapping):
            raise TypeError("Token metadata must be a mapping")
        encoded.append(encode_linearized_token(kind, value, metadata))
    return encoded
