from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from gpu_tokenizer.cpu_packer import BytePacker
from gpu_tokenizer.export import artifacts as export_artifacts
from gpu_tokenizer.code_mode import prepare_corpus

_eval = importlib.import_module("gpu_tokenizer.evaluate")

_DATA_ROOT = Path("tests/data")
_CORPUS_DIR = _DATA_ROOT / "evaluate_corpus"
_MODELS_DIR = _DATA_ROOT / "models" / "bpe"

_BYTE_ENCODER = export_artifacts.byte_level_encoder()
_BYTE_DECODER = export_artifacts.byte_level_decoder()
_VOCAB = export_artifacts.load_vocab(_MODELS_DIR / "vocab.json")
_ID_TO_TOKEN = {int(idx): token for token, idx in _VOCAB.items()}
_MERGES = _eval._load_merges(_MODELS_DIR / "merges.json", _VOCAB)  # type: ignore[attr-defined]


@pytest.mark.parametrize("shard", sorted(_CORPUS_DIR.glob("*.txt")), ids=lambda p: p.name)
def test_plain_corpus_roundtrip(shard: Path) -> None:
    packer = BytePacker()
    for raw_line in shard.read_bytes().splitlines():
        if not raw_line:
            continue
        tokens = list(packer.encode_view(raw_line))
        merged = _eval._apply_merges(tokens, _MERGES)  # type: ignore[attr-defined]
        decoded = _decode_tokens(merged)
        assert decoded == raw_line


def test_code_corpus_roundtrip_bytes() -> None:
    entries = _load_code_entries()
    packer = BytePacker()
    for entry in entries:
        source = entry.get("source")
        assert isinstance(source, str)
        payload = source.encode("utf-8")
        tokens = list(packer.encode_view(payload))
        merged = _eval._apply_merges(tokens, _MERGES)  # type: ignore[attr-defined]
        decoded = _decode_tokens(merged)
        assert decoded == payload


def test_code_corpus_fallback_samples_match_sources() -> None:
    entries = _load_code_entries()
    corpus = prepare_corpus(entries, meta_enabled=True, meta_max_length=6)

    fallback_indices = [idx for idx, sample in enumerate(corpus.samples) if sample.kind == "bytes"]
    assert fallback_indices, "expected at least one fallback sample"

    for index, sample in enumerate(corpus.samples):
        source = entries[index].get("source")
        assert isinstance(source, str)
        payload = source.encode("utf-8")
        if sample.kind == "bytes":
            assert bytes(sample.tokens) == payload
        else:
            assert not sample.metadata.get("fallback", False)


def _load_code_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    with (_CORPUS_DIR / "code.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            entries.append(json.loads(payload))
    return entries


def _decode_tokens(tokens: list[int]) -> bytes:
    output = bytearray()
    for token in tokens:
        token_id = int(token)
        text = _ID_TO_TOKEN.get(token_id)
        if text is None:
            text = _BYTE_ENCODER.get(token_id)
        if text is None:
            raise AssertionError(f"token id {token_id} missing from vocab and byte encoder")
        for char in text:
            decoded = _BYTE_DECODER.get(char)
            if decoded is not None:
                output.append(decoded)
            else:
                output.extend(char.encode("utf-8"))
    return bytes(output)
