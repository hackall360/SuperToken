"""Tests for exporting GPU unigram models to SentencePiece format."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pytest

sentencepiece = pytest.importorskip("sentencepiece")
torch = pytest.importorskip("torch")

from gpu_tokenizer.unigram_trainer import GPUUnigramTrainer


def _as_batch(byte_sequences: Sequence[bytes]) -> list[torch.Tensor]:
    if not byte_sequences:
        return []
    width = max(max(len(seq) for seq in byte_sequences), 1)
    tensor = torch.full((len(byte_sequences), width), -1, dtype=torch.int32)
    for row, seq in enumerate(byte_sequences):
        if not seq:
            continue
        tensor[row, : len(seq)] = torch.tensor(list(seq), dtype=torch.int32)
    return [tensor]


def _latin1_strings(byte_sequences: Iterable[bytes]) -> list[str]:
    return [seq.decode("latin-1") for seq in byte_sequences]


def _normalize_piece(piece: str) -> bytes:
    if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
        return bytes([int(piece[3:5], 16)])
    return piece.replace("▁", " ").encode("latin-1")


def _viterbi_decode(trainer: GPUUnigramTrainer, sequence: Sequence[int]) -> list[int]:
    vocab = [trainer.id2piece[idx] for idx in range(len(trainer.id2piece))]
    log_probs = trainer.logp.detach().cpu().tolist()
    length = len(sequence)

    best = [float("-inf")] * (length + 1)
    back_index = [-1] * (length + 1)
    back_piece = [-1] * (length + 1)
    best[0] = 0.0

    for start in range(length):
        current = best[start]
        if current == float("-inf"):
            continue
        for pid, piece in enumerate(vocab):
            piece_len = len(piece)
            if piece_len == 0 or start + piece_len > length:
                continue
            if list(piece) != list(sequence[start : start + piece_len]):
                continue
            score = current + log_probs[pid]
            end = start + piece_len
            if score > best[end]:
                best[end] = score
                back_index[end] = start
                back_piece[end] = pid

    if best[length] == float("-inf"):
        return list(sequence)

    ids: list[int] = []
    position = length
    while position > 0:
        prev = back_index[position]
        if prev < 0:
            prev = position - 1
            piece_id = sequence[prev]
        else:
            piece_id = back_piece[position]
        ids.append(piece_id)
        position = prev
    ids.reverse()
    return ids


@pytest.mark.parametrize(
    "samples",
    [
        [b"banana", b"bandana", b"banana band", bytes(range(32))],
        [b"hello world", b"low", b"helloworld"],
    ],
)
def test_sentencepiece_export_round_trip(tmp_path: Path, samples: list[bytes]) -> None:
    batches = _as_batch(samples + [bytes(range(256))])
    trainer = GPUUnigramTrainer(
        base_vocab=256,
        vocab_size=256 + 32,
        max_subword_len=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    if batches:
        trainer.fit_epoch(batches)

    model_path = trainer.save(tmp_path)
    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    log_probs = trainer.logp.detach().cpu().tolist()

    assert processor.get_piece_size() == len(trainer.id2piece)

    for idx in range(len(trainer.id2piece)):
        exported_piece = _normalize_piece(processor.id_to_piece(idx))
        assert exported_piece == trainer.id2piece[idx]
        assert processor.get_score(idx) == pytest.approx(log_probs[idx], abs=1e-6, rel=1e-6)

    for sample in samples:
        encoded = processor.encode(_latin1_strings([sample])[0], out_type=int)
        expected = _viterbi_decode(trainer, list(sample))
        assert encoded == expected
