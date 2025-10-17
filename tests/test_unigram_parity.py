"""Parity checks between the GPU unigram trainer and SentencePiece.

This module mirrors the approach used for the BPE parity tests but targets
the unigram trainer.  The reference implementation is SentencePiece's Python
API which we configure to mirror the GPU trainer's byte-oriented assumptions.
The tests rely on a tightly controlled seeding strategy so that both
implementations resolve tie-breakers deterministically on every platform.

The corpora are materialized as raw byte sequences, decoded via ``latin-1`` for
SentencePiece so that every byte value survives the round-trip without
normalization.  A synthetic sentence containing all byte values is appended to
the training data to ensure the reference model assigns probabilities to the
entire base vocabulary, matching the GPU trainer's unconditional byte coverage.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, cast

import pytest

sentencepiece = pytest.importorskip("sentencepiece")
torch = pytest.importorskip("torch")

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

try:  # NumPy is optional, but we seed it when available for completeness.
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - exercised only when NumPy is absent.
    _np = None

from gpu_tokenizer.unigram_trainer import GPUUnigramTrainer
from tests.adversarial_corpora import AdversarialCorpus, get_adversarial_corpora


BASE_VOCAB = 256
TARGET_VOCAB = 320
MAX_SUBWORD_LEN = 8
GLOBAL_SEED = 2024_06_17


@dataclass(frozen=True)
class _SentencePieceArtifacts:
    vocab: List[bytes]
    log_probs: List[float]
    encoded: List[List[int]]


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs (CPU and CUDA when present).

    SentencePiece, PyTorch, and Python use independent PRNGs.  The parity tests
    reset all of them for each training run so that sampling-based tie breakers
    produce reproducible vocabularies and log probabilities.
    """

    random.seed(seed)
    if _np is not None:
        _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_byte_sequences(samples: Sequence[str]) -> List[bytes]:
    return [sample.encode("utf-8") for sample in samples]


def _latin1_strings(byte_sequences: Sequence[bytes]) -> List[str]:
    return [seq.decode("latin-1") for seq in byte_sequences]


def _build_batches(byte_sequences: Sequence[bytes]) -> List[torch.Tensor]:
    if not byte_sequences:
        return []

    width = max(1, max((len(seq) for seq in byte_sequences), default=0))
    tokens = torch.full((len(byte_sequences), width), -1, dtype=torch.int32)
    for row, seq in enumerate(byte_sequences):
        if not seq:
            continue
        tokens[row, : len(seq)] = torch.tensor(list(seq), dtype=torch.int32)
    return [tokens]


def _normalize_sentencepiece_piece(piece: str) -> bytes:
    """Convert a SentencePiece token into the byte representation the project expects."""

    if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
        return bytes([int(piece[3:5], 16)])
    normalized = piece.replace("▁", " ")
    try:
        return normalized.encode("latin-1")
    except UnicodeEncodeError:
        # ``latin-1`` is sufficient for byte preservation, but we fall back to
        # UTF-8 for completeness in the unlikely event SentencePiece emits
        # arbitrary Unicode outside the Latin-1 range.
        return normalized.encode("utf-8")


def _train_sentencepiece(
    training_sentences: Sequence[str],
    eval_sentences: Sequence[str],
    *,
    vocab_size: int,
    max_subword_len: int,
    seed: int,
) -> _SentencePieceArtifacts:
    model_buffer = io.BytesIO()

    class _Iterator:
        def __iter__(self) -> Iterable[str]:
            return iter(training_sentences)

    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=_Iterator(),
        model_writer=model_buffer,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        split_by_whitespace=False,
        remove_extra_whitespaces=False,
        byte_fallback=True,
        hard_vocab_limit=True,
        max_sentencepiece_length=max_subword_len,
        shuffle_input_sentence=False,
        seed_sentencepiece_size=0,
        input_sentence_size=0,
        random_seed=seed,
        pad_id=-1,
        unk_id=-1,
        bos_id=-1,
        eos_id=-1,
    )

    model_buffer.seek(0)
    processor = sentencepiece.SentencePieceProcessor(model_proto=model_buffer.read())

    vocab: List[bytes] = []
    scores: List[float] = []
    for idx in range(processor.get_piece_size()):
        piece = processor.id_to_piece(idx)
        vocab.append(_normalize_sentencepiece_piece(piece))
        scores.append(processor.get_score(idx))

    encoded = [processor.encode(sample, out_type=int) for sample in eval_sentences]
    return _SentencePieceArtifacts(vocab=vocab, log_probs=scores, encoded=encoded)


def _trainer_vocab_and_log_probs(trainer: GPUUnigramTrainer) -> tuple[List[bytes], List[float]]:
    vocab = [trainer.id2piece[idx] for idx in range(len(trainer.id2piece))]
    log_probs = trainer.logp.detach().cpu().tolist()
    return vocab, log_probs


def _viterbi_decode(trainer: GPUUnigramTrainer, sequence: Sequence[int]) -> List[int]:
    pieces = [trainer.id2piece[idx] for idx in range(len(trainer.id2piece))]
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
        for piece_id, piece in enumerate(pieces):
            piece_len = len(piece)
            if piece_len == 0 or start + piece_len > length:
                continue
            if list(piece) != list(sequence[start : start + piece_len]):
                continue
            score = current + log_probs[piece_id]
            end = start + piece_len
            if score > best[end]:
                best[end] = score
                back_index[end] = start
                back_piece[end] = piece_id

    if best[length] == float("-inf"):
        return list(sequence)

    ids: List[int] = []
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


def _train_unigram_trainer(
    training_sequences: Sequence[bytes],
    eval_sequences: Sequence[bytes],
    *,
    base_vocab: int,
    target_vocab: int,
    max_subword_len: int,
    seed: int,
    device: str,
) -> tuple[GPUUnigramTrainer, List[bytes], List[float], List[List[int]]]:
    _seed_everything(seed)
    trainer = GPUUnigramTrainer(
        base_vocab=base_vocab,
        vocab_size=target_vocab,
        max_subword_len=max_subword_len,
        device=device,
        seed=seed,
    )

    batches = _build_batches(training_sequences)
    if batches:
        trainer.fit_epoch(batches)

    vocab, log_probs = _trainer_vocab_and_log_probs(trainer)
    encoded = [_viterbi_decode(trainer, list(seq)) for seq in eval_sequences]
    return trainer, vocab, log_probs, encoded


@pytest.mark.parametrize(
    "corpus",
    get_adversarial_corpora(),
    ids=lambda corpus: corpus.name,
)
def test_unigram_matches_sentencepiece(corpus: AdversarialCorpus) -> None:
    """Ensure both CPU and GPU unigram trainers mirror SentencePiece."""

    corpus_bytes = _as_byte_sequences(corpus.corpus)
    sentinel_bytes = bytes(range(BASE_VOCAB))
    training_bytes = corpus_bytes + [sentinel_bytes]

    training_sentences = _latin1_strings(training_bytes)
    eval_sentences = _latin1_strings(corpus_bytes)

    reference = _train_sentencepiece(
        training_sentences,
        eval_sentences,
        vocab_size=TARGET_VOCAB,
        max_subword_len=MAX_SUBWORD_LEN,
        seed=GLOBAL_SEED,
    )

    results: dict[str, dict[str, object]] = {}
    for device in DEVICES:
        trainer_a, vocab_a, logp_a, encoded_a = _train_unigram_trainer(
            training_bytes,
            corpus_bytes,
            base_vocab=BASE_VOCAB,
            target_vocab=TARGET_VOCAB,
            max_subword_len=MAX_SUBWORD_LEN,
            seed=GLOBAL_SEED,
            device=device,
        )
        trainer_b, vocab_b, logp_b, _ = _train_unigram_trainer(
            training_bytes,
            corpus_bytes,
            base_vocab=BASE_VOCAB,
            target_vocab=TARGET_VOCAB,
            max_subword_len=MAX_SUBWORD_LEN,
            seed=GLOBAL_SEED,
            device=device,
        )
        assert vocab_a == vocab_b
        assert logp_a == pytest.approx(logp_b, abs=1e-7, rel=1e-7)
        results[device] = {
            "trainer": trainer_a,
            "vocab": vocab_a,
            "logp": logp_a,
            "encoded": encoded_a,
        }

    if "cuda" in results and "cpu" in results:
        cpu_res = results["cpu"]
        gpu_res = results["cuda"]
        assert cpu_res["vocab"] == gpu_res["vocab"]
        assert cpu_res["logp"] == pytest.approx(gpu_res["logp"], abs=1e-6, rel=1e-6)
        assert cpu_res["encoded"] == gpu_res["encoded"]
        eval_batch = _build_batches(corpus_bytes)[0]
        mask = eval_batch >= 0
        cpu_trainer = cast(GPUUnigramTrainer, cpu_res["trainer"])
        gpu_trainer = cast(GPUUnigramTrainer, gpu_res["trainer"])
        cpu_logZ, _ = cpu_trainer._forward_backward_batch(eval_batch, mask)
        gpu_batch = eval_batch.to(gpu_trainer.device)
        gpu_mask = mask.to(gpu_trainer.device)
        gpu_logZ, _ = gpu_trainer._forward_backward_batch(gpu_batch, gpu_mask)
        assert torch.allclose(gpu_logZ.cpu(), cpu_logZ, atol=1e-5, rtol=1e-5)

    for payload in results.values():
        vocab = payload["vocab"]
        logp = payload["logp"]
        encoded = payload["encoded"]
        assert vocab == reference.vocab
        assert logp == pytest.approx(reference.log_probs, abs=1e-5, rel=1e-5)
        assert encoded == reference.encoded


@pytest.mark.parametrize("device", DEVICES)
def test_unigram_seed_reproducibility(device: str) -> None:
    """Multiple runs with the same seed should stay in lock-step across epochs."""

    seed = GLOBAL_SEED
    training_sequences = [
        b"banana bandana",
        b"bandana banana",
        b"ananas banana",
        bytes(range(32)),
    ]
    eval_sequences = training_sequences[:-1]
    batches = _build_batches(training_sequences)

    def _run() -> tuple[list[list[bytes]], list[bytes], dict[bytes, int], list[list[int]]]:
        _seed_everything(seed)
        trainer = GPUUnigramTrainer(
            base_vocab=BASE_VOCAB,
            vocab_size=BASE_VOCAB + 128,
            max_subword_len=MAX_SUBWORD_LEN,
            device=device,
            seed=seed,
        )

        candidate_history: list[list[bytes]] = []
        vocab_history: list[dict[bytes, int]] = []
        epoch_seeds = [None, seed + 1, None]
        for epoch_seed in epoch_seeds:
            if epoch_seed is None:
                trainer.reset_rng()
            else:
                trainer.reset_rng(seed=epoch_seed)
            trainer.fit_epoch(batches)
            candidate_history.append(
                [trainer.id2piece[idx] for idx in range(BASE_VOCAB, len(trainer.id2piece))]
            )
            vocab_history.append(dict(trainer.piece2id))

        vocab = [trainer.id2piece[idx] for idx in range(len(trainer.id2piece))]
        encoded = [_viterbi_decode(trainer, list(seq)) for seq in eval_sequences]
        return candidate_history, vocab, vocab_history[-1], encoded

    history_a, vocab_a, table_a, encoded_a = _run()
    history_b, vocab_b, table_b, encoded_b = _run()

    assert history_a == history_b
    assert vocab_a == vocab_b
    assert table_a == table_b
    assert encoded_a == encoded_b
