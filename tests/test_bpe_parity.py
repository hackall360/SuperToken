"""Deterministic parity checks between the GPU BPE trainer and tokenizers.

This module exercises the adversarial corpora against Hugging Face's
``tokenizers`` implementation and the project-local :class:`GPUBPETrainer`.
The tests rely on a shared seed for Python's ``random`` module, NumPy, and
PyTorch (including CUDA when available) in addition to the trainer's own seed
parameter.  The explicit seeding ensures that tie-breaking behavior inside the
reference trainer remains reproducible across platforms.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import pytest

np = pytest.importorskip("numpy")
tokenizers = pytest.importorskip("tokenizers")
torch = pytest.importorskip("torch")

from gpu_tokenizer.bpe_trainer import GPUBPETrainer
from gpu_tokenizer.utils import apply_merge_once
from tests.adversarial_corpora import AdversarialCorpus, get_adversarial_corpora


BASE_VOCAB = 256
GLOBAL_SEED = 1337

BYTE_DECODER = tokenizers.decoders.ByteLevel()


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA) RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _corpus_to_byte_sequences(samples: Sequence[str]) -> List[List[int]]:
    """Encode each UTF-8 sample as a list of byte integers."""

    return [list(sample.encode("utf-8")) for sample in samples]


def _build_tensor_batches(byte_sequences: Sequence[Sequence[int]]):
    """Create a single deterministic batch suitable for ``GPUBPETrainer``."""

    if not byte_sequences:
        return []

    rows = len(byte_sequences)
    width = max(1, max((len(seq) for seq in byte_sequences), default=0))
    tokens = torch.full((rows, width), -1, dtype=torch.long)
    valid = torch.zeros((rows, width), dtype=torch.long)
    lengths = torch.zeros((rows,), dtype=torch.long)
    for row, seq in enumerate(byte_sequences):
        if not seq:
            continue
        length = len(seq)
        tokens[row, :length] = torch.tensor(seq, dtype=torch.long)
        valid[row, :length] = 1
        lengths[row] = length
    return [(tokens, valid, lengths)]


def _huggingface_vocab_bytes(tokenizer: tokenizers.Tokenizer) -> List[bytes]:
    """Return the tokenizer vocabulary ordered by ID as byte sequences."""

    size = tokenizer.get_vocab_size()
    vocab_bytes: List[bytes] = []
    for idx in range(size):
        token = tokenizer.id_to_token(idx)
        decoded = BYTE_DECODER.decode([token])
        vocab_bytes.append(decoded.encode("utf-8"))
    return vocab_bytes


def _gpu_vocab_bytes(merges: Sequence[Tuple[int, int]]) -> List[bytes]:
    """Reconstruct GPU token byte sequences from the merge table."""

    vocab: List[bytes] = [bytes([token_id]) for token_id in range(BASE_VOCAB)]
    for idx, (a_id, b_id) in enumerate(merges):
        new_id = BASE_VOCAB + idx
        if len(vocab) <= new_id:
            vocab.append(b"")
        vocab[new_id] = vocab[a_id] + vocab[b_id]
    return vocab


def _encode_with_merges(
    byte_sequences: Sequence[Sequence[int]], merges: Sequence[Tuple[int, int]]
) -> List[List[int]]:
    """Apply the merge table to produce token ID sequences."""

    if not byte_sequences:
        return []

    batch = _build_tensor_batches(byte_sequences)
    if not batch:
        return []

    tokens, valid, lengths = batch[0]
    tokens = tokens.clone()
    valid = valid.clone()
    lengths = lengths.clone()

    for idx, (a_id, b_id) in enumerate(merges):
        apply_merge_once(
            tokens,
            valid,
            lengths,
            int(a_id),
            int(b_id),
            BASE_VOCAB + idx,
        )

    encoded: List[List[int]] = []
    for row in range(tokens.size(0)):
        length = int(lengths[row].item())
        if length <= 0:
            encoded.append([])
        else:
            encoded.append(tokens[row, :length].tolist())
    return encoded


def _format_bytes(value: bytes) -> str:
    return value.hex() or "<empty>"


def _first_mismatch(
    lhs: Sequence, rhs: Sequence
) -> Tuple[int, object, object] | Tuple[None, None, None]:
    if len(lhs) != len(rhs):
        return len(lhs), None, None
    for idx, (a_val, b_val) in enumerate(zip(lhs, rhs)):
        if a_val != b_val:
            return idx, a_val, b_val
    return None, None, None


def _assert_byte_pairs_equal(
    label: str,
    actual: Sequence[Tuple[bytes, bytes]],
    expected: Sequence[Tuple[bytes, bytes]],
) -> None:
    if actual == expected:
        return
    idx, actual_pair, expected_pair = _first_mismatch(actual, expected)
    if idx is None:
        raise AssertionError(f"{label} length mismatch: {len(actual)} != {len(expected)}")
    actual_str = (
        f"({_format_bytes(actual_pair[0])}, {_format_bytes(actual_pair[1])})"
        if actual_pair is not None
        else "<missing>"
    )
    expected_str = (
        f"({_format_bytes(expected_pair[0])}, {_format_bytes(expected_pair[1])})"
        if expected_pair is not None
        else "<missing>"
    )
    raise AssertionError(
        f"{label} differ at index {idx}: actual={actual_str} expected={expected_str}"
    )


def _assert_bytes_equal(label: str, actual: Sequence[bytes], expected: Sequence[bytes]) -> None:
    if actual == expected:
        return
    idx, actual_val, expected_val = _first_mismatch(actual, expected)
    if idx is None:
        raise AssertionError(f"{label} length mismatch: {len(actual)} != {len(expected)}")
    actual_str = _format_bytes(actual_val if actual_val is not None else b"")
    expected_str = _format_bytes(expected_val if expected_val is not None else b"")
    raise AssertionError(
        f"{label} differ at index {idx}: actual={actual_str} expected={expected_str}"
    )


def _assert_nested_bytes_equal(
    label: str,
    actual: Sequence[Sequence[bytes]],
    expected: Sequence[Sequence[bytes]],
) -> None:
    if actual == expected:
        return
    if len(actual) != len(expected):
        raise AssertionError(f"{label} length mismatch: {len(actual)} != {len(expected)}")
    for row, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        if actual_row == expected_row:
            continue
        idx, actual_val, expected_val = _first_mismatch(actual_row, expected_row)
        actual_str = _format_bytes(actual_val if actual_val is not None else b"")
        expected_str = _format_bytes(expected_val if expected_val is not None else b"")
        raise AssertionError(
            f"{label} differ for sample {row} at token {idx}: actual={actual_str} expected={expected_str}"
        )


def _train_reference_tokenizer(
    corpus: AdversarialCorpus, merge_budget: int
) -> dict[str, object]:
    _seed_everything(GLOBAL_SEED)
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = BYTE_DECODER
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=BASE_VOCAB + merge_budget,
        min_frequency=2,
        show_progress=False,
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        seed=GLOBAL_SEED,
    )
    tokenizer.train_from_iterator(corpus.corpus, trainer=trainer)
    vocab_bytes = _huggingface_vocab_bytes(tokenizer)
    merges_ids: List[Tuple[int, int]] = [
        (tokenizer.token_to_id(a_token), tokenizer.token_to_id(b_token))
        for (a_token, b_token) in tokenizer.model.get_merges()
    ]
    merges_bytes = [
        (vocab_bytes[a_id], vocab_bytes[b_id])
        for (a_id, b_id) in merges_ids
    ]
    encoded_ids = [tokenizer.encode(sample).ids for sample in corpus.corpus]
    encoded_bytes = [[vocab_bytes[idx] for idx in seq] for seq in encoded_ids]
    return {
        "merges_ids": merges_ids,
        "merges_bytes": merges_bytes,
        "vocab_bytes": vocab_bytes,
        "encoded_ids": encoded_ids,
        "encoded_bytes": encoded_bytes,
    }


def _train_gpu_trainer(
    byte_sequences: Sequence[Sequence[int]], merge_budget: int, device: str
) -> GPUBPETrainer:
    _seed_everything(GLOBAL_SEED)
    trainer = GPUBPETrainer(base_vocab=BASE_VOCAB, merges=merge_budget, device=device)
    batches = _build_tensor_batches(byte_sequences)
    trainer.fit(batches, log_every=max(merge_budget, 1))
    return trainer


_CORPORA = list(get_adversarial_corpora())


@pytest.mark.parametrize("corpus", _CORPORA, ids=[corpus.name for corpus in _CORPORA])
def test_gpu_trainer_matches_huggingface(corpus: AdversarialCorpus) -> None:
    reference = _train_reference_tokenizer(corpus, corpus.target_merge_operations)
    byte_sequences = _corpus_to_byte_sequences(corpus.corpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    first_trainer = _train_gpu_trainer(byte_sequences, corpus.target_merge_operations, device)
    first_merges = list(first_trainer.merges)
    gpu_vocab_bytes = _gpu_vocab_bytes(first_merges)
    gpu_merges_bytes = [(gpu_vocab_bytes[a_id], gpu_vocab_bytes[b_id]) for (a_id, b_id) in first_merges]
    encoded_gpu_ids = _encode_with_merges(byte_sequences, first_merges)
    encoded_gpu_bytes = [[gpu_vocab_bytes[idx] for idx in seq] for seq in encoded_gpu_ids]

    _assert_byte_pairs_equal("Merge bytes", gpu_merges_bytes, reference["merges_bytes"])
    _assert_bytes_equal("Vocabulary bytes", gpu_vocab_bytes, reference["vocab_bytes"])
    assert encoded_gpu_ids == reference["encoded_ids"], "Encoded token IDs diverged"
    _assert_nested_bytes_equal("Encoded token bytes", encoded_gpu_bytes, reference["encoded_bytes"])

    second_trainer = _train_gpu_trainer(byte_sequences, corpus.target_merge_operations, device)
    second_merges = list(second_trainer.merges)
    second_encoded_ids = _encode_with_merges(byte_sequences, second_merges)

    assert second_merges == first_merges, "GPUBPETrainer merges were not deterministic"
    assert (
        second_encoded_ids == encoded_gpu_ids
    ), "GPUBPETrainer encodings were not deterministic"
