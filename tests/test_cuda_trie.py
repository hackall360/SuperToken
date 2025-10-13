import collections

import pytest

torch = pytest.importorskip("torch")

from gpu_tokenizer.unigram_trainer import GPUUnigramTrainer


def _cpu_counts(sequences: torch.Tensor, max_len: int) -> dict[bytes, int]:
    counts: dict[bytes, int] = collections.Counter()
    B, L = sequences.shape
    for b in range(B):
        for i in range(L):
            if sequences[b, i] < 0:
                continue
            for n in range(2, max_len + 1):
                if i + n > L:
                    break
                if (sequences[b, i : i + n] < 0).any():
                    break
                piece = bytes(int(x) for x in sequences[b, i : i + n])
                counts[piece] += 1
    return counts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_gpu_trie_histogram_matches_cpu():
    trainer = GPUUnigramTrainer(max_subword_len=4, device="cuda")
    sequences = torch.tensor(
        [
            [1, 2, 3, 4, -1, -1],
            [1, 2, 2, 3, 4, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    cpu_reference = _cpu_counts(sequences.cpu(), trainer.max_len)
    trainer._extend_candidates(sequences)
    new_pieces = {piece for piece in trainer.piece2id if len(piece) > 1}
    expected = {piece for piece, count in cpu_reference.items() if count > 0}
    assert expected.issuperset(new_pieces)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_candidate_extension_respects_vocab_cap():
    trainer = GPUUnigramTrainer(base_vocab=2, vocab_size=4, max_subword_len=3, device="cuda")
    sequences = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32, device="cuda")
    trainer._extend_candidates(sequences)
    assert len(trainer.id2piece) <= trainer.target_vocab
