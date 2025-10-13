import pytest

torch = pytest.importorskip("torch")

from gpu_tokenizer.bpe_trainer import GPUBPETrainer, _aggregate_pair_keys


def test_aggregate_pair_keys_repeated_counts():
    keys = torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.long)
    counts = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)

    aggregated_keys, aggregated_counts = _aggregate_pair_keys(keys, counts)

    assert torch.equal(aggregated_keys, torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(
        aggregated_counts, torch.tensor([1 + 2 + 3, 4 + 5, 6], dtype=torch.long)
    )


def test_aggregate_pair_keys_unsorted_input():
    keys = torch.tensor([2, 1, 3, 1, 2], dtype=torch.long)
    counts = torch.tensor([5, 1, 2, 3, 4], dtype=torch.long)

    aggregated_keys, aggregated_counts = _aggregate_pair_keys(keys, counts)

    assert torch.equal(aggregated_keys, torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(aggregated_counts, torch.tensor([1 + 3, 5 + 4, 2], dtype=torch.long))


class OneShotIterable:
    def __init__(self, batches):
        self._batches = batches
        self._iterated = False

    def __iter__(self):
        if self._iterated:
            raise AssertionError("Batch stream should only be iterated once")
        self._iterated = True
        for batch in self._batches:
            yield batch


def _make_batch(seqs: list[list[int]]):
    max_len = max(len(seq) for seq in seqs)
    tokens = torch.full((len(seqs), max_len), -1, dtype=torch.long)
    valid = torch.zeros((len(seqs), max_len), dtype=torch.long)
    lengths = torch.zeros(len(seqs), dtype=torch.long)
    for row, seq in enumerate(seqs):
        L = len(seq)
        if L == 0:
            continue
        tokens[row, :L] = torch.tensor(seq, dtype=torch.long)
        valid[row, :L] = 1
        lengths[row] = L
    if torch.cuda.is_available():
        tokens = tokens.pin_memory()
        valid = valid.pin_memory()
    return tokens, valid, lengths


def test_fit_supports_streaming_iterator():
    seqs = [
        [1, 2, 1, 2, 3],
        [1, 2, 4, 2],
        [1, 2, 1, 2],
        [1, 2, 1, 2],
    ]
    first_batch = _make_batch(seqs[:2])
    second_batch = _make_batch(seqs[2:])
    stream = OneShotIterable([first_batch, second_batch])

    trainer = GPUBPETrainer(base_vocab=256, merges=1, device="cpu")
    trainer.fit(stream, log_every=10)

    assert trainer.merges[:1] == [(1, 2)]
