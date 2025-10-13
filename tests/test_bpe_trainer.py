import pytest

torch = pytest.importorskip("torch")

from gpu_tokenizer.bpe_trainer import _aggregate_pair_keys


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
