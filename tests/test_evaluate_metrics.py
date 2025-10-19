import pytest

from gpu_tokenizer.evaluate_metrics import (
    compute_code_mode_reduction,
    compute_compression_ratio,
    compute_morphology_purity,
    compute_oov_rate,
)


def test_compression_ratio_from_scalars() -> None:
    stats = compute_compression_ratio(30, 10)

    assert stats["tokens_per_byte"] == pytest.approx(3.0)
    assert stats["bytes_per_token"] == pytest.approx(1 / 3)


def test_compression_ratio_from_sequences() -> None:
    stats = compute_compression_ratio([5, 10, 5], [4, 8, 8])

    assert stats["tokens_per_byte"] == pytest.approx(20 / 20)
    assert stats["bytes_per_token"] == pytest.approx(1.0)


def test_oov_rate_handles_zero_division() -> None:
    assert compute_oov_rate(0, 0) == 0.0
    assert compute_oov_rate(5, 25) == pytest.approx(0.2)


def test_morphology_purity_handles_empty_segments() -> None:
    assert compute_morphology_purity(0, 0) == 0.0
    assert compute_morphology_purity(7, 10) == pytest.approx(0.7)


def test_code_mode_reduction_counts_meta_tokens() -> None:
    sequences = [
        ["META0", "{" , "}"],
        ["META1", "(" , ")", "META1"],
        [1, 2, 3],
    ]
    meta = {"META0": ["def", "identifier"], "META1": ["identifier", "=", "value"]}

    reduction = compute_code_mode_reduction(sequences, meta)

    compressed = 3 + 4  # first two sequences
    expanded = (2 + 1 + 1) + (3 + 1 + 1 + 3)
    expected = 1 - (compressed / expanded)

    assert reduction == pytest.approx(expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compression_ratio_with_torch_tensors(device: str) -> None:
    torch = pytest.importorskip("torch")
    if getattr(torch, "__super_token_stub__", False):
        pytest.skip("torch stub does not implement tensor math")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")

    tokens = torch.tensor([4, 6], device=device)
    bytes_ = torch.tensor([10, 10], device=device)

    stats = compute_compression_ratio(tokens, bytes_)

    assert stats["tokens_per_byte"] == pytest.approx(1.0)
    assert stats["bytes_per_token"] == pytest.approx(1.0)
