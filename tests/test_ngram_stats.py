from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Tuple

import pytest

torch = pytest.importorskip("torch")
if getattr(torch, "_SUPERTOKEN_TORCH_STUB", False) or not hasattr(torch, "tensor"):
    pytest.skip(
        "PyTorch with tensor factories is required for n-gram statistics tests",
        allow_module_level=True,
    )

from gpu_tokenizer.ngram_stats import compute_ngram_histograms


def _cpu_reference_histograms(
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    max_order: int,
    bits_per_symbol: int,
) -> Dict[int, Dict[int, int]]:
    histograms: Dict[int, Dict[int, int]] = {n: defaultdict(int) for n in range(1, max_order + 1)}

    for tokens, valid in batches:
        tokens_cpu = tokens.to(torch.int64)
        valid_cpu = valid.to(torch.bool)
        B, width = tokens_cpu.shape
        for b in range(B):
            for order in range(1, max_order + 1):
                if width < order:
                    break
                for start in range(width - order + 1):
                    if torch.all(valid_cpu[b, start : start + order]):
                        values = tokens_cpu[b, start : start + order]
                        key = 0
                        for idx, val in enumerate(values):
                            shift = bits_per_symbol * (order - 1 - idx)
                            key |= int(val.item()) << shift
                        histograms[order][key] += 1

    return histograms


@pytest.mark.parametrize(
    "device",
    [torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")],
)
def test_compute_ngram_histograms_matches_cpu(device: torch.device) -> None:
    tokens1 = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
        ],
        dtype=torch.int32,
    )
    valid1 = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1, 1],
        ],
        dtype=torch.uint8,
    )

    tokens2 = torch.tensor([[30, 31, 32, 33, 34, 35]], dtype=torch.int32)
    valid2 = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.uint8)

    batches = [
        (tokens1, valid1, torch.tensor([4, 5], dtype=torch.int32)),
        (tokens2, valid2, torch.tensor([6], dtype=torch.int32)),
    ]

    histograms = compute_ngram_histograms(batches, max_order=4, device=device)

    bits_per_symbol = max(16, 64 // 4)
    cpu_histograms = _cpu_reference_histograms(
        [(tokens1, valid1), (tokens2, valid2)], max_order=4, bits_per_symbol=bits_per_symbol
    )

    for order in (2, 3, 4):
        keys, counts = histograms[order]
        assert keys.dtype == torch.long
        assert counts.dtype == torch.int64
        assert keys.device == device
        assert counts.device == device

        observed = {int(k): int(c) for k, c in zip(keys.cpu(), counts.cpu())}
        assert observed == cpu_histograms[order]


def test_compute_ngram_histograms_respects_padding() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokens = torch.tensor([[7, 8, 9, 10, 11]], dtype=torch.int32)
    valid = torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.uint8)

    histograms = compute_ngram_histograms([(tokens, valid)], max_order=4, device=device)

    # Only the first bigram is valid; no trigrams or 4-grams should be emitted.
    bigram_keys, bigram_counts = histograms[2]
    assert bigram_keys.numel() == 1
    assert torch.equal(bigram_counts.cpu(), torch.tensor([1], dtype=torch.int64))

    trigram_keys, trigram_counts = histograms[3]
    assert trigram_keys.numel() == 0
    assert trigram_counts.numel() == 0

    fourgram_keys, fourgram_counts = histograms[4]
    assert fourgram_keys.numel() == 0
    assert fourgram_counts.numel() == 0
