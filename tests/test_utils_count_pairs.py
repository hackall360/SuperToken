import pytest

torch = pytest.importorskip("torch")

from gpu_tokenizer.utils import count_pairs


def _allocate_pair_buffers(seqs: torch.Tensor):
    B, L = seqs.shape
    capacity = int(B * max(L - 1, 0))
    device = seqs.device
    dtype = seqs.dtype
    pair_keys_buffer = torch.empty((capacity, 2), dtype=dtype, device=device)
    pair_counts_buffer = torch.empty((capacity,), dtype=torch.long, device=device)
    pair_count_length = torch.zeros((1,), dtype=torch.long, device=device)
    return pair_keys_buffer, pair_counts_buffer, pair_count_length


def _run_count_pairs(seqs: torch.Tensor, valid: torch.Tensor):
    pair_keys_buffer, pair_counts_buffer, pair_count_length = _allocate_pair_buffers(seqs)
    count_pairs(seqs, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length)
    length = int(pair_count_length.item())
    pairs = pair_keys_buffer[:length].clone()
    counts = pair_counts_buffer[:length].clone()
    return pairs, counts, pair_keys_buffer, pair_counts_buffer, pair_count_length


def _cpu_reference_count_pairs(seqs: torch.Tensor, valid: torch.Tensor):
    counts = {}
    B, L = seqs.shape
    for b in range(B):
        for i in range(L - 1):
            if valid[b, i].item() and valid[b, i + 1].item():
                pair = (int(seqs[b, i].item()), int(seqs[b, i + 1].item()))
                counts[pair] = counts.get(pair, 0) + 1

    if not counts:
        return (
            torch.empty((0, 2), dtype=seqs.dtype),
            torch.empty((0,), dtype=torch.long),
        )

    ordered_items = sorted(counts.items())
    pairs = torch.tensor([p for p, _ in ordered_items], dtype=seqs.dtype)
    freq = torch.tensor([c for _, c in ordered_items], dtype=torch.long)
    return pairs, freq


@pytest.mark.parametrize("dtype", [torch.int32, torch.long])
def test_count_pairs_matches_cpu_reference(dtype):
    torch.manual_seed(0)
    seqs = torch.randint(0, 512, (4, 32), dtype=dtype)
    valid = torch.randint(0, 2, (4, 32), dtype=torch.long)

    pairs, counts, _, _, _ = _run_count_pairs(seqs, valid)
    ref_pairs, ref_counts = _cpu_reference_count_pairs(seqs.cpu(), valid.cpu())

    assert torch.equal(pairs.cpu(), ref_pairs)
    assert torch.equal(counts.cpu(), ref_counts)


def test_count_pairs_device_matches_input():
    torch.manual_seed(0)
    seqs = torch.randint(0, 100, (2, 5), dtype=torch.long)
    valid = torch.ones_like(seqs, dtype=torch.long)

    pairs, counts, keys_buffer, counts_buffer, length_buffer = _run_count_pairs(seqs, valid)
    assert keys_buffer.device == seqs.device
    assert counts_buffer.device == seqs.device
    assert length_buffer.device == seqs.device
    assert pairs.device == seqs.device
    assert counts.device == seqs.device

    if torch.cuda.is_available():
        seqs_cuda = seqs.cuda()
        valid_cuda = valid.cuda()
        pairs_cuda, counts_cuda, _, _, _ = _run_count_pairs(seqs_cuda, valid_cuda)
        assert pairs_cuda.device.type == "cuda"
        assert counts_cuda.device.type == "cuda"
        assert torch.equal(pairs, pairs_cuda.cpu())
        assert torch.equal(counts, counts_cuda.cpu())
