import pytest

torch = pytest.importorskip("torch")

import gpu_tokenizer.triton_kernels as triton_kernels
import gpu_tokenizer.utils as utils
from gpu_tokenizer.utils import (
    _count_pairs_pytorch,
    aggregate_pair_keys,
    count_pairs,
    reduce_pair_histograms,
)


def _allocate_pair_buffers(seqs: torch.Tensor):
    B, L = seqs.shape
    capacity = int(B * max(L - 1, 0))
    device = seqs.device
    dtype = seqs.dtype
    pair_keys_buffer = torch.empty((capacity, 2), dtype=dtype, device=device)
    pair_counts_buffer = torch.empty((capacity,), dtype=torch.int32, device=device)
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
            torch.empty((0,), dtype=torch.int32),
        )

    ordered_items = sorted(counts.items())
    pairs = torch.tensor([p for p, _ in ordered_items], dtype=seqs.dtype)
    freq = torch.tensor([c for _, c in ordered_items], dtype=torch.int32)
    return pairs, freq


@pytest.mark.parametrize("dtype", [torch.int32, torch.long])
def test_count_pairs_matches_cpu_reference(dtype):
    torch.manual_seed(0)
    seqs = torch.randint(0, 512, (4, 32), dtype=dtype)
    valid = torch.randint(0, 2, (4, 32), dtype=torch.uint8)

    pairs, counts, _, _, _ = _run_count_pairs(seqs, valid)
    ref_pairs, ref_counts = _cpu_reference_count_pairs(seqs.cpu(), valid.cpu())

    assert torch.equal(pairs.cpu(), ref_pairs)
    assert torch.equal(counts.cpu(), ref_counts)


def test_count_pairs_device_matches_input():
    torch.manual_seed(0)
    seqs = torch.randint(0, 100, (2, 5), dtype=torch.long)
    valid = torch.ones_like(seqs, dtype=torch.uint8)

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


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_kernels.is_triton_available(),
    reason="CUDA and Triton are required for GPU parity",
)
def test_count_pairs_triton_matches_cpu_reference():
    torch.manual_seed(1)
    seqs = torch.randint(0, 1024, (8, 257), dtype=torch.int32, device="cuda")
    valid = torch.randint(0, 2, seqs.shape, dtype=torch.uint8, device="cuda")

    pair_keys_buffer, pair_counts_buffer, pair_count_length = _allocate_pair_buffers(seqs)
    count_pairs(seqs, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length)

    length = int(pair_count_length.item())
    pairs = pair_keys_buffer[:length].cpu()
    counts = pair_counts_buffer[:length].cpu()

    ref_pairs, ref_counts = _cpu_reference_count_pairs(seqs.cpu(), valid.cpu())

    assert torch.equal(pairs, ref_pairs)
    assert torch.equal(counts, ref_counts)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_kernels.is_triton_available(),
    reason="CUDA and Triton are required for performance comparison",
)
def test_count_pairs_triton_outperforms_pytorch():
    torch.manual_seed(123)
    batch, length = 64, 513
    seqs = torch.randint(0, 4096, (batch, length), dtype=torch.int32, device="cuda")
    valid = torch.ones_like(seqs, dtype=torch.uint8)

    buffers = _allocate_pair_buffers(seqs)

    count_pairs(seqs, valid, *buffers)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def _time_call(fn):
        durations = []
        for _ in range(5):
            start.record()
            fn(seqs, valid, *buffers)
            end.record()
            torch.cuda.synchronize()
            durations.append(start.elapsed_time(end))
        return sum(durations) / len(durations)

    triton_ms = _time_call(count_pairs)
    pytorch_ms = _time_call(_count_pairs_pytorch)

    assert triton_ms < pytorch_ms


def test_count_pairs_saturates_and_logs(monkeypatch, caplog):
    monkeypatch.setattr(utils, "INT32_MAX", 5)
    caplog.set_level("WARNING", logger="gpu_tokenizer.utils")

    seqs = torch.ones((1, 12), dtype=torch.int32)
    valid = torch.ones_like(seqs, dtype=torch.uint8)

    pairs, counts, _, _, _ = _run_count_pairs(seqs, valid)

    assert counts.dtype == torch.int32
    assert counts.tolist() == [5]
    assert any("count_pairs" in record.message for record in caplog.records)


def test_aggregate_pair_keys_saturates(monkeypatch, caplog):
    monkeypatch.setattr(utils, "INT32_MAX", 5)
    caplog.set_level("WARNING", logger="gpu_tokenizer.utils")

    keys = torch.tensor([1, 1, 1], dtype=torch.long)
    counts = torch.tensor([4, 4, 4], dtype=torch.int64)

    aggregated_keys, aggregated_counts = aggregate_pair_keys(keys, counts)

    assert aggregated_keys.tolist() == [1]
    assert aggregated_counts.dtype == torch.int32
    assert aggregated_counts.tolist() == [5]
    assert any("aggregate_pair_keys" in record.message for record in caplog.records)


def test_reduce_pair_histograms_matches_clipped_reference(monkeypatch, caplog):
    monkeypatch.setattr(utils, "INT32_MAX", 5)
    caplog.set_level("WARNING", logger="gpu_tokenizer.utils")

    keys = torch.tensor([1, 2, 2, 3], dtype=torch.long)
    counts = torch.tensor([3, 4, 4, 7], dtype=torch.int64)

    reduced_keys, reduced_counts = reduce_pair_histograms(keys, counts)

    assert reduced_keys.tolist() == [1, 2, 3]
    assert reduced_counts.dtype == torch.int32
    assert reduced_counts.tolist() == [3, 5, 5]
    assert any("aggregate_pair_keys" in record.message for record in caplog.records)
