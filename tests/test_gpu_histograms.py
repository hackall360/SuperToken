import pytest

torch = pytest.importorskip("torch")

from gpu_tokenizer.triton_kernels import is_triton_available
from gpu_tokenizer.utils import aggregate_pair_keys, count_pairs
from gpu_tokenizer.trainers.bpe_gpu import (
    PairHistogramResult,
    combine_histogram_results,
    count_pairs_on_device,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA + Triton are required to compare GPU histograms",
)
def test_triton_histogram_matches_cpu_fallback():
    torch.manual_seed(1234)

    device = torch.device("cuda", 0)
    for batch, length in [(4, 16), (2, 7), (1, 5)]:
        tokens = torch.randint(0, 512, (batch, length), device=device, dtype=torch.int32)
        valid = torch.randint(0, 2, (batch, length), device=device, dtype=torch.uint8)

        width = max(length - 1, 0)
        capacity = max(1, batch * width)

        pair_keys_buffer = torch.empty((capacity, 2), dtype=torch.int32, device=device)
        pair_counts_buffer = torch.empty((capacity,), dtype=torch.int64, device=device)
        pair_count_length = torch.zeros((1,), dtype=torch.long, device=device)

        gpu_hist = count_pairs_on_device(
            tokens, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length
        )

        tokens_cpu = tokens.to("cpu")
        valid_cpu = valid.to("cpu")
        cpu_pair_keys = torch.empty((capacity, 2), dtype=torch.int64)
        cpu_pair_counts = torch.empty((capacity,), dtype=torch.int64)
        cpu_length = torch.zeros((1,), dtype=torch.long)

        count_pairs(tokens_cpu, valid_cpu, cpu_pair_keys, cpu_pair_counts, cpu_length)
        length_cpu = int(cpu_length.item())
        if length_cpu == 0:
            cpu_hist = PairHistogramResult.empty("cpu")
        else:
            pairs_view = cpu_pair_keys.narrow(0, 0, length_cpu)
            counts_view = cpu_pair_counts.narrow(0, 0, length_cpu)
            a_ids = pairs_view[:, 0].to(torch.long)
            b_ids = pairs_view[:, 1].to(torch.long)
            packed = (a_ids << 32) | b_ids
            cpu_hist = PairHistogramResult(packed, counts_view.to(torch.int64))

        reduced_gpu = combine_histogram_results([gpu_hist], target_device=device)
        reduced_cpu = combine_histogram_results([cpu_hist], target_device="cpu")
        cpu_keys, cpu_counts = aggregate_pair_keys(
            reduced_cpu.keys, reduced_cpu.counts
        )
        gpu_keys, gpu_counts = aggregate_pair_keys(
            reduced_gpu.keys, reduced_gpu.counts
        )

        assert torch.equal(gpu_keys.to("cpu"), cpu_keys)
        assert torch.equal(gpu_counts.to("cpu"), cpu_counts)
