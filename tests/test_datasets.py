import pytest

torch = pytest.importorskip("torch")
if getattr(torch, "_SUPERTOKEN_TORCH_STUB", False) or not hasattr(torch, "full"):
    pytest.skip(
        "PyTorch with tensor factories is required for dataset batching tests",
        allow_module_level=True,
    )

from gpu_tokenizer.datasets import PackedBatcher


def test_packed_batcher_double_buffer_reuse():
    sequences = [[i] for i in range(6)]
    batcher = PackedBatcher(sequences, batch_size=1, seed=42)

    iterator = iter(batcher)
    first_tokens, first_valid, first_lengths = next(iterator)
    second_tokens, second_valid, second_lengths = next(iterator)
    third_tokens, third_valid, third_lengths = next(iterator)

    # Two alternating buffers should be reused in order
    assert first_tokens.data_ptr() == third_tokens.data_ptr()
    assert first_tokens.data_ptr() != second_tokens.data_ptr()
    assert first_valid.data_ptr() == third_valid.data_ptr()
    assert first_valid.data_ptr() != second_valid.data_ptr()

    assert first_tokens.dtype == torch.int32
    assert first_valid.dtype == torch.uint8
    assert first_lengths.dtype == torch.uint16

    for lengths in (first_lengths, second_lengths, third_lengths):
        assert torch.all(lengths == 1)


def test_packed_batcher_device_iterator_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for device-aware batch views")

    sequences = [[1, 2, 3], [4], [5, 6]]
    cpu_batcher = PackedBatcher(sequences, batch_size=2, seed=1337)
    cpu_tokens, cpu_valid, cpu_lengths = next(iter(cpu_batcher))

    device = torch.device("cuda", 0)
    gpu_batcher = PackedBatcher(sequences, batch_size=2, seed=1337)
    tokens_dev, valid_dev, lengths_dev = next(gpu_batcher.iter_device(device))

    assert tokens_dev.device == device
    assert valid_dev.device == device
    assert lengths_dev.device == device

    assert torch.equal(tokens_dev.to("cpu"), cpu_tokens)
    assert torch.equal(valid_dev.to("cpu"), cpu_valid)
    assert torch.equal(lengths_dev.to("cpu"), cpu_lengths)
