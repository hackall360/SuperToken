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
