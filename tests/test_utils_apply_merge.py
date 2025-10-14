from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from gpu_tokenizer.dtypes import length_storage_dtype
from gpu_tokenizer.utils import apply_merge_once



def _prepare_batch(seqs: list[list[int]]):
    max_len = max((len(seq) for seq in seqs), default=0)
    tokens = torch.zeros((len(seqs), max_len), dtype=torch.int32)
    valid = torch.zeros((len(seqs), max_len), dtype=torch.uint8)
    length_dtype = length_storage_dtype(max_len)
    lengths = torch.zeros(len(seqs), dtype=length_dtype)
    for row, seq in enumerate(seqs):
        if not seq:
            continue
        vals = torch.tensor(seq, dtype=torch.int64)
        L = vals.numel()
        tokens[row, :L] = vals.to(torch.int32)
        valid[row, :L] = 1
        lengths[row] = L
    return tokens, valid, lengths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for GPU parity check")
def test_apply_merge_inplace_matches_cpu_multi_step():
    seqs = [
        [1, 2, 3, 1, 2, 3, 1],
        [1, 2, 3, 3, 1, 2, 3],
    ]
    merges = [(1, 2, 256), (256, 3, 257), (257, 1, 258)]

    tokens_cpu, valid_cpu, lengths_cpu = _prepare_batch(seqs)
    tokens_ref = tokens_cpu.clone()
    valid_ref = valid_cpu.clone()
    lengths_ref = lengths_cpu.clone()

    B, L = tokens_cpu.shape
    width = max(L - 1, 0)
    pair_workspace_cpu = torch.zeros((B, width), dtype=torch.bool)
    prefix_workspace_cpu = torch.zeros((B,), dtype=lengths_ref.dtype)
    span_workspace_cpu = torch.zeros((B, width), dtype=torch.bool)
    cpu_spans: list[torch.Tensor] = []

    for a_id, b_id, new_id in merges:
        prev_tokens = tokens_ref.clone()
        prev_valid = valid_ref.clone()
        _, _, _, span_mask = apply_merge_once(
            tokens_ref,
            valid_ref,
            lengths_ref,
            a_id,
            b_id,
            new_id,
            pair_workspace_cpu,
            prefix_workspace_cpu,
            span_workspace_cpu,
        )
        expected_mask = (
            (prev_tokens[:, :-1] == a_id)
            & (prev_tokens[:, 1:] == b_id)
            & prev_valid[:, :-1].to(torch.bool)
            & prev_valid[:, 1:].to(torch.bool)
        )
        assert torch.equal(span_mask.to(torch.bool), expected_mask)
        cpu_spans.append(span_mask.to(torch.bool).clone())

    tokens_gpu = tokens_cpu.to("cuda")
    valid_gpu = valid_cpu.to("cuda")
    lengths_gpu = lengths_cpu.to("cuda")
    pair_workspace_gpu = torch.zeros((B, width), dtype=torch.bool, device="cuda")
    prefix_workspace_gpu = torch.zeros((B,), dtype=lengths_gpu.dtype, device="cuda")
    span_workspace_gpu = torch.zeros((B, width), dtype=torch.bool, device="cuda")
    gpu_spans: list[torch.Tensor] = []

    for a_id, b_id, new_id in merges:
        prev_tokens = tokens_gpu.clone()
        prev_valid = valid_gpu.clone()
        _, _, _, span_mask = apply_merge_once(
            tokens_gpu,
            valid_gpu,
            lengths_gpu,
            a_id,
            b_id,
            new_id,
            pair_workspace_gpu,
            prefix_workspace_gpu,
            span_workspace_gpu,
        )
        expected_mask = (
            (prev_tokens[:, :-1] == a_id)
            & (prev_tokens[:, 1:] == b_id)
            & prev_valid[:, :-1].to(torch.bool)
            & prev_valid[:, 1:].to(torch.bool)
        )
        assert torch.equal(span_mask.to(torch.bool), expected_mask)
        gpu_spans.append(span_mask.to(torch.bool).clone())

    assert torch.equal(tokens_ref, tokens_gpu.cpu())
    assert torch.equal(valid_ref, valid_gpu.cpu())
    assert torch.equal(lengths_ref, lengths_gpu.cpu())

    assert len(cpu_spans) == len(gpu_spans) == len(merges)
    for idx, (cpu_mask, gpu_mask) in enumerate(zip(cpu_spans, gpu_spans)):
        assert torch.equal(cpu_mask, gpu_mask.cpu()), f"Mismatch in span mask for merge {idx}"

    for row in range(B):
        keep = int(lengths_ref[row].item())
        assert torch.all(valid_ref[row, keep:] == 0)
        assert torch.all(tokens_ref[row, keep:] == 0)

    assert tokens_ref.dtype == torch.int32
    assert valid_ref.dtype == torch.uint8


def test_apply_merge_guard_raises_for_uint32_overflow():
    tokens = torch.tensor([[0, 1]], dtype=torch.int32)
    valid = torch.ones_like(tokens, dtype=torch.uint8)
    lengths = torch.tensor([2], dtype=torch.int32)

    with pytest.raises(OverflowError):
        apply_merge_once(tokens, valid, lengths, 0, 1, (1 << 32))


def test_apply_merge_handles_uint16_capacity_limit():
    width = 65535
    tokens = torch.arange(width, dtype=torch.int32).unsqueeze(0)
    valid = torch.ones_like(tokens, dtype=torch.uint8)
    lengths = torch.full((1,), width, dtype=torch.uint16)
    pair_workspace = torch.zeros((1, width - 1), dtype=torch.bool)
    prefix_workspace = torch.zeros((1,), dtype=torch.uint16)
    overflow_workspace = torch.zeros((1,), dtype=torch.bool)

    apply_merge_once(
        tokens,
        valid,
        lengths,
        int(width + 1),
        int(width + 2),
        int(width + 3),
        pair_workspace,
        prefix_workspace,
        None,
        overflow_workspace,
    )

    assert int(lengths.item()) == width
    assert not bool(overflow_workspace.item())


def test_apply_merge_matches_between_uint16_and_int32():
    seqs = [[1, 2, 3, 4], [4, 3, 2, 1]]
    tokens_u16, valid_u16, lengths_u16 = _prepare_batch(seqs)
    tokens_i32 = tokens_u16.clone()
    valid_i32 = valid_u16.clone()
    lengths_i32 = lengths_u16.to(torch.int32)

    B, L = tokens_u16.shape
    width = max(L - 1, 0)
    pair_workspace_u16 = torch.zeros((B, width), dtype=torch.bool)
    pair_workspace_i32 = torch.zeros((B, width), dtype=torch.bool)
    prefix_u16 = torch.zeros((B,), dtype=torch.uint16)
    prefix_i32 = torch.zeros((B,), dtype=torch.int32)

    for a_id, b_id, new_id in [(1, 2, 300), (300, 3, 301)]:
        apply_merge_once(
            tokens_u16,
            valid_u16,
            lengths_u16,
            a_id,
            b_id,
            new_id,
            pair_workspace_u16,
            prefix_u16,
        )
        apply_merge_once(
            tokens_i32,
            valid_i32,
            lengths_i32,
            a_id,
            b_id,
            new_id,
            pair_workspace_i32,
            prefix_i32,
        )

    assert torch.equal(tokens_u16, tokens_i32)
    assert torch.equal(valid_u16, valid_i32)
    assert torch.equal(lengths_u16.to(torch.int32), lengths_i32)
