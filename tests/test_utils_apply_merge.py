import pytest


torch = pytest.importorskip("torch")

from gpu_tokenizer.utils import apply_merge_once



def _prepare_batch(seqs: list[list[int]]):
    max_len = max((len(seq) for seq in seqs), default=0)
    tokens = torch.zeros((len(seqs), max_len), dtype=torch.long)
    valid = torch.zeros((len(seqs), max_len), dtype=torch.long)
    lengths = torch.zeros(len(seqs), dtype=torch.long)
    for row, seq in enumerate(seqs):
        if not seq:
            continue
        vals = torch.tensor(seq, dtype=torch.long)
        L = vals.numel()
        tokens[row, :L] = vals
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
    prefix_workspace_cpu = torch.zeros((B,), dtype=torch.long)

    for a_id, b_id, new_id in merges:
        apply_merge_once(
            tokens_ref,
            valid_ref,
            lengths_ref,
            a_id,
            b_id,
            new_id,
            pair_workspace_cpu,
            prefix_workspace_cpu,
        )

    tokens_gpu = tokens_cpu.to("cuda")
    valid_gpu = valid_cpu.to("cuda")
    lengths_gpu = lengths_cpu.to("cuda")
    pair_workspace_gpu = torch.zeros((B, width), dtype=torch.bool, device="cuda")
    prefix_workspace_gpu = torch.zeros((B,), dtype=torch.long, device="cuda")

    for a_id, b_id, new_id in merges:
        apply_merge_once(
            tokens_gpu,
            valid_gpu,
            lengths_gpu,
            a_id,
            b_id,
            new_id,
            pair_workspace_gpu,
            prefix_workspace_gpu,
        )

    assert torch.equal(tokens_ref, tokens_gpu.cpu())
    assert torch.equal(valid_ref, valid_gpu.cpu())
    assert torch.equal(lengths_ref, lengths_gpu.cpu())

    for row in range(B):
        keep = int(lengths_ref[row].item())
        assert torch.all(valid_ref[row, keep:] == 0)
        assert torch.all(tokens_ref[row, keep:] == 0)
