"""CUDA kernels for GPU unigram trainer."""

from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


@lru_cache(maxsize=1)
def _load_module() -> torch._C.ScriptModule:
    cuda_src = r"""
    extern "C" __global__ void traverse_trie(
        const int32_t* sequences,
        const uint8_t* valid,
        const int32_t B,
        const int32_t L,
        const int32_t max_len,
        const int32_t* next_state,
        const int32_t* terminal_ids,
        int32_t* counts
    ) {
        int b = blockIdx.x;
        if (b >= B) {
            return;
        }
        int base = b * L;
        for (int pos = threadIdx.x; pos < L; pos += blockDim.x) {
            if (!valid[base + pos]) {
                continue;
            }
            int state = 0;
            for (int step = 0; step < max_len; ++step) {
                int idx = pos + step;
                if (idx >= L) {
                    break;
                }
                if (!valid[base + idx]) {
                    break;
                }
                unsigned char byte = (unsigned char)sequences[base + idx];
                int next = next_state[state * 256 + (int)byte];
                if (next < 0) {
                    break;
                }
                state = next;
                int term = terminal_ids[state];
                if (term >= 0) {
                    atomicAdd(&counts[term], 1);
                }
            }
        }
    }
    """
    return load_inline(
        name="gpu_trie_kernels",
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=["traverse_trie"],
        extra_cuda_cflags=["-lineinfo"],
    )


def traverse_trie(
    sequences: torch.Tensor,
    valid: torch.Tensor,
    next_state: torch.Tensor,
    terminal_ids: torch.Tensor,
    counts: torch.Tensor,
    B: int,
    L: int,
    max_len: int,
) -> None:
    if not sequences.is_cuda:
        raise RuntimeError("trie traversal requires CUDA tensors")
    module = _load_module()
    threads = 128
    blocks = B
    module.traverse_trie(
        sequences.contiguous(),
        valid.contiguous(),
        B,
        L,
        max_len,
        next_state.contiguous().view(-1),
        terminal_ids.contiguous(),
        counts.contiguous(),
        grid=(blocks,),
        block=(threads,),
    )


__all__ = ["traverse_trie"]
