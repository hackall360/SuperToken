"""CUDA kernels for GPU unigram trainer."""

from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


@lru_cache(maxsize=1)
def _load_module() -> torch._C.ScriptModule:
    cuda_src = r"""
    #include <math.h>

    __device__ __forceinline__ float _log_add(float a, float b, const float neg_inf) {
        if (a <= neg_inf) {
            return b;
        }
        if (b <= neg_inf) {
            return a;
        }
        float mx = a > b ? a : b;
        float mn = a > b ? b : a;
        return mx + log1pf(expf(mn - mx));
    }

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

    extern "C" __global__ void forward_logz(
        const int32_t* sequences,
        const uint8_t* valid,
        const int32_t B,
        const int32_t L,
        const int32_t max_len,
        const int32_t* next_state,
        const int32_t* terminal_piece,
        const float* logp,
        const int32_t* piece_lens,
        float* forward_out
    ) {
        int b = blockIdx.x;
        if (b >= B) {
            return;
        }
        extern __shared__ float alpha[];
        const float NEG_INF = -1e30f;
        for (int idx = threadIdx.x; idx < L + 1; idx += blockDim.x) {
            alpha[idx] = NEG_INF;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            alpha[0] = 0.0f;
        }
        __syncthreads();

        const int32_t* seq = sequences + b * L;
        const uint8_t* mask = valid + b * L;

        for (int pos = 0; pos < L; ++pos) {
            if (!mask[pos]) {
                continue;
            }
            float current = alpha[pos];
            if (current <= NEG_INF) {
                continue;
            }
            int state = 0;
            for (int step = 0; step < max_len; ++step) {
                int idx = pos + step;
                if (idx >= L) {
                    break;
                }
                if (!mask[idx]) {
                    break;
                }
                unsigned char byte = (unsigned char)seq[idx];
                int next = next_state[state * 256 + (int)byte];
                if (next < 0) {
                    break;
                }
                state = next;
                int piece_id = terminal_piece[state];
                if (piece_id >= 0) {
                    int length = piece_lens[piece_id];
                    int target = pos + length;
                    if (target > L) {
                        continue;
                    }
                    float candidate = current + logp[piece_id];
                    alpha[target] = _log_add(alpha[target], candidate, NEG_INF);
                }
            }
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < L + 1; idx += blockDim.x) {
            forward_out[b * (L + 1) + idx] = alpha[idx];
        }
    }

    extern "C" __global__ void backward_logz(
        const int32_t* sequences,
        const uint8_t* valid,
        const int32_t B,
        const int32_t L,
        const int32_t max_len,
        const int32_t* next_state,
        const int32_t* terminal_piece,
        const float* logp,
        const int32_t* piece_lens,
        float* backward_out
    ) {
        int b = blockIdx.x;
        if (b >= B) {
            return;
        }
        extern __shared__ float beta[];
        const float NEG_INF = -1e30f;
        for (int idx = threadIdx.x; idx < L + 1; idx += blockDim.x) {
            beta[idx] = NEG_INF;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            beta[L] = 0.0f;
        }
        __syncthreads();

        const int32_t* seq = sequences + b * L;
        const uint8_t* mask = valid + b * L;

        for (int pos = L - 1; pos >= 0; --pos) {
            if (!mask[pos]) {
                continue;
            }
            float accum = NEG_INF;
            int state = 0;
            for (int step = 0; step < max_len; ++step) {
                int idx = pos + step;
                if (idx >= L) {
                    break;
                }
                if (!mask[idx]) {
                    break;
                }
                unsigned char byte = (unsigned char)seq[idx];
                int next = next_state[state * 256 + (int)byte];
                if (next < 0) {
                    break;
                }
                state = next;
                int piece_id = terminal_piece[state];
                if (piece_id >= 0) {
                    int length = piece_lens[piece_id];
                    int target = pos + length;
                    if (target > L) {
                        continue;
                    }
                    float candidate = logp[piece_id] + beta[target];
                    accum = _log_add(accum, candidate, NEG_INF);
                }
            }
            if (accum > NEG_INF) {
                beta[pos] = accum;
            }
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < L + 1; idx += blockDim.x) {
            backward_out[b * (L + 1) + idx] = beta[idx];
        }
    }

    extern "C" __global__ void accumulate_expectations(
        const int32_t* sequences,
        const uint8_t* valid,
        const int32_t B,
        const int32_t L,
        const int32_t max_len,
        const int32_t* next_state,
        const int32_t* terminal_piece,
        const float* logp,
        const int32_t* piece_lens,
        const float* forward_vals,
        const float* backward_vals,
        const float* log_z,
        float* expectations,
        const int32_t V
    ) {
        const float NEG_INF = -1e30f;
        int b = blockIdx.x;
        if (b >= B) {
            return;
        }
        if (log_z[b] <= NEG_INF) {
            return;
        }
        const int32_t* seq = sequences + b * L;
        const uint8_t* mask = valid + b * L;
        const float* alpha = forward_vals + b * (L + 1);
        const float* beta = backward_vals + b * (L + 1);
        float* exp_row = expectations + b * V;
        float logz = log_z[b];

        for (int pos = threadIdx.x; pos < L; pos += blockDim.x) {
            if (!mask[pos]) {
                continue;
            }
            float current = alpha[pos];
            if (current <= NEG_INF) {
                continue;
            }
            int state = 0;
            for (int step = 0; step < max_len; ++step) {
                int idx = pos + step;
                if (idx >= L) {
                    break;
                }
                if (!mask[idx]) {
                    break;
                }
                unsigned char byte = (unsigned char)seq[idx];
                int next = next_state[state * 256 + (int)byte];
                if (next < 0) {
                    break;
                }
                state = next;
                int piece_id = terminal_piece[state];
                if (piece_id >= 0) {
                    int length = piece_lens[piece_id];
                    int target = pos + length;
                    if (target > L) {
                        continue;
                    }
                    float val = current + logp[piece_id] + beta[target] - logz;
                    float weight = expf(val);
                    if (weight > 0.0f) {
                        atomicAdd(&exp_row[piece_id], weight);
                    }
                }
            }
        }
    }
    """
    return load_inline(
        name="gpu_trie_kernels",
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=[
            "traverse_trie",
            "forward_logz",
            "backward_logz",
            "accumulate_expectations",
        ],
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


def forward_logz(
    sequences: torch.Tensor,
    valid: torch.Tensor,
    max_len: int,
    next_state: torch.Tensor,
    terminal_piece: torch.Tensor,
    logp: torch.Tensor,
    piece_lens: torch.Tensor,
) -> torch.Tensor:
    if not sequences.is_cuda:
        raise RuntimeError("forward_logz expects CUDA tensors")
    module = _load_module()
    B, L = sequences.shape
    forward = torch.empty((B, L + 1), device=sequences.device, dtype=torch.float32)
    threads = 128
    shared_mem = (L + 1) * 4
    module.forward_logz(
        sequences.contiguous(),
        valid.contiguous(),
        B,
        L,
        max_len,
        next_state.contiguous().view(-1),
        terminal_piece.contiguous(),
        logp.contiguous(),
        piece_lens.contiguous(),
        forward,
        grid=(B,),
        block=(threads,),
        shared_mem=shared_mem,
    )
    return forward


def backward_logz(
    sequences: torch.Tensor,
    valid: torch.Tensor,
    max_len: int,
    next_state: torch.Tensor,
    terminal_piece: torch.Tensor,
    logp: torch.Tensor,
    piece_lens: torch.Tensor,
) -> torch.Tensor:
    if not sequences.is_cuda:
        raise RuntimeError("backward_logz expects CUDA tensors")
    module = _load_module()
    B, L = sequences.shape
    backward = torch.empty((B, L + 1), device=sequences.device, dtype=torch.float32)
    threads = 128
    shared_mem = (L + 1) * 4
    module.backward_logz(
        sequences.contiguous(),
        valid.contiguous(),
        B,
        L,
        max_len,
        next_state.contiguous().view(-1),
        terminal_piece.contiguous(),
        logp.contiguous(),
        piece_lens.contiguous(),
        backward,
        grid=(B,),
        block=(threads,),
        shared_mem=shared_mem,
    )
    return backward


def accumulate_expectations(
    sequences: torch.Tensor,
    valid: torch.Tensor,
    max_len: int,
    next_state: torch.Tensor,
    terminal_piece: torch.Tensor,
    logp: torch.Tensor,
    piece_lens: torch.Tensor,
    forward: torch.Tensor,
    backward: torch.Tensor,
    logz: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if not sequences.is_cuda:
        raise RuntimeError("accumulate_expectations expects CUDA tensors")
    module = _load_module()
    B, L = sequences.shape
    V = out.size(1)
    threads = 128
    module.accumulate_expectations(
        sequences.contiguous(),
        valid.contiguous(),
        B,
        L,
        max_len,
        next_state.contiguous().view(-1),
        terminal_piece.contiguous(),
        logp.contiguous(),
        piece_lens.contiguous(),
        forward.contiguous(),
        backward.contiguous(),
        logz.contiguous(),
        out.contiguous(),
        V,
        grid=(B,),
        block=(threads,),
    )


__all__ = [
    "traverse_trie",
    "forward_logz",
    "backward_logz",
    "accumulate_expectations",
]
