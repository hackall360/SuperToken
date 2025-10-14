"""CUDA kernels for GPU unigram trainer."""

from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


@lru_cache(maxsize=1)
def _load_module() -> torch._C.ScriptModule:
    cuda_src = r"""
    #include <math.h>
    #include <stdint.h>

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

    extern "C" __global__ void apply_merge_and_compact_u32(
        uint32_t* tokens,
        uint8_t* valid,
        int32_t* prefix_workspace,
        bool* pair_mask,
        const int32_t B,
        const int32_t L,
        const int32_t width,
        const uint32_t a_id,
        const uint32_t b_id,
        const uint32_t new_id
    ) {
        int32_t row = static_cast<int32_t>(blockIdx.x);
        if (row >= B) {
            return;
        }
        uint32_t* row_tokens = tokens + row * L;
        uint8_t* row_valid = valid + row * L;
        bool* row_mask = nullptr;
        if (width > 0 && pair_mask != nullptr) {
            row_mask = pair_mask + row * width;
        }

        if (row_mask != nullptr) {
            for (int32_t col = 0; col < width; ++col) {
                bool left_valid = row_valid[col] != 0;
                bool right_valid = row_valid[col + 1] != 0;
                bool match =
                    left_valid && right_valid && (row_tokens[col] == a_id) && (row_tokens[col + 1] == b_id);
                row_mask[col] = match;
            }
            for (int32_t col = 0; col < width; ++col) {
                if (row_mask[col]) {
                    row_tokens[col] = new_id;
                    row_valid[col + 1] = 0;
                    row_tokens[col + 1] = 0u;
                }
            }
        }

        int32_t length = 0;
        for (int32_t col = 0; col < L; ++col) {
            uint8_t is_valid = row_valid[col];
            if (is_valid != 0) {
                uint32_t value = row_tokens[col];
                row_tokens[length] = value;
                row_valid[length] = 1;
                if (length != col) {
                    row_tokens[col] = 0u;
                    row_valid[col] = 0;
                }
                length += 1;
            } else {
                row_tokens[col] = 0u;
                row_valid[col] = 0;
            }
        }
        for (int32_t col = length; col < L; ++col) {
            row_tokens[col] = 0u;
            row_valid[col] = 0;
        }
        prefix_workspace[row] = length;
    }

    extern "C" __global__ void apply_merge_and_compact_u32_u16(
        uint32_t* tokens,
        uint8_t* valid,
        uint16_t* prefix_workspace,
        bool* pair_mask,
        const int32_t B,
        const int32_t L,
        const int32_t width,
        const uint32_t a_id,
        const uint32_t b_id,
        const uint32_t new_id
    ) {
        int32_t row = static_cast<int32_t>(blockIdx.x);
        if (row >= B) {
            return;
        }
        uint32_t* row_tokens = tokens + row * L;
        uint8_t* row_valid = valid + row * L;
        bool* row_mask = nullptr;
        if (width > 0 && pair_mask != nullptr) {
            row_mask = pair_mask + row * width;
        }

        if (row_mask != nullptr) {
            for (int32_t col = 0; col < width; ++col) {
                bool left_valid = row_valid[col] != 0;
                bool right_valid = row_valid[col + 1] != 0;
                bool match =
                    left_valid && right_valid && (row_tokens[col] == a_id) && (row_tokens[col + 1] == b_id);
                row_mask[col] = match;
            }
            for (int32_t col = 0; col < width; ++col) {
                if (row_mask[col]) {
                    row_tokens[col] = new_id;
                    row_valid[col + 1] = 0;
                    row_tokens[col + 1] = 0u;
                }
            }
        }

        int32_t length = 0;
        for (int32_t col = 0; col < L; ++col) {
            uint8_t is_valid = row_valid[col];
            if (is_valid != 0) {
                uint32_t value = row_tokens[col];
                row_tokens[length] = value;
                row_valid[length] = 1;
                if (length != col) {
                    row_tokens[col] = 0u;
                    row_valid[col] = 0;
                }
                length += 1;
            } else {
                row_tokens[col] = 0u;
                row_valid[col] = 0;
            }
        }
        for (int32_t col = length; col < L; ++col) {
            row_tokens[col] = 0u;
            row_valid[col] = 0;
        }
        const uint16_t max_val = 65535u;
        uint16_t stored = length > static_cast<int32_t>(max_val) ? max_val : static_cast<uint16_t>(length);
        prefix_workspace[row] = stored;
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
            "apply_merge_and_compact_u32",
            "apply_merge_and_compact_u32_u16",
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


def apply_merge_and_compact(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    prefix_workspace: torch.Tensor,
    pair_workspace: torch.Tensor,
    a_id: int,
    b_id: int,
    new_id: int,
) -> None:
    if not tokens.is_cuda or not valid.is_cuda:
        raise RuntimeError("apply_merge_and_compact expects CUDA tensors")
    if prefix_workspace is None or not prefix_workspace.is_cuda:
        raise RuntimeError("prefix_workspace must be a CUDA tensor")
    if pair_workspace is None or not pair_workspace.is_cuda:
        raise RuntimeError("pair_workspace must be a CUDA tensor")

    B, L = tokens.shape
    width = max(L - 1, 0)

    if prefix_workspace.shape[0] != B:
        raise ValueError("prefix_workspace shape mismatch")
    if width > 0 and (pair_workspace.shape[0] != B or pair_workspace.shape[1] != width):
        raise ValueError("pair_workspace shape mismatch")
    if width == 0 and pair_workspace.shape[0] != B:
        raise ValueError("pair_workspace batch mismatch")

    if tokens.dtype != torch.int32:
        raise TypeError("tokens must use torch.int32 dtype for CUDA merge kernel")
    if valid.dtype not in (torch.uint8, torch.bool):
        raise TypeError("valid must use torch.uint8 or torch.bool dtype for CUDA merge kernel")
    if prefix_workspace.dtype not in (torch.int32, torch.uint16):
        raise TypeError("prefix_workspace must use torch.int32 or torch.uint16 dtype")
    if pair_workspace.dtype != torch.bool:
        raise TypeError("pair_workspace must use torch.bool dtype")

    if not tokens.is_contiguous() or not valid.is_contiguous():
        raise RuntimeError("tokens and valid must be contiguous")
    if not prefix_workspace.is_contiguous() or (pair_workspace.numel() > 0 and not pair_workspace.is_contiguous()):
        raise RuntimeError("workspaces must be contiguous")

    module = _load_module()

    if B == 0:
        prefix_workspace.zero_()
        if pair_workspace.numel() > 0:
            pair_workspace.zero_()
        return

    if prefix_workspace.dtype == torch.int32:
        module.apply_merge_and_compact_u32(
            tokens,
            valid,
            prefix_workspace,
            pair_workspace,
            B,
            L,
            width,
            int(a_id),
            int(b_id),
            int(new_id),
            grid=(B,),
            block=(1,),
        )
    else:
        module.apply_merge_and_compact_u32_u16(
            tokens,
            valid,
            prefix_workspace,
            pair_workspace,
            B,
            L,
            width,
            int(a_id),
            int(b_id),
            int(new_id),
            grid=(B,),
            block=(1,),
        )


__all__ = [
    "traverse_trie",
    "forward_logz",
    "backward_logz",
    "accumulate_expectations",
    "apply_merge_and_compact",
]
