# `count_pairs` Performance Notes

## Measurement setup
- **Environment:** PyTorch 2.2.0+cpu, NumPy 1.26.4, Python 3.12.10.
- **Hardware:** Single CPU worker provided by the evaluation container (no CUDA device available).
- **Workload:** 10,000,000 token sequence (1 × 10M) with all positions marked valid.
- **Commands:** Measurements executed with `PYTORCH_JIT=0` to avoid TorchScript compilation overhead and `torch.randint` to synthesize the corpus.

## Runtime comparison
| Implementation | Wall time (s) |
| --- | ---: |
| Baseline (`torch.unique` on stacked pairs) | 53.774 |
| Packed-key implementation | 0.794 |

The packed-key approach is ~67× faster on the 10M token synthetic corpus while producing identical counts to the CPU reference implementation.

## Memory considerations
The previous implementation materialized a `(num_pairs, 2)` tensor in the active dtype (typically `torch.long`), requiring roughly `num_pairs × 16` bytes. By contrast, the packed-key path stores a single `torch.long` vector before run-length encoding, requiring `num_pairs × 8` bytes. For the 9,999,999 adjacent pairs in the 10M token benchmark, this reduces intermediate activation memory from ~160 MB to ~80 MB before accounting for allocator overhead.

Although no CUDA device was available in this environment, the reduction in intermediate tensor size directly translates to lower VRAM pressure during GPU execution because no additional host-only allocations are introduced.
