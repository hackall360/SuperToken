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

## Multi-GPU execution

The `GPUBPETrainer` can aggregate pair histograms across multiple NCCL-backed
workers. To launch on multiple GPUs you must initialize
`torch.distributed` *before* constructing the trainer:

```python
import torch.distributed as dist

dist.init_process_group("nccl")
trainer = GPUBPETrainer(devices=[f"cuda:{dist.get_rank()}"])
```

When launching with `torchrun`, the required environment variables are set
automatically. Manual launches must export the following variables consistently
across processes:

* `MASTER_ADDR` and `MASTER_PORT` – the rendezvous host/port.
* `WORLD_SIZE` – total number of ranks in the job.
* `RANK` – the global rank of the current process.
* `LOCAL_RANK` – the device index on the local node (used to select the GPU).

Example invocation on a single node with four GPUs:

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
WORLD_SIZE=4 torchrun --nproc_per_node=4 \
    python -m gpu_tokenizer.cli_train_bpe --merges 10000 --devices cuda
```

Each rank must contribute identically shaped tensors to the reduction; the
trainer handles padding internally, so no additional user code is required
beyond the distributed initialization.

## Streaming corpus ingestion

`main.py` now wires the `CorpusStreamer` into the training loop. Shards are
memory-mapped (or decompressed with `zstandard`/`lz4.frame`) on background
workers before they reach the GPU packer. The streamer exposes both sync and
async consumers via a bounded queue controlled by live GPU utilization, so
prefetching automatically throttles when VRAM pressure climbs above the
configured target. Use `--compression`, `--io-workers`, and
`--prefetch-batches` to tailor throughput for your hardware.
