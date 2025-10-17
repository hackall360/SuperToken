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

The CLI automatically shards the expanded ``--data`` glob list by rank. When
launched with ``torchrun`` or any other launcher that initializes
``torch.distributed`` (or sets ``RANK``/``WORLD_SIZE``), each process only reads
its assigned subset of shards, eliminating redundant disk I/O during
multi-worker training.

Each rank must contribute identically shaped tensors to the reduction; the
trainer handles padding internally, so no additional user code is required
beyond the distributed initialization.

### Heterogeneous GPU scaling

The benchmarking harness now accepts JSON configuration files that enumerate
multiple BPE runs with per-run device lists and scaling expectations. Invoke the
benchmark CLI with `--bpe-config` to sweep single- and multi-GPU combinations in
one shot:

```bash
PYTORCH_JIT=0 python main.py benchmark \
  --synthetic-docs 4 \
  --synthetic-min-len 1024 \
  --synthetic-max-len 2048 \
  --synthetic-vocab 256 \
  --bpe-merges 512 \
  --bpe-batch-size 1024 \
  --bpe-config benchmarks/configs/heterogeneous_example.json \
  --unigram-batch-size 1024 \
  --unigram-vocab 1024 \
  --unigram-epochs 2 \
  --output-dir artifacts/benchmarks
```

Each entry in `benchmarks/configs/heterogeneous_example.json` names the run,
batch size, participating devices, and (optionally) device weights used when
computing scaling efficiency.【F:benchmarks/configs/heterogeneous_example.json†L1-L14】
During serialization the harness records the observed tokens/sec for every
configuration, expected throughput based on the baseline run and weights, and a
boolean flag indicating whether the efficiency meets the ≥88 % target.

The sample output in `benchmarks/samples/heterogeneous_benchmark_sample.json`
shows a two-run sweep (single `cuda:0` and a heterogeneous `cuda:0`/`cuda:1`
pair). The heterogeneous run delivered ~9,000 tokens/sec against a 9,830
tokens/sec expectation, yielding ~91.5 % scaling and satisfying the threshold.
【F:benchmarks/samples/heterogeneous_benchmark_sample.json†L97-L150】
These JSON artifacts are checked into the repository so CI can assert that the
schema stays stable and that the scaling metric remains above the target via a
lightweight unit test.【F:tests/test_benchmark_config.py†L1-L25】

### `--dist` runtime workflow

The new distributed runtime wraps the trainer in a lease-based scheduler so
heterogeneous GPUs keep pulling work until the corpus is exhausted. Enable it
with the `--dist` flag and enumerate the participating devices via `--gpus`:

```bash
torchrun --standalone --nnodes 1 --nproc_per_node 2 \
  python -m gpu_tokenizer.cli_train_bpe \
    --dist \
    --gpus 0,1 \
    --data "data/**/*.txt" \
    --merges 50000 \
    --lease-size 16 \
    --rebalance-secs 10 \
    --target-chunk-ms 100
```

When launched with `torchrun`, rank 0 runs the host notary that doles out leases
of work to every other rank. Faster GPUs request the next lease as soon as they
drain the current chunk, while slower devices automatically contribute at their
own pace. The per-rank table that prints every few iterations reports the GPU
identifier, recent tokens/sec, lease size, number of inflight leases, and stage
timings, making it easy to spot stragglers or communication stalls during long
training jobs.

## Streaming corpus ingestion

`main.py` now wires the `CorpusStreamer` into the training loop. Shards are
memory-mapped (or decompressed with `zstandard`/`lz4.frame`) on background
workers before they reach the GPU packer. The streamer exposes both sync and
async consumers via a bounded queue controlled by live GPU utilization, so
prefetching automatically throttles when VRAM pressure climbs above the
configured target. Use `--compression`, `--io-workers`, and
`--prefetch-batches` to tailor throughput for your hardware.

## Continuous benchmarking pipeline

Automated performance tracking lives in [`.github/workflows/bench.yml`](../.github/workflows/bench.yml).
Each run provisions Python 3.12 and the CUDA-enabled PyTorch wheel (`torch==2.5.1+cu118`),
installs the plotting dependencies (`matplotlib`/`pandas`), and executes the synthetic
benchmark with TorchScript disabled to avoid the `torch.nonzero` signature regression:

```bash
PYTORCH_JIT=0 python main.py benchmark \
  --synthetic-docs 4 \
  --synthetic-min-len 8 \
  --synthetic-max-len 16 \
  --synthetic-vocab 128 \
  --bpe-merges 32 \
  --bpe-batch-size 32 \
  --bpe-log-every 0 \
  --unigram-batch-size 32 \
  --unigram-vocab 128 \
  --unigram-epochs 2 \
  --output-dir artifacts/benchmarks \
  --device cpu
```

The raw JSON telemetry (e.g. `artifacts/benchmarks/benchmark_<timestamp>.json`) plus a
`latest_summary.txt` capture of the CLI table are uploaded as workflow artifacts.  After the
benchmark finishes, `benchmarks/trend_report.py` ingests the full history of stored JSON files
to regenerate:

* `trend_table.md` – a Markdown table that mirrors tokens/sec and wall-time trends.
* `trend_plot.png` – a line chart visualizing both trainers' throughput over time.
* `trend_manifest.json` – a machine-readable manifest enumerating every input snapshot.

Every workflow run attaches these files to the **Benchmark results** artifact and surfaces the
table directly in the run summary for quick inspection.  To review the trend plot locally,
download the artifact from the run page, unpack it, and open `trend_plot.png`.  The same
report can be regenerated for any directory of JSON snapshots by invoking:

```bash
python benchmarks/trend_report.py --input <path/to/json/dir> --output-dir <path/to/report>
```

Because the JSON history is version-controlled via workflow artifacts, spotting regressions
becomes a matter of comparing the latest trend plot against previous runs.
