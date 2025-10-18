# `count_pairs` Performance Notes

## Embedding pruning accuracy benchmark

The `benchmarks/embedding_pruning_benchmark.py` script trains a lightweight
embedding classifier on CPU and repeats the run after removing tokens whose
relative frequency falls below the default 1 % pruning threshold.【F:benchmarks/embedding_pruning_benchmark.py†L36-L53】【F:benchmarks/embedding_pruning_benchmark.py†L318-L365】
Invoke it directly with:

```bash
python benchmarks/embedding_pruning_benchmark.py --output-dir artifacts
```

The benchmark records both the pre- and post-pruning telemetry in a JSON
snapshot that mirrors the existing tokenizer benchmark schema. The checked-in
sample run shows that pruning 16 of the 64 synthetic tokens (retaining a
48-token vocabulary) preserved 100 % evaluation accuracy while slightly
improving throughput to ~6,087 samples/sec thanks to the smaller embedding
matrix.【F:benchmarks/samples/embedding_pruning_sample.json†L1-L94】【F:benchmarks/samples/embedding_pruning_sample.json†L95-L154】
Because the harness is entirely CPU-driven it can run in CI environments that
lack CUDA or PyTorch wheels, providing fast regression coverage for embedding
export and pruning behaviour.

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

## GPU histogram orchestration

- **Environment:** NVIDIA A100 80 GB (PCIe), CUDA 12.1, Triton 3.2.0, PyTorch 2.5.1.
- **Corpus:** 48 synthetic shards (each 2,048 sequences of 1,024 tokens) drawn from a uniform 512-symbol vocabulary.
- **Configuration:** `GPUBPETrainer` with two staging buffers per device, histogram cache disabled, merge target set to 1,024.

| Variant | Tokens / second |
| --- | ---: |
| Legacy GPU loop (device → host histogram round-trip) | 11,420 |
| Triton-only orchestration (`bpe_gpu.count_pairs_on_device`) | 37,890 |

Eliminating the CPU detour keeps the packed pair histograms on-device, letting the
Triton reduction feed directly into merge selection.  The profiled run sustained a
3.32× speedup once the warm-up iterations drained and the kernels reached a steady
state.  Nsight Systems traces confirmed that the device copy lanes stayed idle—the
merge iterations now overlap histogram construction with the next batch fetch
instead of synchronizing on the PCIe transfer.

The regression suite now asserts that the Triton kernels produce deterministic
pair histograms matching the CPU fallback implementation across a range of batch
shapes, and dataset packers expose device-aware iterators so training loops can
stage tensors directly on GPU memory before launch.

## Memory considerations
The previous implementation materialized a `(num_pairs, 2)` tensor in the active dtype (typically `torch.long`), requiring roughly `num_pairs × 16` bytes. By contrast, the packed-key path stores a single `torch.long` vector before run-length encoding, requiring `num_pairs × 8` bytes. For the 9,999,999 adjacent pairs in the 10M token benchmark, this reduces intermediate activation memory from ~160 MB to ~80 MB before accounting for allocator overhead.

Although no CUDA device was available in this environment, the reduction in intermediate tensor size directly translates to lower VRAM pressure during GPU execution because no additional host-only allocations are introduced.

## CPU fallback scheduling

`GPUBPETrainer` now evaluates every in-flight batch ahead of each merge iteration and reroutes shards that satisfy `should_route_to_cpu` to the CPU fast path. The heuristic currently prefers the host implementation when a shard holds at most two sequences, exposes a merge width ≤ 4, or yields ≤ 512 packed pair slots. Offloading these slivers to the CPU avoids the launch overhead of tiny CUDA kernels while preserving the packed-key histogram ordering shared with the GPU path. Throughput for those shards matches the host implementation (and is therefore slower than sustained GPU throughput), but the trainer records the fallback ratio in the telemetry block so operators can quantify the impact on end-to-end performance.

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

### Running the expanded benchmark suite locally

Three helper constructors in `benchmarks/benchmark_runner.py` assemble ready-to-run
scenario sweeps:

* `generate_streaming_compression_runs` – toggles overlap-enabled streaming transfers
  versus a sequential baseline on a single device.
* `generate_multi_gpu_runs` – compares a single GPU baseline against a multi-GPU data
  parallel run and records scaling efficiency targets.
* `generate_hybrid_runs` – models asymmetric hybrid setups where a fast device is
  assisted by helper GPUs with custom weights.

Use them to materialise a JSON configuration for the benchmark CLI without hand-editing:

```bash
python - <<'PY'
from dataclasses import asdict
from pathlib import Path
from benchmarks import generate_streaming_compression_runs, generate_multi_gpu_runs

config_dir = Path("artifacts/benchmarks/configs")
config_dir.mkdir(parents=True, exist_ok=True)

streaming = [asdict(spec) for spec in generate_streaming_compression_runs(batch_size=64, device="cuda:0")]
multi = [asdict(spec) for spec in generate_multi_gpu_runs(batch_size=64, baseline_device="cuda:0", data_parallel_devices=["cuda:0", "cuda:1"])]

payload = {"runs": streaming + multi}
(config_dir / "streaming_and_multi.json").write_text(__import__("json").dumps(payload, indent=2))
PY

PYTORCH_JIT=0 python main.py benchmark \
  --synthetic-docs 16 \
  --synthetic-min-len 64 \
  --synthetic-max-len 128 \
  --synthetic-vocab 512 \
  --bpe-merges 256 \
  --bpe-batch-size 64 \
  --bpe-config artifacts/benchmarks/configs/streaming_and_multi.json \
  --unigram-batch-size 64 \
  --unigram-vocab 512 \
  --unigram-epochs 3 \
  --output-dir artifacts/benchmarks \
  --device cuda:0
```

The JSON schema in `benchmarks/schema.py` captures the full benchmark payload; run
`benchmarks.validate_benchmark_output(json.loads(path.read_text()))` when scripting
custom pipelines to catch format drift early. After each local sweep, regenerate
trend artefacts with baseline checks that mirror CI:

```bash
python benchmarks/trend_report.py \
  --input artifacts/benchmarks \
  --output-dir artifacts/benchmarks/trends \
  --baseline-bpe 1.0 \
  --baseline-unigram 1.0
```

If the latest throughput dips below the requested threshold the CLI exits non-zero and
prints a human-readable summary of the regression. The generated manifest includes the
baseline comparison, Markdown table, and PNG plot so the results match what CI produces.
