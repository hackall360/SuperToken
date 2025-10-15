# Architecture Overview

SuperToken is organized around a GPU-first tokenization toolkit with a command-line interface and supporting benchmarks. This guide sketches how the major components interact so you can extend them or slot in new modules.

## Quick Navigation
- [CLI entry points](#cli-entry-points)
- [GPU tokenizer package](#gpu-tokenizer-package)
- [Benchmark utilities](#benchmark-utilities)
- [Related guides](#related-guides)

## CLI Entry Points
The `main.py` script exposes the public CLI surface. It wires the subcommands (`train-bpe`, `train-unigram`, `benchmark`) to the concrete trainers and benchmarking helpers while handling shared arguments such as dataset globs, checkpoint configuration, and autoscaling targets. Each subcommand delegates to functions in `gpu_tokenizer.cli_train_bpe`, `gpu_tokenizer.unigram_trainer`, and the benchmarking helpers to keep the CLI thin and composable.

Key responsibilities:
- Parse global and subcommand-specific options via `argparse`.
- Instantiate the appropriate trainer (`GPUBPETrainer` or `GPUUnigramTrainer`).
- Configure autoscaling, dataset ingestion, and checkpoint paths.
- Dispatch benchmarking runs that emit telemetry tables and JSON artifacts.

## GPU Tokenizer Package
The `gpu_tokenizer/` package houses the GPU kernels, trainers, and utilities that power the CLI. The modules are loosely grouped by concern:

- **Training frontends**: `bpe_trainer.py` and `unigram_trainer.py` implement the high-level training loops for BPE and unigram vocabularies. They coordinate dataset streaming, invoke CUDA/Triton kernels, and persist vocab artifacts.
- **Autoscaling**: `autoscaler.py` monitors throughput statistics and adjusts batch sizes to saturate the GPU within a target utilization band.
- **Dataset ingestion**: `datasets.py` and `io.py` provide streaming readers with optional compression, memory mapping, and worker-based prefetch.
- **Packing and fast paths**: `cpu_packer.py`, `cpu_fastpath.py`, and `utils.py` manage host-side preparation of byte sequences before they reach the GPU.
- **Kernels and dtype helpers**: `cuda_kernels.py`, `triton_kernels.py`, and `dtypes.py` collect specialized kernels plus dtype utilities shared across trainers.
- **Analytics and statistics**: `ngram_stats.py` gathers corpus statistics that feed into tokenizer initialization and evaluation.

Each trainer composes these modules: data loaders stream shards into packed batches, autoscaling suggests the next batch size, and GPU kernels execute merge or unigram scoring steps. The package layout is intentionally modular so new trainers or kernels can be dropped into the same scaffolding.

## Benchmark Utilities
Benchmark entry points live alongside the CLI (`main.py benchmark`) and reuse the trainers with synthetic or sampled datasets. The helper functions in `benchmarks/` generate reproducible corpora, run both trainers under comparable settings, and emit:

- Tabular summaries printed to stdout.
- JSON telemetry payloads under the requested output directory.

When adding new benchmarking scenarios, place reusable helpers in `benchmarks/` and expose them via the CLI subcommand. This keeps benchmarking logic isolated from the training loops while still sharing dataset and autoscaling utilities.

## Related Guides
- [Performance notes and benchmarks](performance.md): Deep dive into throughput expectations, tuning knobs, and representative benchmark results.
- [README](../README.md#quick-start): Quick-start examples and command invocations for common workflows.

> Looking for something else? Future guides can extend this section with links to API references, kernel deep dives, or integration tutorials.
