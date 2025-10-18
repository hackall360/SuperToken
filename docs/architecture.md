# Architecture Overview

SuperToken couples GPU-centric tokenizer trainers with a streaming data engine and automation layers that keep accelerators saturated. This document drills into how the moving parts cooperate so you can extend or swap components confidently.

## Quick Navigation
- [Trainer pipeline](#trainer-pipeline)
- [Autoscaler lifecycle](#autoscaler-lifecycle)
- [Dataset and streaming layers](#dataset-and-streaming-layers)
- [CLI integration](#cli-integration)
- [Benchmarking workflow](#benchmarking-workflow)
- [Related guides](#related-guides)

## Trainer Pipeline
The core training loop is shared between the [BPE](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/bpe_trainer.py) and [unigram](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/unigram_trainer.py) implementations. Each step is designed to maximize GPU residency:

1. **Shard discovery** – Dataset providers enumerate files or glob patterns and create an iterator of logical shards.
2. **Streaming ingestion** – Reader workers pull shards through the I/O adapters, decompress data when required, and feed bytes into the host staging buffers.
3. **Packing** – The packing utilities collate sequences into contiguous device-ready tensors, applying special tokens or padding rules.
4. **Autoscaled batching** – The [`AutoScaler`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/autoscaler.py) recommends the next batch size based on throughput telemetry captured in the previous iteration.
5. **GPU kernels** – Trainer-specific CUDA/Triton kernels execute merge scoring or unigram probability updates.
6. **Checkpointing and metrics** – Trainers optionally snapshot state (merge tables, unigram weights, autoscaler state) and surface progress to the CLI logger.

Because the trainers conform to a shared interface, you can prototype new algorithms by reusing the streaming and autoscaling layers while plugging in new kernel invocations. Refer to the [API reference](api.md) for class signatures and extensibility hooks.

## Autoscaler Lifecycle
Maintaining high device utilization is the autoscaler's primary objective:

- **Initialization** – When a trainer starts, it seeds the autoscaler with a target utilization band and warm-up window.
- **Measurement** – After each batch, trainers report throughput counters (tokens processed, kernel duration, host wait time).
- **Decision** – The autoscaler smooths the metrics, compares them to targets, and proposes the next batch size within configured bounds.
- **Persistence** – On checkpoint, autoscaler state is serialized so resumed runs maintain momentum.

The autoscaler exposes hooks for alternative policies (e.g., PID-style controllers or reinforcement learners). Implementation details and configuration knobs are documented in the [API reference](api.md#autoscaler).

## Dataset and Streaming Layers
SuperToken treats data as an infinite stream:

- **Datasets module** – [`gpu_tokenizer/datasets`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/datasets/__init__.py) exports high-level abstractions such as `IterableCorpus`, synthetic generators, and shard samplers.
- **I/O adapters** – [`gpu_tokenizer/io`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/io/__init__.py) centralizes compression handling, memory mapping, and threaded prefetch.
- **Prefetch workers** – Configurable worker pools overlap disk reads with GPU execution, reducing idle windows.
- **Backpressure** – Autoscaler suggestions feed back into the dataset layer to avoid queue overflows or starvation.

For more operational details—including configuration flags and expected throughput—see the [CLI usage guide](cli.md#streaming-options) and [performance notes](performance.md).

## CLI Integration
`main.py` orchestrates the trainers through user-facing commands:

- Each subcommand (e.g., `train-bpe`, `train-unigram`, `benchmark`) wires parser arguments to trainer factories.
- Common flags configure dataset paths, compression, autoscaler targets, checkpoint cadence, and output directories.
- Structured logging surfaces autoscaler decisions, throughput summaries, and checkpoint events for observability.

The [CLI usage guide](cli.md) contains end-to-end examples and option tables, while the [API reference](api.md#cli-helpers) covers the helper functions exposed for reuse in other entry points.

## Benchmarking Workflow
Benchmarks share the training backbone but focus on repeatability:

1. Configure synthetic and real corpus sources, output directories, and runtime limits.
2. Instantiate trainer pairs with consistent autoscaler targets.
3. Execute each trainer sequentially while recording telemetry snapshots.
4. Emit tabular summaries and JSON artifacts that can be parsed by external dashboards.

Custom benchmarks should live under the `benchmarks/` package, reusing the dataset and autoscaling primitives described above. Consult the [CLI usage guide](cli.md#benchmark-command) for invocation recipes and the [API reference](api.md#benchmarking-utilities) for programmatic entry points.

## Related Guides
- [CLI usage guide](cli.md)
- [API reference](api.md)
- [Module guide](modules.md)
- [Performance notes and benchmarks](performance.md)
- [Project README](https://github.com/example/SuperToken/blob/main/README.md)
