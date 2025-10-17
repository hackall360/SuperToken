# API Reference

This reference collects the most commonly extended classes and functions in SuperToken. It complements the system-level view from the [architecture overview](architecture.md) and the workflows covered in the [CLI usage guide](cli.md).

## Quick Navigation
- [Trainers](#trainers)
- [Autoscaler](#autoscaler)
- [Datasets](#datasets)
- [I/O and streaming](#io-and-streaming)
- [Checkpointing](#checkpointing)
- [CLI helpers](#cli-helpers)
- [Benchmarking utilities](#benchmarking-utilities)
- [Related guides](#related-guides)

## Trainers
Located in [`gpu_tokenizer/bpe_trainer.py`](../gpu_tokenizer/bpe_trainer.py) and [`gpu_tokenizer/unigram_trainer.py`](../gpu_tokenizer/unigram_trainer.py).

### `GPUBPETrainer`
- **Purpose**: Learns merge operations using GPU kernels.
- **Key methods**:
  - `__init__(dataset, autoscaler, *, merges, token_bytes, ...)`
  - `train(progress_logger)` – Executes the training loop, emitting checkpoints and telemetry.
  - `save(out_dir)` – Persists learned vocabulary artifacts.
- **Extension points**: Override merge scoring kernels, customize checkpoint schema, or plug in alternative token filters.

### `GPUUnigramTrainer`
- **Purpose**: Performs expectation-maximization to learn unigram vocabularies.
- **Key methods**:
  - `__init__(dataset, autoscaler, *, vocab_size, epochs, ...)`
  - `train(progress_logger)` – Coordinates EM updates, pruning, and autoscaler feedback.
  - `save(out_dir)` – Persists vocabulary and probability tables.
- **Extension points**: Swap out scoring heuristics, customize smoothing, or integrate additional pruning stages.
- **Device support**: Executes the candidate search, forward/backward scoring, and pruning loops on either CUDA or CPU based on the configured `device`.

Both trainers rely on the dataset and autoscaler layers described below. Reusing them is the fastest way to implement new algorithms; see the [architecture overview](architecture.md#trainer-pipeline) for lifecycle context.

## Autoscaler
Defined in [`gpu_tokenizer/autoscaler.py`](../gpu_tokenizer/autoscaler.py).

### `AutoScaler`
- **Purpose**: Adjusts batch sizes to maintain target GPU utilization.
- **Key methods**:
  - `suggest_batch_size(metrics)` – Returns the next batch size given throughput telemetry.
  - `update(metrics)` – Feeds raw measurements into the controller.
  - `state_dict()` / `load_state_dict(state)` – Serialize/restore autoscaler state for checkpointing.
- **Configuration**: Accepts target utilization, smoothing windows, minimum/maximum batch bounds, and cooldown parameters.
- **Customization**: Subclass to implement alternative control policies or override smoothing strategies.

The autoscaler is invoked by trainers and referenced by CLI options such as `--target-util`. Review the [CLI usage guide](cli.md#global-flags) for how users configure it.

## Datasets
Entry points live in [`gpu_tokenizer/datasets/__init__.py`](../gpu_tokenizer/datasets/__init__.py).

- **`IterableCorpus`**: Abstract base that yields byte sequences for packing.
- **`FileShardDataset`**: Streams text files using glob patterns and optional compression.
- **`SyntheticCorpus`**: Generates random documents for benchmarking and stress tests.
- **`ChainedCorpus`**: Combines multiple corpora while maintaining consistent interfaces.

Datasets are composable: you can wrap corpora with filters, sampling policies, or metadata enrichers before handing them to a trainer. The [architecture overview](architecture.md#dataset-and-streaming-layers) illustrates how these iterables feed the GPU.

## I/O and Streaming
Defined across [`gpu_tokenizer/io/__init__.py`](../gpu_tokenizer/io/__init__.py) and helper modules.

- **Compression adapters**: Dispatch to codec-specific readers (`zstd`, `lz4`, or plain text).
- **Memory mapping**: Utilities for mapping shards into shared buffers when the filesystem permits.
- **Prefetch workers**: Threaded or process-based pools that populate queues consumed by the packers.
- **Backpressure signals**: Interfaces that allow the autoscaler to inform dataset loaders about pending work.

These components are wired together by CLI options described in the [Streaming options](cli.md#streaming-options) section of the usage guide.

## Checkpointing
Shared helpers reside alongside trainers and datasets.

- **`CheckpointManager`** (if present) coordinates writing trainer state, autoscaler snapshots, and metadata.
- **Trainer hooks**: `save_checkpoint()` / `load_checkpoint()` methods capture merge tables, unigram weights, and dataset cursors.
- **CLI integration**: `--checkpoint-dir`, `--checkpoint-every`, and `--resume-from` map to the helpers listed above.

When designing new trainers, implement `state_dict()`/`load_state_dict()` pairs so checkpoints remain compatible with the existing resume logic. The [CLI usage guide](cli.md#checkpointing-and-resume) documents the user-facing switches.

## CLI Helpers
Within [`main.py`](../main.py):

- **`build_parser()`** – Declares subcommands and shared options.
- **`main(argv=None)`** – Entry point that dispatches to subcommand handlers.
- **Handler functions** – Each subcommand has a dedicated handler that constructs datasets, trainers, and autoscalers.

Reuse these helpers when embedding SuperToken into larger applications or notebooks. The design expectations are summarized in the [architecture overview](architecture.md#cli-integration).

## Benchmarking Utilities
Located under [`benchmarks/`](../benchmarks/).

- **`run_benchmark()`** – Executes trainers against provided corpora and aggregates metrics.
- **Telemetry writers** – Format tabular output and JSON payloads for downstream consumption.
- **Scenario builders** – Compose synthetic and real datasets to match specific evaluation goals.

Benchmarking relies on the same dataset and autoscaling primitives, ensuring metrics remain comparable. Consult the [CLI usage guide](cli.md#benchmark) for command-line invocation and the [architecture overview](architecture.md#benchmarking-workflow) for conceptual flow.

## Related Guides
- [Architecture overview](architecture.md)
- [CLI usage guide](cli.md)
- [Performance notes and benchmarks](performance.md)
- [Module guide](modules.md)
