# CLI Usage Guide

SuperToken ships a single `main.py` entry point that exposes tokenizer training and benchmarking workflows. This guide expands on the quick-start snippets from the [README](../README.md#quick-start) and connects them to the system design described in the [architecture overview](architecture.md).

## Quick Navigation
- [Global flags](#global-flags)
- [`train-bpe`](#train-bpe)
- [`train-unigram`](#train-unigram)
- [`benchmark`](#benchmark)
- [Streaming options](#streaming-options)
- [Checkpointing and resume](#checkpointing-and-resume)
- [Extending the CLI](#extending-the-cli)
- [Related guides](#related-guides)

## Global Flags
Each subcommand inherits a shared set of options that configure inputs, autoscaling, and logging:

| Flag | Description |
| --- | --- |
| `--data` | One or more glob patterns pointing to UTF-8 text shards. |
| `--compression` | Compression codec (`none`, `zstd`, or `lz4`) to apply while reading shards. |
| `--io-workers` | Number of background workers used for streaming ingestion. |
| `--prefetch-batches` | Size of the host-side batch queue that feeds the GPU. |
| `--target-util` | Desired GPU utilization ratio (0–1) forwarded to the autoscaler. |
| `--log-json` | Emit JSON-formatted logs suitable for ingestion by dashboards. |

These options map directly onto the streaming and autoscaling layers documented in the [architecture overview](architecture.md#dataset-and-streaming-layers) and the [API reference](api.md#autoscaler).

## `train-bpe`
Train a byte-pair encoding tokenizer backed by GPU kernels:

```bash
python main.py train-bpe \
  --data "data/**/*.txt" \
  --merges 50000 \
  --token-bytes 8192 \
  --target-util 0.85 \
  --out-dir ./artifacts/bpe
```

Key arguments:

- `--merges`: Number of merge operations to learn.
- `--token-bytes`: Maximum packed byte length per batch.
- `--checkpoint-dir`: Directory for periodic checkpoints.
- `--checkpoint-every`: Number of steps between checkpoints.

Behind the scenes the [`GPUBPETrainer`](../gpu_tokenizer/bpe_trainer.py) collaborates with the autoscaler, dataset readers, and checkpoint writers described in the [architecture overview](architecture.md#trainer-pipeline).

## `train-unigram`
Train a unigram tokenizer with GPU-accelerated expectation-maximization loops:

```bash
python main.py train-unigram \
  --data "data/**/*.txt" \
  --vocab-size 50000 \
  --epochs 3 \
  --out-dir ./artifacts/unigram
```

Important options:

- `--vocab-size`: Target vocabulary size after pruning.
- `--epochs`: Number of passes over the corpus.
- `--min-prob`: Optional probability floor for retention.

The [`GPUUnigramTrainer`](../gpu_tokenizer/unigram_trainer.py) reuses the same streaming and autoscaling primitives; refer to the [API reference](api.md#trainers) for extension hooks.

## `benchmark`
Compare trainers using real and synthetic corpora in a single run:

```bash
python main.py benchmark \
  --data "data/**/*.txt" \
  --max-real-docs 1000 \
  --synthetic-docs 2000 \
  --synthetic-min-len 16 \
  --synthetic-max-len 64 \
  --output-dir ./artifacts/benchmarks
```

This command orchestrates both trainers with consistent autoscaler targets, records throughput telemetry, and emits JSON summaries. The benchmarking harness is described in the [architecture overview](architecture.md#benchmarking-workflow) and the [API reference](api.md#benchmarking-utilities).

## Streaming Options
Fine-tune ingestion to match storage and hardware characteristics:

- `--compression`: Choose codecs compatible with your shards. The I/O layer auto-detects file extensions when possible.
- `--io-workers`: Increase for high-latency storage so that GPU kernels stay busy.
- `--prefetch-batches`: Buffer batches on the host to absorb variability in shard sizes.
- `--stream-chunk-bytes`: (if enabled) Control chunk size when reading from object storage.

These flags interact with the dataset abstractions outlined in the [architecture overview](architecture.md#dataset-and-streaming-layers). Inspect the [API reference](api.md#datasets) for programmatic configuration.

## Checkpointing and Resume
Both trainers support checkpointing so long-running runs can survive interruptions:

- `--checkpoint-dir`: Location where checkpoints and autoscaler snapshots are stored.
- `--checkpoint-every`: Step interval between snapshots.
- `--resume-from`: Restore the latest checkpoint from a directory.

When resuming, the CLI rebuilds trainers, reloads autoscaler state, and seeks the dataset streams accordingly. Implementation specifics live in the [API reference](api.md#checkpointing).

## Extending the CLI
To add a new command:

1. Open [`main.py`](../main.py) and extend the `build_parser` function with your subcommand and options.
2. Implement a handler that wires command-line arguments to a trainer or benchmarking routine.
3. Reuse the shared logging, autoscaling, and dataset helpers exposed in the [API reference](api.md).

Because trainers are composed of modular building blocks, new commands can focus on domain-specific orchestration without rewriting streaming or autoscaling code. Consult the [architecture overview](architecture.md) for design constraints and expectations around batch lifecycles.

## Related Guides
- [Architecture overview](architecture.md)
- [API reference](api.md)
- [Performance notes and benchmarks](performance.md)
- [Module guide](modules.md)
