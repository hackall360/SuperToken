# SuperToken

<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white&style=for-the-badge" alt="Python 3.10+" />
<img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Status: Active" />
<a href="docs/performance.md"><img src="https://img.shields.io/badge/Docs-Performance-blue?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation" /></a>
<img src="https://img.shields.io/badge/Made%20with-%E2%9D%A4-red?style=for-the-badge" alt="Made with Love" />

</div>

SuperToken is a GPU-accelerated tokenizer toolkit that offers high-throughput byte-pair and unigram training pipelines. It combines streaming data ingestion, adaptive batch sizing, and GPU-friendly packing utilities to keep your accelerators busy while you iterate on vocabulary design.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
- [Benchmarking](#benchmarking)
- [Architecture & API Overview](#architecture--api-overview)
- [Project Layout](#project-layout)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features
- **GPU-native trainers** for both Byte Pair Encoding (BPE) and unigram vocabularies via `GPUBPETrainer` and `GPUUnigramTrainer`.
- **Adaptive autoscaling** batch suggestion system to maintain target GPU utilization using the `AutoScaler` utility.
- **Streaming corpus ingestion** with optional compression, memory-mapped shards, and background worker prefetch.
- **Packed sequence helpers** that minimize host-device transfers and keep kernels fed with contiguous bytes.

## Installation
This project requires Python 3.10+ and a working PyTorch installation with CUDA support.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

> **Note:** Editable installs make it easy to iterate on the library modules in `gpu_tokenizer/` while running the CLI.

## Quick Start
Train a BPE model against a directory of text shards:

```bash
python main.py train-bpe \
  --data "data/**/*.txt" \
  --merges 50000 \
  --token-bytes 8192 \
  --target-util 0.85 \
  --out-dir ./artifacts/bpe
```

Need resilience against interruptions? The BPE trainer now supports periodic checkpointing and seamless resume:

```bash
python main.py train-bpe \
  --data "data/**/*.txt" \
  --merges 50000 \
  --checkpoint-dir ./artifacts/bpe-checkpoints \
  --checkpoint-every 2000

# Later, continue from the most recent checkpoint:
python main.py train-bpe \
  --data "data/**/*.txt" \
  --merges 50000 \
  --resume-from ./artifacts/bpe-checkpoints \
  --checkpoint-dir ./artifacts/bpe-checkpoints \
  --checkpoint-every 2000
```

The CLI will restore the autoscaler state, on-device batches, and resume streaming where it left off while logging each checkpoint save/restore event.

Train a unigram model with a fixed vocabulary size:

```bash
python main.py train-unigram \
  --data "data/**/*.txt" \
  --vocab-size 50000 \
  --epochs 3 \
  --out-dir ./artifacts/unigram
```

Both commands will automatically adapt the batch size in response to your GPU throughput and persist the resulting vocabulary files.

## Command Reference
The CLI is organized into subcommands that share a common set of arguments.

| Command | Description | Highlights |
| --- | --- | --- |
| `train-bpe` | Trains a GPU-accelerated BPE tokenizer. | Autoscaled batch sizing, streaming ingestion, optional on-the-fly merges export. |
| `train-unigram` | Trains a GPU-accelerated unigram tokenizer. | Epoch-based training with configurable vocab size and subword length. |
| `benchmark` | Runs both trainers against synthetic and/or real corpora. | Emits comparative tables and JSON telemetry snapshots. |

## Benchmarking

Run the bundled benchmark to compare the BPE and unigram trainers with a single command. The example below synthesizes 2,000 sentences while also sampling up to 1,000 documents from your dataset globs:

```bash
python main.py benchmark \
  --data "data/**/*.txt" \
  --max-real-docs 1000 \
  --synthetic-docs 2000 \
  --synthetic-min-len 16 \
  --synthetic-max-len 64 \
  --output-dir ./artifacts/benchmarks
```

Sample output:

```
Corpus → 2500 sequences, 102400 tokens (max len 128)
Trainer           | Wall time (s) | Tokens/s    | Final vocab
------------------+---------------+-------------+------------
GPUBPETrainer     | 12.84         | 7975.19     | 50256
GPUUnigramTrainer | 8.42          | 12158.52    | 50000
Saved benchmark metadata → artifacts/benchmarks/benchmark_20240101T120000Z.json
```

The benchmark will always emit a pretty-printed comparison table and serialize the full telemetry payloads into timestamped JSON files under the requested output directory. Those JSON artifacts capture the raw trainer metadata, corpus descriptors, and the CLI configuration so runs are fully reproducible.

Common flags include:

- `--data`: One or more glob patterns pointing at UTF-8 text shards.
- `--compression`: Choose between `none`, `zstd`, or `lz4` for shard decoding.
- `--io-workers` & `--prefetch-batches`: Control the background streaming pipeline.
- `--bos`/`--eos`: Optionally inject special token IDs during packing.

Run `python main.py --help` for a full list of options.

## Architecture & API Overview

SuperToken is organized into modular layers that can be reused independently or combined through the CLI:

- **Autoscaling (`gpu_tokenizer.autoscaler`)** – Provides the `AutoScaler` class that tracks throughput telemetry and surfaces `suggest_batch_size` helpers for trainers. See the inline docstrings in [`gpu_tokenizer/autoscaler.py`](gpu_tokenizer/autoscaler.py) for configuration knobs and extension hooks around utilization targets.
- **BPE training (`gpu_tokenizer.bpe_trainer`)** – Implements `GPUBPETrainer`, merging heuristics, and checkpoint serialization. This module integrates directly with the autoscaler and exposes hooks for custom merge filters; refer to [`gpu_tokenizer/bpe_trainer.py`](gpu_tokenizer/bpe_trainer.py).
- **Unigram training (`gpu_tokenizer.unigram_trainer`)** – Offers `GPUUnigramTrainer` plus scoring utilities for probabilistic vocabularies. Docstrings in [`gpu_tokenizer/unigram_trainer.py`](gpu_tokenizer/unigram_trainer.py) describe how to plug in custom smoothing or constraint logic.
- **Datasets & packing (`gpu_tokenizer.datasets`)** – Houses streaming dataset abstractions, packing helpers, and synthetic corpus generators used by both trainers. See [`gpu_tokenizer/datasets/__init__.py`](gpu_tokenizer/datasets/__init__.py) and the submodules it re-exports.
- **I/O pipeline (`gpu_tokenizer.io`)** – Encapsulates shard decoding, compression handling, and background workers. Start with [`gpu_tokenizer/io/__init__.py`](gpu_tokenizer/io/__init__.py) and follow the module-level docs for extension points.
- **CLI composition (`main.py`)** – Declares the `train-bpe`, `train-unigram`, and `benchmark` subcommands. You can register new commands by extending the `build_parser` function and wiring your trainers to the shared autoscaler utilities.
- **Benchmark utilities (`benchmarks/`)** – Contains reusable benchmarking harnesses and report formatters. Module docstrings point to upcoming narrative guides under `docs/benchmarks/` for more complex scenarios.

Future deep dives will land in the `docs/` directory (see [`docs/architecture.md`](docs/architecture.md)) and will mirror the high-level flow described here.

## Project Layout
```
.
├── main.py              # CLI entry point tying together trainers and utilities
├── gpu_tokenizer/       # Core GPU trainers, packing utilities, and dataset helpers
├── docs/                # Design notes and performance documentation
└── tests/               # Unit tests covering packing, IO, and trainer behavior
```

## Documentation
- [Architecture & API Overview](#architecture--api-overview): Start here for a quick summary of major modules, CLI extension points, and benchmarking utilities.
- [Architecture overview](docs/architecture.md): High-level map of the CLI, core modules, and benchmarks so you can navigate the codebase quickly.
- [Performance notes and benchmarks](docs/performance.md): Guidance on measuring throughput, interpreting telemetry, and tuning GPU utilization.

Additional guides and API notes can be added under the `docs/` directory as the project grows.

## Contributing
1. Fork the repository and create a virtual environment.
2. Install development dependencies (see `pyproject.toml` if present).
3. Format your changes and ensure tests pass via `pytest`.
4. Open a pull request describing your changes and include benchmark results when appropriate.

## License
This project is licensed under the [Mozilla Public License 2.0](LICENSE).
