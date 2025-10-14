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

Common flags include:

- `--data`: One or more glob patterns pointing at UTF-8 text shards.
- `--compression`: Choose between `none`, `zstd`, or `lz4` for shard decoding.
- `--io-workers` & `--prefetch-batches`: Control the background streaming pipeline.
- `--bos`/`--eos`: Optionally inject special token IDs during packing.

Run `python main.py --help` for a full list of options.

## Project Layout
```
.
├── main.py              # CLI entry point tying together trainers and utilities
├── gpu_tokenizer/       # Core GPU trainers, packing utilities, and dataset helpers
├── docs/                # Design notes and performance documentation
└── tests/               # Unit tests covering packing, IO, and trainer behavior
```

## Documentation
- [Performance notes and benchmarks](docs/performance.md)

Additional guides and API notes can be added under the `docs/` directory as the project grows.

## Contributing
1. Fork the repository and create a virtual environment.
2. Install development dependencies (see `pyproject.toml` if present).
3. Format your changes and ensure tests pass via `pytest`.
4. Open a pull request describing your changes and include benchmark results when appropriate.

## License
This project is distributed under the terms specified in the repository's license file.
