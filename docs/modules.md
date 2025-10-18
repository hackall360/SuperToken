# Module Guide

This guide provides deeper context for the core modules that power SuperToken's GPU-native tokenization workflows. Each section points to the primary modules, common extension points, and related helper utilities so you can navigate the package quickly.

## Trainers

The training frontends live in [`gpu_tokenizer/bpe_trainer.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/bpe_trainer.py) and [`gpu_tokenizer/unigram_trainer.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/unigram_trainer.py). They orchestrate dataset streaming, invoke CUDA/Triton kernels, and persist vocabularies and merge tables. When extending the trainers:

- Look at the `TrainerConfig` dataclasses for knobs around checkpointing, merge export, and normalization behaviors.
- Reuse the packing helpers from [`gpu_tokenizer/cpu_packer.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/cpu_packer.py) to prepare byte tensors efficiently.
- Integrate new kernels by following the call sites in the `step` and `finalize` methods.

## Autoscaler

[`gpu_tokenizer/autoscaler.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/autoscaler.py) houses the adaptive batching logic that keeps GPUs saturated. The `AutoScaler` class exposes:

- `observe_throughput` hooks you can call with step timing information.
- `suggest_batch_size` to derive the next batch size based on utilization targets and guardrails.
- Configuration flags for warmup steps, min/max batch sizes, and smoothing windows.

If you need to plug in custom heuristics, subclass `AutoScaler` and override `_compute_adjustment`. The trainers accept drop-in replacements through their configuration objects.

## Streaming I/O

The streaming pipeline is split across [`gpu_tokenizer/datasets.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/datasets.py) and [`gpu_tokenizer/io.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/io.py). Together they:

- Support globbed input shards with optional compression (`none`, `zstd`, `lz4`).
- Provide background worker pools for asynchronous prefetch and decompression.
- Expose synthetic corpus generators used by the benchmarking suite for reproducible experiments.

For large-scale deployments, start with the dataset factories in `datasets.py` and customize the `StreamingShardLoader` or compression codecs. The IO layer is intentionally modular so you can bring your own storage backends or sampling strategies without rewriting the trainers.
