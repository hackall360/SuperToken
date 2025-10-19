# CLI Usage Guide

SuperToken ships a single `main.py` entry point that exposes tokenizer training and benchmarking workflows. This guide expands on the quick-start snippets from the [README](https://github.com/example/SuperToken/blob/main/README.md#quick-start) and connects them to the system design described in the [architecture overview](architecture.md).

## Quick Navigation
- [Global flags](#global-flags)
- [Code mode workflows](#code-mode-workflows)
- [`train-bpe`](#train-bpe)
- [`resume-bpe`](#resume-bpe)
- [`train-unigram`](#train-unigram)
- [`train-hybrid`](#train-hybrid)
- [Privacy options](#privacy-options)
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
| `--code-mode` | Enable the AST-based code preprocessing pipeline for JSON or source inputs. |
| `--code-langs` | Optional allowlist of languages to process when code mode is active. |
| `--meta-compress` | Discover and apply meta-token compression when code mode is active. |

These options map directly onto the streaming and autoscaling layers documented in the [architecture overview](architecture.md#dataset-and-streaming-layers) and the [API reference](api.md#autoscaler).

## Code mode workflows

Pass `--code-mode` to any training subcommand to swap the default byte-stream ingestion for an AST-aware pipeline tailored to source code repositories. Code mode accepts newline-delimited JSON (`*.jsonl`/`*.ndjson`), JSON manifests containing an `entries` array, or raw source files (`.py`, `.pyi`, `.ts`, `.tsx`, `.js`, `.jsx`). Each structured entry must provide a `language`, `source`, and optional `filename`. When raw files are supplied the language is inferred from the extension; the `--code-langs` allowlist filters entries that the current run should process.

AST linearisation produces canonical placeholder tokens for identifiers and literals while preserving metadata about the originating language and file. When `--meta-compress` is present the pipeline also discovers repeating token patterns and replaces them with synthetic `META*` markers to reduce downstream token counts. Any entry that fails AST parsing automatically falls back to byte-level tokenisation; the CLI flags these fallbacks in the emitted summaries so corpus issues remain visible.

Because code-mode corpora are pre-packed in memory, the BPE trainer ignores autoscaler resize requests and disables `--resume-from` checkpoints for these runs. Checkpointing continues to work for unigram and hybrid trainers because their batches are replayed locally.

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
- `--code-mode`: Preprocess structured code entries instead of byte streams. When enabled the trainer loads corpora eagerly and ignores autoscaler resize events.
- `--code-langs`: Restrict code-mode runs to specific languages (for example `--code-langs python typescript`).
- `--meta-compress`: Discover and apply meta-token compression on AST sequences.
- `--morphology-lang` / `--morphology-case-markers` / `--morphology-affix-tags`: Opt into language-aware segmentation and optional annotations.
- `--privacy` / `--privacy-salt` / `--tie-seed`: Opt into merge redaction and tie randomization (see [Privacy options](#privacy-options)).

Behind the scenes the [`GPUBPETrainer`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/bpe_trainer.py) collaborates with the autoscaler, dataset readers, and checkpoint writers described in the [architecture overview](architecture.md#trainer-pipeline).

### Distributed execution flags

Enable the heterogeneous multi-GPU workflow by invoking the dedicated launcher module with `--dist`:

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

The additional flags configure the distributed runtime:

| Flag | Description |
| --- | --- |
| `--dist` | Switch the trainer into distributed mode with a lease-based work queue. |
| `--gpus` | Comma-separated list of CUDA device indices claimed by the job. |
| `--lease-size` | Number of chunks granted per lease (per rank) before requesting more work. |
| `--rebalance-secs` | Interval between EWMA-based throughput rebalancing passes. |
| `--target-chunk-ms` | Desired processing time per chunk used to size leases dynamically. |

Every few iterations the CLI prints a per-rank status table that surfaces GPU IDs, the most recent tokens/sec readings, current lease sizes, inflight work count, and stage-level timings. Monitoring the table helps operators confirm that faster GPUs keep receiving new leases, slower cards remain productive, and no rank stalls on communication or reduction phases.

## `resume-bpe`

Resume a previously interrupted BPE run without retyping the original arguments:

```bash
python main.py resume-bpe \
  --data "data/**/*.txt" \
  --merges 50000 \
  --token-bytes 8192 \
  --resume-from ./artifacts/bpe/checkpoints
```

The command mirrors every flag accepted by [`train-bpe`](#train-bpe)—including autoscaler knobs such as `--target-util`, batch limits (`--min-batch`/`--max-batch`), privacy guards, and checkpoint writers—so operators can tweak settings or hand a run off to another machine without rewriting scripts. The only additional requirement is `--resume-from`, which must point at the checkpoint directory created by `--checkpoint-dir`. The handler validates the presence of both `--resume-from` and the normal `--data` globs before delegating to the same training routine used by a fresh run.

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
- `--base-vocab`: Size of the seed alphabet supplied to the trainer.
- `--max-subword-len`: Maximum subword length considered when constructing the candidate lattice.
- `--batch-size`: Number of packed documents replayed per batch.
- `--seed`: Shuffle seed controlling batch materialisation order.
- `--dry-run`: Instantiate the trainer, log the resolved configuration, and exit without fitting epochs.
- `--code-mode`: Enable the code-mode ingestion pipeline for JSON or source code repositories.
- `--code-langs`: Optional allowlist applied when `--code-mode` is set.
- `--meta-compress`: Toggle meta-token discovery while preparing code-mode corpora.
- `--morphology-lang` / `--morphology-case-markers` / `--morphology-affix-tags`: Apply morphology segmentation before packing batches.

The [`GPUUnigramTrainer`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/unigram_trainer.py) reuses the same streaming and autoscaling primitives; refer to the [API reference](api.md#trainers) for extension hooks.

## `train-hybrid`
Alternate between BPE warm-up phases and unigram refinement without leaving the CLI:

```bash
python main.py train-hybrid \
  --data "data/**/*.txt" \
  --merges 50000 \
  --cycles 2 \
  --unigram-epochs 2 \
  --out-dir ./artifacts/hybrid
```

Key arguments:

- `--cycles`: Number of BPE→unigram handoff rounds to execute.
- `--unigram-epochs`: Epochs to run inside each unigram phase.
- `--warm-start`: Optional path to a JSON manifest containing seeded merge pairs.
- `--checkpoint-dir`: Directory where per-cycle checkpoints are stored.
- `--code-mode`: Run both phases on AST-linearised corpora.
- `--code-langs`: Filter the code-mode corpus to specific languages.
- `--meta-compress`: Share meta-token compression dictionaries across BPE and unigram phases.
- `--morphology-lang` / `--morphology-case-markers` / `--morphology-affix-tags`: Carry morphology preprocessing through both phases.
- `--privacy` / `--privacy-salt` / `--tie-seed`: Apply the same privacy guard as the standalone BPE trainer to hybrid manifests.

After training the command emits a `hybrid_manifest.json`, Hugging Face-compatible `merges.txt`/`tokenizer.json`, and a SentencePiece-style `unigram.prob`/`unigram.model` pair for downstream consumers. The [`HybridTrainer`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/trainers/hybrid.py) section of the API reference describes the orchestration hooks in more detail.

## Privacy options

Both the `train-bpe` and `train-hybrid` subcommands accept `--privacy` with three modes:

- `none` *(default)* exports raw merge pairs and preserves deterministic tie-breaks.
- `hash-merges` replaces merge IDs with salted hashes in `bpe_merges.json`, `state.json`, and `hybrid_manifest.json`. Supply `--privacy-salt` with a hex or UTF-8 value to avoid identical hashes across runs; the salt itself is never written to disk.
- `tie-randomize` hashes merges and randomizes tie-breaks. This sacrifices deterministic parity across devices; provide `--tie-seed` to reproduce the stochastic ordering when needed.

Every manifest now emits a `privacy` block summarizing the active mode, whether merges were redacted, the effective tie seed, and whether a salt was configured. Downstream tooling should inspect this block instead of inferring privacy status from merge contents. Remember that enabling tie randomization changes merge selection order even when `--tie-seed` is supplied—use this mode only when deterministic parity is not required.

## `export-embeddings`

Transform a trained vocabulary into an embedding package, optionally pruning rarely used tokens:

```bash
python main.py export-embeddings \
  --vocab artifacts/bpe/vocab.json \
  --stats artifacts/bpe/stats.json \
  --dedupe-similarity 0.98 \
  --min-frequency 5 \
  --dimension 256 \
  --dtype float32 \
  --seed 7 \
  --out-dir artifacts/bpe/embeddings
```

> **Heads up:** Earlier documentation referenced `--token-stats`, `--embedding-dim`, and `--embedding-seed`. These options were
> renamed to `--stats`, `--dimension`, and `--seed` respectively. Update any saved scripts to keep exports reproducible.

Key switches:

- `--vocab`: Path to a tokenizer vocabulary JSON file (the CLI accepts both BPE and unigram outputs).
- `--stats`: Optional token usage statistics gathered during co-training; counts drive pruning and weight seeding.
- `--dedupe-similarity`: Cosine similarity threshold used to merge redundant tokens before pruning. Identical or near-identical vectors collapse into a single canonical token, and the pruning report records each merge alongside traditional removals.
- `--min-frequency`: Drop tokens whose observed frequency falls below the provided threshold (set to `0` to disable pruning).
- `--keep-token`: Repeatable flag that pins tokens regardless of frequency (for example `--keep-token <pad>`).
- `--dimension` / `--dtype` / `--seed`: Control the shape and initialization of synthesized vectors.

The command writes four artifacts—`vocab.json`, `embeddings.json`, `manifest.json`, and `pruning.json`—and logs a summary describing how many tokens were deduplicated, how many were pruned after deduplication, and which specials were preserved. Deduplication runs before pruning, honours `--keep-token`, and the resulting pruning log interleaves merged-token metadata with frequency-based removals so downstream tooling can distinguish between the two actions. See [docs/api.md](api.md#embedding-exports) for programmatic access to the export helpers.

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

When resuming, the CLI rebuilds trainers, reloads autoscaler state, and seeks the dataset streams accordingly. The dedicated [`resume-bpe`](#resume-bpe) subcommand wires these options into the same handler as `train-bpe`, making it easy to pick up an interrupted job or continue training on a different node. Implementation specifics live in the [API reference](api.md#checkpointing).

## Extending the CLI
To add a new command:

1. Open [`main.py`](https://github.com/example/SuperToken/blob/main/main.py) and extend the `build_parser` function with your subcommand and options.
2. Implement a handler that wires command-line arguments to a trainer or benchmarking routine.
3. Reuse the shared logging, autoscaling, and dataset helpers exposed in the [API reference](api.md).

Because trainers are composed of modular building blocks, new commands can focus on domain-specific orchestration without rewriting streaming or autoscaling code. Consult the [architecture overview](architecture.md) for design constraints and expectations around batch lifecycles.

## Related Guides
- [Architecture overview](architecture.md)
- [API reference](api.md)
- [Performance notes and benchmarks](performance.md)
- [Module guide](modules.md)
