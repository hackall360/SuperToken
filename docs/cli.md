# CLI Usage Guide

SuperToken ships a single `main.py` entry point that exposes tokenizer training and benchmarking workflows. This guide expands on the quick-start snippets from the [README](https://github.com/example/SuperToken/blob/main/README.md#quick-start) and connects them to the system design described in the [architecture overview](architecture.md).

## Quick Navigation
- [Command matrix](#command-matrix)
- [Global flags](#global-flags)
- [Code mode workflows](#code-mode-workflows)
- [Augmentation options](#augmentation-options)
- [`train-bpe`](#train-bpe)
- [`resume-bpe`](#resume-bpe)
- [`train-unigram`](#train-unigram)
- [`train-hybrid`](#train-hybrid)
- [Augmentation cookbook](#augmentation-cookbook)
- [Privacy options](#privacy-options)
- [`benchmark`](#benchmark)
- [`evaluate`](#evaluate)
- [Streaming options](#streaming-options)
- [Checkpointing and resume](#checkpointing-and-resume)
- [Extending the CLI](#extending-the-cli)
- [Related guides](#related-guides)

## Command matrix

| Command | Primary workflow | Notable flags |
| --- | --- | --- |
| `train-bpe` | Fit a merge-based vocabulary with GPU acceleration. | `--merges`, `--token-bytes`, `--checkpoint-dir`, `--augmentation`, `--code-mode`, `--morphology-*`, `--privacy*` |
| `resume-bpe` | Continue a previous BPE run using stored checkpoints. | `--resume-from`, `--checkpoint-dir`, `--augmentation`, `--code-mode`, `--morphology-*` |
| `train-unigram` | Optimise unigram vocabularies with EM loops. | `--vocab-size`, `--epochs`, `--base-vocab`, `--max-subword-len`, `--augmentation`, `--code-mode`, `--morphology-*` |
| `train-hybrid` | Alternate between BPE warm-up and unigram refinement. | `--cycles`, `--unigram-epochs`, `--warm-start`, `--checkpoint-dir`, `--augmentation`, `--code-mode`, `--morphology-*`, `--privacy*` |
| `export-embeddings` | Convert vocabularies into embedding packages with pruning. | `--vocab`, `--stats`, `--dedupe-similarity`, `--min-frequency`, `--keep-token`, `--dimension`, `--dtype`, `--seed` |
| `evaluate` | Produce JSON reports that score artifacts against a reference corpus. | `--data`, `--artifacts`/`--vocab`/`--merges`/`--tokenizer`, `--deterministic`, `--meta-max-length`, `--code-mode`, `--code-langs`, `--meta-compress`, `--morphology-*`, `--output` |
| `benchmark` | Compare trainers on synthetic and real corpora. | `--scenarios`, `--gpus`, `--steps`, `--profile`, `--dist`, `--unigram-*`, `--hybrid-*` |

Use the matrix to spot which CLI options overlap across workflows. Each command also inherits the [global flags](#global-flags) listed below.

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

## Augmentation options

Training subcommands accept a pair of flags that inject lightweight data augmentation into the packing pipeline:

| Flag | Description |
| --- | --- |
| `--augmentation` | Selects the augmentation policy. Valid values are `none` (default), `entropy`, and `diffusion`. |
| `--aug-strength` | Floating-point knob that controls the intensity of the chosen policy; `0` disables augmentation. |

Both parameters feed the shared augmentation module used by the pre-packed (`PackedBatcher`) and streaming (`StreamingPackedBatcher`) datasets. Augmentations respect the global `--seed`, ensuring that runs remain reproducible when deterministic shuffles are required. The current release includes two policies:

- **entropy** randomly drops tokens with probability equal to `--aug-strength` (clamped to `[0, 1]`). At least one token always survives.
- **diffusion** performs local token swaps; `--aug-strength` represents the fraction of the sequence that participates in swaps (again clamped to `[0, 1]`).

Changing augmentation settings mid-run does not retroactively mutate batches restored from checkpoints—the CLI reuses the serialized data recorded on disk. Streaming jobs and freshly materialised batches honour the requested policy on their next pass through the dataset. High strengths can noticeably distort short documents, so start with conservative values (for example `0.1`) when experimenting.

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
- `--augmentation` / `--aug-strength`: Enable on-the-fly corpus augmentation for each packed sequence (see [Augmentation options](#augmentation-options)).
- `--code-mode`: Preprocess structured code entries instead of byte streams. When enabled the trainer loads corpora eagerly and ignores autoscaler resize events.
- `--code-langs`: Restrict code-mode runs to specific languages (for example `--code-langs python typescript`).
- `--meta-compress`: Discover and apply meta-token compression on AST sequences.
- `--morphology-lang` / `--morphology-case-markers` / `--morphology-affix-tags`: Opt into language-aware segmentation and optional annotations.
- `--privacy` / `--privacy-salt` / `--tie-seed`: Opt into merge redaction and tie randomization (see [Privacy options](#privacy-options)).
- `--warm-start`: Prime the trainer with merges described by a JSON manifest, a TikToken bundle (`merges.tiktoken`), or Hugging Face tokenizer artifacts (`tokenizer.json` or `vocab.json`/`merges.txt`).

Both `--warm-start` and `--resume-from` recognise Hugging Face tokenizer bundles, allowing existing vocabularies to bootstrap or continue GPU training runs without manual conversion.

The `train-bpe` command writes Hugging Face compatible `vocab.json`, `merges.txt`, and `tokenizer.json` files alongside a TikToken-formatted `merges.tiktoken` table that can be consumed by the [`tiktoken`](https://github.com/openai/tiktoken) package or reused as a warm-start seed.

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

The command mirrors every flag accepted by [`train-bpe`](#train-bpe)—including autoscaler knobs such as `--target-util`, batch limits (`--min-batch`/`--max-batch`), privacy guards, and checkpoint writers—so operators can tweak settings or hand a run off to another machine without rewriting scripts. The only additional requirement is `--resume-from`, which may point at a checkpoint directory created by `--checkpoint-dir` or a Hugging Face tokenizer bundle containing `tokenizer.json` or `vocab.json`/`merges.txt`. The handler validates the presence of both `--resume-from` and the normal `--data` globs before delegating to the same training routine used by a fresh run.

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
- `--augmentation` / `--aug-strength`: Apply augmentation to each batch before it reaches the trainer (see [Augmentation options](#augmentation-options)).
- `--dry-run`: Instantiate the trainer, log the resolved configuration, and exit without fitting epochs.
- `--code-mode`: Enable the code-mode ingestion pipeline for JSON or source code repositories.
- `--code-langs`: Optional allowlist applied when `--code-mode` is set.
- `--meta-compress`: Toggle meta-token discovery while preparing code-mode corpora.
- `--morphology-lang` / `--morphology-case-markers` / `--morphology-affix-tags`: Apply morphology segmentation before packing batches.
- `--checkpoint-dir`: Directory where the trainer writes checkpoints. Combine with `--resume-from` to continue a run.
- `--checkpoint-every`: Emit checkpoints every N epochs when `--checkpoint-dir` is set (defaults to final-only when zero).
- `--warm-start`: Seed the trainer from a SentencePiece `.model` or `.vocab`/`.prob` bundle.
- `--resume-from`: Restore optimiser state and epoch history from a previous run or import a SentencePiece model before continuing. The remaining epochs honour the current CLI flags.
- `--time-minutes`: Optional wall-clock budget that pauses training after the requested number of minutes, writing a checkpoint when configured.

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
- `--warm-start`: Optional path to a JSON manifest or TikToken bundle containing seeded merge pairs.
- `--checkpoint-dir`: Directory where per-cycle checkpoints are stored.
- `--checkpoint-every`: Write the hybrid checkpoint after every N cycles; when omitted or zero only the final state is saved.
- `--resume-from`: Restore the previous cycle state and merge history before continuing with additional cycles.
- `--time-minutes`: Optional wall-clock guard that stops training once the elapsed time reaches the requested number of minutes.
- `--code-mode`: Run both phases on AST-linearised corpora.
- `--code-langs`: Filter the code-mode corpus to specific languages.
- `--meta-compress`: Share meta-token compression dictionaries across BPE and unigram phases.
- `--augmentation` / `--aug-strength`: Apply the augmentation pipeline consistently across the BPE and unigram phases.
- `--morphology-lang` / `--morphology-case-markers` / `--morphology-affix-tags`: Carry morphology preprocessing through both phases.
- `--privacy` / `--privacy-salt` / `--tie-seed`: Apply the same privacy guard as the standalone BPE trainer to hybrid manifests.

After training the command emits a `hybrid_manifest.json`, Hugging Face-compatible `merges.txt`/`tokenizer.json`, and a SentencePiece-style `unigram.prob`/`unigram.model` pair for downstream consumers. The [`HybridTrainer`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/trainers/hybrid.py) section of the API reference describes the orchestration hooks in more detail.

## Augmentation cookbook

Augmentation is opt-in; the examples below illustrate common scenarios:

- **Stochastic masking for regularisation** – remove a small percentage of tokens to encourage robustness: `python main.py train-bpe --data "data/**/*.txt" --merges 50000 --augmentation entropy --aug-strength 0.1 --dry-run` (swap `train-bpe` for `train-unigram`/`train-hybrid` as needed).
- **Order-robust shuffling** – soften the importance of local token order without destroying content: `python main.py train-unigram --data data/train.txt --augmentation diffusion --aug-strength 0.3 --epochs 3`.
- **Disable augmentation** – omit both flags (or set `--augmentation none`) to retain the original corpus. This is the default behaviour for all commands.

Remember that augmentation parameters combine with `--seed`; supply an explicit seed to reproduce the same stochastic transformations across reruns.

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

## `evaluate`

Generate a deterministic JSON report that benchmarks trained artifacts against a reference corpus. The command mirrors the CLI help banner by spelling out the required inputs, supported artifact layouts, and resulting outputs:

```bash
python main.py evaluate \
  --data tests/data/evaluate_corpus/plain.txt \
  --artifacts tests/data/models/bpe \
  --morphology-lang tr \
  --deterministic \
  --summary-format table
```

Key switches:

- `--artifacts`: Directory containing `vocab.json`, `merges.(json|txt)`, and optionally `tokenizer.json`. Override any component via `--vocab`, `--merges`, or `--tokenizer`. SentencePiece exports expose `unigram.vocab` and can omit merge histories.
- `--model-type`: Force `bpe` or `unigram` detection when the defaults (`auto`) are ambiguous. BPE runs require a merge history; unigram runs expect SentencePiece-style vocabularies and ignore merges.
- `--output`: File path where the structured report should be written. When omitted the command writes to `reports/evaluate.json`; use `--output -` to stream JSON to stdout.
- `--deterministic`: Sort OOV listings and meta-token summaries so repeated runs remain byte-identical.
- `--meta-max-length`: Upper bound on meta-token discovery length when `--code-mode` corpora are evaluated.
- `--code-mode` / `--code-langs` / `--meta-compress`: Reuse the AST-aware pipeline described in [Code mode workflows](#code-mode-workflows) when evaluating source code manifests.
- `--morphology-*`: The same segmentation toggles exposed by the training commands. When enabled the report records both the aggregate statistics and the resolved morphology configuration.
- `--summary-format`: Choose between a human-readable banner (`table`), machine-friendly output (`json`), or silence (`none`).

The command writes a JSON object with several top-level sections:

- `artifacts`: Canonical paths, vocabulary size, resolved `model_type`, and the number of merge rules applied while compressing byte sequences.
- `corpus`: Document counts, total bytes processed, and per-document averages.
- `compression`: Derived ratios such as `tokens_per_byte` and its reciprocal `bytes_per_token` after merge application.
- `oov`: Raw and relative out-of-vocabulary counts alongside the set of offending token ids or strings.
- `morphology`: Segment counts (and optional role/tag summaries) plus the captured CLI configuration.
- `code_mode`: The active ingestion mode (`plain` or `code`), sample breakdown, and the resolved code-mode configuration including meta-token statistics.

Schema-validated output ensures downstream consumers can rely on these sections. The canonical JSON schema lives at
[`docs/schemas/evaluate_report.schema.json`](schemas/evaluate_report.schema.json) and the CLI refuses to emit reports that do
not conform to it. The schema requires explicit configuration mirrors such as `morphology.config.enabled` and
`code_mode.config.languages_filter`, so dashboards can audit which toggles were active. A compact example looks like this:

```json
{
  "artifacts": {
    "merge_rules": 10,
    "merges": null,
    "tokenizer": "tokenizer.json",
    "vocab": "vocab.json",
    "model_type": "bpe",
    "vocab_size": 42000
  },
  "code_mode": {
    "config": {
      "enabled": false,
      "languages_filter": null,
      "meta_compress": false,
      "meta_max_length": 8
    },
    "documents": 2,
    "fallback_samples": 0,
    "languages": ["python"],
    "meta_compress": {},
    "meta_enabled": false,
    "meta_max_length": 8,
    "meta_token_count": 0,
    "mode": "plain",
    "reduction": 0.0
  },
  "compression": {
    "bytes_per_token": 1.23,
    "tokens_per_byte": 0.81
  },
  "corpus": {
    "average_bytes": 123.0,
    "average_tokens": 99.0,
    "documents": 2,
    "total_bytes": 246,
    "total_tokens": 198
  },
  "morphology": {
    "config": {"enabled": false},
    "enabled": false
  },
  "oov": {
    "instances": 0,
    "rate": 0.0,
    "unique": []
  }
}
```

Downstream workflows can load the JSON directly (for example with `json.load`) to integrate evaluation metrics into dashboards or CI assertions. Deterministic mode guarantees that identical corpora, artifacts, and flags produce identical files—ideal for golden snapshot testing.

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
All training commands support checkpointing so long-running runs can survive interruptions:

- `--checkpoint-dir`: Location where checkpoints and autoscaler snapshots are stored.
- `--checkpoint-every`: Step interval between snapshots (epochs for unigram, cycles for hybrid, merges for BPE).
- `--resume-from`: Restore the latest checkpoint from a directory.
- `--time-minutes`: Optional wall-clock guard that pauses training once the threshold is reached and leaves a checkpoint behind.

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
