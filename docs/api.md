# API Reference

This reference collects the most commonly extended classes and functions in SuperToken. It complements the system-level view from the [architecture overview](architecture.md) and the workflows covered in the [CLI usage guide](cli.md).

## Quick Navigation
- [Trainers](#trainers)
- [HybridTrainer](#hybridtrainer)
- [Autoscaler](#autoscaler)
- [Datasets](#datasets)
- [Morphology plugins](#morphology-plugins)
- [Code mode helpers](#code-mode-helpers)
- [I/O and streaming](#io-and-streaming)
- [Checkpointing](#checkpointing)
- [Privacy controls](#privacy-controls)
- [Embedding exports](#embedding-exports)
- [Evaluation reports](#evaluation-reports)
- [CLI helpers](#cli-helpers)
- [Benchmarking utilities](#benchmarking-utilities)
- [Related guides](#related-guides)

## Trainers
Located in [`gpu_tokenizer/bpe_trainer.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/bpe_trainer.py) and [`gpu_tokenizer/unigram_trainer.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/unigram_trainer.py).

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

### `HybridTrainer`
Located in [`gpu_tokenizer/trainers/hybrid.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/trainers/hybrid.py).

- **Purpose**: Alternates BPE warm-start phases with unigram refinement without reloading corpora.
- **Key methods**:
  - `__init__(dataset, autoscaler, *, merges, cycles, unigram_epochs, ...)` wires shared resources into the phased schedule.
  - `train(progress_logger)` orchestrates the BPE→unigram hand-offs, persisting per-cycle checkpoints and manifest updates.
  - `save(out_dir)` exports BPE merges, SentencePiece-compatible unigram artifacts, and the consolidated `hybrid_manifest.json`.
- **Privacy & morphology**: The trainer respects `privacy` configuration when emitting manifests and forwards morphology plugins to both phases so segmentation remains consistent.
- **Extension points**: Override `run_cycle()` to plug in additional evaluation passes or custom export hooks after each iteration.

## Autoscaler
Defined in [`gpu_tokenizer/autoscaler.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/autoscaler.py).

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
Entry points live in [`gpu_tokenizer/datasets/__init__.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/datasets/__init__.py).

- **`IterableCorpus`**: Abstract base that yields byte sequences for packing.
- **`FileShardDataset`**: Streams text files using glob patterns and optional compression.
- **`SyntheticCorpus`**: Generates random documents for benchmarking and stress tests.
- **`ChainedCorpus`**: Combines multiple corpora while maintaining consistent interfaces.

Datasets are composable: you can wrap corpora with filters, sampling policies, or metadata enrichers before handing them to a trainer. The [architecture overview](architecture.md#dataset-and-streaming-layers) illustrates how these iterables feed the GPU.

## Morphology plugins

Morphology preprocessing hooks live in [`gpu_tokenizer/morphology/`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/morphology). They are **opt-in**; the CLI
does not activate any plugin unless `--morphology-lang` is specified. Leaving the flag unset guarantees that byte sequences fed
into the trainers stay untouched so baseline token statistics remain reproducible.

- **`MorphologyPlugin`** – Abstract base defining the `presegment()` and `recompose()` contract.
- **`MorphologySegment`** – Lightweight dataclass describing a surface form, optional tags, and roles. Segments emitted from
  `presegment()` are consumed by the `BytePacker` when a plugin is active.
- **`available_plugins()` / `create_plugin(name, **config)`** – Discovery helpers used by the CLI. Plugins register themselves
  via `register_plugin(name, cls)` on import.

The built-in [`TurkishMorphologyPlugin`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/morphology/turkish.py) demonstrates the expected segmentation API and
honours `case_markers`/`affix_tags` toggles. When experimenting with new languages, start with `create_plugin("tr")` to inspect
the emitted segments and ensure `plugin.recompose(plugin.presegment(sample)) == sample` before feeding data into the trainers.
Refer to [docs/cookbook/morphology.md](cookbook/morphology.md) for an end-to-end training example with fidelity checks.

## Code mode helpers
The code-mode pipeline lives under [`gpu_tokenizer/code_mode/`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/code_mode). It transforms structured source samples into canonical token sequences that downstream trainers can consume.

- **`prepare_corpus(entries, meta_enabled=True, meta_max_length=8)`**: Accepts dictionaries containing `language`, `source`, and optional `filename` fields. Returns a `CodeModeCorpus` with AST-linearised samples, byte-level fallbacks, and an optional meta-token dictionary.
- **`MetaTokenCompressor`**: Discovers frequently occurring token runs and rewrites them as compact `META*` placeholders when `meta_enabled` is true. The compressor enforces a configurable maximum pattern length so merges remain interpretable.
- **`linearize_python_source` / `linearize_typescript_source`**: Language-specific front-ends that produce placeholder-rich token streams and symbol sidecars.

The CLI integrates these helpers via `--code-mode`, `--code-langs`, and `--meta-compress`. When AST parsing fails the pipeline emits byte-level fallbacks that retain metadata (`fallback=True`) so you can audit corpus quality. BPE code-mode runs load the entire corpus eagerly and therefore ignore autoscaler resize events and `--resume-from` checkpoints, whereas unigram and hybrid trainers operate on reusable in-memory batches.

## I/O and Streaming
Defined across [`gpu_tokenizer/io/__init__.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/io/__init__.py) and helper modules.

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

## Privacy controls
The BPE and hybrid trainers expose optional privacy guards that redact merge histories from exported artifacts:

- **`privacy_mode`** (bool) enables hashing of merge pairs in `bpe_merges.json` and `hybrid_manifest.json`. When active, the manifests emit a `privacy` block summarizing the redaction.
- **`randomize_ties`** (bool) toggles stochastic tie-breaking. The effective seed is recorded in the same `privacy` block so downstream consumers know when deterministic parity no longer holds. Pair this with `tie_seed` to reproduce stochastic merges across runs.
- **`privacy_salt`** (bytes/str) injects a caller-provided salt into the merge hashes. The salt itself is never written to disk; manifests only advertise that a salt was configured.

Every `state.json` produced by `save_checkpoint` now includes `payload["trainer"]["privacy"]` with the mode, merge redaction flag, tie seed metadata, and salt status. Consumers should inspect this section instead of inferring behavior from raw merge tables. Consult [docs/cli.md](cli.md#privacy-options) for CLI examples and the trade-offs between `none`, `hash-merges`, and `tie-randomize`.

## Embedding exports

Export helpers live in [`gpu_tokenizer/export/artifacts.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/export/artifacts.py) and bridge trained vocabularies with downstream embedding trainers.

- **`load_vocab(path)`**: Reads a tokenizer vocabulary JSON (token→id mapping) from disk.
- **`load_token_stats(path)`**: Loads optional token usage metadata produced during co-training. Counts and per-token vectors inform pruning and embedding seeding.
- **`dedupe_vocabulary(vocab, stats, *, similarity_threshold, dimension, seed, keep_tokens)`**: Clusters tokens whose usage vectors (or synthesized embeddings when vectors are absent) exceed the cosine similarity threshold. Returns a `DedupeResult` containing the merged vocabulary, aggregated statistics, and a log of deduplicated tokens.
- **`prune_vocabulary(vocab, stats, *, min_frequency, keep_tokens, original_size=None)`**: Produces a renumbered vocabulary while collecting pruned token metadata. The optional `original_size` parameter ensures downstream manifests retain visibility into the vocabulary size before deduplication.
- **`generate_embedding_matrix(vocab, stats, *, dimension, seed, dtype)`**: Synthesizes an embedding matrix aligned with the surviving vocabulary, reusing stored vectors when available.
- **`build_manifest(...)`**: Summarises export parameters (dimension, dtype, seed, min frequency, preserved tokens) for logging and reproducibility.
- **`write_export_package(out_dir, embeddings, vocab, manifest, pruned)`**: Persists the embeddings, pruned vocabulary, manifest, and bookkeeping JSON sidecars.

`dedupe_vocabulary` runs ahead of pruning so any merged tokens disappear from the exported vocabulary, yet the combined pruning log records both the deduplicated entries and subsequent frequency-based removals. Consumers that only care about frequency pruning can ignore entries whose `action` equals `deduped`.

The [`export-embeddings` CLI command](cli.md#export-embeddings) composes these helpers. Downstream applications can import `gpu_tokenizer.export` directly to embed SuperToken vocabularies into custom pipelines or notebooks.

When invoking the CLI, the helper arguments surface as `--stats`, `--dimension`, `--dtype`, and `--seed` so scripting aligns with the programmatic API. Prior drafts of the documentation referenced legacy `--token-stats`/`--embedding-*` flags; consult the [CLI usage guide](cli.md#export-embeddings) for the current spellings.

## Evaluation reports

The evaluation helpers live in [`gpu_tokenizer/evaluate.py`](https://github.com/example/SuperToken/blob/main/gpu_tokenizer/evaluate.py).

- **`evaluate(data_files, *, vocab_path, merges_path=None, tokenizer_path=None, bos=None, eos=None, morphology=None, code_mode=False, code_languages=None, meta_compress=False, meta_max_length=8, deterministic=False)`**
  - Loads the provided corpus (plain text or code manifests), applies optional morphology preprocessing, and materialises token sequences.
  - Applies merge rules to integer token streams when a merge file is supplied and computes aggregate metrics such as total tokens, `tokens_per_byte`, and out-of-vocabulary counts.
  - Returns a dictionary with the sections documented in [docs/cookbook/evaluate.md](cookbook/evaluate.md): `artifacts`, `corpus`, `compression`, `oov`, `morphology`, and `code_mode`. Each block is serialisable with `json.dump` and matches the CLI output.
  - Respecting `deterministic=True` seeds the random module so OOV listings and meta-token summaries remain stable—ideal for regression tests or golden reports.

Supporting dataclasses used by the implementation are also exported for advanced integrations:

- **`MergeRule(left, right, new_id)`** – typed representation of a merge pair applied during compression.
- **`LoadedCorpus(documents, tokens, raw_bytes, summary)`** – in-memory view of the evaluated corpus including raw byte totals and per-sample metadata.

Import the module with `from gpu_tokenizer import evaluate as eval_mod` to reuse the same logic that backs `python main.py evaluate`.

## CLI Helpers
Within [`main.py`](https://github.com/example/SuperToken/blob/main/main.py):

- **`build_parser()`** – Declares subcommands and shared options.
- **`main(argv=None)`** – Entry point that dispatches to subcommand handlers.
- **Handler functions** – Each subcommand has a dedicated handler that constructs datasets, trainers, and autoscalers.

Reuse these helpers when embedding SuperToken into larger applications or notebooks. The design expectations are summarized in the [architecture overview](architecture.md#cli-integration).

## Benchmarking Utilities
Located under [`benchmarks/`](https://github.com/example/SuperToken/blob/main/benchmarks/).

- **`run_benchmark()`** – Executes trainers against provided corpora and aggregates metrics.
- **Telemetry writers** – Format tabular output and JSON payloads for downstream consumption.
- **Scenario builders** – Compose synthetic and real datasets to match specific evaluation goals.

Benchmarking relies on the same dataset and autoscaling primitives, ensuring metrics remain comparable. Consult the [CLI usage guide](cli.md#benchmark) for command-line invocation and the [architecture overview](architecture.md#benchmarking-workflow) for conceptual flow.

## Related Guides
- [Architecture overview](architecture.md)
- [CLI usage guide](cli.md)
- [Performance notes and benchmarks](performance.md)
- [Module guide](modules.md)
