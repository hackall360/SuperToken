"""Main entry-point that ties together the GPU tokenizer components."""

from __future__ import annotations

import argparse
import json
import glob
import sys
import types
from dataclasses import asdict
from pathlib import Path
from contextlib import ExitStack
from typing import Iterable, Iterator, Sequence

import torch

from gpu_tokenizer import (
    AutoScaler,
    BytePacker,
    GPUBPETrainer,
    GPUUnigramTrainer,
    PackedBatcher,
    StreamingPackedBatcher,
    utils,
)
from gpu_tokenizer.io import CorpusStreamer, MemoryMappedShard
from gpu_tokenizer.dtypes import length_storage_dtype
from gpu_tokenizer.trainers.base import BaseTrainer
from benchmarks import benchmark_runner

__all__ = [
    "AutoScaler",
    "BytePacker",
    "GPUBPETrainer",
    "GPUUnigramTrainer",
    "PackedBatcher",
    "utils",
    "main",
]


def _load_sequences(
    paths: Iterable[Path], bos: int | None, eos: int | None
) -> Iterator[Iterator[int]]:
    """Yield tokenized documents drawn from memory-mapped shard files.

    Args:
        paths: Iterable of shard file paths to open and encode.
        bos: Optional beginning-of-sequence token id to prefix onto each document.
        eos: Optional end-of-sequence token id to suffix onto each document.

    Returns:
        Iterator that lazily yields iterables of integer token ids for each
        document contained in the supplied shards. The outer iterator streams
        shards while the inner iterator walks the token sequence of each
        document.

    Side Effects:
        Opens :class:`MemoryMappedShard` handles via an :class:`ExitStack` so
        shard files stay memory-mapped for the lifetime of the generator.
    """
    packer = BytePacker(bos=bos, eos=eos)

    def _generator() -> Iterator[Iterator[int]]:
        with ExitStack() as stack:
            for path in paths:
                shard = stack.enter_context(MemoryMappedShard(path))
                yield packer.encode_shard(shard)

    return _generator()


def _expand_data_patterns(patterns: Sequence[str]) -> list[Path]:
    """Resolve input glob patterns into a deterministic list of shard paths.

    Args:
        patterns: Glob expressions pointing at data shards.

    Returns:
        List of concrete file paths that matched at least one pattern. The
        results are ordered by discovery so subsequent iteration is stable.

    Side Effects:
        Touches the filesystem to discover matching shard files.

    Raises:
        SystemExit: If no input files match the provided glob patterns.
    """
    files: list[Path] = []
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            p = Path(path)
            if p.is_file():
                files.append(p)
    if not files:
        raise SystemExit("No input files matched the provided --data globs")
    return files


def _iter_packed_batches(
    sequences: Iterable[Iterable[int]],
    batch_size: int,
    seed: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Construct a streaming batch iterator backed by :class:`PackedBatcher`.

    Args:
        sequences: Tokenized documents to be batched.
        batch_size: Maximum number of documents per batch.
        seed: Shuffle seed forwarded to :class:`PackedBatcher` for determinism.

    Returns:
        Iterable that yields ``(tokens, mask, lengths)`` tensors suitable for
        GPU consumption. Each iteration produces a packed token matrix, a mask
        indicating valid positions, and per-document lengths.

    Side Effects:
        None.
    """
    return PackedBatcher(sequences, batch_size=batch_size, seed=seed)


def _build_unigram_batches(
    sequences: Iterable[Iterable[int]],
    batch_size: int,
    seed: int,
) -> list[torch.Tensor]:
    """Materialize packed token batches for the unigram trainer.

    Args:
        sequences: Tokenized documents to feed into the unigram objective.
        batch_size: Number of documents per packed batch.
        seed: Shuffle seed used to stabilize batch ordering.

    Returns:
        List containing the token payload tensor for every packed batch. Masks
        and length tensors are intentionally dropped because the unigram
        objective only consumes token ids.

    Side Effects:
        Loads the entire packed representation into host memory so batches can
        be replayed across epochs without rebuilding.
    """
    packed = PackedBatcher(sequences, batch_size=batch_size, seed=seed)
    return [x for (x, _mask, _lengths) in packed]


def _stringify_config_value(value: object) -> object:
    """Convert config payloads into JSON-serialisable structures."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _stringify_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_config_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_stringify_config_value(v) for v in value)
    return value


def _log_resolved_config(command: str, config: dict[str, object]) -> None:
    """Render structured configuration dictionaries prior to training."""

    payload = _stringify_config_value(config)
    try:
        rendered = json.dumps(payload, indent=2, sort_keys=True)
    except TypeError:
        rendered = json.dumps(payload, indent=2, sort_keys=True, default=str)  # pragma: no cover - defensive fallback
    print(f"[config][{command}] {rendered}")


def _build_bpe_trainer(
    args: argparse.Namespace,
) -> tuple[BaseTrainer, AutoScaler, int, dict[str, object]]:
    """Factory that materialises a :class:`GPUBPETrainer` from CLI args."""

    autoscaler = AutoScaler(
        target_util=args.target_util,
        min_bs=args.min_batch,
        max_bs=args.max_batch,
    )
    suggestion_state = autoscaler.suggest(token_bytes_per_example=args.token_bytes)
    resolved_batch = min(args.max_batch, max(args.min_batch, suggestion_state.batch_size))
    trainer = GPUBPETrainer(
        base_vocab=args.base_vocab,
        merges=args.merges,
        device=args.device,
        autoscaler=autoscaler,
    )
    suggestion = asdict(suggestion_state)
    suggestion["requested_batch_size"] = suggestion_state.batch_size
    suggestion["resolved_batch_size"] = resolved_batch
    config: dict[str, object] = {
        "trainer": {
            "base_vocab": args.base_vocab,
            "merges": args.merges,
            "device": args.device,
        },
        "autoscaler": {
            "target_util": args.target_util,
            "min_batch": args.min_batch,
            "max_batch": args.max_batch,
            "token_bytes_per_example": args.token_bytes,
            "suggestion": suggestion,
        },
    }
    return trainer, autoscaler, resolved_batch, config


def _build_unigram_trainer(args: argparse.Namespace) -> tuple[BaseTrainer, dict[str, object]]:
    """Factory that materialises a :class:`GPUUnigramTrainer` from CLI args."""

    trainer = GPUUnigramTrainer(
        base_vocab=args.base_vocab,
        vocab_size=args.vocab_size,
        max_subword_len=args.max_subword_len,
        device=args.device,
    )
    config: dict[str, object] = {
        "trainer": {
            "base_vocab": args.base_vocab,
            "vocab_size": args.vocab_size,
            "max_subword_len": args.max_subword_len,
            "device": args.device,
        },
        "runtime": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
    }
    return trainer, config


def _cmd_benchmark(args: argparse.Namespace) -> None:
    """Run synthetic and real-corpus benchmarks and emit a serialized report.

    Args:
        args: Parsed CLI arguments configuring synthetic data, datasets, and
            trainer hyper-parameters.

    Returns:
        ``None``. The function prints progress summaries and writes a structured
        report for downstream inspection.

    Side Effects:
        Generates synthetic corpora when requested, reads optional datasets,
        writes benchmark summaries under ``args.output_dir``, and prints a
        human-readable summary to stdout.

    Raises:
        SystemExit: If no synthetic or real corpora are supplied for the run.
    """
    sequences: list[list[int]] = []
    sources: list[dict[str, object]] = []

    if args.synthetic_docs > 0:
        synthetic = benchmark_runner.synthesize_corpus(
            documents=args.synthetic_docs,
            min_length=args.synthetic_min_len,
            max_length=args.synthetic_max_len,
            vocab_size=args.synthetic_vocab,
            seed=args.seed,
        )
        sequences.extend(synthetic)
        sources.append(
            {
                "type": "synthetic",
                "documents": args.synthetic_docs,
                "min_length": args.synthetic_min_len,
                "max_length": args.synthetic_max_len,
                "vocab_size": args.synthetic_vocab,
            }
        )

    real_paths: list[Path] = []
    if args.data:
        real_paths = _expand_data_patterns(args.data)
        real_sequences = benchmark_runner.load_real_corpus(
            real_paths,
            bos=args.bos,
            eos=args.eos,
            limit=args.max_real_docs,
        )
        if real_sequences:
            sequences.extend(real_sequences)
            sources.append(
                {
                    "type": "dataset",
                    "paths": [str(p) for p in real_paths],
                    "limit": args.max_real_docs,
                }
            )

    if not sequences:
        raise SystemExit(
            "Benchmark requires at least one corpus source via --synthetic-docs or --data"
        )

    corpus = benchmark_runner.summarize_corpus(sequences, sources=sources)
    # The synthetic/real corpus mixtures above feed both trainers, allowing
    # benchmarking code to compare how each algorithm responds to identical
    # token streams.
    bpe_suite: dict[str, object] | None = None
    if getattr(args, "bpe_config", None):
        config_path = Path(args.bpe_config)
        run_specs = benchmark_runner.load_bpe_run_config(config_path)
        if not run_specs:
            raise SystemExit(f"No runnable BPE entries found in {config_path}")
        bpe_suite = benchmark_runner.run_bpe_suite(
            sequences,
            base_vocab=args.bpe_base_vocab,
            merges=args.bpe_merges,
            seed=args.seed,
            log_every=args.bpe_log_every,
            run_configs=run_specs,
        )
        runs = bpe_suite.get("runs", []) if isinstance(bpe_suite, dict) else []
        if not runs:
            raise SystemExit(f"BPE suite {config_path} did not produce any runs")
        bpe = runs[0]
    else:
        bpe = benchmark_runner.run_bpe_benchmark(
            sequences,
            base_vocab=args.bpe_base_vocab,
            merges=args.bpe_merges,
            batch_size=args.bpe_batch_size,
            device=args.device,
            seed=args.seed,
            log_every=args.bpe_log_every,
        )
    unigram = benchmark_runner.run_unigram_benchmark(
        sequences,
        base_vocab=args.unigram_base_vocab,
        vocab_size=args.unigram_vocab,
        max_subword_len=args.unigram_max_subword,
        batch_size=args.unigram_batch_size,
        epochs=args.unigram_epochs,
        device=args.device,
        seed=args.seed,
    )
    print(benchmark_runner.emit_benchmark_summary(corpus, bpe, unigram, bpe_suite))
    # Persist full benchmark metadata so checkpointing infrastructure can re-use
    # the exact same corpora, hyper-parameters, and timing metrics in later
    # automation runs.
    output_path = benchmark_runner.serialize_run(
        Path(args.output_dir),
        corpus=corpus,
        config={
            "seed": args.seed,
            "device": args.device,
            "synthetic": {
                "documents": args.synthetic_docs,
                "min_length": args.synthetic_min_len,
                "max_length": args.synthetic_max_len,
                "vocab_size": args.synthetic_vocab,
            }
            if args.synthetic_docs
            else None,
            "data": [str(p) for p in real_paths],
            "max_real_docs": args.max_real_docs,
            "bpe": bpe["config"],
            "unigram": unigram["config"],
        },
        bpe=bpe,
        unigram=unigram,
        bpe_runs=bpe_suite,
    )
    print(f"Saved benchmark metadata → {output_path}")


def _cmd_train_bpe(args: argparse.Namespace) -> None:
    """Train a GPU-accelerated BPE model with autoscaling batch management.

    Args:
        args: Parsed CLI arguments describing data sources, autoscaler targets,
            checkpointing configuration, and trainer hyper-parameters.

    Returns:
        ``None``. Progress and metadata are surfaced via stdout and optional
        checkpoint directories while the trained model may be exported.

    Side Effects:
        Uses :class:`AutoScaler` suggestions to resize the active batch size,
        starts a :class:`CorpusStreamer` that must be closed when training
        completes, and loads/saves checkpoints when ``--resume-from`` or
        ``--checkpoint-dir`` are provided.

    Raises:
        SystemExit: If no ``--data`` globs are supplied or none yield readable
            shard files.
    """
    data_patterns = getattr(args, "data", None)
    if not data_patterns:
        raise SystemExit("train-bpe requires at least one --data glob pattern")
    data_files = _expand_data_patterns(data_patterns)
    trainer, autoscaler, batch_size, trainer_config = _build_bpe_trainer(args)
    config = dict(trainer_config)
    config.update(
        {
            "data": {
                "patterns": list(data_patterns),
                "files": [str(path) for path in data_files],
                "bos": args.bos,
                "eos": args.eos,
                "compression": args.compression,
                "io_workers": args.io_workers,
                "prefetch_batches": args.prefetch_batches,
            },
            "checkpointing": {
                "checkpoint_dir": args.checkpoint_dir,
                "checkpoint_every": args.checkpoint_every,
                "resume_from": str(args.resume_from) if args.resume_from else None,
                "out_dir": args.out_dir,
            },
            "dry_run": bool(getattr(args, "dry_run", False)),
        }
    )
    _log_resolved_config("train-bpe", config)
    if getattr(args, "dry_run", False):
        print("[dry-run] train-bpe initialization complete")
        return
    packer = BytePacker(bos=args.bos, eos=args.eos)

    def _build_serialized_batches(
        serialized: dict[str, object],
        default_bs: int,
    ) -> tuple[int, Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None]:
        sequences_raw = serialized.get("sequences", [])
        sequences: list[list[int]] = []
        if isinstance(sequences_raw, list):
            for seq in sequences_raw:
                if isinstance(seq, list):
                    sequences.append([int(token) for token in seq])
        if not sequences:
            return 0, None
        resume_bs = int(serialized.get("active_batch_size") or 0)
        if resume_bs <= 0:
            resume_bs = default_bs

        class _SerializedIterable:
            def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                pin_memory = torch.cuda.is_available()
                storage_width = 1
                length_dtype = length_storage_dtype(storage_width)
                tokens = torch.full(
                    (resume_bs, storage_width),
                    -1,
                    dtype=torch.int32,
                    pin_memory=pin_memory,
                )
                valid = torch.zeros(
                    (resume_bs, storage_width),
                    dtype=torch.uint8,
                    pin_memory=pin_memory,
                )
                lengths = torch.zeros(
                    (resume_bs,),
                    dtype=length_dtype,
                    pin_memory=pin_memory,
                )

                for start in range(0, len(sequences), resume_bs):
                    chunk = sequences[start : start + resume_bs]
                    if not chunk:
                        continue
                    max_len = max((len(seq) for seq in chunk), default=0)
                    width = max(1, max_len)
                    if width > storage_width:
                        storage_width = width
                        length_dtype = length_storage_dtype(storage_width)
                        tokens = torch.full(
                            (resume_bs, storage_width),
                            -1,
                            dtype=torch.int32,
                            pin_memory=pin_memory,
                        )
                        valid = torch.zeros(
                            (resume_bs, storage_width),
                            dtype=torch.uint8,
                            pin_memory=pin_memory,
                        )
                        lengths = torch.zeros(
                            (resume_bs,),
                            dtype=length_dtype,
                            pin_memory=pin_memory,
                        )
                    count = len(chunk)
                    tokens[:count].fill_(-1)
                    valid[:count].zero_()
                    lengths[:count].zero_()
                    for row, seq in enumerate(chunk):
                        L = len(seq)
                        if L == 0:
                            continue
                        lengths[row] = L
                        vals = torch.as_tensor(seq, dtype=torch.int32)
                        tokens[row, :L] = vals
                        valid[row, :L] = 1
                    yield tokens[:count, :width], valid[:count, :width], lengths[:count]

        return resume_bs, _SerializedIterable()

    def _build_streamer() -> CorpusStreamer:
        streamer = CorpusStreamer(
            data_files,
            compression=args.compression,
            num_workers=args.io_workers,
            max_prefetch=args.prefetch_batches,
            autoscaler=autoscaler,
        )
        streamer.start()
        return streamer

    resume_state: dict[str, object] | None = None
    resume_batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    # Checkpoint resume flow:
    # 1. Optionally load serialized state, including packed batches, from the
    #    requested ``--resume-from`` directory.
    # 2. When batches are available in the checkpoint metadata we replay them
    #    directly so the autoscaler restarts from the previous batch size.
    # 3. Otherwise we fall back to building a fresh :class:`CorpusStreamer`
    #    pipeline that will be recreated whenever the autoscaler resizes.
    if args.resume_from:
        resume_state = trainer.load_checkpoint(str(args.resume_from))
        metadata = dict(resume_state.get("metadata", {}))
        merge_step = metadata.get("merge_step")
        print(
            f"[checkpoint] Restored checkpoint from {args.resume_from}"
            + (f" at merge {merge_step}" if merge_step is not None else "")
        )
        serialized_batches = metadata.get("batches")
        if isinstance(serialized_batches, dict):
            restored_bs, restored_iter = _build_serialized_batches(
                serialized_batches,
                batch_size,
            )
            if restored_bs > 0:
                batch_size = restored_bs
            resume_batches = restored_iter

    streamer: CorpusStreamer | None = None
    if resume_batches is None:
        streamer = _build_streamer()
        batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = StreamingPackedBatcher(
            streamer,
            packer.encode_view,
            batch_size=batch_size,
        )
    else:
        batches = resume_batches
    current_batch_size = batch_size

    def _handle_batch_resize(new_bs: int) -> None:
        nonlocal batches, current_batch_size, streamer
        if new_bs <= 0 or new_bs == current_batch_size:
            return
        current_batch_size = new_bs
        if streamer is None:
            return
        # Autoscaler triggered a resize: tear down the current streamer so it
        # restarts with the new batch size and refreshed packing layout.
        streamer.close()
        streamer = _build_streamer()
        batches = StreamingPackedBatcher(
            streamer,
            packer.encode_view,
            batch_size=new_bs,
        )

    try:
        if args.checkpoint_dir:
            original_save_checkpoint = trainer.save_checkpoint

            def _save_checkpoint_with_log(self, path: str, *cargs, **ckwargs):
                state = original_save_checkpoint(path, *cargs, **ckwargs)
                print(f"[checkpoint] Saved checkpoint → {path}")
                return state

            trainer.save_checkpoint = types.MethodType(_save_checkpoint_with_log, trainer)
        meta = trainer.fit(
            batches,
            log_every=args.log_every,
            on_batch_size_change=_handle_batch_resize,
            checkpoint_interval=(
                args.checkpoint_every if args.checkpoint_every and args.checkpoint_every > 0 else None
            ),
            checkpoint_dir=args.checkpoint_dir,
            resume_state=resume_state,
        )
    finally:
        if streamer is not None:
            streamer.close()
    if args.out_dir:
        trainer.save(args.out_dir)
    print(meta)


def _cmd_train_unigram(args: argparse.Namespace) -> None:
    """Train a unigram tokenizer model over prepacked batches.

    Args:
        args: Parsed CLI arguments providing data patterns, batching options,
            and model hyper-parameters.

    Returns:
        ``None``. Training progress is printed for each epoch and the trained
        state is optionally saved to disk.

    Side Effects:
        Loads all packed batches into host memory and writes the trained model
        to ``args.out_dir`` when provided. The trainer's internal state is
        mutated across epochs.

    Raises:
        SystemExit: If ``--data`` is omitted or the globs do not resolve to at
            least one shard.
    """
    data_patterns = getattr(args, "data", None)
    if not data_patterns:
        raise SystemExit("train-unigram requires at least one --data glob pattern")
    data_files = _expand_data_patterns(data_patterns)
    trainer, trainer_config = _build_unigram_trainer(args)
    config = dict(trainer_config)
    config.update(
        {
            "data": {
                "patterns": list(data_patterns),
                "files": [str(path) for path in data_files],
                "bos": args.bos,
                "eos": args.eos,
            },
            "output": args.out_dir,
            "dry_run": bool(getattr(args, "dry_run", False)),
        }
    )
    _log_resolved_config("train-unigram", config)
    if getattr(args, "dry_run", False):
        print("[dry-run] train-unigram initialization complete")
        return

    sequences = _load_sequences(data_files, bos=args.bos, eos=args.eos)

    batches = _build_unigram_batches(sequences, batch_size=args.batch_size, seed=args.seed)
    for epoch in range(args.epochs):
        stats = trainer.fit_epoch(batches)
        print(f"epoch {epoch + 1}: {stats}")
    if args.out_dir:
        trainer.save(args.out_dir)


def _cmd_stream_batches(args: argparse.Namespace) -> None:
    """Stream packed batches and report their tensor dimensions.

    Args:
        args: Parsed CLI arguments providing data globs, optional BOS/EOS
            tokens, seed, and the desired ``batch_size`` and ``max_batches``
            attributes.

    Returns:
        ``None``. Batch metadata is emitted to stdout for inspection.

    Side Effects:
        Opens shard files, materializes packed tensors on the host, and prints
        batch shapes until ``max_batches`` is reached (when provided).

    Raises:
        SystemExit: If no ``--data`` patterns are supplied or the globs resolve
            to no shard files.
    """

    data_patterns = getattr(args, "data", None)
    if not data_patterns:
        raise SystemExit("stream-batches requires at least one --data glob pattern")
    data_files = _expand_data_patterns(data_patterns)
    sequences = _load_sequences(
        data_files,
        bos=getattr(args, "bos", None),
        eos=getattr(args, "eos", None),
    )
    batch_size = getattr(args, "batch_size", 1024)
    seed = getattr(args, "seed", 1337)
    batches = _iter_packed_batches(sequences, batch_size=batch_size, seed=seed)
    max_batches = getattr(args, "max_batches", None)
    for idx, (tokens, mask, lengths) in enumerate(batches):
        print(
            f"batch {idx}: tokens={tuple(tokens.shape)} mask={tuple(mask.shape)} lengths={tuple(lengths.shape)}"
        )
        if max_batches is not None and max_batches > 0 and idx + 1 >= max_batches:
            break


def _cmd_resume_bpe(args: argparse.Namespace) -> None:
    """Resume a BPE training run from an on-disk checkpoint.

    Args:
        args: Parsed CLI arguments expected to include ``--resume-from`` and all
            parameters required by :func:`_cmd_train_bpe`.

    Returns:
        ``None``. All work is delegated to :func:`_cmd_train_bpe`.

    Side Effects:
        Loads checkpoint state via :class:`GPUBPETrainer`, potentially rebuilds
        :class:`CorpusStreamer` instances, and produces the same outputs as a
        standard ``train-bpe`` invocation.

    Raises:
        SystemExit: If ``--resume-from`` or ``--data`` are missing before
            dispatching to :func:`_cmd_train_bpe`, or if the delegated training
            invocation encounters its own fatal CLI condition.
    """

    if not getattr(args, "resume_from", None):
        raise SystemExit("--resume-from is required when invoking resume-bpe")
    if not getattr(args, "data", None):
        raise SystemExit("resume-bpe requires --data globs to stream training shards")
    _cmd_train_bpe(args)


def _parser() -> argparse.ArgumentParser:
    """Build the CLI parser that exposes training and benchmarking commands.

    Returns:
        Configured :class:`argparse.ArgumentParser` with subcommands registered
        for ``train-bpe``, ``train-unigram``, and ``benchmark``.

    Side Effects:
        None.
    """
    parser = argparse.ArgumentParser(description="GPU tokenizer toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", nargs="+", required=True, help="Input glob patterns")
    common.add_argument("--bos", type=int, default=None, help="Optional BOS token id")
    common.add_argument("--eos", type=int, default=None, help="Optional EOS token id")
    common.add_argument("--seed", type=int, default=1337, help="Shuffle seed")
    common.add_argument("--device", type=str, default=None, help="Torch device override")
    common.add_argument(
        "--compression",
        type=str,
        default="none",
        choices=["none", "zstd", "lz4"],
        help="Compression codec for input shards",
    )
    common.add_argument(
        "--io-workers",
        type=int,
        default=2,
        help="Number of background workers for shard decoding",
    )
    common.add_argument(
        "--prefetch-batches",
        type=int,
        default=4,
        help="Maximum number of prefetched batches before backpressure engages",
    )

    train_bpe = subparsers.add_parser("train-bpe", parents=[common], help="Train a BPE model")
    train_bpe.set_defaults(func=_cmd_train_bpe)
    train_bpe.add_argument("--merges", type=int, default=50_000)
    train_bpe.add_argument("--base-vocab", type=int, default=256)
    train_bpe.add_argument("--target-util", type=float, default=0.80)
    train_bpe.add_argument("--min-batch", type=int, default=512)
    train_bpe.add_argument("--max-batch", type=int, default=4096)
    train_bpe.add_argument("--token-bytes", type=int, default=8 * 1024)
    train_bpe.add_argument("--log-every", type=int, default=100)
    train_bpe.add_argument("--out-dir", type=str, default="./bpe_out")
    train_bpe.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory where periodic training checkpoints are written",
    )
    train_bpe.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Frequency (in merges) for checkpoint writes; disabled when set to 0",
    )
    train_bpe.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint directory created by --checkpoint-dir",
    )
    train_bpe.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate the trainer, log resolved configuration, and exit",
    )

    train_unigram = subparsers.add_parser(
        "train-unigram", parents=[common], help="Train a unigram model"
    )
    train_unigram.set_defaults(func=_cmd_train_unigram)
    train_unigram.add_argument("--vocab-size", type=int, default=50_000)
    train_unigram.add_argument("--base-vocab", type=int, default=256)
    train_unigram.add_argument("--max-subword-len", type=int, default=8)
    train_unigram.add_argument("--batch-size", type=int, default=1024)
    train_unigram.add_argument("--epochs", type=int, default=1)
    train_unigram.add_argument("--out-dir", type=str, default="./unigram_out")
    train_unigram.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate the trainer, log resolved configuration, and exit",
    )

    benchmark = subparsers.add_parser("benchmark", help="Run tokenizer training benchmarks")
    benchmark.set_defaults(func=_cmd_benchmark)
    benchmark.add_argument(
        "--data",
        nargs="+",
        default=[],
        help="Optional glob patterns pointing at real datasets",
    )
    benchmark.add_argument("--bos", type=int, default=None, help="Optional BOS token id")
    benchmark.add_argument("--eos", type=int, default=None, help="Optional EOS token id")
    benchmark.add_argument("--synthetic-docs", type=int, default=0, help="Number of synthetic documents")
    benchmark.add_argument(
        "--synthetic-min-len", type=int, default=32, help="Minimum synthetic document length"
    )
    benchmark.add_argument(
        "--synthetic-max-len", type=int, default=256, help="Maximum synthetic document length"
    )
    benchmark.add_argument(
        "--synthetic-vocab", type=int, default=256, help="Vocabulary range for synthetic corpora"
    )
    benchmark.add_argument(
        "--max-real-docs",
        type=int,
        default=None,
        help="Optional cap on the number of real documents to load",
    )
    benchmark.add_argument("--seed", type=int, default=1337, help="Shuffle seed")
    benchmark.add_argument("--device", type=str, default=None, help="Torch device override")
    benchmark.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where benchmark JSON outputs are written",
    )
    benchmark.add_argument(
        "--bpe-base-vocab", type=int, default=256, help="Base vocabulary size for BPE"
    )
    benchmark.add_argument("--bpe-merges", type=int, default=10_000, help="Number of BPE merges")
    benchmark.add_argument(
        "--bpe-batch-size", type=int, default=1024, help="Batch size to use for BPE"
    )
    benchmark.add_argument(
        "--bpe-log-every", type=int, default=250, help="Logging cadence for BPE trainer"
    )
    benchmark.add_argument(
        "--bpe-config",
        type=str,
        default=None,
        help="Optional JSON config describing heterogeneous BPE benchmark runs",
    )
    benchmark.add_argument(
        "--unigram-base-vocab",
        type=int,
        default=256,
        help="Base vocabulary size for unigram trainer",
    )
    benchmark.add_argument(
        "--unigram-vocab", type=int, default=50_000, help="Target unigram vocabulary size"
    )
    benchmark.add_argument(
        "--unigram-max-subword",
        type=int,
        default=8,
        help="Maximum unigram subword length",
    )
    benchmark.add_argument(
        "--unigram-batch-size", type=int, default=1024, help="Batch size for unigram batches"
    )
    benchmark.add_argument(
        "--unigram-epochs", type=int, default=1, help="Number of unigram epochs"
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point that dispatches to tokenizer subcommands.

    Args:
        argv: Optional argument vector override, mirroring ``sys.argv[1:]``
            when ``None``.

    Returns:
        ``None``. The selected subcommand performs all work.

    Side Effects:
        Parses CLI arguments, writes help text and command output to stdout, and
        may exit the interpreter via :func:`argparse.ArgumentParser.parse_args`.

    Raises:
        SystemExit: If argument parsing fails or no subcommand is provided.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        raise SystemExit(1)
    func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
