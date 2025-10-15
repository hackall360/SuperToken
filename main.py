"""Main entry-point that ties together the GPU tokenizer components."""

from __future__ import annotations

import argparse
import glob
import sys
import types
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
    packer = BytePacker(bos=bos, eos=eos)

    def _generator() -> Iterator[Iterator[int]]:
        with ExitStack() as stack:
            for path in paths:
                shard = stack.enter_context(MemoryMappedShard(path))
                yield packer.encode_shard(shard)

    return _generator()


def _expand_data_patterns(patterns: Sequence[str]) -> list[Path]:
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
    return PackedBatcher(sequences, batch_size=batch_size, seed=seed)


def _build_unigram_batches(
    sequences: Iterable[Iterable[int]],
    batch_size: int,
    seed: int,
) -> list[torch.Tensor]:
    packed = PackedBatcher(sequences, batch_size=batch_size, seed=seed)
    return [x for (x, _mask, _lengths) in packed]


def _cmd_benchmark(args: argparse.Namespace) -> None:
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
    print(benchmark_runner.emit_benchmark_summary(corpus, bpe, unigram))
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
    )
    print(f"Saved benchmark metadata → {output_path}")


def _cmd_train_bpe(args: argparse.Namespace) -> None:
    data_files = _expand_data_patterns(args.data)
    autoscaler = AutoScaler(
        target_util=args.target_util,
        min_bs=args.min_batch,
        max_bs=args.max_batch,
    )
    suggestion = autoscaler.suggest(token_bytes_per_example=args.token_bytes)
    batch_size = min(args.max_batch, max(args.min_batch, suggestion.batch_size))
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

    trainer = GPUBPETrainer(
        base_vocab=args.base_vocab,
        merges=args.merges,
        device=args.device,
        autoscaler=autoscaler,
    )
    resume_state: dict[str, object] | None = None
    resume_batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
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
    data_files = _expand_data_patterns(args.data)
    sequences = _load_sequences(data_files, bos=args.bos, eos=args.eos)

    batches = _build_unigram_batches(sequences, batch_size=args.batch_size, seed=args.seed)
    trainer = GPUUnigramTrainer(
        base_vocab=args.base_vocab,
        vocab_size=args.vocab_size,
        max_subword_len=args.max_subword_len,
        device=args.device,
    )
    for epoch in range(args.epochs):
        stats = trainer.fit_epoch(batches)
        print(f"epoch {epoch + 1}: {stats}")
    if args.out_dir:
        trainer.save(args.out_dir)


def _parser() -> argparse.ArgumentParser:
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
    parser = _parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        raise SystemExit(1)
    func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
