"""Main entry-point that ties together the GPU tokenizer components."""

from __future__ import annotations

import argparse
import glob
import sys
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

    streamer = _build_streamer()
    batches = StreamingPackedBatcher(
        streamer,
        packer.encode_view,
        batch_size=batch_size,
    )
    current_batch_size = batch_size

    def _handle_batch_resize(new_bs: int) -> None:
        nonlocal batches, current_batch_size, streamer
        if new_bs <= 0 or new_bs == current_batch_size:
            return
        streamer.close()
        streamer = _build_streamer()
        batches = StreamingPackedBatcher(
            streamer,
            packer.encode_view,
            batch_size=new_bs,
        )
        current_batch_size = new_bs

    trainer = GPUBPETrainer(
        base_vocab=args.base_vocab,
        merges=args.merges,
        device=args.device,
        autoscaler=autoscaler,
    )
    try:
        meta = trainer.fit(
            batches,
            log_every=args.log_every,
            on_batch_size_change=_handle_batch_resize,
        )
    finally:
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
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        vocab_file = out_path / "unigram_vocab.txt"
        with vocab_file.open("w", encoding="utf-8") as f:
            for idx, piece in trainer.id2piece.items():
                f.write(f"{idx}\t{piece.hex()}\n")
        print(f"Saved unigram vocab → {vocab_file}")


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
