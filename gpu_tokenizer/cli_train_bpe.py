"""Command line interface for training BPE merges."""

from __future__ import annotations

import argparse
import glob
from contextlib import ExitStack
from typing import Iterable, Iterator, List

from .autoscaler import AutoScaler
from .bpe_trainer import GPUBPETrainer
from .cpu_packer import BytePacker
from .datasets import PackedBatcher
from .io import MemoryMappedShard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", help="Globs of input files (any bytes)")
    parser.add_argument("--merges", type=int, default=10_000)
    parser.add_argument(
        "--bs",
        type=int,
        default=2048,
        help="Max initial batch size; autoscaler may select smaller values",
    )
    parser.add_argument(
        "--warm-start-ngrams",
        type=int,
        default=0,
        help="Seed merges using the top-N bigrams from an n-gram histogram",
    )
    parser.add_argument(
        "--freeze-warm-start",
        action="store_true",
        help="Prevent seeded merges from being reconsidered during training",
    )
    args = parser.parse_args()

    packer = BytePacker()
    paths: List[str] = []
    for pattern in args.data:
        paths.extend(glob.glob(pattern, recursive=True))

    def _iter_sequences() -> Iterator[Iterator[int]]:
        with ExitStack() as stack:
            for path in paths:
                shard = stack.enter_context(MemoryMappedShard(path))
                yield packer.encode_shard(shard)

    seqs: Iterable[Iterable[int]] = _iter_sequences()

    scaler = AutoScaler(target_util=0.80)
    init = scaler.suggest(token_bytes_per_example=8 * 1024)
    bs = min(args.bs, init.batch_size)
    batcher = PackedBatcher(seqs, batch_size=bs)

    warm_plan = None
    if args.warm_start_ngrams:
        warm_plan = GPUBPETrainer.precompute_warm_start_plan(
            batcher, args.warm_start_ngrams
        )
        if warm_plan["merges"]:
            print(
                f"Seeding {len(warm_plan['merges'])} merges from top "
                f"{warm_plan['requested_top_k']} bigrams"
            )

    trainer = GPUBPETrainer(
        base_vocab=256,
        merges=args.merges,
        autoscaler=scaler,
        warm_start_merges=(warm_plan["merges"] if warm_plan else None),
        freeze_warm_start=args.freeze_warm_start,
    )
    meta = trainer.fit(
        batcher,
        log_every=100,
        warm_start_plan=warm_plan,
        freeze_warm_start=args.freeze_warm_start if warm_plan else None,
    )
    trainer.save("./bpe_out")
    print(meta)


if __name__ == "__main__":
    main()
