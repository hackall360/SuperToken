"""Command line interface for training BPE merges."""

from __future__ import annotations

import argparse
import glob
from typing import List

from .autoscaler import AutoScaler
from .bpe_trainer import GPUBPETrainer
from .cpu_packer import BytePacker
from .datasets import PackedBatcher


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
    args = parser.parse_args()

    packer = BytePacker()
    paths: List[str] = []
    for pattern in args.data:
        paths.extend(glob.glob(pattern, recursive=True))
    seqs = [packer.encode_file(path) for path in paths]

    scaler = AutoScaler(target_util=0.80)
    init = scaler.suggest(token_bytes_per_example=8 * 1024)
    bs = min(args.bs, init.batch_size)
    batcher = list(PackedBatcher(seqs, batch_size=bs))

    trainer = GPUBPETrainer(base_vocab=256, merges=args.merges, autoscaler=scaler)
    meta = trainer.fit(batcher, log_every=100)
    trainer.save("./bpe_out")
    print(meta)


if __name__ == "__main__":
    main()
