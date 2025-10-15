"""Command line interface for training BPE merges."""

from __future__ import annotations

import argparse
import glob
import os
from contextlib import ExitStack
from typing import Iterable, Iterator, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency for distributed launches
    import torch.distributed as dist
except Exception:  # pragma: no cover - torch may be unavailable in CI
    dist = None


def resolve_distributed_rank() -> Tuple[int, int]:
    """Return the (rank, world_size) tuple for the current process.

    The function first consults ``torch.distributed`` if it is both available and
    initialized. When PyTorch's distributed runtime has not yet been configured,
    ``RANK``/``WORLD_SIZE`` or ``LOCAL_RANK``/``WORLD_SIZE`` environment variables
    are honored as a fallback. Absent any distributed hints, the function returns
    ``(0, 1)``.
    """

    if dist is not None and dist.is_available():
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()

    env_rank = os.environ.get("RANK")
    env_world = os.environ.get("WORLD_SIZE")
    if env_rank is not None and env_world is not None:
        return int(env_rank), int(env_world)

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        world_size = os.environ.get("WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE")
        if world_size:
            return int(local_rank), int(world_size)
        return int(local_rank), 1

    return 0, 1


def partition_paths_for_rank(
    paths: Sequence[str], *, rank: int, world_size: int
) -> List[str]:
    """Partition ``paths`` such that every rank processes a disjoint subset.

    The function always returns a sorted list to make the sharding order
    deterministic. ``world_size`` must be positive; if ``rank`` falls outside the
    range ``[0, world_size)``, a ``ValueError`` is raised.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} must satisfy 0 <= rank < {world_size}")

    unique_sorted = sorted(dict.fromkeys(paths))
    return [unique_sorted[i] for i in range(rank, len(unique_sorted), world_size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        nargs="+",
        help=(
            "Globs of input files (any bytes). When launched with torch.distributed "
            "each rank deterministically shards the expanded file list."
        ),
    )
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

    from .autoscaler import AutoScaler
    from .bpe_trainer import GPUBPETrainer
    from .cpu_packer import BytePacker
    from .datasets import PackedBatcher
    from .io import MemoryMappedShard

    packer = BytePacker()
    paths: List[str] = []
    for pattern in args.data:
        paths.extend(glob.glob(pattern, recursive=True))

    paths = sorted(set(paths))

    rank, world_size = resolve_distributed_rank()
    shard_paths = partition_paths_for_rank(paths, rank=rank, world_size=world_size)

    if world_size > 1:
        print(
            f"[rank {rank}/{world_size}] Assigned {len(shard_paths)} of {len(paths)} shards",
            flush=True,
        )
    if not shard_paths:
        raise RuntimeError(
            f"Rank {rank} (world_size={world_size}) did not receive any input shards."
        )

    def _iter_sequences() -> Iterator[Iterator[int]]:
        with ExitStack() as stack:
            for path in shard_paths:
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

