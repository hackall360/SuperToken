"""Command line interface for training BPE merges."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from contextlib import ExitStack
from typing import Iterable, Iterator, List, Sequence, Tuple

if __package__:
    from . import dist_runtime  # pragma: no cover - package import path
else:  # pragma: no cover - support direct module loading in tests
    import importlib.util
    from pathlib import Path

    _dist_spec = importlib.util.spec_from_file_location(
        "gpu_tokenizer.dist_runtime", Path(__file__).with_name("dist_runtime.py")
    )
    if _dist_spec is None or _dist_spec.loader is None:  # pragma: no cover - defensive
        raise ImportError("Unable to load dist_runtime module")
    dist_runtime = importlib.util.module_from_spec(_dist_spec)
    sys.modules.setdefault(_dist_spec.name, dist_runtime)
    _dist_spec.loader.exec_module(dist_runtime)

try:  # pragma: no cover - optional dependency for distributed launches
    import torch.distributed as dist
except Exception:  # pragma: no cover - torch may be unavailable in CI
    dist = None


DEFAULT_LOG_EVERY = 100


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


def _expand_device_argument(raw: str) -> List[int]:
    """Return a list of unique, zero-based GPU indices parsed from *raw*."""

    values: List[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        piece = item.strip()
        if not piece:
            continue

        lowered = piece.lower()
        for prefix in ("cuda:", "cuda", "gpu:", "gpu"):
            if lowered.startswith(prefix):
                piece = piece[len(prefix) :]
                lowered = piece.lower()
                break

        try:
            index = int(piece)
        except ValueError as exc:
            raise ValueError("GPU indices must be integers") from exc
        if index < 0:
            raise ValueError("GPU indices must be non-negative")
        if index not in seen:
            values.append(index)
            seen.add(index)

    if not values:
        raise ValueError("At least one GPU index must be provided with --gpus")

    return values


def _sanitize_cli_args(argv: Sequence[str] | None) -> List[str]:
    """Return *argv* without ``--dist``/``--gpus`` flags for worker launches."""

    if argv is None:
        raw_args: Sequence[str] = sys.argv[1:]
    else:
        raw_args = argv

    cleaned: List[str] = []
    skip_next = False
    for token in raw_args:
        if skip_next:
            skip_next = False
            continue
        if token == "--dist":
            continue
        if token.startswith("--gpus"):
            if token == "--gpus":
                skip_next = True
            continue
        cleaned.append(token)

    return cleaned


def _load_sibling_attr(module: str, attr: str):
    """Load *attr* from a sibling module, supporting direct file execution."""

    if __package__:
        mod = __import__(f"{__package__}.{module}", fromlist=[attr])
    else:  # pragma: no cover - exercised indirectly in tests
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            f"gpu_tokenizer.{module}", Path(__file__).with_name(f"{module}.py")
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load sibling module {module!r}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(spec.name, mod)
        spec.loader.exec_module(mod)

    return getattr(mod, attr)


def _format_iteration_summary(summary: dict[str, object]) -> str:
    merge_idx = int(summary.get("merge", 0) or 0)
    tokens = int(summary.get("tokens", 0) or 0)
    leases = int(summary.get("leases", 0) or 0)
    tokens_per_s = float(summary.get("tokens_per_s", 0.0) or 0.0)
    leases_per_s = float(summary.get("lease_per_s", 0.0) or 0.0)
    h2d = float(summary.get("h2d_s", 0.0) or 0.0)
    kernel = float(summary.get("kernel_s", 0.0) or 0.0)
    d2h = float(summary.get("d2h_s", 0.0) or 0.0)
    reduction = float(summary.get("reduction_s", 0.0) or 0.0)
    return (
        f"[timings] merge {merge_idx:6d} | "
        f"tokens={tokens:,} ({tokens_per_s:,.0f}/s) | "
        f"leases={leases:,} ({leases_per_s:,.0f}/s) | "
        f"h2d={h2d:.3f}s kernel={kernel:.3f}s d2h={d2h:.3f}s reduce={reduction:.3f}s"
    )


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./bpe_out",
        help="Directory where the trained tokenizer artifacts will be stored",
    )
    parser.add_argument(
        "--dist",
        action="store_true",
        help="Enable the experimental distributed launcher",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help=(
            "Comma-separated list of GPU devices (e.g. 0,1 or cuda:0,cuda:1) to use"
            " for distributed runs"
        ),
    )
    parser.add_argument(
        "--target-chunk-ms",
        type=float,
        default=None,
        help="Preferred processing time per chunk in milliseconds",
    )
    parser.add_argument(
        "--log-stage-timings",
        action="store_true",
        help="Print EWMA timing summaries alongside merge progress logs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.target_chunk_ms is not None and args.target_chunk_ms <= 0:
        parser.error("--target-chunk-ms must be positive when provided")

    if args.gpus is not None and not args.dist:
        parser.error("--gpus may only be specified together with --dist")

    if args.dist:
        if args.gpus is None:
            parser.error("--dist requires --gpus to specify at least one device")
        try:
            device_ids = _expand_device_argument(args.gpus)
        except ValueError as exc:
            parser.error(str(exc))

        world_size = len(device_ids)
        config = dist_runtime.DistributedLaunchConfig(
            device_ids=tuple(device_ids),
            world_size=world_size,
        )
        cli_args = _sanitize_cli_args(argv if argv is not None else sys.argv[1:])
        dist_runtime.launch_training(config, cli_args)
        return

    log_every = DEFAULT_LOG_EVERY

    AutoScalerType = globals().get("AutoScaler")
    if AutoScalerType is None:
        AutoScalerType = _load_sibling_attr("autoscaler", "AutoScaler")
        globals()["AutoScaler"] = AutoScalerType

    GPUBPETrainerType = globals().get("GPUBPETrainer")
    if GPUBPETrainerType is None:
        GPUBPETrainerType = _load_sibling_attr("bpe_trainer", "GPUBPETrainer")
        globals()["GPUBPETrainer"] = GPUBPETrainerType

    BytePackerType = globals().get("BytePacker")
    if BytePackerType is None:
        BytePackerType = _load_sibling_attr("cpu_packer", "BytePacker")
        globals()["BytePacker"] = BytePackerType

    PackedBatcherType = globals().get("PackedBatcher")
    if PackedBatcherType is None:
        PackedBatcherType = _load_sibling_attr("datasets", "PackedBatcher")
        globals()["PackedBatcher"] = PackedBatcherType

    MemoryMappedShardType = globals().get("MemoryMappedShard")
    if MemoryMappedShardType is None:
        MemoryMappedShardType = _load_sibling_attr("io", "MemoryMappedShard")
        globals()["MemoryMappedShard"] = MemoryMappedShardType

    packer = BytePackerType()
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
                shard = stack.enter_context(MemoryMappedShardType(path))
                yield packer.encode_shard(shard)

    seqs: Iterable[Iterable[int]] = _iter_sequences()

    scaler = AutoScalerType(target_util=0.80)
    init = scaler.suggest(token_bytes_per_example=8 * 1024)
    bs = min(args.bs, init.batch_size)
    batcher = PackedBatcherType(seqs, batch_size=bs)

    warm_plan = None
    if args.warm_start_ngrams:
        warm_plan = GPUBPETrainerType.precompute_warm_start_plan(
            batcher, args.warm_start_ngrams
        )
        if warm_plan["merges"]:
            print(
                f"Seeding {len(warm_plan['merges'])} merges from top "
                f"{warm_plan['requested_top_k']} bigrams"
            )

    trainer = GPUBPETrainerType(
        base_vocab=256,
        merges=args.merges,
        autoscaler=scaler,
        warm_start_merges=(warm_plan["merges"] if warm_plan else None),
        freeze_warm_start=args.freeze_warm_start,
    )
    iteration_callback = None
    if args.log_stage_timings:
        trainer.metrics.enabled = True

        def _log_iteration(summary: dict[str, object]) -> None:
            if summary.get("kind") != "merge":
                return
            merge_idx = int(summary.get("merge", 0) or 0)
            if merge_idx <= 0:
                return
            if (merge_idx - 1) % max(1, log_every) != 0:
                return
            print(_format_iteration_summary(summary), flush=True)

        iteration_callback = _log_iteration
    meta = trainer.fit(
        batcher,
        log_every=log_every,
        warm_start_plan=warm_plan,
        freeze_warm_start=args.freeze_warm_start if warm_plan else None,
        on_iteration_summary=iteration_callback,
    )
    trainer.save(args.out_dir)
    print(meta)


if __name__ == "__main__":
    main()

