"""Command line interface for training BPE merges."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import types
from contextlib import ExitStack
from typing import Iterable, Iterator, List, Mapping, Sequence, Tuple

if __package__:
    from . import dist_runtime  # pragma: no cover - package import path
else:  # pragma: no cover - support direct module loading in tests
    import importlib.util
    from pathlib import Path

    _dist_spec = importlib.util.spec_from_file_location(
        "gpu_tokenizer.dist_runtime", Path(__file__).with_name("dist_runtime.py")
    )
    dist_runtime = None
    if _dist_spec is not None and _dist_spec.loader is not None:
        module = importlib.util.module_from_spec(_dist_spec)
        sys.modules.setdefault(_dist_spec.name, module)
        try:
            _dist_spec.loader.exec_module(module)
            dist_runtime = module
        except Exception:  # pragma: no cover - fallback when optional deps missing
            sys.modules.pop(_dist_spec.name, None)
    if dist_runtime is None:  # pragma: no cover - exercised in CLI tests
        dist_runtime = types.SimpleNamespace(
            compute_lease_job_id=lambda paths: "stub-job",
            iterate_leased_shards=lambda *args, **kwargs: iter(()),
            plan_chunk_slices=lambda *args, **kwargs: [],
            register_lease_client=lambda **kwargs: None,
            launch_training=lambda *args, **kwargs: None,
            DistributedLaunchConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
            RendezvousSettings=lambda **kwargs: types.SimpleNamespace(**kwargs),
        )

try:  # pragma: no cover - optional dependency for distributed launches
    import torch.distributed as dist
except Exception:  # pragma: no cover - torch may be unavailable in CI
    dist = None


DEFAULT_LOG_EVERY = 100
DEFAULT_CHUNK_TARGET_MS = 100.0


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

    dist_flags = {
        "--dist",
        "--gpus",
        "--dist-init-method",
        "--dist-timeout",
        "--dist-log-level",
    }
    cleaned: List[str] = []
    skip_next = False
    for token in raw_args:
        if skip_next:
            skip_next = False
            continue
        if token in dist_flags:
            if token in {"--gpus", "--dist-init-method", "--dist-timeout", "--dist-log-level"}:
                skip_next = True
            continue
        if any(token.startswith(prefix + "=") for prefix in dist_flags if prefix != "--dist"):
            continue
        cleaned.append(token)

    return cleaned


def render_rank_metrics_table(snapshots: Sequence[Mapping[str, object]]) -> str:
    """Return a formatted table summarising per-rank throughput metrics."""

    aggregated: dict[int, tuple[float, float, float]] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        per_rank = snapshot.get("per_rank")
        if isinstance(per_rank, Mapping):
            for rank_key, stats in per_rank.items():
                try:
                    rank_idx = int(rank_key)
                except (TypeError, ValueError):
                    continue
                if not isinstance(stats, Mapping):
                    continue
                tokens_val = stats.get("tokens_per_s", 0.0)
                leases_val = stats.get("lease_per_s", 0.0)
                samples_val = stats.get("samples", 0.0)
                try:
                    tokens = float(tokens_val if tokens_val is not None else 0.0)
                except (TypeError, ValueError):
                    tokens = 0.0
                try:
                    leases = float(leases_val if leases_val is not None else 0.0)
                except (TypeError, ValueError):
                    leases = 0.0
                try:
                    samples = float(samples_val if samples_val is not None else 0.0)
                except (TypeError, ValueError):
                    samples = 0.0
                aggregated[rank_idx] = (tokens, leases, samples)

        rank_val = snapshot.get("rank")
        rank_idx: int | None
        try:
            rank_idx = int(rank_val) if rank_val is not None else None
        except (TypeError, ValueError):
            rank_idx = None
        if rank_idx is None:
            continue
        tokens_val = snapshot.get("tokens_per_s", 0.0)
        leases_val = snapshot.get("lease_per_s", 0.0)
        samples_val = snapshot.get("samples", 0.0)
        try:
            tokens = float(tokens_val if tokens_val is not None else 0.0)
        except (TypeError, ValueError):
            tokens = 0.0
        try:
            leases = float(leases_val if leases_val is not None else 0.0)
        except (TypeError, ValueError):
            leases = 0.0
        try:
            samples = float(samples_val if samples_val is not None else 0.0)
        except (TypeError, ValueError):
            samples = 0.0
        aggregated.setdefault(rank_idx, (tokens, leases, samples))

    if not aggregated:
        return ""

    rows: list[list[str]] = []
    for rank_idx in sorted(aggregated):
        tokens, leases, samples = aggregated[rank_idx]
        rows.append(
            [
                str(rank_idx),
                f"{tokens:,.2f}",
                f"{leases:,.2f}",
                f"{samples:,.2f}",
            ]
        )

    headers = ["Rank", "Tokens/s", "Leases/s", "Samples"]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _format_row(cells: Sequence[str]) -> str:
        pieces = [f" {cell.rjust(widths[idx])} " for idx, cell in enumerate(cells)]
        return "|" + "|".join(pieces) + "|"

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [border, _format_row(headers), border]
    for row in rows:
        lines.append(_format_row(row))
    lines.append(border)
    return "\n".join(lines)


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
    metrics_payload = summary.get("metrics")

    def _to_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    tokens_rate_obj = summary.get("tokens_per_s", 0.0)
    leases_rate_obj = summary.get("lease_per_s", 0.0)
    if isinstance(metrics_payload, Mapping):
        throughput = metrics_payload.get("throughput")
        if isinstance(throughput, Mapping):
            tokens_rate_obj = throughput.get("tokens_per_s", tokens_rate_obj)
            leases_rate_obj = throughput.get("lease_per_s", leases_rate_obj)

    tokens_per_s = _to_float(tokens_rate_obj)
    leases_per_s = _to_float(leases_rate_obj)
    h2d = float(summary.get("h2d_s", 0.0) or 0.0)
    kernel = float(summary.get("kernel_s", 0.0) or 0.0)
    d2h = float(summary.get("d2h_s", 0.0) or 0.0)
    reduction = float(summary.get("reduction_s", 0.0) or 0.0)
    copy_time = float(summary.get("copy_s", 0.0) or 0.0)
    compute_time = float(summary.get("compute_s", 0.0) or 0.0)
    overlap = "on" if bool(summary.get("overlap", True)) else "off"
    return (
        f"[timings] merge {merge_idx:6d} | "
        f"tokens={tokens:,} ({tokens_per_s:,.0f}/s) | "
        f"leases={leases:,} ({leases_per_s:,.0f}/s) | "
        f"h2d={h2d:.3f}s kernel={kernel:.3f}s d2h={d2h:.3f}s reduce={reduction:.3f}s | "
        f"copy={copy_time:.3f}s compute={compute_time:.3f}s overlap={overlap}"
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
        "--dist-init-method",
        type=str,
        default="env://",
        help="Initialization method used for torch.distributed rendezvous",
    )
    parser.add_argument(
        "--dist-timeout",
        type=float,
        default=300.0,
        help="Timeout (in seconds) for distributed process group setup; non-positive disables",
    )
    parser.add_argument(
        "--dist-log-level",
        type=str,
        default="info",
        help="Logging level applied to distributed worker processes",
    )
    parser.add_argument(
        "--lease-min-inflight",
        dest="min_inflight",
        type=int,
        default=1,
        help="Minimum number of outstanding chunk leases to keep queued per rank",
    )
    parser.add_argument(
        "--lease-prefetch-slack-ms",
        dest="prefetch_slack_ms",
        type=float,
        default=50.0,
        help="Idle threshold in milliseconds before eagerly requesting another lease",
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
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Disable copy/compute overlap to compare throughput against the pipelined path",
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
        timeout_seconds = None
        if args.dist_timeout is not None and float(args.dist_timeout) > 0:
            timeout_seconds = float(args.dist_timeout)
        rendezvous = dist_runtime.RendezvousSettings(
            init_method=args.dist_init_method,
            timeout_seconds=timeout_seconds,
        )
        config = dist_runtime.DistributedLaunchConfig(
            device_ids=tuple(device_ids),
            world_size=world_size,
            log_level=args.dist_log_level,
            rendezvous=rendezvous,
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

    lease_client = None
    chunk_slices: List[Tuple[int, int]] | None = None
    preferred_lease_size = max(1, int(os.environ.get("SUPERTOKEN_LEASE_SIZE", "1")))
    lease_prefetch_threshold = max(
        0, int(os.environ.get("SUPERTOKEN_LEASE_PREFETCH_THRESHOLD", "0"))
    )
    lease_min_inflight = max(1, int(getattr(args, "min_inflight", 1)))
    lease_prefetch_slack_ms = max(0.0, float(getattr(args, "prefetch_slack_ms", 50.0)))
    if world_size > 1:
        chunk_target_ms = float(
            os.environ.get("SUPERTOKEN_CHUNK_TARGET_MS", DEFAULT_CHUNK_TARGET_MS)
        )
        chunk_slices = dist_runtime.plan_chunk_slices(
            len(shard_paths),
            target_ms=chunk_target_ms,
            batch_tokens=max(1, args.bs),
            ewma=None,
        )
        job_id = dist_runtime.compute_lease_job_id(shard_paths)
        lease_client = dist_runtime.register_lease_client(
            job_id=job_id,
            total_chunks=len(chunk_slices),
            rank=rank,
            world_size=world_size,
            max_active_leases=max(2, lease_min_inflight),
        )
        os.environ.setdefault("SUPERTOKEN_LEASE_JOB", job_id)
        os.environ.setdefault(
            "SUPERTOKEN_LEASE_TOTAL_CHUNKS", str(lease_client.total_chunks)
        )

    trainer_ref: dict[str, object | None] = {"value": None}

    def _iter_sequences() -> Iterator[Iterator[int]]:
        if lease_client is None or chunk_slices is None:
            with ExitStack() as stack:
                for path in shard_paths:
                    shard = stack.enter_context(MemoryMappedShardType(path))
                    yield packer.encode_shard(shard)
            return

        def _on_chunk(info) -> None:
            trainer_obj = trainer_ref.get("value")
            handler = getattr(trainer_obj, "handle_chunk_start", None)
            if handler is None:
                return
            try:
                handler(
                    getattr(info, "chunk_id", -1),
                    reprocessed=bool(getattr(info, "reprocessed", False)),
                    attempts=getattr(info, "attempts", None),
                )
            except Exception:
                pass

        yield from dist_runtime.iterate_leased_shards(
            shard_paths,
            chunk_slices,
            lease_client=lease_client,
            encode_shard=packer.encode_shard,
            shard_opener=MemoryMappedShardType,
            preferred_lease_size=preferred_lease_size,
            prefetch_threshold=lease_prefetch_threshold,
            min_inflight=lease_min_inflight,
            prefetch_slack_ms=lease_prefetch_slack_ms,
            on_chunk_start=_on_chunk,
        )

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
    trainer_ref["value"] = trainer
    iteration_callback = None
    if args.log_stage_timings:
        metrics_registry = trainer.metrics()
        throughput_tracker = None
        if isinstance(metrics_registry, Mapping):
            throughput_tracker = metrics_registry.get("throughput")
            if throughput_tracker is None and metrics_registry:
                throughput_tracker = next(iter(metrics_registry.values()))
        if throughput_tracker is not None:
            throughput_tracker.enabled = True

        metrics_tracker = throughput_tracker
        metrics_rank = rank
        metrics_world_size = world_size

        def _log_iteration(summary: dict[str, object]) -> None:
            if summary.get("kind") != "merge":
                return
            merge_idx = int(summary.get("merge", 0) or 0)
            if merge_idx <= 0:
                return
            if (merge_idx - 1) % max(1, log_every) != 0:
                return
            print(_format_iteration_summary(summary), flush=True)

            if metrics_tracker is None or not getattr(metrics_tracker, "enabled", False):
                return

            try:
                snapshot = metrics_tracker.snapshot()
            except Exception:
                return

            snapshots: Sequence[Mapping[str, object]]
            if (
                dist is not None
                and hasattr(dist, "gather_object")
                and dist.is_available()
                and dist.is_initialized()
                and metrics_world_size > 1
            ):
                gather_list: List[Mapping[str, object] | None] | None
                gather_list = [None] * metrics_world_size if metrics_rank == 0 else None
                try:
                    dist.gather_object(snapshot, gather_list, dst=0)  # type: ignore[arg-type]
                except Exception:
                    if metrics_rank != 0:
                        return
                    snapshots = [snapshot]
                else:
                    if metrics_rank != 0:
                        return
                    snapshots = [s for s in gather_list or [] if isinstance(s, Mapping)]
                    if not snapshots:
                        snapshots = [snapshot]
            else:
                snapshots = [snapshot]

            if metrics_rank != 0:
                return

            table = render_rank_metrics_table(snapshots)
            if table:
                print(table, flush=True)

        iteration_callback = _log_iteration
    meta = trainer.fit(
        batcher,
        log_every=log_every,
        warm_start_plan=warm_plan,
        freeze_warm_start=args.freeze_warm_start if warm_plan else None,
        overlap_transfers=not bool(getattr(args, "no_overlap", False)),
        on_iteration_summary=iteration_callback,
    )
    trainer.save(args.out_dir)
    print(meta)


if __name__ == "__main__":
    main()

