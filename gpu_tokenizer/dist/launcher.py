"""Utilities for launching distributed tokenizer trainers."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover - allow import without torch in CI
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]

from ..trainers.base import BaseTrainer
from ..dist_runtime import DistributedLeaseClient, register_lease_client
from .. import utils

logger = logging.getLogger(__name__)


ReducerFn = Callable[["torch.Tensor", "torch.Tensor"], tuple["torch.Tensor", "torch.Tensor"]]


@dataclass(frozen=True)
class RankLaunchConfig:
    """Configuration describing a rank-local trainer launch."""

    rank: int
    world_size: int
    init_method: Optional[str] = "env://"
    timeout_seconds: Optional[float] = None
    lease_job_id: Optional[str] = None
    lease_total_chunks: int = 0
    lease_max_active_leases: Optional[int] = None
    lease_root_rank: int = 0
    device: Optional[object] = None
    histogram_reducer: Optional[ReducerFn] = None

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if self.timeout_seconds is not None and self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative when provided")
        if self.lease_total_chunks < 0:
            raise ValueError("lease_total_chunks must be non-negative")
        if self.lease_max_active_leases is not None and self.lease_max_active_leases <= 0:
            raise ValueError("lease_max_active_leases must be positive when provided")


@dataclass
class LaunchContext:
    """Runtime context provided to trainer factories."""

    rank: int
    world_size: int
    lease_client: Optional[DistributedLeaseClient]
    device: Optional["torch.device"]


class LaunchHandle:
    """Container bundling a constructed trainer and distributed context."""

    def __init__(
        self,
        trainer: BaseTrainer,
        context: LaunchContext,
        cleanup_callbacks: Iterable[Callable[[], None]],
    ) -> None:
        self.trainer = trainer
        self.context = context
        self._cleanup: List[Callable[[], None]] = list(cleanup_callbacks)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for callback in reversed(self._cleanup):
            with contextlib.suppress(Exception):
                callback()

    def __enter__(self) -> "LaunchHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()


_histogram_reducer: Optional[ReducerFn] = None


def register_histogram_reducer(fn: Optional[ReducerFn]) -> None:
    """Register *fn* as the active histogram reducer for trainers."""

    global _histogram_reducer
    _histogram_reducer = fn


def get_histogram_reducer() -> ReducerFn:
    """Return the histogram reducer registered by the distributed launcher."""

    if _histogram_reducer is not None:
        return _histogram_reducer
    return utils.reduce_pair_histograms


def launch_rank(
    factory: Callable[[LaunchContext], BaseTrainer], *, config: RankLaunchConfig
) -> LaunchHandle:
    """Instantiate a rank-local trainer under distributed process context."""

    if not callable(factory):
        raise TypeError("factory must be callable")

    context, cleanup_callbacks = _initialise_rank_context(config)
    reducer = config.histogram_reducer or utils.reduce_pair_histograms
    register_histogram_reducer(reducer)
    cleanup_callbacks.append(lambda: register_histogram_reducer(None))

    try:
        trainer = factory(context)
    except Exception:
        for callback in reversed(cleanup_callbacks):
            with contextlib.suppress(Exception):
                callback()
        raise

    if not isinstance(trainer, BaseTrainer):
        for callback in reversed(cleanup_callbacks):
            with contextlib.suppress(Exception):
                callback()
        raise TypeError("factory must return an instance of BaseTrainer")

    return LaunchHandle(trainer=trainer, context=context, cleanup_callbacks=cleanup_callbacks)


def _initialise_rank_context(config: RankLaunchConfig) -> tuple[LaunchContext, List[Callable[[], None]]]:
    cleanup: List[Callable[[], None]] = []

    lease_client: Optional[DistributedLeaseClient] = None
    device_obj: Optional["torch.device"] = _resolve_device(config.device)

    if config.world_size > 1:
        _ensure_distributed_runtime()
        created_pg = False
        if dist is not None and not dist.is_initialized():
            init_kwargs = {
                "backend": "nccl",
                "init_method": config.init_method or "env://",
                "world_size": config.world_size,
                "rank": config.rank,
            }
            if config.timeout_seconds is not None:
                init_kwargs["timeout"] = timedelta(seconds=float(config.timeout_seconds))
            dist.init_process_group(**init_kwargs)  # type: ignore[arg-type]
            created_pg = True
        if created_pg:
            cleanup.append(_destroy_process_group)
        if config.lease_total_chunks:
            lease_client = register_lease_client(
                job_id=config.lease_job_id,
                total_chunks=config.lease_total_chunks,
                rank=config.rank,
                world_size=config.world_size,
                root=config.lease_root_rank,
                max_active_leases=config.lease_max_active_leases,
            )

    context = LaunchContext(
        rank=config.rank,
        world_size=config.world_size,
        lease_client=lease_client,
        device=device_obj,
    )

    return context, cleanup


def _resolve_device(spec: Optional[object]) -> Optional["torch.device"]:
    if torch is None or spec is None:
        return None
    try:
        device_obj = torch.device(spec)
    except Exception:
        logger.debug("failed_to_resolve_device", exc_info=True)
        return None
    if device_obj.type == "cuda":
        cuda_mod = getattr(torch, "cuda", None)
        set_device = getattr(cuda_mod, "set_device", None) if cuda_mod is not None else None
        if set_device is not None:
            with contextlib.suppress(Exception):
                set_device(device_obj)
    return device_obj


def _ensure_distributed_runtime() -> None:
    if torch is None or dist is None:
        raise RuntimeError("PyTorch distributed runtime is unavailable")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")


def _destroy_process_group() -> None:
    if dist is None:
        return
    destroy = getattr(dist, "destroy_process_group", None)
    if destroy is None:
        return
    with contextlib.suppress(Exception):
        destroy()


__all__ = [
    "RankLaunchConfig",
    "LaunchContext",
    "LaunchHandle",
    "launch_rank",
    "get_histogram_reducer",
    "register_histogram_reducer",
]
