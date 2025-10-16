"""Distributed runtime configuration and helpers for GPU tokenization workers."""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import queue
import signal
from dataclasses import dataclass, field
from datetime import timedelta
from types import FrameType
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - torch is optional in some environments
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
except Exception:  # pragma: no cover - fallback when torch missing
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    mp = None  # type: ignore[assignment]

try:  # pragma: no cover - optional type for static analysis
    from torch.multiprocessing.spawn import ProcessContext
except Exception:  # pragma: no cover - fallback when torch missing
    ProcessContext = object  # type: ignore[misc,assignment]

# Configure a module level logger. Downstream applications can adjust the
# configuration to taste but this ensures we at least emit something when the
# module is used in isolation.
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:  # pragma: no cover - configuration guard
    logging.basicConfig(level=logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


SignalHandler = Callable[[int, Optional[FrameType]], None]
_SHUTDOWN_SIGNALS: Tuple[int, int] = (signal.SIGINT, signal.SIGTERM)


def _destroy_process_group_if_available() -> None:
    """Best-effort destruction of an initialized torch.distributed group."""

    if dist is None:  # pragma: no cover - guarded by optional dependency
        return
    with contextlib.suppress(Exception):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _terminate_outstanding_leases(scope: str) -> None:
    """Log a placeholder shutdown event for pending work leases."""

    logger.info("lease_shutdown", extra={"scope": scope})


def _install_signal_handlers(handler: SignalHandler) -> Dict[int, SignalHandler | int | None]:
    """Register *handler* for shutdown signals, returning previous handlers."""

    previous: Dict[int, SignalHandler | int | None] = {}
    for sig in _SHUTDOWN_SIGNALS:
        try:
            prior = signal.getsignal(sig)
        except Exception:  # pragma: no cover - defensive fallback
            prior = None
        previous[sig] = prior
        signal.signal(sig, handler)
    return previous


def _restore_signal_handlers(previous: Dict[int, SignalHandler | int | None]) -> None:
    """Restore handlers captured by :func:`_install_signal_handlers`."""

    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - defensive fallback
            logger.debug("failed_restoring_signal_handler", extra={"signal": sig})


def _summarize_exception(payload: object) -> str:
    """Return a concise human readable description of *payload*."""

    if payload is None:
        return ""
    if isinstance(payload, BaseException):
        return f"{payload.__class__.__name__}: {payload}"
    for attr in ("message", "msg", "exc_msg", "reason", "detail"):
        value = getattr(payload, attr, None)
        if value:
            return str(value)
    if hasattr(payload, "exc_type") and hasattr(payload, "exc_value"):
        return f"{getattr(payload, 'exc_type')}: {getattr(payload, 'exc_value')}"
    return str(payload)


def _drain_error_queue(queue_obj: object) -> Optional[str]:
    """Fetch and format the first exception payload from *queue_obj*."""

    if queue_obj is None:
        return None
    getter = getattr(queue_obj, "get_nowait", None)
    if getter is None:
        getter = getattr(queue_obj, "get", None)
        if getter is None:
            return None
        try:
            item = getter(False)
        except TypeError:
            with contextlib.suppress(Exception):
                item = getter(timeout=0.0)
        except Exception:
            return None
        else:
            return _summarize_exception(item)
    try:
        item = getter()
    except queue.Empty:
        return None
    except Exception:
        return None
    return _summarize_exception(item)


def _collect_process_failures(context: ProcessContext) -> List[str]:
    """Return human readable failure descriptions for *context* workers."""

    failures: List[str] = []

    processes = list(getattr(context, "processes", []) or [])
    error_queues = list(getattr(context, "error_queues", []) or [])

    for rank, proc in enumerate(processes):
        exitcode = getattr(proc, "exitcode", None)
        errors: List[str] = []
        if rank < len(error_queues):
            summary = _drain_error_queue(error_queues[rank])
            if summary:
                errors.append(summary)
        if exitcode not in (None, 0):
            errors.insert(0, f"exit code {exitcode}")
        if errors:
            message = f"Rank {rank} failed: {'; '.join(errors)}"
            logger.error(
                "worker_exit_nonzero",
                extra={"worker_rank": rank, "details": errors},
            )
            failures.append(message)

    # Capture any additional error queues beyond len(processes)
    for offset, queue_obj in enumerate(error_queues[len(processes):], start=len(processes)):
        summary = _drain_error_queue(queue_obj)
        if summary:
            message = f"Rank {offset} reported error: {summary}"
            logger.error(
                "worker_error_only",
                extra={"worker_rank": offset, "details": summary},
            )
            failures.append(message)

    return failures


@dataclass(frozen=True)
class DeviceRequest:
    """Describe a type of accelerator device that should be provisioned."""

    kind: str
    count: int
    capability: Optional[str] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.count <= 0:
            raise ValueError("DeviceRequest.count must be positive")
        if not self.kind:
            raise ValueError("DeviceRequest.kind must be non-empty")


@dataclass(frozen=True)
class ChunkTarget:
    """Declare how many samples each worker should accumulate per chunk."""

    tokens: int
    sequences: int

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.tokens <= 0:
            raise ValueError("ChunkTarget.tokens must be positive")
        if self.sequences <= 0:
            raise ValueError("ChunkTarget.sequences must be positive")


@dataclass(frozen=True)
class LaunchConfig:
    """Aggregate the full configuration for launching distributed workers."""

    devices: List[DeviceRequest] = field(default_factory=list)
    chunk_target: Optional[ChunkTarget] = None
    rebalance_cadence: Optional[timedelta] = None
    environment: dict[str, str] = field(default_factory=dict)

    def iter_env(self) -> Iterable[tuple[str, str]]:
        """Iterate over environment variables with defaults applied."""

        for key, value in self.environment.items():
            yield key, value


@dataclass(frozen=True)
class RendezvousSettings:
    """Describe how distributed workers rendezvous for process group creation."""

    init_method: str = "env://"
    timeout_seconds: float | None = 300.0

    def __post_init__(self) -> None:  # pragma: no cover - basic validation
        if not self.init_method:
            raise ValueError("init_method must be a non-empty string")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")

    def resolve_timeout(self) -> Optional[timedelta]:
        """Return a :class:`~datetime.timedelta` suitable for torch.distributed."""

        if self.timeout_seconds is None:
            return None
        return timedelta(seconds=float(self.timeout_seconds))


@dataclass(frozen=True)
class DistributedLaunchConfig:
    """Configuration payload for launching distributed BPE training workers."""

    device_ids: Sequence[int]
    world_size: int
    log_level: int | str = logging.INFO
    rendezvous: RendezvousSettings = field(default_factory=RendezvousSettings)

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not self.device_ids:
            raise ValueError("device_ids must be a non-empty sequence")
        canonical: Tuple[int, ...] = tuple(int(idx) for idx in self.device_ids)
        if len(canonical) < self.world_size:
            raise ValueError("device_ids must cover all ranks in world_size")
        object.__setattr__(self, "device_ids", canonical)

        level = self.log_level
        if isinstance(level, str):
            numeric = logging.getLevelName(level.upper())
            if isinstance(numeric, str):
                raise ValueError(f"Unknown log level {level!r}")
            level = int(numeric)
        object.__setattr__(self, "log_level", int(level))


def get_env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean from an environment variable."""

    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw in {"1", "true", "yes", "on"}


def get_env_int(name: str, default: int) -> int:
    """Return an integer from an environment variable, with a default."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Environment variable {name!r} must be an integer") from exc


def launch_training(config: DistributedLaunchConfig, cli_args: Sequence[str]) -> None:
    """Launch distributed tokenizer training workers using torch.multiprocessing."""

    if mp is None or torch is None:
        raise RuntimeError("PyTorch is required for distributed training")
    if not torch.cuda.is_available():  # pragma: no cover - requires CUDA hardware
        raise RuntimeError("CUDA must be available to launch distributed training")

    if not hasattr(mp, "start_processes"):
        raise RuntimeError("torch.multiprocessing.start_processes is required for distributed launch")

    logger.info(
        "spawning_workers",
        extra={
            "world_size": config.world_size,
            "devices": list(config.device_ids),
            "rendezvous": {
                "init_method": config.rendezvous.init_method,
                "timeout_seconds": config.rendezvous.timeout_seconds,
            },
        },
    )

    args = (config, tuple(cli_args))
    context_holder: Dict[str, Optional[ProcessContext]] = {"context": None}

    def _parent_signal_handler(signum: int, _frame: Optional[FrameType]) -> None:
        logger.warning(
            "launcher_signal_received",
            extra={"signal": signum, "world_size": config.world_size},
        )
        _terminate_outstanding_leases("launcher")
        _destroy_process_group_if_available()
        ctx = context_holder.get("context")
        if ctx is not None:
            with contextlib.suppress(Exception):
                ctx.terminate()

    previous_handlers = _install_signal_handlers(_parent_signal_handler)

    try:
        start_processes = getattr(mp, "start_processes")
        context = start_processes(  # type: ignore[operator]
            _worker_entry,
            args=args,
            nprocs=config.world_size,
            join=False,
            daemon=False,
            start_method="spawn",
        )
        context_holder["context"] = context

        join_exception: Optional[BaseException] = None
        try:
            context.join()
        except BaseException as exc:  # pragma: no cover - join only fails on error
            join_exception = exc

        failures = _collect_process_failures(context)
        if failures:
            message = "One or more distributed workers exited non-zero:\n" + "\n".join(failures)
            aggregated_error = RuntimeError(message)
            if join_exception is not None:
                aggregated_error.__cause__ = join_exception
            raise aggregated_error
        if join_exception is not None:
            raise join_exception
    finally:
        _restore_signal_handlers(previous_handlers)


def _worker_entry(rank: int, config: DistributedLaunchConfig, cli_args: Sequence[str]) -> None:
    """Worker entry point invoked by :func:`launch_training`."""

    if torch is None or dist is None:
        raise RuntimeError("PyTorch distributed runtime is unavailable")

    device_index = config.device_ids[rank % len(config.device_ids)]
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)

    logging.getLogger().setLevel(config.log_level)
    worker_logger = logging.getLogger(f"{__name__}.worker")

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(config.world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    timeout = config.rendezvous.resolve_timeout()
    dist.init_process_group(
        backend="nccl",
        init_method=config.rendezvous.init_method,
        world_size=config.world_size,
        rank=rank,
        timeout=timeout,
    )

    shutdown_state = {"triggered": False}

    def _initiate_shutdown(reason: str) -> None:
        if shutdown_state["triggered"]:
            return
        shutdown_state["triggered"] = True
        worker_logger.info(
            "worker_shutdown",
            extra={
                "worker_rank": rank,
                "world_size": config.world_size,
                "device": str(device),
                "reason": reason,
            },
        )
        _terminate_outstanding_leases("worker")
        _destroy_process_group_if_available()

    def _worker_signal_handler(signum: int, _frame: Optional[FrameType]) -> None:
        worker_logger.warning(
            "worker_signal_received",
            extra={"worker_rank": rank, "signal": signum},
        )
        _initiate_shutdown(f"signal:{signum}")

    atexit.register(lambda: _initiate_shutdown("atexit"))
    worker_previous_handlers = _install_signal_handlers(_worker_signal_handler)

    try:
        from .autoscaler import AutoScaler
        from .bpe_trainer import GPUBPETrainer

        autoscaler = AutoScaler(device=str(device))
        trainer = GPUBPETrainer(devices=[str(device)], autoscaler=autoscaler)

        worker_logger.info(
            "worker_ready",
            extra={
                "worker_rank": rank,
                "world_size": config.world_size,
                "device": str(device),
                "cli_args": list(cli_args),
                "autoscaler": autoscaler.state_dict(),
                "trainer_device_count": len(trainer.devices),
            },
        )
    finally:
        _restore_signal_handlers(worker_previous_handlers)


def launch_workers(config: LaunchConfig) -> None:
    """Spawn worker processes as described by *config*.

    Future iterations of the distributed runtime will implement orchestration
    logic here. For now this placeholder makes the API explicit while clearly
    signalling that the functionality is not yet implemented.
    """

    raise NotImplementedError("Distributed worker launch is not implemented yet")


def worker_main() -> None:
    """Entry point for a tokenizer worker process."""

    raise NotImplementedError("Distributed worker execution is not implemented yet")
