"""Distributed runtime configuration and helpers for GPU tokenization workers."""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import logging
import os
import queue
import signal
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from types import FrameType
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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

from .io import make_chunker
from .lease_queue import LeaseNotary

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


@dataclass
class _LeaseHostState:
    """Container storing the shared :class:`LeaseNotary` for a job."""

    notary: LeaseNotary
    lock: threading.Lock

    @property
    def total_chunks(self) -> int:
        return self.notary.total_chunks


_LEASE_REGISTRY_LOCK = threading.Lock()
_LEASE_REGISTRY: Dict[str, _LeaseHostState] = {}


def _normalize_job_id(job_id: str | None) -> str:
    if not job_id:
        return "default"
    return job_id


def _get_or_create_host_state(job_id: str, total_chunks: int) -> _LeaseHostState:
    if total_chunks < 0:
        raise ValueError("total_chunks must be non-negative")
    with _LEASE_REGISTRY_LOCK:
        state = _LEASE_REGISTRY.get(job_id)
        if state is None:
            notary = LeaseNotary(total_chunks=total_chunks)
            state = _LeaseHostState(notary=notary, lock=threading.Lock())
            _LEASE_REGISTRY[job_id] = state
        elif state.total_chunks != total_chunks:
            raise ValueError(
                f"Existing lease registry for job {job_id!r} was initialised with "
                f"{state.total_chunks} chunks but {total_chunks} were requested"
            )
    return state


class DistributedLeaseClient:
    """Thin wrapper around :class:`LeaseNotary` shared across distributed ranks."""

    def __init__(
        self,
        *,
        job_id: str,
        rank: int,
        world_size: int,
        host_state: _LeaseHostState,
        root: int = 0,
    ) -> None:
        self.job_id = job_id
        self.rank = rank
        self.world_size = world_size
        self._host_state = host_state
        self._root = root

    @property
    def total_chunks(self) -> int:
        return self._host_state.total_chunks

    def _dist_ready(self) -> bool:
        return (
            dist is not None
            and hasattr(dist, "broadcast_object_list")
            and dist.is_available()
            and dist.is_initialized()
            and self.world_size > 1
        )

    def _broadcast_event(self, event: str, **payload: object) -> None:
        if not self._dist_ready():
            return
        message = [{"event": event, "rank": self.rank, **payload}]
        try:
            dist.broadcast_object_list(message, src=self.rank)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - best effort notification
            logger.debug(
                "lease_client_broadcast_failed",
                extra={"event": event, "rank": self.rank},
                exc_info=True,
            )

    def request_lease(self, preferred_size: int) -> Optional[Tuple[int, int]]:
        if preferred_size <= 0:
            raise ValueError("preferred_size must be positive")
        with self._host_state.lock:
            return self._host_state.notary.grant_lease(self.rank, preferred_size)

    def complete_lease(self, start: int, end: int) -> None:
        with self._host_state.lock:
            self._host_state.notary.complete_lease(self.rank, start, end)
        self._broadcast_event("complete", lease=(int(start), int(end)))

    def requeue_lease(self, start: int, end: int) -> None:
        with self._host_state.lock:
            self._host_state.notary.requeue_lease(self.rank, start, end)
        self._broadcast_event("requeue", lease=(int(start), int(end)))

    def heartbeat(self) -> float:
        with self._host_state.lock:
            return self._host_state.notary.heartbeat(self.rank)

    def requeue_outstanding(self) -> Optional[Tuple[int, int]]:
        """Best-effort requeue of the inflight lease for this rank."""

        with self._host_state.lock:
            snapshot = self._host_state.notary.state_dict()
        inflight_entry = snapshot["inflight"].get(self.rank)
        if not inflight_entry:
            return None
        start, end = inflight_entry["lease"]
        try:
            with self._host_state.lock:
                self._host_state.notary.requeue_lease(self.rank, int(start), int(end))
        except Exception:
            logger.debug(
                "lease_requeue_on_shutdown_failed",
                extra={"rank": self.rank, "lease": (start, end)},
                exc_info=True,
            )
            return None
        self._broadcast_event("requeue", lease=(int(start), int(end)))
        return (int(start), int(end))

    def iter_leases(self, preferred_size: int) -> Iterator[Tuple[int, int]]:
        while True:
            lease = self.request_lease(preferred_size)
            if lease is None:
                break
            yield lease


def register_lease_client(
    *,
    job_id: str | None,
    total_chunks: int,
    rank: int,
    world_size: int,
    root: int = 0,
) -> DistributedLeaseClient:
    """Return a :class:`DistributedLeaseClient` registered with rank ``root``."""

    normalized_job = _normalize_job_id(job_id)

    should_broadcast = (
        dist is not None
        and hasattr(dist, "broadcast_object_list")
        and dist.is_available()
        and dist.is_initialized()
        and world_size > 1
    )

    if should_broadcast:
        payload: List[object] = [
            {"job_id": normalized_job, "total_chunks": int(total_chunks)}
            if rank == root
            else None
        ]
        dist.broadcast_object_list(payload, src=root)  # type: ignore[arg-type]
        info = payload[0]
        if isinstance(info, dict):
            normalized_job = _normalize_job_id(str(info.get("job_id")))
            total_chunks = int(info.get("total_chunks", total_chunks))

    host_state = _get_or_create_host_state(normalized_job, int(total_chunks))
    return DistributedLeaseClient(
        job_id=normalized_job,
        rank=rank,
        world_size=world_size,
        host_state=host_state,
        root=root,
    )


def plan_chunk_slices(
    total_items: int,
    *,
    target_ms: float,
    batch_tokens: int,
    ewma: "TrainerMetricsEWMA | None" = None,
) -> List[Tuple[int, int]]:
    """Return contiguous slices describing how ``total_items`` should be chunked."""

    if total_items < 0:
        raise ValueError("total_items must be non-negative")
    chunker = make_chunker(target_ms=target_ms, batch_tokens=batch_tokens, ewma=ewma)
    slices: List[Tuple[int, int]] = []
    cursor = 0
    for spec in chunker:
        if cursor >= total_items:
            break
        width = max(1, int(getattr(spec, "batches", 1)))
        end = min(total_items, cursor + width)
        slices.append((cursor, end))
        cursor = end
    if total_items > 0 and not slices:
        raise RuntimeError("make_chunker did not yield any slices for the corpus")
    if slices and slices[-1][1] < total_items:
        slices.append((slices[-1][1], total_items))
    return slices


def compute_lease_job_id(paths: Sequence[str]) -> str:
    """Return a stable job identifier derived from *paths*."""

    digest = hashlib.blake2s(digest_size=8)
    for path in paths:
        digest.update(path.encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
    return f"lease:{digest.hexdigest()}"


def iterate_leased_shards(
    shard_paths: Sequence[str],
    chunk_slices: Sequence[Tuple[int, int]],
    *,
    lease_client: DistributedLeaseClient,
    encode_shard: Callable[[object], Iterator[int]],
    shard_opener: Callable[[str], contextlib.AbstractContextManager],
    preferred_lease_size: int = 1,
) -> Iterator[Iterator[int]]:
    """Yield shard iterators governed by leases from ``lease_client``."""

    if preferred_lease_size <= 0:
        raise ValueError("preferred_lease_size must be positive")
    total_chunks = len(chunk_slices)
    while True:
        lease = lease_client.request_lease(preferred_lease_size)
        if lease is None:
            break
        start, end = lease
        if not (0 <= start <= end <= total_chunks):
            lease_client.requeue_lease(start, end)
            raise ValueError(
                f"Lease [{start}, {end}) lies outside the planned chunk range"
            )
        try:
            for chunk_idx in range(start, end):
                slice_start, slice_end = chunk_slices[chunk_idx]
                if slice_start >= slice_end:
                    continue
                stack = contextlib.ExitStack()
                shards: List[object] = []
                try:
                    for path in shard_paths[slice_start:slice_end]:
                        shard = stack.enter_context(shard_opener(path))
                        shards.append(shard)
                    for shard in shards:
                        yield encode_shard(shard)
                finally:
                    stack.close()
        except GeneratorExit:
            lease_client.requeue_lease(start, end)
            raise
        except Exception:
            lease_client.requeue_lease(start, end)
            raise
        else:
            lease_client.complete_lease(start, end)


def _reset_lease_registry(job_id: str | None = None) -> None:
    """Test helper that clears cached lease state."""

    with _LEASE_REGISTRY_LOCK:
        if job_id is None:
            _LEASE_REGISTRY.clear()
        else:
            _LEASE_REGISTRY.pop(_normalize_job_id(job_id), None)


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

    lease_client: Optional[DistributedLeaseClient]
    lease_client = None
    lease_metadata: Dict[str, object] | None = None
    total_chunks_env = os.getenv("SUPERTOKEN_LEASE_TOTAL_CHUNKS")
    if total_chunks_env is not None:
        try:
            total_chunks_value = int(total_chunks_env)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError("SUPERTOKEN_LEASE_TOTAL_CHUNKS must be an integer") from exc
        job_id_env = os.getenv("SUPERTOKEN_LEASE_JOB")
        lease_client = register_lease_client(
            job_id=job_id_env,
            total_chunks=total_chunks_value,
            rank=rank,
            world_size=config.world_size,
        )
        lease_metadata = {
            "job_id": lease_client.job_id,
            "total_chunks": lease_client.total_chunks,
        }
        if lease_client.total_chunks > 0:
            initial = lease_client.request_lease(1)
            if initial is not None:
                lease_client.requeue_lease(*initial)

    shutdown_state = {"triggered": False}

    def _initiate_shutdown(reason: str) -> None:
        if shutdown_state["triggered"]:
            return
        shutdown_state["triggered"] = True
        if lease_client is not None:
            with contextlib.suppress(Exception):
                lease_client.requeue_outstanding()
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
                "lease_metadata": lease_metadata,
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
