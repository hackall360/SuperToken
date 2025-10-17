"""I/O helpers for zero-copy dataset access and streaming pipelines."""

from __future__ import annotations

import asyncio
import json
import logging
import mmap
import os
import queue
import random
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Callable, Iterable, Iterator, Optional

try:  # pragma: no cover - optional dependency
    import zstandard as _zstd  # type: ignore
except Exception:  # pragma: no cover - allow runtime fallback
    _zstd = None

try:  # pragma: no cover - optional dependency
    import lz4.frame as _lz4f  # type: ignore
except Exception:  # pragma: no cover - allow runtime fallback
    _lz4f = None

try:  # pragma: no cover - optional torch dependency
    import torch
except Exception:  # pragma: no cover - torch optional at runtime
    torch = None


if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .trainers.metrics import TrainerMetricsEWMA


CompressionType = str

_SENTINEL = object()

logger = logging.getLogger(__name__)

AutoscaleCallback = Callable[[dict[str, object], int, int, int], Optional[int]]


class DecompressionError(RuntimeError):
    """Raised when a shard cannot be decompressed with the requested codec."""


class GPUUtilizationMonitor:
    """Minimal GPU utilization helper used for queue backpressure."""

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None and torch is not None and torch.cuda.is_available():
            device = torch.device("cuda", index=torch.cuda.current_device()).index  # type: ignore[arg-type]
            device = f"cuda:{device if device is not None else 0}"
        self.device = device

    def utilization(self) -> Optional[float]:
        if torch is None or self.device is None or not str(self.device).startswith("cuda"):
            return None
        try:
            free, total = torch.cuda.mem_get_info(self.device)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - defensive fallback
            try:
                free, total = torch.cuda.mem_get_info()  # type: ignore[misc]
            except Exception:
                return None
        if total == 0:
            return None
        used = max(0, total - free)
        return max(0.0, min(1.0, used / float(total)))


class BackpressureController:
    """Dynamically gate producer throughput based on observed GPU load."""

    def __init__(
        self,
        target_util: float = 0.8,
        min_depth: int = 1,
        max_depth: int = 8,
        monitor: Optional[GPUUtilizationMonitor] = None,
        poll_interval_s: float = 0.02,
    ) -> None:
        self.target_util = max(0.1, min(0.99, target_util))
        self.min_depth = max(1, min_depth)
        self.max_depth = max(self.min_depth, max_depth)
        self.monitor = monitor or GPUUtilizationMonitor()
        self._poll = max(0.005, poll_interval_s)

    def allowed_depth(self) -> int:
        util = self.monitor.utilization()
        if util is None:
            return self.max_depth
        if util >= self.target_util:
            return self.min_depth
        headroom = max(0.0, self.target_util - util)
        scale = headroom / max(self.target_util, 1e-6)
        depth = self.min_depth + int(round(scale * (self.max_depth - self.min_depth)))
        return max(self.min_depth, min(self.max_depth, depth))

    def update_limits(
        self, *, min_depth: Optional[int] = None, max_depth: Optional[int] = None
    ) -> None:
        if min_depth is not None:
            self.min_depth = max(1, min_depth)
        if max_depth is not None:
            self.max_depth = max(self.min_depth, max_depth)

    def wait_for_slot(self, q: "queue.Queue[object]") -> None:
        while True:
            allowed = self.allowed_depth()
            if q.qsize() < allowed:
                return
            time.sleep(self._poll)


class MemoryMappedShard(AbstractContextManager["MemoryMappedShard"]):
    """Memory-map a corpus file and provide zero-copy slices."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._fd: Optional[int] = None
        self._mmap: Optional[mmap.mmap] = None
        self._size: Optional[int] = None
        self._empty = memoryview(b"")
        self._views: list[memoryview] = []

    def _ensure_open(self) -> None:
        if self._mmap is not None or self._size is not None:
            return
        fd = os.open(self.path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            self._size = size
            if size == 0:
                self._fd = fd
                self._mmap = None
                return
            mm = mmap.mmap(fd, length=0, access=mmap.ACCESS_READ)
        except Exception:
            os.close(fd)
            raise
        else:
            self._fd = fd
            self._mmap = mm

    def __enter__(self) -> "MemoryMappedShard":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for view in self._views:
            if hasattr(view, "release"):
                try:
                    view.release()
                except (BufferError, ValueError):  # pragma: no cover - defensive fallback
                    pass
        self._views.clear()
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._size = None

    def __len__(self) -> int:
        if self._size is None:
            self._ensure_open()
        assert self._size is not None
        return self._size

    def view(self, start: int = 0, end: Optional[int] = None) -> memoryview:
        if self._size is None:
            self._ensure_open()
        if self._size == 0:
            view = self._empty[start:end]
            self._views.append(view)
            return view
        assert self._mmap is not None
        base = memoryview(self._mmap)
        view = base[start:end]
        self._views.append(view)
        return view

    def iter_bytes(self, start: int = 0, end: Optional[int] = None) -> Iterator[int]:
        view = self.view(start, end)
        for byte in view.cast("B"):
            yield byte


class _ReusableBufferPool:
    """Thread-safe buffer pool used for compression codecs."""

    def __init__(self, size: int, initial_bytes: int = 1 << 20):
        self._buffers: "queue.LifoQueue[bytearray]" = queue.LifoQueue(maxsize=size)
        for _ in range(size):
            self._buffers.put(bytearray(initial_bytes))

    def acquire(self, required_bytes: int) -> bytearray:
        buf = self._buffers.get()
        if len(buf) < required_bytes:
            buf.extend(b"\x00" * (required_bytes - len(buf)))
        return buf

    def release(self, buf: bytearray) -> None:
        try:
            self._buffers.put_nowait(buf)
        except queue.Full:  # pragma: no cover - defensive fallback
            pass


@dataclass
class ChunkSpec:
    """Describe a workload chunk fed into the trainer pipeline."""

    batches: int
    tokens: int
    target_ms: float
    expected_ms: Optional[float]
    tokens_per_s_hint: Optional[float]
    leases_per_s_hint: Optional[float]


def make_chunker(
    target_ms: float,
    batch_tokens: int,
    ewma: "TrainerMetricsEWMA | None",
) -> Iterator[ChunkSpec]:
    """Return a generator producing chunk specifications near ``target_ms``.

    ``batch_tokens`` represents the token count processed per trainer batch.  The
    generator consults the optional :class:`TrainerMetricsEWMA` to adapt the chunk
    size based on recent throughput measurements.  When EWMA metrics are
    unavailable the chunker yields fixed-size chunks containing exactly one
    batch.
    """

    try:
        target_ms_value = float(target_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_ms must be convertible to float") from exc
    if not (target_ms_value > 0.0):
        raise ValueError("target_ms must be positive")

    try:
        batch_tokens_value = int(batch_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch_tokens must be an integer") from exc
    if batch_tokens_value <= 0:
        raise ValueError("batch_tokens must be positive")

    def _iter_chunks() -> Iterator[ChunkSpec]:
        while True:
            tokens_per_s = None
            leases_per_s = None
            if ewma is not None and getattr(ewma, "enabled", True):
                tokens_per_s = ewma.tokens_per_s
                leases_per_s = ewma.lease_per_s

            if tokens_per_s is not None and tokens_per_s > 0:
                tokens_per_ms = tokens_per_s / 1000.0
                estimated_tokens = max(
                    batch_tokens_value,
                    int(round(tokens_per_ms * target_ms_value)),
                )
            else:
                estimated_tokens = batch_tokens_value

            batches = max(1, int(round(estimated_tokens / batch_tokens_value)))
            tokens = batches * batch_tokens_value

            expected_ms: Optional[float]
            if tokens_per_s is not None and tokens_per_s > 0:
                expected_ms = (tokens / tokens_per_s) * 1000.0
            else:
                expected_ms = None

            yield ChunkSpec(
                batches=batches,
                tokens=tokens,
                target_ms=target_ms_value,
                expected_ms=expected_ms,
                tokens_per_s_hint=tokens_per_s,
                leases_per_s_hint=leases_per_s,
            )

    return _iter_chunks()


@dataclass
class DecodedShard:
    """Representation of a decoded corpus shard delivered by the streamer."""

    path: Path
    view: memoryview
    release_fn: Callable[[], None]

    def release(self) -> None:
        self.release_fn()


def _decompress_zstd(data: memoryview, pool: _ReusableBufferPool) -> tuple[memoryview, Callable[[], None]]:
    if _zstd is None:
        raise DecompressionError("zstandard support requires the `zstandard` package")
    dctx = _zstd.ZstdDecompressor()
    result = dctx.decompress(data.tobytes())
    buffer = pool.acquire(len(result))
    buffer[: len(result)] = result
    view = memoryview(buffer)[: len(result)]

    def _release() -> None:
        pool.release(buffer)

    return view, _release


def _decompress_lz4(data: memoryview, pool: _ReusableBufferPool) -> tuple[memoryview, Callable[[], None]]:
    if _lz4f is None:
        raise DecompressionError("lz4 support requires the `lz4` package")
    result = _lz4f.decompress(data.tobytes())
    buffer = pool.acquire(len(result))
    buffer[: len(result)] = result
    view = memoryview(buffer)[: len(result)]

    def _release() -> None:
        pool.release(buffer)

    return view, _release


class CorpusStreamer(AbstractContextManager["CorpusStreamer"]):
    """Asynchronously decode shards into a bounded producer queue."""

    def __init__(
        self,
        shards: Iterable[Path],
        *,
        compression: CompressionType = "none",
        num_workers: int = 2,
        max_prefetch: int = 8,
        autoscaler: Optional[object] = None,
        autoscaler_callback: Optional[AutoscaleCallback] = None,
        gpu_monitor: Optional[GPUUtilizationMonitor] = None,
        prefetch_jitter: float = 0.0,
        buffer_pool_size: Optional[int] = None,
        buffer_bytes: int = 1 << 20,
    ) -> None:
        self.paths = [Path(p) for p in shards]
        self.compression = compression
        self.num_workers = max(1, num_workers)
        target_util = getattr(autoscaler, "tu", 0.8) if autoscaler is not None else 0.8
        self._prefetch_floor = 1
        self._prefetch_ceiling = max(1, max_prefetch)
        self._prefetch_target = self._prefetch_ceiling
        self._prefetch_lock = threading.Lock()
        self._prefetch_jitter = max(0.0, min(0.5, float(prefetch_jitter)))
        self._autoscale_callback: Optional[AutoscaleCallback]
        if autoscaler_callback is None:
            self._autoscale_callback = self._default_autoscale_policy
        else:
            self._autoscale_callback = autoscaler_callback
        self._backpressure = BackpressureController(
            target_util=target_util,
            min_depth=1,
            max_depth=self._prefetch_ceiling,
            monitor=gpu_monitor,
        )
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=self._prefetch_ceiling)
        self._tasks: "queue.Queue[Optional[Path]]" = queue.Queue()
        for path in self.paths:
            self._tasks.put(path)
        for _ in range(self.num_workers):
            self._tasks.put(None)
        pool_size = buffer_pool_size or max(2, self.num_workers)
        self._buffer_pool = _ReusableBufferPool(pool_size, buffer_bytes)
        self._threads: list[threading.Thread] = []
        self._stopped = threading.Event()
        self._autoscaler = autoscaler
        if autoscaler is not None and hasattr(autoscaler, "register_feedback_listener"):
            try:
                autoscaler.register_feedback_listener(self._handle_autoscaler_feedback)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("failed to register autoscaler feedback listener")

    def __enter__(self) -> "CorpusStreamer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _default_autoscale_policy(
        self, payload: dict[str, object], current: int, floor: int, ceiling: int
    ) -> Optional[int]:
        event = str(payload.get("event", "")).lower()
        if event == "oom":
            return max(floor, max(1, current // 2))
        if event == "high_utilization":
            mean_vram = float(payload.get("mean_vram", 0.0) or 0.0)
            target = float(payload.get("target_util", 0.0) or 0.0)
            severity = max(0.0, mean_vram - target)
            scale = 0.75
            if severity > 0.12:
                scale = 0.5
            elif severity > 0.05:
                scale = 0.65
            next_limit = max(floor, int(round(current * scale)))
            return max(floor, min(ceiling, next_limit))
        if event == "low_utilization" and current < ceiling:
            step = max(1, int(round(max(1, ceiling) * 0.1)))
            return min(ceiling, current + step)
        return None

    def _handle_autoscaler_feedback(self, payload: dict[str, object]) -> None:
        callback = self._autoscale_callback
        if callback is None:
            return
        current_limit = self.prefetch_limit()
        try:
            suggestion = callback(payload, current_limit, self._prefetch_floor, self._prefetch_ceiling)
        except TypeError:
            try:
                suggestion = callback(payload)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("autoscale callback failed")
                return
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("autoscale callback failed")
            return
        if suggestion is None:
            return
        limit = int(suggestion)
        limit = max(self._prefetch_floor, min(self._prefetch_ceiling, limit))
        if limit < current_limit and self._prefetch_jitter > 0:
            jitter = random.uniform(0.0, self._prefetch_jitter)
            limit = max(self._prefetch_floor, int(round(limit * (1.0 - jitter))))
        self._apply_prefetch_limit(limit, reason=str(payload.get("event", "autoscale")), metadata=payload)

    def _apply_prefetch_limit(
        self, limit: int, *, reason: str, metadata: Optional[dict[str, object]] = None
    ) -> None:
        with self._prefetch_lock:
            prev = self._prefetch_target
            bounded = max(self._prefetch_floor, min(self._prefetch_ceiling, int(limit)))
            if bounded == prev:
                return
            self._prefetch_target = bounded
            self._backpressure.update_limits(max_depth=bounded)
        if bounded < prev:
            payload: dict[str, object] = {
                "reason": reason,
                "prev_limit": prev,
                "new_limit": bounded,
                "queue_depth": self.queue_depth(),
            }
            if metadata:
                filtered: dict[str, object] = {}
                for key in (
                    "event",
                    "mean_vram",
                    "var_vram",
                    "mean_step_time",
                    "cpu_fallback_rate",
                    "target_util",
                    "state",
                    "prev_state",
                ):
                    if key in metadata:
                        filtered[key] = metadata[key]
                if filtered:
                    payload["autoscaler"] = filtered
            logger.info("stream.prefetch.adjust %s", json.dumps(payload, sort_keys=True))

    def prefetch_limit(self) -> int:
        with self._prefetch_lock:
            return self._prefetch_target

    def start(self) -> None:
        if self._threads:
            return
        for _ in range(self.num_workers):
            thread = threading.Thread(target=self._worker, daemon=True)
            thread.start()
            self._threads.append(thread)

    def close(self) -> None:
        self._stopped.set()
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if isinstance(item, DecodedShard):
                item.release()
        for thread in self._threads:
            thread.join(timeout=0.1)
        self._threads.clear()

    def _worker(self) -> None:
        while not self._stopped.is_set():
            try:
                path = self._tasks.get(timeout=0.1)
            except queue.Empty:
                continue
            if path is None:
                break
            shard = MemoryMappedShard(path)
            shard.__enter__()
            try:
                view = shard.view()
                if self.compression == "none":
                    release_fn = shard.close
                    decoded = DecodedShard(path=path, view=view, release_fn=release_fn)
                elif self.compression == "zstd":
                    decompressed, release_fn = _decompress_zstd(view, self._buffer_pool)
                    shard.close()
                    decoded = DecodedShard(path=path, view=decompressed, release_fn=release_fn)
                elif self.compression == "lz4":
                    decompressed, release_fn = _decompress_lz4(view, self._buffer_pool)
                    shard.close()
                    decoded = DecodedShard(path=path, view=decompressed, release_fn=release_fn)
                else:
                    shard.close()
                    raise ValueError(f"Unsupported compression type: {self.compression}")
                self._backpressure.wait_for_slot(self._queue)
                if self._stopped.is_set():
                    decoded.release()
                    break
                self._queue.put(decoded)
            finally:
                self._tasks.task_done()
        self._queue.put(_SENTINEL)

    def __iter__(self) -> Iterator[DecodedShard]:
        active = self.num_workers
        while active > 0:
            item = self._queue.get()
            if item is _SENTINEL:
                active -= 1
                continue
            assert isinstance(item, DecodedShard)
            yield item

    async def __aiter__(self) -> AsyncIterator[DecodedShard]:
        loop = asyncio.get_running_loop()
        active = self.num_workers
        while active > 0:
            item = await loop.run_in_executor(None, self._queue.get)
            if item is _SENTINEL:
                active -= 1
                continue
            assert isinstance(item, DecodedShard)
            yield item

    def queue_depth(self) -> int:
        return self._queue.qsize()
