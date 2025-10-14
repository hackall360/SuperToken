"""I/O helpers for zero-copy dataset access and streaming pipelines."""

from __future__ import annotations

import asyncio
import mmap
import os
import queue
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable, Iterator, Optional

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


CompressionType = str

_SENTINEL = object()


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
        gpu_monitor: Optional[GPUUtilizationMonitor] = None,
        buffer_pool_size: Optional[int] = None,
        buffer_bytes: int = 1 << 20,
    ) -> None:
        self.paths = [Path(p) for p in shards]
        self.compression = compression
        self.num_workers = max(1, num_workers)
        target_util = getattr(autoscaler, "tu", 0.8) if autoscaler is not None else 0.8
        self._backpressure = BackpressureController(
            target_util=target_util,
            min_depth=1,
            max_depth=max(1, max_prefetch),
            monitor=gpu_monitor,
        )
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=max(1, max_prefetch))
        self._tasks: "queue.Queue[Optional[Path]]" = queue.Queue()
        for path in self.paths:
            self._tasks.put(path)
        for _ in range(self.num_workers):
            self._tasks.put(None)
        pool_size = buffer_pool_size or max(2, self.num_workers)
        self._buffer_pool = _ReusableBufferPool(pool_size, buffer_bytes)
        self._threads: list[threading.Thread] = []
        self._stopped = threading.Event()

    def __enter__(self) -> "CorpusStreamer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

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
