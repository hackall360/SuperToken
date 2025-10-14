"""I/O helpers for zero-copy dataset access."""

from __future__ import annotations

import mmap
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Iterator, Optional


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
