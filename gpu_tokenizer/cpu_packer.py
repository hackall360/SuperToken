"""CPU-side byte-to-id packing utilities."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from .io import MemoryMappedShard

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from .morphology import MorphologyPlugin

BytesLike = memoryview | bytes | bytearray
ChunkLike = BytesLike | Iterable[int]


def _iter_chunk_bytes(chunk: ChunkLike) -> Iterator[int]:
    if isinstance(chunk, (bytes, bytearray, memoryview)):
        view = memoryview(chunk)
        if view.ndim != 1 or view.format != "B":
            view = view.cast("B")
        for byte in view:
            yield int(byte)
        return
    for byte in chunk:
        yield int(byte)


class BytePacker:
    """Convert byte streams to integer id sequences."""

    def __init__(
        self,
        bos: int | None = None,
        eos: int | None = None,
        *,
        morphology: "MorphologyPlugin" | None = None,
    ):
        self.bos = bos
        self.eos = eos
        self.morphology = morphology

    def _iter_chunk_pieces(self, chunk: ChunkLike) -> Iterator[ChunkLike]:
        if isinstance(chunk, (bytes, bytearray, memoryview)):
            if self.morphology is not None:
                yield from self.morphology.iter_surfaces(chunk)
            else:
                yield chunk
            return
        yield chunk

    def encode_view(self, data: BytesLike) -> Iterator[int]:
        def _iterator() -> Iterator[int]:
            if self.bos is not None:
                yield self.bos
            for piece in self._iter_chunk_pieces(data):
                yield from _iter_chunk_bytes(piece)
            if self.eos is not None:
                yield self.eos

        return _iterator()

    def encode_chunks(self, chunks: Iterable[ChunkLike]) -> Iterator[int]:
        def _iterator() -> Iterator[int]:
            if self.bos is not None:
                yield self.bos
            for chunk in chunks:
                for piece in self._iter_chunk_pieces(chunk):
                    yield from _iter_chunk_bytes(piece)
            if self.eos is not None:
                yield self.eos

        return _iterator()

    def encode_shard(self, shard: MemoryMappedShard) -> Iterator[int]:
        return self.encode_view(shard.view())

    def encode_file(self, path: str) -> Iterator[int]:
        def _iterator() -> Iterator[int]:
            with MemoryMappedShard(path) as shard:
                yield from self.encode_shard(shard)

        return _iterator()

    def encode_sequence(self, sequence: BytesLike | Iterable[ChunkLike]) -> Iterator[int]:
        if isinstance(sequence, (bytes, bytearray, memoryview)):
            return self.encode_view(sequence)
        if isinstance(sequence, Iterable):
            return self.encode_chunks(sequence)
        raise TypeError(f"Unsupported input type: {type(sequence)!r}")
