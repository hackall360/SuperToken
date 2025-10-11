"""CPU-side byte-to-id packing utilities."""

from __future__ import annotations

class BytePacker:
    """Convert byte streams to integer id sequences."""

    def __init__(self, bos: int | None = None, eos: int | None = None):
        self.bos = bos
        self.eos = eos

    def encode_file(self, path: str) -> list[int]:
        with open(path, "rb") as f:
            data = f.read()
        out: list[int] = []
        if self.bos is not None:
            out.append(self.bos)
        out.extend(data)
        if self.eos is not None:
            out.append(self.eos)
        return out
