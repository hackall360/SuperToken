from __future__ import annotations

import importlib.util
import mmap
import sys
import types
from collections.abc import Iterator
from pathlib import Path

if "torch" not in sys.modules:  # pragma: no cover - lightweight stub for tests
    def _unavailable(*_args, **_kwargs):  # pragma: no cover - placeholder stub
        raise RuntimeError("torch stub is unavailable in tests")

    def _iinfo(dtype: object) -> types.SimpleNamespace:  # pragma: no cover
        if dtype == "uint16":
            return types.SimpleNamespace(max=65535)
        if dtype == "int32":
            return types.SimpleNamespace(max=2**31 - 1)
        raise RuntimeError(f"unsupported dtype for stub: {dtype!r}")

    torch_stub = types.SimpleNamespace(
        uint16="uint16",
        int32="int32",
        int16="int16",
        int8="int8",
        uint8="uint8",
        iinfo=_iinfo,
        full=_unavailable,
        zeros=_unavailable,
        as_tensor=_unavailable,
    )
    torch_stub._SUPERTOKEN_TORCH_STUB = True
    sys.modules["torch"] = torch_stub

ROOT = Path(__file__).resolve().parents[1]

if "gpu_tokenizer" not in sys.modules:
    pkg = types.ModuleType("gpu_tokenizer")
    pkg.__path__ = [str(ROOT / "gpu_tokenizer")]
    sys.modules["gpu_tokenizer"] = pkg
else:  # pragma: no cover - reuse existing stub if provided elsewhere
    pkg = sys.modules["gpu_tokenizer"]

cpu_packer = importlib.import_module("gpu_tokenizer.cpu_packer")
io_module = importlib.import_module("gpu_tokenizer.io")

pkg.BytePacker = cpu_packer.BytePacker
pkg.PackedBatcher = object()
pkg.AutoScaler = object()
pkg.GPUBPETrainer = object()
pkg.GPUUnigramTrainer = object()

main = importlib.import_module("main")

BytePacker = cpu_packer.BytePacker
MemoryMappedShard = io_module.MemoryMappedShard


def test_memory_mapped_shard_provides_zero_copy_view(tmp_path):
    data = b"abc" * 1_048_576  # ~3 MiB
    shard_path = tmp_path / "large.bin"
    shard_path.write_bytes(data)

    with MemoryMappedShard(shard_path) as shard:
        view = shard.view()
        assert len(view) == len(data)
        assert isinstance(view.obj, mmap.mmap)
        assert view[:5].tobytes() == data[:5]


def test_byte_packer_encode_shard_yields_bos_eos(tmp_path):
    shard_path = tmp_path / "sample.bin"
    shard_path.write_bytes(b"hello")

    packer = BytePacker(bos=1, eos=2)
    with MemoryMappedShard(shard_path) as shard:
        seq = list(packer.encode_shard(shard))

    assert seq[0] == 1
    assert seq[-1] == 2
    assert seq[1:-1] == [ord(c) for c in "hello"]


def test_load_sequences_streams_iterators(tmp_path):
    paths: list[Path] = []
    for idx in range(3):
        path = tmp_path / f"file_{idx}.bin"
        path.write_bytes(bytes([idx]) * (idx + 1))
        paths.append(path)

    seqs = main._load_sequences(paths, bos=100, eos=200)
    first = next(seqs)
    assert isinstance(first, Iterator)

    materialized = [list(first)]
    for seq in seqs:
        materialized.append(list(seq))

    assert materialized == [
        [100, 0, 200],
        [100, 1, 1, 200],
        [100, 2, 2, 2, 200],
    ]

