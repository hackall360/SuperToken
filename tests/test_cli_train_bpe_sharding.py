import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "gpu_tokenizer" / "cli_train_bpe.py"
spec = importlib.util.spec_from_file_location("cli_train_bpe_test_module", MODULE_PATH)
cli_train_bpe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cli_train_bpe)


class _FakeDist:
    def __init__(self, *, available=True, initialized=True, rank=0, world_size=1):
        self._available = available
        self._initialized = initialized
        self._rank = rank
        self._world_size = world_size

    def is_available(self):
        return self._available

    def is_initialized(self):
        return self._initialized

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size


def test_resolve_distributed_rank_prefers_torch(monkeypatch):
    fake = _FakeDist(rank=2, world_size=8)
    monkeypatch.setattr(cli_train_bpe, "dist", fake)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert cli_train_bpe.resolve_distributed_rank() == (2, 8)


def test_resolve_distributed_rank_env_fallback(monkeypatch):
    monkeypatch.setattr(
        cli_train_bpe, "dist", _FakeDist(initialized=False), raising=False
    )
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert cli_train_bpe.resolve_distributed_rank() == (3, 16)


def test_resolve_distributed_rank_local_rank(monkeypatch):
    monkeypatch.setattr(
        cli_train_bpe, "dist", _FakeDist(initialized=False), raising=False
    )
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert cli_train_bpe.resolve_distributed_rank() == (1, 4)


def test_partition_paths_for_rank_validates_inputs():
    with pytest.raises(ValueError):
        cli_train_bpe.partition_paths_for_rank([], rank=0, world_size=0)
    with pytest.raises(ValueError):
        cli_train_bpe.partition_paths_for_rank([], rank=5, world_size=2)


def test_partition_paths_unique_across_ranks(tmp_path):
    paths = []
    for i in range(6):
        file_path = tmp_path / f"shard_{i}.bin"
        file_path.write_bytes(b"data")
        paths.append(str(file_path))

    shard_allocations = [
        cli_train_bpe.partition_paths_for_rank(paths, rank=r, world_size=4)
        for r in range(4)
    ]

    flattened = [path for allocation in shard_allocations for path in allocation]
    assert all(allocation == sorted(allocation) for allocation in shard_allocations)
    assert sorted(flattened) == sorted(set(paths))
    assert len(flattened) == len(set(flattened))

