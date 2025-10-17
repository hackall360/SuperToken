from __future__ import annotations

import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if getattr(torch, "_SUPERTOKEN_TORCH_STUB", False) or not hasattr(torch, "full"):
    pytest.skip(
        "PyTorch with tensor factories is required for streaming IO tests",
        allow_module_level=True,
    )

from gpu_tokenizer.cpu_packer import BytePacker
from gpu_tokenizer.datasets import StreamingPackedBatcher
from gpu_tokenizer.io import CorpusStreamer, GPUUtilizationMonitor


class _ConstantMonitor(GPUUtilizationMonitor):
    def __init__(self, value: float) -> None:
        super().__init__(device=None)
        self._value = value

    def utilization(self) -> float:
        return self._value


@pytest.mark.parametrize("compression", ["none", "zstd", "lz4"])
def test_corpus_streamer_decodes(tmp_path: Path, compression: str) -> None:
    data = b"hello streamer\n"
    shard = tmp_path / "sample.bin"
    if compression == "none":
        shard.write_bytes(data)
    elif compression == "zstd":
        zstd = pytest.importorskip("zstandard")
        compressor = zstd.ZstdCompressor(level=1)
        shard.write_bytes(compressor.compress(data))
    else:
        lz4 = pytest.importorskip("lz4.frame")
        shard.write_bytes(lz4.compress(data))

    streamer = CorpusStreamer([shard], compression=compression, num_workers=1, max_prefetch=2)
    streamer.start()
    try:
        decoded = next(iter(streamer))
        assert decoded.view.tobytes() == data
        decoded.release()
    finally:
        streamer.close()


def test_backpressure_limits_queue(tmp_path: Path) -> None:
    data = b"x" * 16
    shards = []
    for idx in range(4):
        path = tmp_path / f"sample_{idx}.bin"
        path.write_bytes(data)
        shards.append(path)

    monitor = _ConstantMonitor(0.99)
    streamer = CorpusStreamer(
        shards,
        compression="none",
        num_workers=2,
        max_prefetch=4,
        gpu_monitor=monitor,
    )
    streamer.start()
    try:
        time.sleep(0.1)
        assert streamer.queue_depth() <= 1
    finally:
        streamer.close()


def test_streaming_batcher_emits_batches(tmp_path: Path) -> None:
    shards = []
    contents = [b"abc", b"defg"]
    for idx, payload in enumerate(contents):
        path = tmp_path / f"shard_{idx}.bin"
        path.write_bytes(payload)
        shards.append(path)

    packer = BytePacker(bos=1, eos=2)
    streamer = CorpusStreamer(shards, compression="none", num_workers=1, max_prefetch=2)
    streamer.start()
    try:
        batcher = StreamingPackedBatcher(streamer, packer.encode_view, batch_size=2)
        batches = list(batcher)
    finally:
        streamer.close()

    assert len(batches) == 1
    tokens, valid, lengths = batches[0]
    assert lengths.tolist() == [5, 6]
    assert tokens.shape[1] >= 6
    assert valid[0, : lengths[0]].sum().item() == lengths[0]
    assert valid[1, : lengths[1]].sum().item() == lengths[1]
