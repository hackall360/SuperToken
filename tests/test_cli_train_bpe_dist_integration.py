"""Integration test ensuring distributed CLI parity with single-device runs."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Distributed CLI parity test requires at least two CUDA devices",
)
def test_cli_train_bpe_dist_matches_single_gpu(tmp_path: Path) -> None:
    """Train a toy corpus with and without ``--dist`` and compare outputs."""

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    shards = [
        corpus_dir / "shard0.bin",
        corpus_dir / "shard1.bin",
        corpus_dir / "shard2.bin",
    ]
    samples = [
        b"abracadabra abracadabra",
        b"alakazam alakazam",
        b"shazam shazam",
    ]
    for path, payload in zip(shards, samples):
        path.write_bytes(payload)

    def _run_cli(out_dir: Path, extra: list[str], env: dict[str, str] | None = None) -> dict[str, object]:
        cmd = [
            sys.executable,
            "-m",
            "gpu_tokenizer.cli_train_bpe",
            "--data",
            *(str(path) for path in shards),
            "--merges",
            "32",
            "--bs",
            "16",
            "--out-dir",
            str(out_dir),
        ]
        cmd.extend(extra)
        subprocess.run(cmd, check=True, env=env)
        meta_path = out_dir / "bpe_merges.json"
        assert meta_path.exists(), "Expected metadata file to be written"
        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    single_dir = tmp_path / "single"
    single_dir.mkdir()
    baseline = _run_cli(single_dir, extra=["--device", "cuda:0"])

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        env.setdefault("MASTER_PORT", str(sock.getsockname()[1]))

    dist_meta = _run_cli(
        dist_dir,
        extra=["--dist", "--gpus", "0,1", "--device", "cuda:0"],
        env=env,
    )

    assert dist_meta["vocab_size"] == baseline["vocab_size"]
    assert dist_meta["merges"] == baseline["merges"]
