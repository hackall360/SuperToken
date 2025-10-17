import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if getattr(torch, "_SUPERTOKEN_TORCH_STUB", False) or not hasattr(torch, "tensor"):
    pytest.skip(
        "PyTorch with tensor support is required for CLI checkpoint resume tests",
        allow_module_level=True,
    )


def _wait_for_checkpoint(
    state_path: Path,
    tensor_path: Path,
    proc: subprocess.Popen[str],
    timeout_s: float = 30.0,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if state_path.exists() and tensor_path.exists():
            return
        if proc.poll() is not None:
            raise RuntimeError("training process exited before writing a checkpoint")
        time.sleep(0.1)
    raise TimeoutError("checkpoint files were not created in time")


def test_train_bpe_resume_cli(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shard_path = data_dir / "shard.txt"
    shard_path.write_text("hello world\nthis is a tiny dataset\n", encoding="utf-8")

    checkpoint_dir = tmp_path / "checkpoints"
    merges = 12

    base_cmd = [
        sys.executable,
        "main.py",
        "train-bpe",
        "--data",
        str(shard_path),
        "--device",
        "cpu",
        "--merges",
        str(merges),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-every",
        "4",
        "--min-batch",
        "2",
        "--max-batch",
        "4",
        "--token-bytes",
        "32",
        "--log-every",
        "2",
    ]

    env = os.environ.copy()
    proc = subprocess.Popen(
        base_cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    state_path = checkpoint_dir / "state.json"
    tensor_path = checkpoint_dir / "tensors.pt"
    _wait_for_checkpoint(state_path, tensor_path, proc)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    partial_stdout, partial_stderr = proc.communicate()

    with state_path.open("r", encoding="utf-8") as f:
        partial_state = json.load(f)
    partial_merges = partial_state.get("merges", [])
    assert partial_merges, "expected checkpoint to contain completed merges"
    assert len(partial_merges) < merges

    resume_cmd = base_cmd + ["--resume-from", str(checkpoint_dir)]
    completed = subprocess.run(
        resume_cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )

    assert "Restored checkpoint" in completed.stdout

    with state_path.open("r", encoding="utf-8") as f:
        final_state = json.load(f)
    final_merges = final_state.get("merges", [])
    assert len(final_merges) == merges
    assert final_merges[: len(partial_merges)] == partial_merges
