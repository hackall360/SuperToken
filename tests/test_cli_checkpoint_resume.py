import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

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

    def _extract_merges(payload: Mapping[str, object]) -> list[object]:
        trainer = payload.get("trainer") if isinstance(payload, Mapping) else None
        if isinstance(trainer, Mapping):
            model = trainer.get("model")
            if isinstance(model, Mapping):
                merges = model.get("merges")
                if isinstance(merges, list):
                    return merges
            merges = trainer.get("merges")
            if isinstance(merges, list):
                return merges
        merges = payload.get("merges") if isinstance(payload, Mapping) else None
        return merges if isinstance(merges, list) else []

    with state_path.open("r", encoding="utf-8") as f:
        partial_state = json.load(f)
    dataset_section = partial_state.get("dataset") if isinstance(partial_state, Mapping) else None
    assert isinstance(dataset_section, Mapping)
    stream_offsets = dataset_section.get("stream_offsets")
    assert isinstance(stream_offsets, Mapping) and stream_offsets
    partial_merges = _extract_merges(partial_state)
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
    assert "Restored dataset cursor" in completed.stdout

    with state_path.open("r", encoding="utf-8") as f:
        final_state = json.load(f)
    final_dataset = final_state.get("dataset") if isinstance(final_state, Mapping) else None
    assert isinstance(final_dataset, Mapping)
    final_stream_offsets = final_dataset.get("stream_offsets")
    assert isinstance(final_stream_offsets, Mapping) and final_stream_offsets
    final_merges = _extract_merges(final_state)
    assert len(final_merges) == merges
    assert final_merges[: len(partial_merges)] == partial_merges


def test_resume_cli_rejects_merge_mismatch(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shard = data_dir / "sample.txt"
    shard.write_text("hello\nworld\n", encoding="utf-8")

    checkpoint_dir = tmp_path / "checkpoints"
    merges = 6

    base_cmd = [
        sys.executable,
        "main.py",
        "train-bpe",
        "--data",
        str(shard),
        "--device",
        "cpu",
        "--merges",
        str(merges),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-every",
        "2",
        "--min-batch",
        "2",
        "--max-batch",
        "2",
        "--token-bytes",
        "16",
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
    proc.communicate()

    mismatch_cmd = base_cmd.copy()
    mismatch_cmd[mismatch_cmd.index("--merges") + 1] = str(merges + 1)
    mismatch_cmd.extend(["--resume-from", str(checkpoint_dir)])
    result = subprocess.run(
        mismatch_cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "Checkpoint configuration mismatch" in result.stderr
    assert "CLI --merges does not match" in result.stderr
