from __future__ import annotations

import importlib
from pathlib import Path

import argparse

import pytest

from tests._stubs import install_torch_stub

install_torch_stub()
main = importlib.import_module("main")


def test_resume_bpe_delegates_to_train(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[argparse.Namespace] = []

    def _fake_train(args: argparse.Namespace) -> None:
        calls.append(args)

    monkeypatch.setattr(main, "_cmd_train_bpe", _fake_train)

    shard = tmp_path / "shard.txt"
    shard.write_text("hello world\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    main.main(
        [
            "resume-bpe",
            "--data",
            str(shard),
            "--merges",
            "24",
            "--target-util",
            "0.5",
            "--min-batch",
            "8",
            "--max-batch",
            "16",
            "--token-bytes",
            "64",
            "--resume-from",
            str(checkpoint_dir),
        ]
    )

    assert len(calls) == 1
    called_args = calls[0]
    assert called_args.command == "resume-bpe"
    assert called_args.resume_from == str(checkpoint_dir)
    assert called_args.target_util == 0.5
    assert called_args.min_batch == 8
    assert called_args.max_batch == 16
    assert called_args.token_bytes == 64


