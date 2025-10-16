from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "gpu_tokenizer" / "cli_train_bpe.py"
spec = importlib.util.spec_from_file_location("cli_train_bpe_arg_parser", MODULE_PATH)
cli_train_bpe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cli_train_bpe)


def test_parser_defaults() -> None:
    parser = cli_train_bpe.build_parser()
    args = parser.parse_args(["--data", "dummy.bin"])
    assert args.dist is False
    assert args.gpus is None
    assert args.target_chunk_ms is None
    assert args.log_stage_timings is False


def test_parser_parses_distributed_gpu_list() -> None:
    parser = cli_train_bpe.build_parser()
    args = parser.parse_args(["--data", "dummy.bin", "--dist", "--gpus", "0,2,2,3"])
    assert args.dist is True
    assert args.gpus == [0, 2, 3]


def test_main_rejects_gpus_without_dist() -> None:
    with pytest.raises(SystemExit):
        cli_train_bpe.main(["--data", "dummy.bin", "--gpus", "0"])


def test_main_rejects_non_positive_chunk_target() -> None:
    with pytest.raises(SystemExit):
        cli_train_bpe.main(["--data", "dummy.bin", "--target-chunk-ms", "0"])


def test_parser_enables_log_stage_flag() -> None:
    parser = cli_train_bpe.build_parser()
    args = parser.parse_args(
        ["--data", "dummy.bin", "--log-stage-timings", "--target-chunk-ms", "10"]
    )
    assert args.log_stage_timings is True
    assert args.target_chunk_ms == pytest.approx(10.0)
