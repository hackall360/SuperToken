import pytest

cli_train_bpe = pytest.importorskip("gpu_tokenizer.cli_train_bpe")


def test_parser_defaults() -> None:
    parser = cli_train_bpe.build_parser()
    args = parser.parse_args(["--data", "dummy.bin"])
    assert args.dist is False
    assert args.gpus is None
    assert args.target_chunk_ms is None
    assert args.log_stage_timings is False
    assert args.no_overlap is False


def test_parser_no_overlap_flag() -> None:
    parser = cli_train_bpe.build_parser()
    args = parser.parse_args(["--data", "dummy.bin", "--no-overlap"])
    assert args.no_overlap is True
