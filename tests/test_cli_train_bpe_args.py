from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


def test_parser_preserves_gpu_argument() -> None:
    parser = cli_train_bpe.build_parser()
    args = parser.parse_args(
        ["--data", "dummy.bin", "--dist", "--gpus", "cuda:0,2,2,3"]
    )
    assert args.dist is True
    assert args.gpus == "cuda:0,2,2,3"


def test_main_rejects_gpus_without_dist() -> None:
    with pytest.raises(SystemExit):
        cli_train_bpe.main(["--data", "dummy.bin", "--gpus", "0"])


def test_main_rejects_non_positive_chunk_target() -> None:
    with pytest.raises(SystemExit):
        cli_train_bpe.main(["--data", "dummy.bin", "--target-chunk-ms", "0"])


def test_main_invokes_distributed_launcher(monkeypatch, tmp_path) -> None:
    launched: dict[str, object] = {}

    def _fake_launch(config, cli_args):
        launched["config"] = config
        launched["cli_args"] = list(cli_args)

    monkeypatch.setattr(cli_train_bpe.dist_runtime, "launch_training", _fake_launch)

    data_path = tmp_path / "corpus.bin"
    data_path.write_bytes(b"")

    cli_train_bpe.main(
        [
            "--data",
            str(data_path),
            "--target-chunk-ms",
            "5",
            "--dist",
            "--gpus",
            "cuda:0,1",
        ]
    )

    config = launched["config"]
    assert config.device_ids == (0, 1)
    assert config.world_size == 2
    assert launched["cli_args"] == [
        "--data",
        str(data_path),
        "--target-chunk-ms",
        "5",
    ]


def test_main_runs_single_process_path(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _FakeMemoryMappedShard:
        def __init__(self, path: str):
            self.path = path

        def __enter__(self):
            calls.setdefault("opened", []).append(self.path)
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.setdefault("closed", []).append(self.path)
            return False

    class _FakeBytePacker:
        def encode_shard(self, shard: _FakeMemoryMappedShard):
            calls.setdefault("encoded", []).append(shard.path)
            return iter([[1, 2, 3]])

    class _FakePackedBatcher:
        def __init__(self, seqs, batch_size):
            calls["batch_size"] = batch_size
            self.sequences = [list(seq) for seq in seqs]

        def __iter__(self):
            return iter(self.sequences)

    class _FakeAutoScaler:
        def __init__(self, target_util=0.8, device=None):
            calls.setdefault("autoscaler", []).append(
                {"target_util": target_util, "device": device}
            )

        def suggest(self, token_bytes_per_example: int):
            calls["suggested_tokens"] = token_bytes_per_example
            return SimpleNamespace(batch_size=4)

        def state_dict(self):
            return {"batch_size": 4}

    class _FakeTrainer:
        def __init__(
            self,
            *,
            base_vocab,
            merges,
            autoscaler,
            warm_start_merges=None,
            freeze_warm_start=False,
        ):
            calls["trainer_init"] = {
                "base_vocab": base_vocab,
                "merges": merges,
                "autoscaler": autoscaler,
                "warm_start_merges": warm_start_merges,
                "freeze_warm_start": freeze_warm_start,
            }
            self.metrics = SimpleNamespace(enabled=False)

        def fit(self, batcher, **kwargs):
            calls["fit_kwargs"] = {"batcher": batcher, **kwargs}
            return {"status": "ok"}

        def save(self, out_dir: str):
            calls["saved_dir"] = out_dir

        @staticmethod
        def precompute_warm_start_plan(batcher, warm_start_ngrams):
            return {"merges": [], "requested_top_k": warm_start_ngrams}

    def _fake_glob(pattern, recursive):
        calls.setdefault("globs", []).append((pattern, recursive))
        return ["/tmp/shard_a.bin", "/tmp/shard_b.bin"]

    monkeypatch.setattr(cli_train_bpe, "MemoryMappedShard", _FakeMemoryMappedShard, raising=False)
    monkeypatch.setattr(cli_train_bpe, "BytePacker", _FakeBytePacker, raising=False)
    monkeypatch.setattr(cli_train_bpe, "PackedBatcher", _FakePackedBatcher, raising=False)
    monkeypatch.setattr(cli_train_bpe, "AutoScaler", _FakeAutoScaler, raising=False)
    monkeypatch.setattr(cli_train_bpe, "GPUBPETrainer", _FakeTrainer, raising=False)
    monkeypatch.setattr(cli_train_bpe.glob, "glob", _fake_glob)

    cli_train_bpe.main(["--data", "dummy.bin", "--merges", "10"])

    assert calls["batch_size"] == 4
    assert calls["suggested_tokens"] == 8 * 1024
    assert calls["trainer_init"]["merges"] == 10
    assert calls["saved_dir"] == "./bpe_out"


def test_parser_enables_log_stage_flag() -> None:
    parser = cli_train_bpe.build_parser()
    args = parser.parse_args(
        ["--data", "dummy.bin", "--log-stage-timings", "--target-chunk-ms", "10"]
    )
    assert args.log_stage_timings is True
    assert args.target_chunk_ms == pytest.approx(10.0)
