import contextlib
import sys
import types

import pytest

torch_stub = types.ModuleType("torch")
dist_stub = types.ModuleType("torch.distributed")
dist_stub.is_available = lambda: False
dist_stub.is_initialized = lambda: False
dist_stub.get_rank = lambda: 0
dist_stub.get_world_size = lambda: 1
dist_stub.ReduceOp = types.SimpleNamespace(SUM=0, MAX=1)


class _FakeDType:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeDType) and self.name == other.name


torch_stub.uint16 = _FakeDType("uint16")
torch_stub.int32 = _FakeDType("int32")
torch_stub.int16 = _FakeDType("int16")
torch_stub.int8 = _FakeDType("int8")
torch_stub.uint8 = _FakeDType("uint8")
torch_stub.int64 = _FakeDType("int64")
def _fake_iinfo(dtype):
    name = getattr(dtype, "name", "")
    if name == "int32":
        return types.SimpleNamespace(max=2_147_483_647)
    if name == "int64":
        return types.SimpleNamespace(max=9_223_372_036_854_775_807)
    return types.SimpleNamespace(max=65_535)


torch_stub.iinfo = _fake_iinfo
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
torch_stub.distributed = dist_stub
torch_stub.jit = types.SimpleNamespace(script=lambda fn, *a, **k: fn)
torch_stub.Tensor = type("Tensor", (), {})
torch_stub.no_grad = contextlib.nullcontext
torch_stub.device = lambda *a, **k: None
torch_stub.float32 = _FakeDType("float32")
torch_stub.float64 = _FakeDType("float64")
torch_stub.zeros = lambda *a, **k: torch_stub.Tensor()
torch_stub.tensor = lambda *a, **k: torch_stub.Tensor()
utils_stub = sys.modules.setdefault("torch.utils", types.ModuleType("torch.utils"))
torch_stub.utils = utils_stub
cpp_stub = types.ModuleType("torch.utils.cpp_extension")
cpp_stub.load = lambda *a, **k: None  # type: ignore[attr-defined]
cpp_stub.load_inline = lambda *a, **k: None  # type: ignore[attr-defined]
cpp_stub.include_paths = lambda: []  # type: ignore[attr-defined]
utils_stub.cpp_extension = cpp_stub
sys.modules.setdefault("torch.utils.cpp_extension", cpp_stub)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.distributed", dist_stub)

cli_train_bpe = pytest.importorskip("gpu_tokenizer.cli_train_bpe")


def test_parser_defaults() -> None:
    parser = cli_train_bpe.build_parser()
    args = parser.parse_args(["--data", "dummy.bin"])
    assert args.dist is False
    assert args.gpus is None
    assert args.dist_init_method == "env://"
    assert args.dist_timeout == 300.0
    assert args.dist_log_level == "info"
    assert args.target_chunk_ms is None
    assert args.log_stage_timings is False
    assert args.no_overlap is False


def test_parser_no_overlap_flag() -> None:
    parser = cli_train_bpe.build_parser()
    args = parser.parse_args(["--data", "dummy.bin", "--no-overlap"])
    assert args.no_overlap is True


def test_dist_launch_propagates_cli_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_config: dict[str, object] = {}
    captured_args: dict[str, list[str]] = {}

    def _fake_config(**kwargs):
        captured_config.update(kwargs)
        return types.SimpleNamespace(**kwargs)

    def _fake_launch(config, cli_args):
        captured_args["cli"] = list(cli_args)

    dist_stub = types.SimpleNamespace(
        DistributedLaunchConfig=_fake_config,
        RendezvousSettings=lambda **kwargs: types.SimpleNamespace(**kwargs),
        launch_training=_fake_launch,
    )

    monkeypatch.setattr(cli_train_bpe, "dist_runtime", dist_stub, raising=False)

    argv = [
        "--data",
        "dummy.bin",
        "--dist",
        "--gpus",
        "0,1",
        "--dist-init-method",
        "tcp://127.0.0.1:1234",
        "--dist-timeout",
        "42",
        "--dist-log-level",
        "DEBUG",
    ]

    cli_train_bpe.main(argv)

    assert captured_config["device_ids"] == (0, 1)
    assert captured_config["world_size"] == 2
    assert captured_config["log_level"] == "DEBUG"
    rendezvous = captured_config["rendezvous"]
    assert rendezvous.init_method == "tcp://127.0.0.1:1234"
    assert rendezvous.timeout_seconds == 42.0
    assert captured_args["cli"] == ["--data", "dummy.bin"]


def test_render_rank_metrics_table_formats_rows() -> None:
    snapshots = [
        {
            "rank": 0,
            "tokens_per_s": 1234.567,
            "lease_per_s": 890.123,
            "samples": 4.0,
            "idle_ms": 12.5,
            "lease_width": 8,
        },
        {
            "rank": 1,
            "tokens_per_s": 2222.0,
            "lease_per_s": 111.5,
            "samples": 8.0,
            "idle_ms": 50.0,
            "lease_width": 4,
            "per_rank": {
                0: {
                    "tokens_per_s": 1234.567,
                    "lease_per_s": 890.123,
                    "samples": 4.0,
                    "idle_ms": 12.5,
                    "lease_width": 8,
                },
                1: {
                    "tokens_per_s": 2222.0,
                    "lease_per_s": 111.5,
                    "samples": 8.0,
                    "idle_ms": 50.0,
                    "lease_width": 4,
                },
            },
        },
    ]

    table = cli_train_bpe.render_rank_metrics_table(snapshots)
    assert "Rank" in table
    assert "Tokens/s" in table
    assert "Leases/s" in table
    assert "Samples" in table
    assert "Idle EWMA (ms)" in table
    assert "Lease Width" in table
    assert "0" in table
    assert "1" in table
    assert "1,234.57" in table
    assert "111.50" in table
    assert "12.50 |           8" in table


def test_format_iteration_summary_reads_nested_metrics() -> None:
    summary = {
        "merge": 5,
        "tokens": 1_000,
        "leases": 12,
        "h2d_s": 0.1,
        "kernel_s": 0.2,
        "d2h_s": 0.1,
        "reduction_s": 0.0,
        "copy_s": 0.2,
        "compute_s": 0.3,
        "overlap": True,
        "metrics": {
            "throughput": {"tokens_per_s": 4321.0, "lease_per_s": 21.5}
        },
    }

    line = cli_train_bpe._format_iteration_summary(summary)
    assert "4,321" in line
    assert "22/s" in line
