from datetime import datetime
from pathlib import Path
import json

import pytest

pytest.importorskip("matplotlib")

from benchmarks.trend_report import TrendPoint, evaluate_baseline, main as trend_main


def _make_point(tokens: int = 100, bpe_wall: float = 10.0, unigram_wall: float = 5.0) -> TrendPoint:
    return TrendPoint(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        source=Path("benchmark_20240101T120000Z.json"),
        tokens=tokens,
        bpe_wall=bpe_wall,
        unigram_wall=unigram_wall,
    )


def test_evaluate_baseline_flags_regressions() -> None:
    point = _make_point(tokens=100, bpe_wall=20.0, unigram_wall=25.0)  # 5 and 4 tokens/s
    summary = evaluate_baseline([point], baseline_bpe=6.0, baseline_unigram=2.0)
    assert summary["bpe_met"] is False
    assert summary["unigram_met"] is True
    assert "bpe" in summary["failed_metrics"]
    assert "unigram" not in summary["failed_metrics"]


def _write_snapshot(path: Path, *, tokens: int, bpe_wall: float, unigram_wall: float) -> None:
    payload = {
        "timestamp": "20240101T120000Z",
        "config": {},
        "corpus": {"sequences": 1, "tokens": tokens, "max_length": tokens, "sources": []},
        "bpe": {
            "config": {},
            "wall_time_s": bpe_wall,
            "result": {},
            "overlap_enabled": True,
            "tokens_processed": tokens,
            "tokens_per_s": tokens / bpe_wall if bpe_wall else None,
            "autoscaler_window": [],
        },
        "unigram": {
            "config": {},
            "wall_time_s": unigram_wall,
            "epochs": [],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_trend_report_cli_writes_baseline_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    snapshot = input_dir / "benchmark_20240101T120000Z.json"
    _write_snapshot(snapshot, tokens=100, bpe_wall=10.0, unigram_wall=20.0)

    trend_main(
        [
            "--input",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--baseline-bpe",
            "5",
            "--baseline-unigram",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert "meets baseline" in captured.out
    manifest = json.loads((output_dir / "trend_manifest.json").read_text(encoding="utf-8"))
    baseline = manifest["baseline_evaluation"]
    assert baseline["bpe_met"] is True
    assert baseline["unigram_met"] is True


def test_trend_report_cli_exits_on_baseline_failure(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    snapshot = input_dir / "benchmark_20240101T120000Z.json"
    _write_snapshot(snapshot, tokens=100, bpe_wall=10.0, unigram_wall=20.0)

    with pytest.raises(SystemExit):
        trend_main(
            [
                "--input",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--baseline-bpe",
                "20",
                "--baseline-unigram",
                "20",
            ]
        )
