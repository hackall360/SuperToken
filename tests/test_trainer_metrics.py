import pytest

pytest.importorskip("torch")

from gpu_tokenizer.trainers.metrics import TrainerMetricsEWMA


def test_trainer_metrics_ewma_smoothing_and_window():
    metrics = TrainerMetricsEWMA(alpha=0.5, window_size=3, enabled=True)

    metrics.record_tokens(tokens=100, duration_s=1.0, leases=10)
    metrics.record_tokens(tokens=200, duration_s=1.0, leases=20)
    metrics.record_tokens(tokens=50, duration_s=1.0, leases=5)

    metrics.record_stage("h2d", 0.25)
    metrics.record_stage("d2h", 0.5)
    metrics.record_stage("kernel", 0.5)
    metrics.record_stage("kernel", 1.0)
    metrics.record_stage("kernel", 1.5)
    metrics.record_stage("kernel", 2.0)

    summary = metrics.summaries()

    assert summary["enabled"] is True
    assert summary["overlap_enabled"] is True
    assert summary["tokens_per_s"] == pytest.approx(100.0, rel=1e-6)
    assert summary["lease_per_s"] == pytest.approx(10.0, rel=1e-6)

    kernel_summary = summary["stages"].get("kernel", {})
    assert kernel_summary.get("samples") == 3
    assert kernel_summary.get("window") == [1.0, 1.5, 2.0]

    copy_summary = summary["copy"]
    compute_summary = summary["compute"]
    assert copy_summary["samples"] == 2
    assert copy_summary["latest_s"] == pytest.approx(0.5)
    assert compute_summary["samples"] == 3
    assert compute_summary["latest_s"] == pytest.approx(2.0)
    reduction_summary = summary["reduction"]
    assert reduction_summary["samples"] == 0
    assert reduction_summary["share_ewma"] is None

    metrics.reset()
    reset_summary = metrics.summaries()
    assert reset_summary["tokens_per_s"] is None
    assert reset_summary["lease_per_s"] is None
    assert reset_summary["stages"] == {}
    assert reset_summary["copy"]["samples"] == 0
    assert reset_summary["compute"]["samples"] == 0


def test_trainer_metrics_handles_disabled_state():
    metrics = TrainerMetricsEWMA(enabled=False)
    metrics.record_tokens(tokens=100, duration_s=1.0, leases=5)
    metrics.record_stage("kernel", 0.5)

    summary = metrics.summaries()
    assert summary["enabled"] is False
    assert summary["tokens_per_s"] is None
    assert summary["lease_per_s"] is None
    assert summary["stages"] == {}
    assert summary["copy"]["samples"] == 0
    assert summary["compute"]["samples"] == 0


def test_overlap_toggle_changes_throughput_breakdown():
    metrics = TrainerMetricsEWMA(alpha=1.0, window_size=4, enabled=True)

    metrics.record_tokens(tokens=1_000, duration_s=1.0)
    metrics.record_stage("kernel", 0.4)
    metrics.record_stage("h2d", 0.2)
    metrics.record_stage("d2h", 0.2)
    overlap_summary = metrics.summaries()

    metrics.reset()
    metrics.overlap_enabled = False
    metrics.record_tokens(tokens=1_000, duration_s=2.0)
    metrics.record_stage("kernel", 0.4)
    metrics.record_stage("h2d", 0.2)
    metrics.record_stage("d2h", 0.2)
    no_overlap_summary = metrics.summaries()

    assert overlap_summary["overlap_enabled"] is True
    assert no_overlap_summary["overlap_enabled"] is False
    assert no_overlap_summary["tokens_per_s"] < overlap_summary["tokens_per_s"]
    assert no_overlap_summary["copy"]["avg_s"] >= overlap_summary["copy"]["avg_s"]
    assert no_overlap_summary["compute"]["avg_s"] == pytest.approx(
        overlap_summary["compute"]["avg_s"]
    )


def test_reduction_overhead_tracking_and_summary():
    metrics = TrainerMetricsEWMA(alpha=0.5, window_size=4, enabled=True)

    metrics.record_iteration(total_duration_s=1.0, reduction_s=0.2)
    metrics.record_iteration(total_duration_s=1.0, reduction_s=0.4)

    summary = metrics.summaries()["reduction"]

    assert summary["samples"] == 2
    assert summary["avg_total_s"] == pytest.approx(1.0)
    assert summary["avg_reduction_s"] == pytest.approx(0.3)
    assert summary["latest_total_s"] == pytest.approx(1.0)
    assert summary["latest_reduction_s"] == pytest.approx(0.4)
    assert summary["share_latest"] == pytest.approx(0.4)
    assert summary["share_ewma"] == pytest.approx(0.3)


def test_reduction_cadence_adjustment_limits():
    metrics = TrainerMetricsEWMA(alpha=0.2, window_size=8, enabled=True)

    for _ in range(5):
        metrics.record_iteration(total_duration_s=1.0, reduction_s=0.25)

    cadence = metrics.recommend_reduction_cadence(
        8, min_cadence=8, max_cadence=10
    )
    assert cadence == 9

    cadence = metrics.recommend_reduction_cadence(
        cadence, min_cadence=8, max_cadence=9
    )
    assert cadence == 9

    for _ in range(6):
        metrics.record_iteration(total_duration_s=1.0, reduction_s=0.01)

    cadence = metrics.recommend_reduction_cadence(
        9, min_cadence=8, max_cadence=9
    )
    assert cadence == 8


def test_snapshot_tracks_per_rank_metrics():
    metrics = TrainerMetricsEWMA(alpha=0.5, window_size=4, enabled=True)
    metrics.set_rank(0)

    metrics.record_tokens(tokens=200, duration_s=2.0, leases=4)
    metrics.record_tokens(tokens=100, duration_s=1.0, leases=2)

    peer_snapshot = {
        "rank": 1,
        "tokens_per_s": 150.0,
        "lease_per_s": 3.0,
        "samples": 6.0,
    }
    metrics.update_rank_snapshot(peer_snapshot)

    snapshot = metrics.snapshot()

    assert snapshot["rank"] == 0
    assert snapshot["tokens_per_s"] == pytest.approx(metrics.tokens_per_s)
    per_rank = snapshot["per_rank"]
    assert 0 in per_rank and 1 in per_rank
    assert per_rank[0]["tokens_per_s"] == pytest.approx(metrics.tokens_per_s)
    assert per_rank[1]["tokens_per_s"] == pytest.approx(150.0)
    assert per_rank[1]["samples"] == pytest.approx(6.0)
