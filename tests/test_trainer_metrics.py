import pytest

pytest.importorskip("torch")

from gpu_tokenizer.bpe_trainer import TrainerMetricsEWMA


def test_trainer_metrics_ewma_smoothing_and_window():
    metrics = TrainerMetricsEWMA(alpha=0.5, window_size=3, enabled=True)

    metrics.record_tokens(tokens=100, duration_s=1.0, leases=10)
    metrics.record_tokens(tokens=200, duration_s=1.0, leases=20)
    metrics.record_tokens(tokens=50, duration_s=1.0, leases=5)

    metrics.record_stage("kernel", 0.5)
    metrics.record_stage("kernel", 1.0)
    metrics.record_stage("kernel", 1.5)
    metrics.record_stage("kernel", 2.0)

    summary = metrics.summaries()

    assert summary["enabled"] is True
    assert summary["tokens_per_s"] == pytest.approx(100.0, rel=1e-6)
    assert summary["lease_per_s"] == pytest.approx(10.0, rel=1e-6)

    kernel_summary = summary["stages"].get("kernel", {})
    assert kernel_summary.get("samples") == 3
    assert kernel_summary.get("window") == [1.0, 1.5, 2.0]

    metrics.reset()
    reset_summary = metrics.summaries()
    assert reset_summary["tokens_per_s"] is None
    assert reset_summary["lease_per_s"] is None
    assert reset_summary["stages"] == {}


def test_trainer_metrics_handles_disabled_state():
    metrics = TrainerMetricsEWMA(enabled=False)
    metrics.record_tokens(tokens=100, duration_s=1.0, leases=5)
    metrics.record_stage("kernel", 0.5)

    summary = metrics.summaries()
    assert summary["enabled"] is False
    assert summary["tokens_per_s"] is None
    assert summary["lease_per_s"] is None
    assert summary["stages"] == {}
