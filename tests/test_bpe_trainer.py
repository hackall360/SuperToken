import pytest
from types import MethodType
import time

torch = pytest.importorskip("torch")
pytest.importorskip("torch.utils")

from gpu_tokenizer import bpe_trainer as bt
from gpu_tokenizer.autoscaler import AutoScaler, ScaleState
from gpu_tokenizer.bpe_trainer import GPUBPETrainer, _aggregate_pair_keys
from gpu_tokenizer.dtypes import length_storage_dtype
from gpu_tokenizer.datasets import PackedBatcher
from gpu_tokenizer.bpe_trainer import GPUBatchRecord
from gpu_tokenizer.cpu_fastpath import (
    FastPathWorkspaces,
    apply_merge_fastpath,
    count_pairs_fastpath,
    should_route_to_cpu,
)
from gpu_tokenizer.utils import apply_merge_once, count_pairs


def test_aggregate_pair_keys_repeated_counts():
    keys = torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.long)
    counts = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)

    aggregated_keys, aggregated_counts = _aggregate_pair_keys(keys, counts)

    assert torch.equal(aggregated_keys, torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(
        aggregated_counts, torch.tensor([1 + 2 + 3, 4 + 5, 6], dtype=torch.int32)
    )


def test_aggregate_pair_keys_unsorted_input():
    keys = torch.tensor([2, 1, 3, 1, 2], dtype=torch.long)
    counts = torch.tensor([5, 1, 2, 3, 4], dtype=torch.int32)

    aggregated_keys, aggregated_counts = _aggregate_pair_keys(keys, counts)

    assert torch.equal(aggregated_keys, torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(
        aggregated_counts, torch.tensor([1 + 3, 5 + 4, 2], dtype=torch.int32)
    )


class OneShotIterable:
    def __init__(self, batches):
        self._batches = batches
        self._iterated = False

    def __iter__(self):
        if self._iterated:
            raise AssertionError("Batch stream should only be iterated once")
        self._iterated = True
        for batch in self._batches:
            yield batch


def _make_batch(seqs: list[list[int]]):
    max_len = max(len(seq) for seq in seqs)
    tokens = torch.full((len(seqs), max_len), -1, dtype=torch.int32)
    valid = torch.zeros((len(seqs), max_len), dtype=torch.uint8)
    length_dtype = length_storage_dtype(max_len)
    lengths = torch.zeros(len(seqs), dtype=length_dtype)
    for row, seq in enumerate(seqs):
        L = len(seq)
        if L == 0:
            continue
        tokens[row, :L] = torch.tensor(seq, dtype=torch.int32)
        valid[row, :L] = 1
        lengths[row] = L
    if torch.cuda.is_available():
        tokens = tokens.pin_memory()
        valid = valid.pin_memory()
    return tokens, valid, lengths


def test_cpu_fastpath_pair_count_matches_baseline():
    seqs = [[1, 2, 3, 4], [4, 3, 2, 1]]
    tokens, valid, lengths = _make_batch(seqs)
    tokens = tokens.clone()
    valid = valid.clone()
    B, L = tokens.shape
    width = max(L - 1, 0)
    capacity = max(B * width, 1)
    pair_workspace = torch.empty((capacity, 2), dtype=tokens.dtype)
    count_workspace = torch.empty((capacity,), dtype=torch.int32)
    pair_length = torch.zeros((1,), dtype=torch.long)
    count_pairs(tokens, valid, pair_workspace, count_workspace, pair_length)
    length = int(pair_length.item())
    baseline_pairs = pair_workspace[:length]
    baseline_counts = count_workspace[:length]
    baseline_keys = (
        (baseline_pairs[:, 0].to(torch.long) << 32)
        | baseline_pairs[:, 1].to(torch.long)
    )

    fast_keys, fast_counts = count_pairs_fastpath(tokens, valid)
    assert torch.equal(torch.sort(baseline_keys)[0], torch.sort(fast_keys)[0])
    assert torch.equal(
        torch.sort(baseline_counts.to(torch.int32))[0],
        torch.sort(fast_counts.to(torch.int32))[0],
    )


def test_cpu_fastpath_merge_matches_reference():
    seqs = [[1, 2, 1, 2], [2, 1, 2, 1]]
    tokens, valid, lengths = _make_batch(seqs)
    tokens_fast = tokens.clone()
    valid_fast = valid.clone()
    lengths_fast = lengths.clone()
    workspaces = FastPathWorkspaces()
    apply_merge_fastpath(tokens_fast, valid_fast, lengths_fast, 1, 2, 260, workspaces)

    tokens_ref = tokens.clone()
    valid_ref = valid.clone()
    lengths_ref = lengths.clone()
    apply_merge_once(
        tokens_ref,
        valid_ref,
        lengths_ref,
        1,
        2,
        260,
        None,
        None,
        None,
        None,
    )

    assert torch.equal(tokens_fast, tokens_ref)
    assert torch.equal(valid_fast, valid_ref)
    assert torch.equal(lengths_fast.to(torch.int64), lengths_ref.to(torch.int64))


def test_should_route_to_cpu_heuristic():
    assert should_route_to_cpu(1, 10)
    assert should_route_to_cpu(8, 0)
    assert not should_route_to_cpu(32, 64)


def test_autoscaler_cpu_fallback_feedback(monkeypatch):
    scaler = AutoScaler(device="cuda")
    scaler.state = ScaleState(batch_size=512, cpu_workers=4, h2d_mb=512, cpu_fallback_rate=0.0)

    monkeypatch.setattr(scaler, "_gpu_caps", lambda: (900, 1000))
    monkeypatch.setattr(scaler, "_cpu_caps", lambda: (8, 1 << 20, 1 << 20, 10.0))

    scaler.feedback(step_time_s=0.1, cpu_fallback_rate=0.5)
    assert pytest.approx(0.5, rel=1e-3) == scaler.state.cpu_fallback_rate
    assert scaler.state.batch_size <= 512


def test_gpu_batch_record_flags_overflow_for_uint16_lengths():
    width = 65535
    tokens = torch.zeros((1, width), dtype=torch.int32)
    valid = torch.ones((1, width), dtype=torch.uint8)
    lengths = torch.tensor([width + 10], dtype=torch.int64)
    record = GPUBatchRecord.from_cpu(tokens, valid, lengths, torch.device("cpu"))

    assert record.lengths.dtype == torch.uint16
    assert record.length_overflow is not None
    assert bool(record.length_overflow.item())

    host_tokens, host_valid, host_lengths = record.resolve_host()
    assert host_lengths.dtype == torch.uint16
    assert host_tokens.data_ptr() == record.host_tokens.data_ptr()
    assert host_valid.data_ptr() == record.host_valid.data_ptr()
    assert bool(record.host_length_overflow is not None and record.host_length_overflow.item())


def test_fit_supports_streaming_iterator():
    seqs = [
        [1, 2, 1, 2, 3],
        [1, 2, 4, 2],
        [1, 2, 1, 2],
        [1, 2, 1, 2],
    ]
    first_batch = _make_batch(seqs[:2])
    second_batch = _make_batch(seqs[2:])
    stream = OneShotIterable([first_batch, second_batch])

    trainer = GPUBPETrainer(base_vocab=256, merges=1, device="cpu")
    result = trainer.fit(stream, log_every=10)

    assert trainer.merges[:1] == [(1, 2)]
    metrics = result["transfer_metrics"]
    assert metrics["bytes_h2d"] == 0
    assert metrics["bytes_d2h"] == 0
    assert metrics["merge_stats"]


def test_histogram_cache_matches_baseline():
    seqs = [
        [1, 2, 3, 1, 2, 4],
        [1, 2, 3, 1, 2, 3],
        [1, 2, 4, 2, 4],
    ]
    baseline_batch = _make_batch(seqs)
    delta_batch = _make_batch(seqs)

    baseline_trainer = GPUBPETrainer(base_vocab=256, merges=3, device="cpu")
    baseline_trainer._enable_histogram_cache = False
    baseline_trainer.fit([baseline_batch], log_every=10)

    delta_trainer = GPUBPETrainer(base_vocab=256, merges=3, device="cpu")
    delta_trainer.fit([delta_batch], log_every=10)

    assert baseline_trainer.merges == delta_trainer.merges
    assert baseline_trainer.vocab_size == delta_trainer.vocab_size


def test_fit_reports_telemetry_and_transfers():
    seqs = [
        [1, 2, 3, 1, 2, 3],
        [1, 2, 4, 2, 4, 2],
        [1, 2, 1, 2, 5, 6],
    ]
    batch = _make_batch(seqs)
    trainer = GPUBPETrainer(base_vocab=256, merges=2, device="cpu")

    result = trainer.fit([batch], log_every=10)

    telemetry = result["telemetry"]
    assert "timings" in telemetry
    for stage_name in ("pair_count", "apply_merge", "host_sync"):
        assert stage_name in telemetry["timings"]
        stage_payload = telemetry["timings"][stage_name]
        assert stage_payload["stage"] == stage_name
        assert "count" in stage_payload
        assert "total_s" in stage_payload
    autoscaler_payload = telemetry["autoscaler"]
    assert "device" in autoscaler_payload
    assert "state" in autoscaler_payload
    transfer_metrics = result["transfer_metrics"]
    assert "per_stage" in transfer_metrics
    assert "pair_count" in transfer_metrics["per_stage"]


def test_warm_start_plan_reduces_counts(monkeypatch):
    seqs = [
        [1, 2, 1, 2, 3],
        [1, 2, 4, 2, 4],
        [1, 2, 1, 2, 5],
        [1, 2, 6, 2, 7],
    ]
    batch = _make_batch(seqs)
    plan = GPUBPETrainer.precompute_warm_start_plan([batch], top_k=1)
    assert plan["merges"], "expected a warm-start merge to be selected"

    baseline_trainer = GPUBPETrainer(base_vocab=256, merges=2, device="cpu")
    baseline_trainer._enable_histogram_cache = False
    baseline_calls = {"count": 0}
    original_cpu = GPUBPETrainer._invoke_count_pairs_cpu

    def _count_baseline(self, batch_iter, impl):
        baseline_calls["count"] += 1
        return original_cpu(self, batch_iter, impl)

    with monkeypatch.context() as ctx:
        ctx.setattr(GPUBPETrainer, "_invoke_count_pairs_cpu", _count_baseline)
        baseline_trainer.fit([batch], log_every=10)

    seeded_trainer = GPUBPETrainer(
        base_vocab=256,
        merges=2,
        device="cpu",
        warm_start_merges=plan["merges"],
        freeze_warm_start=True,
    )
    seeded_trainer._enable_histogram_cache = False
    seeded_calls = {"count": 0}

    def _count_seeded(self, batch_iter, impl):
        seeded_calls["count"] += 1
        return original_cpu(self, batch_iter, impl)

    with monkeypatch.context() as ctx:
        ctx.setattr(GPUBPETrainer, "_invoke_count_pairs_cpu", _count_seeded)
        meta = seeded_trainer.fit([batch], log_every=10, warm_start_plan=plan)

    assert seeded_calls["count"] < baseline_calls["count"]
    assert seeded_trainer.merges[: len(plan["merges"])] == plan["merges"]
    assert meta["warm_start"]["applied"] == plan["merges"]


def test_histogram_delta_recounts_expected_spans(monkeypatch):
    seqs = [
        [1, 2, 3, 1, 2, 4],
        [1, 2, 3, 1, 2, 3],
    ]
    batch = _make_batch(seqs)
    trainer = GPUBPETrainer(base_vocab=256, merges=2, device="cpu")

    original_compute = GPUBPETrainer._compute_histogram_deltas
    calls: list[dict[str, int]] = []

    def _wrapped(self, span_mask, pre_lhs, pre_rhs, pre_mask, post_lhs, post_rhs, post_mask):
        remove_keys, remove_counts, add_keys, add_counts = original_compute(
            self, span_mask, pre_lhs, pre_rhs, pre_mask, post_lhs, post_rhs, post_mask
        )
        span_bool = span_mask.to(torch.bool)
        affected = self._expand_recount_spans(span_bool)
        expected_remove = int((affected & pre_mask.to(torch.bool)).sum().item())
        expected_add = int((affected & post_mask.to(torch.bool)).sum().item())
        actual_remove = int(remove_counts.sum().item()) if remove_counts.numel() else 0
        actual_add = int(add_counts.sum().item()) if add_counts.numel() else 0
        calls.append(
            {
                "span": int(span_bool.sum().item()),
                "affected": int(affected.sum().item()),
                "expected_remove": expected_remove,
                "actual_remove": actual_remove,
                "expected_add": expected_add,
                "actual_add": actual_add,
            }
        )
        assert actual_remove == expected_remove
        assert actual_add == expected_add
        return remove_keys, remove_counts, add_keys, add_counts

    monkeypatch.setattr(GPUBPETrainer, "_compute_histogram_deltas", _wrapped)

    trainer.fit([batch], log_every=10)

    assert calls, "delta recounts should be recorded"
    assert any(entry["span"] > 0 for entry in calls)
    for entry in calls:
        assert entry["actual_remove"] == entry["expected_remove"]
        assert entry["actual_add"] == entry["expected_add"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_fast_path_reduces_full_recounts():
    seqs = [
        [1, 2] * 20 + [3],
        [1, 2] * 18 + [4],
        [1, 2] * 16 + [5],
        [1, 2] * 15 + [6],
    ]
    fast_batch = _make_batch(seqs)
    baseline_batch = _make_batch(seqs)

    fast_trainer = GPUBPETrainer(base_vocab=256, merges=4, device="cuda")
    baseline_trainer = GPUBPETrainer(base_vocab=256, merges=4, device="cuda")
    baseline_trainer._top_pairs_limit = 0

    fast_calls = {"count": 0}
    baseline_calls = {"count": 0}

    def _count_fast(self, batch_iter, impl):
        fast_calls["count"] += 1
        return impl(batch_iter)

    def _count_baseline(self, batch_iter, impl):
        baseline_calls["count"] += 1
        return impl(batch_iter)

    fast_trainer._invoke_count_pairs_gpu = MethodType(_count_fast, fast_trainer)
    baseline_trainer._invoke_count_pairs_gpu = MethodType(_count_baseline, baseline_trainer)

    fast_trainer.fit([fast_batch], log_every=100)
    baseline_trainer.fit([baseline_batch], log_every=100)

    assert fast_trainer.merges == baseline_trainer.merges
    assert fast_calls["count"] < baseline_calls["count"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_transfer_counters_accumulate():
    seqs = [
        [1, 2, 1, 2, 3],
        [1, 2, 4, 2],
    ]
    batch = _make_batch(seqs)
    trainer = GPUBPETrainer(base_vocab=256, merges=1, device="cuda", sync_every=1)

    result = trainer.fit([batch], log_every=10)

    metrics = result["transfer_metrics"]
    assert metrics["bytes_h2d"] > 0
    assert metrics["bytes_d2h"] > 0
    assert metrics["h2d_events"] >= 1
    assert metrics["d2h_events"] >= 1
    assert metrics["sync_intervals"]


def test_autoscaler_reduces_batch_after_oom(monkeypatch):
    class ShrinkingAutoScaler:
        def __init__(self) -> None:
            self.state = ScaleState(batch_size=4, cpu_workers=2, h2d_mb=512)

        def suggest(self, token_bytes_per_example: int = 0) -> ScaleState:  # pragma: no cover - simple stub
            return self.state

        def feedback(self, step_time_s: float | None = None, oom: bool = False) -> None:
            if oom:
                self.state = ScaleState(batch_size=2, cpu_workers=2, h2d_mb=512)

    seqs = [
        [1, 2, 3, 4],
        [1, 2, 3],
        [1, 2],
        [1, 2, 3, 4, 5],
    ]
    batcher = PackedBatcher(seqs, batch_size=4, seed=42)

    trainer = GPUBPETrainer(base_vocab=256, merges=2, device="cpu", autoscaler=ShrinkingAutoScaler())

    calls = {"count": 0, "rows": []}
    original_apply = bt.apply_merge_once

    def _oom_then_apply(tokens, valid, lengths, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        calls["rows"].append(int(tokens.shape[0]))
        return original_apply(tokens, valid, lengths, *args, **kwargs)

    monkeypatch.setattr(bt, "apply_merge_once", _oom_then_apply)

    trainer.fit(batcher, log_every=10)

    assert trainer._active_batch_size == 2
    assert calls["rows"] and all(row <= 2 for row in calls["rows"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_count_pairs_retries_after_oom(monkeypatch):
    class ShrinkingAutoScaler:
        def __init__(self) -> None:
            self.state = ScaleState(batch_size=4, cpu_workers=2, h2d_mb=512)

        def suggest(self, token_bytes_per_example: int = 0) -> ScaleState:  # pragma: no cover - simple stub
            return self.state

        def feedback(self, step_time_s: float | None = None, oom: bool = False) -> None:
            if oom:
                self.state = ScaleState(batch_size=2, cpu_workers=2, h2d_mb=512)

    seqs = [
        [1, 2, 1, 2, 3, 4],
        [1, 2, 1, 2, 3, 4],
        [1, 2, 1, 2, 3, 4],
        [1, 2, 1, 2, 3, 4],
    ]
    batcher = PackedBatcher(seqs, batch_size=4, seed=123)

    trainer = GPUBPETrainer(base_vocab=256, merges=1, device="cuda", autoscaler=ShrinkingAutoScaler())

    original_count = bt.count_pairs
    call_rows: list[int] = []
    call_count = {"count": 0}

    def _oom_then_count(tokens, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        call_rows.append(int(tokens.shape[0]))
        return original_count(tokens, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length)

    monkeypatch.setattr(bt, "count_pairs", _oom_then_count)

    trainer.fit(batcher, log_every=10)

    assert trainer._active_batch_size == 2
    assert call_count["count"] >= 2
    assert call_rows and all(row <= 2 for row in call_rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_apply_merge_retries_after_oom(monkeypatch):
    class ShrinkingAutoScaler:
        def __init__(self) -> None:
            self.state = ScaleState(batch_size=4, cpu_workers=2, h2d_mb=512)

        def suggest(self, token_bytes_per_example: int = 0) -> ScaleState:  # pragma: no cover - simple stub
            return self.state

        def feedback(self, step_time_s: float | None = None, oom: bool = False) -> None:
            if oom:
                self.state = ScaleState(batch_size=2, cpu_workers=2, h2d_mb=512)

    seqs = [
        [1, 2, 1, 2, 3, 4],
        [1, 2, 1, 2, 3, 4],
        [1, 2, 1, 2, 3, 4],
        [1, 2, 1, 2, 3, 4],
    ]
    batcher = PackedBatcher(seqs, batch_size=4, seed=321)

    trainer = GPUBPETrainer(base_vocab=256, merges=2, device="cuda", autoscaler=ShrinkingAutoScaler())

    original_apply = bt.apply_merge_once
    rows_seen: list[int] = []
    apply_calls = {"count": 0}

    def _oom_then_apply(tokens, valid, lengths, *args, **kwargs):
        apply_calls["count"] += 1
        if apply_calls["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        rows_seen.append(int(tokens.shape[0]))
        return original_apply(tokens, valid, lengths, *args, **kwargs)

    monkeypatch.setattr(bt, "apply_merge_once", _oom_then_apply)

    trainer.fit(batcher, log_every=10)

    assert trainer._active_batch_size == 2
    assert apply_calls["count"] >= 2
    assert rows_seen and all(row <= 2 for row in rows_seen)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >=2 CUDA devices")
def test_multi_gpu_merges_match_single_gpu():
    seqs = [
        [1, 2, 1, 2, 3],
        [1, 2, 4, 2, 4],
        [1, 3, 1, 3, 4],
        [1, 2, 1, 2, 4],
    ]
    single_batch = _make_batch(seqs)
    multi_batch = _make_batch(seqs)

    single_trainer = GPUBPETrainer(base_vocab=256, merges=3, device="cuda:0", sync_every=1)
    single_trainer.fit([single_batch], log_every=10)

    multi_trainer = GPUBPETrainer(
        base_vocab=256,
        merges=3,
        devices=("cuda:0", "cuda:1"),
        sync_every=1,
    )
    result_multi = multi_trainer.fit([multi_batch], log_every=10)

    assert multi_trainer.merges == single_trainer.merges
    assert multi_trainer.vocab_size == single_trainer.vocab_size
    per_device = result_multi["transfer_metrics"]["per_device"]
    assert set(per_device.keys()) == {"cuda:0", "cuda:1"}
    total_h2d = sum(metrics["bytes_h2d"] for metrics in per_device.values())
    assert total_h2d == result_multi["transfer_metrics"]["bytes_h2d"]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >=2 CUDA devices")
def test_multi_gpu_autoscaler_updates_contexts(monkeypatch):
    class ShrinkingAutoScaler:
        def __init__(self) -> None:
            self.state = ScaleState(batch_size=4, cpu_workers=2, h2d_mb=512)

        def suggest(self, token_bytes_per_example: int = 0) -> ScaleState:  # pragma: no cover - simple stub
            return self.state

        def feedback(self, step_time_s: float | None = None, oom: bool = False) -> None:
            if oom:
                self.state = ScaleState(batch_size=2, cpu_workers=2, h2d_mb=512)

    seqs = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 6],
        [1, 2, 3, 4, 7],
        [1, 2, 3, 4, 8],
    ]
    batcher = PackedBatcher(seqs, batch_size=4, seed=999)

    trainer = GPUBPETrainer(
        base_vocab=256,
        merges=2,
        devices=("cuda:0", "cuda:1"),
        autoscaler=ShrinkingAutoScaler(),
        sync_every=1,
    )

    original_count = bt.count_pairs
    call_order: list[int] = []

    def _oom_then_count(tokens, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length):
        call_order.append(int(tokens.shape[0]))
        if len(call_order) == 1:
            raise RuntimeError("CUDA out of memory")
        return original_count(tokens, valid, pair_keys_buffer, pair_counts_buffer, pair_count_length)

    monkeypatch.setattr(bt, "count_pairs", _oom_then_count)

    result = trainer.fit(batcher, log_every=10)

    assert trainer._active_batch_size == 2
    assert len(call_order) >= 2
    assert all(rows <= 2 for rows in call_order[1:])
    per_device = result["transfer_metrics"]["per_device"]
    assert set(per_device.keys()) == {"cuda:0", "cuda:1"}
    assert all(metrics["bytes_h2d"] > 0 for metrics in per_device.values())


def test_checkpoint_resume_matches_uninterrupted(tmp_path, monkeypatch):
    clock = {"perf": 0.0, "wall": 0.0}

    def reset_clock() -> None:
        clock["perf"] = 0.0
        clock["wall"] = 0.0

    def fake_perf_counter() -> float:
        clock["perf"] += 0.001
        return clock["perf"]

    def fake_time() -> float:
        clock["wall"] += 1.0
        return clock["wall"]

    monkeypatch.setattr(time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(time, "time", fake_time)

    seq_batches = [
        [[1, 2, 1, 2], [1, 2, 3, 4]],
        [[2, 3, 2, 3], [3, 4, 3, 4]],
    ]
    base_batches = [_make_batch(seqs) for seqs in seq_batches]

    def fresh_batches():
        cloned = []
        for tokens, valid, lengths in base_batches:
            cloned.append((tokens.clone(), valid.clone(), lengths.clone()))
        return cloned

    reset_clock()
    trainer_full = GPUBPETrainer(base_vocab=256, merges=4, device="cpu")
    result_full = trainer_full.fit(fresh_batches(), log_every=10)

    reset_clock()
    trainer_partial = GPUBPETrainer(base_vocab=256, merges=4, device="cpu")
    ckpt_dir = tmp_path / "ckpt"
    save_calls = {"count": 0}
    original_save = trainer_partial.save_checkpoint

    def intercept(path, include_batches=True, **kwargs):
        save_calls["count"] += 1
        state = original_save(path, include_batches=include_batches, **kwargs)
        if save_calls["count"] == 1:
            raise RuntimeError("checkpoint interrupt")
        return state

    monkeypatch.setattr(trainer_partial, "save_checkpoint", intercept)
    with pytest.raises(RuntimeError, match="checkpoint interrupt"):
        trainer_partial.fit(
            fresh_batches(),
            log_every=10,
            checkpoint_interval=2,
            checkpoint_dir=str(ckpt_dir),
        )

    trainer_resume = GPUBPETrainer(base_vocab=256, merges=4, device="cpu")
    resume_state = trainer_resume.load_checkpoint(str(ckpt_dir))
    resumed_result = trainer_resume.fit(
        [],
        log_every=10,
        resume_state=resume_state,
    )

    assert trainer_resume.merges == trainer_full.merges
    assert resumed_result["transfer_metrics"] == result_full["transfer_metrics"]
    assert resumed_result["telemetry"] == result_full["telemetry"]
