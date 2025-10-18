import json
import time
from pathlib import Path
from types import MethodType
from typing import Sequence

import pytest

torch = pytest.importorskip("torch")
if getattr(torch, "_SUPERTOKEN_TORCH_STUB", False) or not hasattr(torch, "tensor"):
    pytest.skip(
        "PyTorch with tensor factories is required for BPE trainer tests",
        allow_module_level=True,
    )
if not hasattr(torch, "cuda") or not hasattr(torch.cuda, "device_count"):
    pytest.skip(
        "PyTorch CUDA support is unavailable in the current test environment",
        allow_module_level=True,
    )
pytest.importorskip("torch.utils")

from gpu_tokenizer import bpe_trainer as bt
from gpu_tokenizer.autoscaler import AutoScaler, ScaleState
from gpu_tokenizer.bpe_trainer import (
    DeviceContext,
    GPUBPETrainer,
    GPUBatchRecord,
    _aggregate_pair_keys,
)
from gpu_tokenizer.dtypes import length_storage_dtype
from gpu_tokenizer.datasets import PackedBatcher
from gpu_tokenizer.cpu_fastpath import (
    FastPathWorkspaces,
    apply_merge_fastpath,
    count_pairs_fastpath,
    should_route_to_cpu,
)
from gpu_tokenizer.utils import apply_merge_once, count_pairs, hash_merge_pair
from tests.adversarial_corpora import get_adversarial_corpora


def test_aggregate_pair_keys_repeated_counts():
    keys = torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.long)
    counts = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)

    aggregated_keys, aggregated_counts = _aggregate_pair_keys(keys, counts)

    assert torch.equal(aggregated_keys, torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(
        aggregated_counts, torch.tensor([1 + 2 + 3, 4 + 5, 6], dtype=torch.int64)
    )


def test_aggregate_pair_keys_unsorted_input():
    keys = torch.tensor([2, 1, 3, 1, 2], dtype=torch.long)
    counts = torch.tensor([5, 1, 2, 3, 4], dtype=torch.int64)

    aggregated_keys, aggregated_counts = _aggregate_pair_keys(keys, counts)

    assert torch.equal(aggregated_keys, torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(
        aggregated_counts, torch.tensor([1 + 3, 5 + 4, 2], dtype=torch.int64)
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


def _encode_corpus_to_batches(corpus: Sequence[str], batch_rows: int = 2):
    byte_sequences = [list(sample.encode("utf-8")) for sample in corpus]
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(byte_sequences), batch_rows):
        chunk = byte_sequences[start : start + batch_rows]
        if not chunk:
            continue
        width = max(1, max((len(seq) for seq in chunk), default=0))
        tokens = torch.full((len(chunk), width), -1, dtype=torch.int32)
        valid = torch.zeros((len(chunk), width), dtype=torch.uint8)
        length_dtype = length_storage_dtype(width)
        lengths = torch.zeros((len(chunk),), dtype=length_dtype)
        for row, seq in enumerate(chunk):
            if not seq:
                continue
            seq_tensor = torch.tensor(seq, dtype=torch.int32)
            tokens[row, : seq_tensor.numel()] = seq_tensor
            valid[row, : seq_tensor.numel()] = 1
            lengths[row] = seq_tensor.numel()
        batches.append((tokens, valid, lengths))
    return batches


def _clone_batches(
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    cloned: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for tokens, valid, lengths in batches:
        cloned.append((tokens.clone(), valid.clone(), lengths.clone()))
    return cloned


def test_handle_chunk_start_resets_cache_state():
    trainer = GPUBPETrainer(base_vocab=4, merges=2, device="cpu")
    trainer._enable_histogram_cache = True
    trainer._hist_cache_valid = True
    trainer._force_recount = False
    trainer._cached_pair_keys = torch.tensor([1, 2], dtype=torch.long)
    trainer._cached_pair_counts = torch.tensor([3, 4], dtype=torch.int64)
    trainer._cached_pair_keys_per_device[torch.device("cpu")] = torch.tensor([5])
    trainer._cached_pair_counts_per_device[torch.device("cpu")] = torch.tensor([6])

    trainer.handle_chunk_start(7, reprocessed=False)
    assert trainer._hist_cache_valid
    assert not trainer._force_recount

    trainer.handle_chunk_start(7, reprocessed=True, attempts=2)
    assert trainer._last_chunk_id == 7
    assert not trainer._hist_cache_valid
    assert trainer._force_recount
    assert trainer._cached_pair_keys.numel() == 0
    assert trainer._cached_pair_counts.numel() == 0
    assert trainer._cached_pair_keys_per_device == {}
    assert trainer._cached_pair_counts_per_device == {}


def test_reprocessing_chunk_matches_baseline_vocab_and_merges():
    seqs = [
        [1, 2, 1, 2, 3],
        [1, 2, 4, 2, 4],
        [2, 3, 2, 3, 2],
    ]

    baseline_batch = _make_batch(seqs)
    retry_batch = _make_batch(seqs)

    baseline_trainer = GPUBPETrainer(base_vocab=256, merges=4, device="cpu")
    baseline_result = baseline_trainer.fit([baseline_batch], log_every=100)

    assert baseline_result["merges"], "expected at least one merge to be learned"

    retry_trainer = GPUBPETrainer(base_vocab=256, merges=4, device="cpu")
    retry_trainer._enable_histogram_cache = True
    retry_trainer._hist_cache_valid = True
    retry_trainer._force_recount = False
    retry_trainer._cached_pair_keys = torch.tensor([123, 456], dtype=torch.long)
    retry_trainer._cached_pair_counts = torch.tensor([7, 8], dtype=torch.int64)
    retry_trainer._cached_pair_keys_per_device[torch.device("cpu")] = torch.tensor(
        [789], dtype=torch.long
    )
    retry_trainer._cached_pair_counts_per_device[torch.device("cpu")] = torch.tensor(
        [10], dtype=torch.int64
    )

    retry_trainer.handle_chunk_start(5, reprocessed=True, attempts=2)

    retry_result = retry_trainer.fit([retry_batch], log_every=100)

    assert retry_result["vocab_size"] == baseline_result["vocab_size"]
    assert retry_result["merges"] == baseline_result["merges"]
    assert len(retry_trainer.merges) == len(set(retry_trainer.merges))


def test_tie_breaker_respects_randomization(tmp_path: Path):
    seqs = [
        [0, 1, 0, 1],
        [0, 2, 0, 2],
    ]
    batch = _make_batch(seqs)

    deterministic = GPUBPETrainer(
        base_vocab=16, merges=1, device="cpu", randomize_ties=False
    )
    deterministic.fit([batch], log_every=100)
    assert deterministic.merges == [(0, 1)]

    seeded_a = GPUBPETrainer(
        base_vocab=16, merges=1, device="cpu", randomize_ties=True, tie_seed=123
    )
    seeded_b = GPUBPETrainer(
        base_vocab=16, merges=1, device="cpu", randomize_ties=True, tie_seed=123
    )
    seeded_c = GPUBPETrainer(
        base_vocab=16, merges=1, device="cpu", randomize_ties=True, tie_seed=456
    )

    seeded_a.fit([batch], log_every=100)
    seeded_b.fit([batch], log_every=100)
    seeded_c.fit([batch], log_every=100)

    assert seeded_a.merges == seeded_b.merges
    assert seeded_a.merges in ([(0, 1)], [(0, 2)])
    assert seeded_c.merges in ([(0, 1)], [(0, 2)])
    assert seeded_c.merges != seeded_a.merges


def test_privacy_mode_redacts_merge_metadata(tmp_path: Path):
    trainer = GPUBPETrainer(
        base_vocab=32,
        merges=1,
        device="cpu",
        privacy_mode=True,
        randomize_ties=False,
        tie_seed=7,
        privacy_salt=b"secret",
    )
    trainer.merges = [(1, 2)]
    trainer.vocab_size = trainer.base_vocab + len(trainer.merges)
    paths = trainer.save_artifacts(tmp_path)
    meta_path = Path(paths["metadata"])
    payload = json.loads(meta_path.read_text("utf-8"))
    assert payload["privacy_mode"] is True
    assert payload["merge_count"] == 1
    assert payload["merges"][0] == hash_merge_pair((1, 2), b"secret")
    assert all(isinstance(entry, str) for entry in payload["merges"])


def test_cpu_fastpath_pair_count_matches_baseline():
    seqs = [[1, 2, 3, 4], [4, 3, 2, 1]]
    tokens, valid, lengths = _make_batch(seqs)
    tokens = tokens.clone()
    valid = valid.clone()
    B, L = tokens.shape
    width = max(L - 1, 0)
    capacity = max(B * width, 1)
    pair_workspace = torch.empty((capacity, 2), dtype=tokens.dtype)
    count_workspace = torch.empty((capacity,), dtype=torch.int64)
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
        torch.sort(baseline_counts)[0],
        torch.sort(fast_counts)[0],
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


@pytest.mark.parametrize("corpus", get_adversarial_corpora(), ids=lambda c: c.name)
def test_gpu_cpu_parity_tiny_batches_adversarial(corpus):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("CUDA device required for GPU/CPU parity checks")

    base_batches = _encode_corpus_to_batches(corpus.corpus, batch_rows=2)
    if not base_batches:
        pytest.skip("Corpus did not yield any batches")

    merges_budget = min(12, max(4, corpus.target_merge_operations // 4))

    gpu_trainer = GPUBPETrainer(base_vocab=256, merges=merges_budget, device="cuda")
    gpu_result = gpu_trainer.fit(_clone_batches(base_batches), log_every=0)

    cpu_trainer = GPUBPETrainer(base_vocab=256, merges=merges_budget, device="cpu")
    cpu_result = cpu_trainer.fit(_clone_batches(base_batches), log_every=0)

    assert gpu_trainer._cpu_fallback_batches > 0
    assert gpu_result["merges"] == cpu_result["merges"]
    assert gpu_result["vocab_size"] == cpu_result["vocab_size"]


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for stream overlap test")
def test_gpu_copy_and_compute_overlap_across_streams():
    device = torch.device("cuda")
    ctx = DeviceContext(
        device=device,
        compute_stream=torch.cuda.Stream(device=device),
        h2d_stream=torch.cuda.Stream(device=device),
        d2h_stream=torch.cuda.Stream(device=device),
    )

    torch.cuda.synchronize(device)

    rows_copy, width_copy = 256, 8192
    tokens_copy = torch.arange(rows_copy * width_copy, dtype=torch.int32).view(rows_copy, width_copy)
    valid_copy = torch.ones((rows_copy, width_copy), dtype=torch.uint8)
    lengths_copy = torch.full((rows_copy,), width_copy, dtype=torch.int64)
    record_copy = GPUBatchRecord.from_cpu(tokens_copy, valid_copy, lengths_copy, device, ctx=ctx)

    compute1_start = torch.cuda.Event(enable_timing=True)
    compute1_end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(device), torch.cuda.stream(ctx.compute_stream):
        record_copy.wait_for_device(ctx.compute_stream)
        compute1_start.record()
        torch.cuda._sleep(int(5e6))
        compute1_end.record()
    record_copy.mark_device_event(compute1_end)

    copy_start = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(device), torch.cuda.stream(ctx.d2h_stream):
        copy_start.record()
    with torch.cuda.device(device):
        record_copy.schedule_host_sync(ctx.d2h_stream)
    copy_end = record_copy.host_event
    assert copy_end is not None

    rows_compute, width_compute = 16, 256
    tokens_compute = torch.arange(rows_compute * width_compute, dtype=torch.int32).view(
        rows_compute, width_compute
    )
    valid_compute = torch.ones((rows_compute, width_compute), dtype=torch.uint8)
    lengths_compute = torch.full((rows_compute,), width_compute, dtype=torch.int64)
    record_compute = GPUBatchRecord.from_cpu(
        tokens_compute, valid_compute, lengths_compute, device, ctx=ctx
    )

    compute2_start = torch.cuda.Event(enable_timing=True)
    compute2_end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(device), torch.cuda.stream(ctx.compute_stream):
        record_compute.wait_for_device(ctx.compute_stream)
        compute2_start.record()
        torch.cuda._sleep(int(5e6))
        compute2_end.record()
    record_compute.mark_device_event(compute2_end)

    torch.cuda.synchronize(device)

    copy_duration = copy_start.elapsed_time(copy_end)
    start_offset = copy_start.elapsed_time(compute2_start)
    end_offset = copy_start.elapsed_time(compute2_end)

    assert copy_duration > 0.0
    assert 0.0 <= start_offset < copy_duration
    assert end_offset > 0.0


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

        def suggest(self, token_bytes_per_example: int = 0) -> tuple[ScaleState, dict[str, object]]:  # pragma: no cover - simple stub
            return self.state, {}

        def feedback(
            self, step_time_s: float | None = None, oom: bool = False
        ) -> tuple[ScaleState, dict[str, object]]:
            if oom:
                self.state = ScaleState(batch_size=2, cpu_workers=2, h2d_mb=512)
            return self.state, {}

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

        def suggest(self, token_bytes_per_example: int = 0) -> tuple[ScaleState, dict[str, object]]:  # pragma: no cover - simple stub
            return self.state, {}

        def feedback(
            self, step_time_s: float | None = None, oom: bool = False
        ) -> tuple[ScaleState, dict[str, object]]:
            if oom:
                self.state = ScaleState(batch_size=2, cpu_workers=2, h2d_mb=512)
            return self.state, {}

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

        def suggest(self, token_bytes_per_example: int = 0) -> tuple[ScaleState, dict[str, object]]:  # pragma: no cover - simple stub
            return self.state, {}

        def feedback(
            self, step_time_s: float | None = None, oom: bool = False
        ) -> tuple[ScaleState, dict[str, object]]:
            if oom:
                self.state = ScaleState(batch_size=2, cpu_workers=2, h2d_mb=512)
            return self.state, {}

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

        def suggest(self, token_bytes_per_example: int = 0) -> tuple[ScaleState, dict[str, object]]:  # pragma: no cover - simple stub
            return self.state, {}

        def feedback(
            self, step_time_s: float | None = None, oom: bool = False
        ) -> tuple[ScaleState, dict[str, object]]:
            if oom:
                self.state = ScaleState(batch_size=2, cpu_workers=2, h2d_mb=512)
            return self.state, {}

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
