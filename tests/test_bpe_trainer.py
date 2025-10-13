import pytest

torch = pytest.importorskip("torch")

from gpu_tokenizer import bpe_trainer as bt
from gpu_tokenizer.autoscaler import ScaleState
from gpu_tokenizer.bpe_trainer import GPUBPETrainer, _aggregate_pair_keys
from gpu_tokenizer.datasets import PackedBatcher


def test_aggregate_pair_keys_repeated_counts():
    keys = torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.long)
    counts = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)

    aggregated_keys, aggregated_counts = _aggregate_pair_keys(keys, counts)

    assert torch.equal(aggregated_keys, torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(
        aggregated_counts, torch.tensor([1 + 2 + 3, 4 + 5, 6], dtype=torch.long)
    )


def test_aggregate_pair_keys_unsorted_input():
    keys = torch.tensor([2, 1, 3, 1, 2], dtype=torch.long)
    counts = torch.tensor([5, 1, 2, 3, 4], dtype=torch.long)

    aggregated_keys, aggregated_counts = _aggregate_pair_keys(keys, counts)

    assert torch.equal(aggregated_keys, torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(aggregated_counts, torch.tensor([1 + 3, 5 + 4, 2], dtype=torch.long))


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
    tokens = torch.full((len(seqs), max_len), -1, dtype=torch.long)
    valid = torch.zeros((len(seqs), max_len), dtype=torch.long)
    lengths = torch.zeros(len(seqs), dtype=torch.long)
    for row, seq in enumerate(seqs):
        L = len(seq)
        if L == 0:
            continue
        tokens[row, :L] = torch.tensor(seq, dtype=torch.long)
        valid[row, :L] = 1
        lengths[row] = L
    if torch.cuda.is_available():
        tokens = tokens.pin_memory()
        valid = valid.pin_memory()
    return tokens, valid, lengths


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
