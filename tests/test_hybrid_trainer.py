import json
import math
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import pytest

torch = pytest.importorskip("torch")

from gpu_tokenizer import GPUBPETrainer, HybridTrainer
from gpu_tokenizer.dtypes import length_storage_dtype
from gpu_tokenizer.trainers.hybrid import _build_piece_tables
from gpu_tokenizer.utils import apply_merge_once, hash_merge_pair


def _encode_corpus_to_batches(
    corpus: Sequence[str],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    byte_sequences = [list(sample.encode("utf-8")) for sample in corpus]
    if not byte_sequences:
        return []
    width = max(len(seq) for seq in byte_sequences)
    tokens = torch.full((len(byte_sequences), width), -1, dtype=torch.int32)
    valid = torch.zeros((len(byte_sequences), width), dtype=torch.uint8)
    length_dtype = length_storage_dtype(width)
    lengths = torch.zeros((len(byte_sequences),), dtype=length_dtype)
    for row, seq in enumerate(byte_sequences):
        if not seq:
            continue
        seq_tensor = torch.tensor(seq, dtype=torch.int32)
        tokens[row, : seq_tensor.numel()] = seq_tensor
        valid[row, : seq_tensor.numel()] = 1
        lengths[row] = seq_tensor.numel()
    return [(tokens, valid, lengths)]


def _clone_batches(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    cloned = []
    for tokens, valid, lengths in batches:
        cloned.append((tokens.clone(), valid.clone(), lengths.clone()))
    return cloned


def _apply_merges(
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    merges: Sequence[Tuple[int, int]],
    base_vocab: int,
) -> tuple[int, dict[int, int]]:
    total_tokens = 0
    freq: dict[int, int] = {}
    for tokens, valid, lengths in batches:
        work_tokens = tokens.clone()
        work_valid = valid.clone()
        work_lengths = lengths.clone()
        for offset, (left_id, right_id) in enumerate(merges):
            new_id = base_vocab + offset
            work_tokens, work_valid, work_lengths, _ = apply_merge_once(
                work_tokens, work_valid, work_lengths, left_id, right_id, new_id
            )
        for row in range(work_tokens.shape[0]):
            row_len = int(work_lengths[row].item())
            if row_len <= 0:
                continue
            row_tokens = work_tokens[row, :row_len].tolist()
            total_tokens += row_len
            for token_id in row_tokens:
                freq[token_id] = freq.get(token_id, 0) + 1
    return total_tokens, freq


@pytest.mark.skipif(
    GPUBPETrainer is None or HybridTrainer is None,
    reason="GPU trainer components require PyTorch",
)
def test_hybrid_trainer_improves_compression(tmp_path: Path) -> None:
    corpus = [
        "banana bandana banana",
        "bandana banana bandana",
        "anana banana bandana",
    ]

    batches = _encode_corpus_to_batches(corpus)
    baseline_batches = _clone_batches(batches)

    bpe = GPUBPETrainer(base_vocab=256, merges=8, device="cpu")
    baseline_summary = bpe.fit(baseline_batches, log_every=100)
    baseline_merges = baseline_summary["merges"]
    total_tokens, token_freq = _apply_merges(batches, baseline_merges, 256)
    baseline_vocab = baseline_summary.get("vocab_size", 256)
    baseline_bits = total_tokens * math.log2(max(baseline_vocab, 1))

    hybrid = HybridTrainer(
        base_vocab=256,
        merges=8,
        cycles=1,
        unigram_epochs=4,
        max_unigram_len=16,
        bpe_init_kwargs={"device": "cpu"},
    )
    hybrid_summary = hybrid.fit(batches, bpe_fit_kwargs={"log_every": 100})
    hybrid_merges = [tuple(pair) for pair in hybrid_summary["merges"]]
    assert hybrid_merges == [tuple(pair) for pair in baseline_merges]

    logp = torch.tensor(hybrid_summary["unigram_logp"], dtype=torch.float64)
    assert logp.numel() >= baseline_vocab
    logp_probs = logp.exp()
    hybrid_bits = 0.0
    for token_id, count in token_freq.items():
        prob = float(logp_probs[token_id].item())
        if prob <= 0:
            continue
        hybrid_bits += -count * math.log2(prob)

    assert hybrid_bits < baseline_bits

    artifact_paths = hybrid.save_artifacts(tmp_path)
    manifest_path = Path(artifact_paths["manifest"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["merges"] == [list(map(int, pair)) for pair in hybrid_merges]
    assert len(manifest["unigram_logp"]) == len(logp)


def test_hybrid_privacy_manifest_redacts_merges(tmp_path: Path) -> None:
    trainer = HybridTrainer(
        base_vocab=64,
        merges=1,
        cycles=1,
        unigram_epochs=1,
        privacy_mode=True,
        randomize_ties=False,
        tie_seed=11,
        privacy_salt="pepper",
    )
    trainer._final_merges = [(1, 2)]
    id2piece, _, _ = _build_piece_tables(trainer.base_vocab, trainer._final_merges)
    trainer._final_id2piece = id2piece
    trainer._final_logp = torch.zeros(len(id2piece), dtype=torch.float32)
    trainer._completed_cycles = 1

    artifacts = trainer.save_artifacts(tmp_path)
    manifest_path = Path(artifacts["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = hash_merge_pair((1, 2), "pepper")

    assert manifest["privacy_mode"] is True
    assert manifest["merge_count"] == len(trainer._final_merges)
    assert manifest["merges"][0] == expected_hash
    assert all(isinstance(entry, str) for entry in manifest["merges"])

