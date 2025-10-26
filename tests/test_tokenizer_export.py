import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gpu_tokenizer.bpe_trainer import GPUBPETrainer
from gpu_tokenizer.export import artifacts as export_artifacts
from gpu_tokenizer.trainers.hybrid import HybridTrainer, _build_piece_tables

try:
    from tokenizers import Tokenizer
except Exception:  # pragma: no cover - optional dependency in CI
    Tokenizer = None


@pytest.mark.skipif(Tokenizer is None, reason="tokenizers library is unavailable")
def test_save_emits_huggingface_artifacts(tmp_path: Path) -> None:
    trainer = GPUBPETrainer(base_vocab=256, merges=1, device="cpu")
    trainer.merges = [(ord("h"), ord("i"))]
    trainer.vocab_size = trainer.base_vocab + len(trainer.merges)

    trainer.save(str(tmp_path))

    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    tiktoken_path = tmp_path / "merges.tiktoken"

    assert vocab_path.exists()
    assert merges_path.exists()
    assert tokenizer_path.exists()
    assert tiktoken_path.exists()

    with vocab_path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    assert vocab["h"] == ord("h")
    assert vocab["i"] == ord("i")
    assert vocab["hi"] == trainer.base_vocab

    merges_lines = merges_path.read_text(encoding="utf-8").splitlines()
    assert merges_lines[0] == "#version: 0.2"
    assert merges_lines[1] == "h i"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    sample = "hi hi"
    encoding = tokenizer.encode(sample)
    assert tokenizer.decode(encoding.ids) == sample
    assert tokenizer.get_vocab_size(with_added_tokens=True) == trainer.vocab_size

    ranks = export_artifacts.load_tiktoken_bpe(tiktoken_path)
    assert ranks
    assert export_artifacts.mergeable_ranks_to_merges(ranks) == trainer.merges


@pytest.mark.skipif(Tokenizer is None, reason="tokenizers library is unavailable")
def test_tiktoken_roundtrip_matches_reference(tmp_path: Path) -> None:
    tiktoken = pytest.importorskip("tiktoken")

    trainer = GPUBPETrainer(base_vocab=256, merges=2, device="cpu")
    trainer.merges = [
        (ord("h"), ord("i")),
        (trainer.base_vocab, ord("!")),
    ]
    trainer.vocab_size = trainer.base_vocab + len(trainer.merges)

    artifacts = trainer.save_artifacts(tmp_path)

    tiktoken_path = Path(artifacts["tiktoken_merges"])
    assert tiktoken_path.exists()

    ours = export_artifacts.load_tiktoken_bpe(tiktoken_path)
    theirs = tiktoken.load.load_tiktoken_bpe(str(tiktoken_path))
    assert ours == theirs

    merges = export_artifacts.load_tiktoken_merges(tiktoken_path)
    assert merges == trainer.merges


@pytest.mark.skipif(Tokenizer is None, reason="tokenizers library is unavailable")
def test_hybrid_save_emits_combined_artifacts(tmp_path: Path) -> None:
    sentencepiece = pytest.importorskip("sentencepiece")

    trainer = HybridTrainer(base_vocab=256, merges=1, cycles=1, unigram_epochs=1)
    trainer._final_merges = [(ord("h"), ord("i"))]
    id2piece, _, _ = _build_piece_tables(trainer.base_vocab, trainer._final_merges)
    trainer._final_id2piece = id2piece
    trainer._final_logp = torch.full(
        (len(id2piece),),
        -math.log(max(len(id2piece), 1)),
        dtype=torch.float32,
    )
    trainer._completed_cycles = 1

    artifacts = trainer.save(tmp_path)

    merges_path = tmp_path / "merges.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    unigram_prob_path = tmp_path / "unigram.prob"
    unigram_model_path = tmp_path / "unigram.model"
    manifest_path = tmp_path / "hybrid_manifest.json"
    tiktoken_path = tmp_path / "merges.tiktoken"

    assert merges_path.exists()
    assert tokenizer_path.exists()
    assert unigram_prob_path.exists()
    assert manifest_path.exists()
    assert unigram_model_path.exists()
    assert tiktoken_path.exists()

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoding = tokenizer.encode("hi")
    assert tokenizer.decode(encoding.ids) == "hi"

    processor = sentencepiece.SentencePieceProcessor(model_file=str(unigram_model_path))
    loaded = processor.LoadVocabulary(str(unigram_prob_path), threshold=0)
    assert loaded > 0
    sp_ids = processor.encode("hi", out_type=int)
    assert sp_ids

    assert artifacts["merges"] == str(merges_path)
    assert artifacts["unigram_prob"] == str(unigram_prob_path)
    assert artifacts["tiktoken_merges"] == str(tiktoken_path)
