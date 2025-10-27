from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sentencepiece")
pytest.importorskip("tokenizers")

from benchmarks import BaselineCorpus, run_reference_tokenizers


def _train_sentencepiece_model(tmp_path: Path, sentences: list[str]) -> Path:
    import sentencepiece as spm

    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("\n".join(sentences), encoding="utf-8")
    model_prefix = tmp_path / "toy"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        model_type="unigram",
        vocab_size=32,
        character_coverage=1.0,
    )
    return model_prefix.with_suffix(".model")


def test_run_reference_tokenizers_collects_metrics(tmp_path: Path) -> None:
    sentences = ["Hello world!", "Tokenizer benchmarks are informative."]
    model_path = _train_sentencepiece_model(tmp_path, sentences)
    hf_path = Path("tests/data/models/bpe/tokenizer.json")
    corpus = BaselineCorpus(
        name="demo",
        description="Demo corpus",
        documents=tuple(sentences),
    )
    results = run_reference_tokenizers(
        [corpus],
        sentencepiece_model=model_path,
        huggingface_tokenizer=hf_path,
    )
    assert len(results) == 1
    record = results[0]
    assert record["name"] == "demo"
    assert record["total_bytes"] > 0
    tokenizers = record["tokenizers"]
    assert "sentencepiece" in tokenizers
    assert "huggingface" in tokenizers
    sp_stats = tokenizers["sentencepiece"]
    assert sp_stats["tokens"] > 0
    assert sp_stats["tokens_per_s"] is not None
    assert sp_stats["bytes_per_token"] is not None
    assert sp_stats["loss_per_token"] is not None
    hf_stats = tokenizers["huggingface"]
    assert hf_stats["tokens"] > 0
    assert hf_stats["bytes_per_token"] is not None
    assert hf_stats["loss_per_token"] is None
