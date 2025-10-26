from __future__ import annotations

from pathlib import Path

import pytest

from gpu_tokenizer.export import artifacts as export_artifacts


def test_load_hf_bpe_from_tokenizer_matches_reference(tmp_path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")

    tokenizer_path = Path("tests/data/models/bpe/tokenizer.json")
    result = export_artifacts.load_hf_bpe_artifacts(tokenizer_path=tokenizer_path)

    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab()
    assert result.vocab == vocab

    merges_tokens = tokenizer.model.get_merges()
    merges_ids = [(vocab[left], vocab[right]) for left, right in merges_tokens]
    assert result.merges == merges_ids


def test_load_hf_bpe_from_vocab_merges(tmp_path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")

    tokenizer_path = Path("tests/data/models/bpe/tokenizer.json")
    vocab_path = Path("tests/data/models/bpe/vocab.json")
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    merges_tokens = tokenizer.model.get_merges()

    merges_path = tmp_path / "merges.txt"
    with merges_path.open("w", encoding="utf-8") as handle:
        handle.write("#version: 0.2\n")
        for left, right in merges_tokens:
            handle.write(f"{left} {right}\n")

    result = export_artifacts.load_hf_bpe_artifacts(
        vocab_path=vocab_path, merges_path=merges_path
    )

    vocab = tokenizer.get_vocab()
    merges_ids = [(vocab[left], vocab[right]) for left, right in merges_tokens]
    assert result.vocab == vocab
    assert result.merges == merges_ids


def test_load_sentencepiece_unigram_matches_processor(tmp_path: Path) -> None:
    sentencepiece = pytest.importorskip("sentencepiece")

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nhello tokenize\n", encoding="utf-8")

    model_prefix = tmp_path / "toy"
    sentencepiece.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(model_prefix),
        model_type="unigram",
        vocab_size=64,
        character_coverage=1.0,
        byte_fallback=True,
        input_sentence_size=0,
        shuffle_input_sentence=False,
    )

    model_path = model_prefix.with_suffix(".model")
    vocab_path = model_prefix.with_suffix(".vocab")

    result = export_artifacts.load_sentencepiece_unigram(model_path=model_path)
    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))

    expected_size = processor.get_piece_size()
    assert set(result.pieces.keys()) == {idx for idx in range(expected_size)}
    for idx in range(expected_size):
        piece_text = processor.id_to_piece(idx)
        expected_bytes = piece_text.replace("▁", " ").encode("utf-8")
        assert result.pieces[idx] == expected_bytes
        assert pytest.approx(result.scores[idx]) == processor.get_score(idx)

    vocab_only = export_artifacts.load_sentencepiece_unigram(vocab_path=vocab_path)
    assert vocab_only.pieces == result.pieces
    assert vocab_only.scores == pytest.approx(result.scores)
