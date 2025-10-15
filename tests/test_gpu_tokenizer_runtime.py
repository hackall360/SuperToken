import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer

from gpu_tokenizer.bpe_trainer import GPUBPETrainer, GPUBPETokenizer


@pytest.fixture()
def sample_trainer() -> GPUBPETrainer:
    trainer = GPUBPETrainer(base_vocab=256, merges=2, device="cpu")
    trainer.merges = [
        (ord("h"), ord("i")),
        (trainer.base_vocab, ord("!")),
    ]
    trainer.vocab_size = trainer.base_vocab + len(trainer.merges)
    return trainer


def test_export_tokenizer_matches_hf(sample_trainer: GPUBPETrainer) -> None:
    runtime = sample_trainer.export_tokenizer()
    hf_tokenizer = Tokenizer.from_str(json.dumps(runtime.config))

    corpus = [
        "hi!",
        "hi hi!",
        "hihi!!",
        "hello world!",
        "",
    ]

    for text in corpus:
        runtime_encoding = runtime.encode(text)
        hf_encoding = hf_tokenizer.encode(text)
        assert runtime_encoding.ids == hf_encoding.ids
        assert runtime.decode(runtime_encoding.ids) == hf_tokenizer.decode(hf_encoding.ids)


def test_gpubpe_tokenizer_from_file_round_trip(
    sample_trainer: GPUBPETrainer, tmp_path: Path
) -> None:
    sample_trainer.save(str(tmp_path))
    tokenizer_path = tmp_path / "tokenizer.json"
    assert tokenizer_path.exists()

    runtime = GPUBPETokenizer.from_file(str(tokenizer_path))
    hf_tokenizer = Tokenizer.from_file(str(tokenizer_path))

    sample = "hi! hi!"
    runtime_encoding = runtime.encode(sample)
    hf_encoding = hf_tokenizer.encode(sample)
    assert runtime_encoding.ids == hf_encoding.ids
    assert runtime.decode(runtime_encoding.ids) == hf_tokenizer.decode(hf_encoding.ids)
