import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from gpu_tokenizer.bpe_trainer import GPUBPETrainer

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

    assert vocab_path.exists()
    assert merges_path.exists()
    assert tokenizer_path.exists()

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
