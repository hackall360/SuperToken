from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tests._stubs import install_torch_stub

install_torch_stub()
main = importlib.import_module("main")


def test_evaluate_cli_generates_report(tmp_path: Path) -> None:
    data_root = Path(__file__).resolve().parent / "data"
    corpus = data_root / "evaluate_corpus" / "plain.txt"
    artifacts = data_root / "models" / "bpe"
    output = tmp_path / "report.json"

    cwd = Path.cwd()
    corpus_arg = str(corpus.relative_to(cwd))
    artifacts_arg = str(artifacts.relative_to(cwd))

    main.main(
        [
            "evaluate",
            "--data",
            corpus_arg,
            "--artifacts",
            artifacts_arg,
            "--morphology-lang",
            "tr",
            "--morphology-case-markers",
            "--morphology-affix-tags",
            "--deterministic",
            "--output",
            str(output),
        ]
    )

    assert output.exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    expected_path = data_root / "expected" / "evaluate_report.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    assert report == expected


def test_evaluate_help_describes_new_flags(capsys: pytest.CaptureFixture[str]) -> None:
    parser = main._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--help"])
    help_text = capsys.readouterr().out
    assert "Score exported BPE or unigram" in help_text
    assert "--model-type" in help_text
    assert "reports/evaluate.json" in help_text


def test_evaluate_cli_skip_flag(monkeypatch, tmp_path: Path) -> None:
    data_root = Path(__file__).resolve().parent / "data"
    corpus = data_root / "evaluate_corpus" / "plain.txt"
    artifacts = data_root / "models" / "bpe"
    output = tmp_path / "skip.json"

    monkeypatch.setenv("SUPERTOKEN_SKIP_EVALUATION", "1")

    cwd = Path.cwd()
    main.main(
        [
            "evaluate",
            "--data",
            str(corpus.relative_to(cwd)),
            "--artifacts",
            str(artifacts.relative_to(cwd)),
            "--output",
            str(output),
        ]
    )

    assert not output.exists()


def test_evaluate_cli_force_overrides_skip(monkeypatch, tmp_path: Path) -> None:
    data_root = Path(__file__).resolve().parent / "data"
    corpus = data_root / "evaluate_corpus" / "plain.txt"
    artifacts = data_root / "models" / "bpe"
    output = tmp_path / "forced.json"

    monkeypatch.setenv("SUPERTOKEN_SKIP_EVALUATION", "true")

    cwd = Path.cwd()
    main.main(
        [
            "evaluate",
            "--data",
            str(corpus.relative_to(cwd)),
            "--artifacts",
            str(artifacts.relative_to(cwd)),
            "--force-evaluation",
            "--output",
            str(output),
        ]
    )

    assert output.exists()


def test_evaluate_cli_accepts_unigram_exports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_root = Path(__file__).resolve().parent / "data"
    corpus = data_root / "evaluate_corpus" / "plain.txt"
    unigram_vocab = data_root / "models" / "unigram" / "unigram.vocab"
    output = tmp_path / "unigram.json"

    cwd = Path.cwd()
    main.main(
        [
            "evaluate",
            "--data",
            str(corpus.relative_to(cwd)),
            "--vocab",
            str(unigram_vocab.relative_to(cwd)),
            "--model-type",
            "unigram",
            "--summary-format",
            "json",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert "[evaluate][summary]" in captured.out
    assert '"model_type": "unigram"' in captured.out

    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifacts"]["model_type"] == "unigram"
