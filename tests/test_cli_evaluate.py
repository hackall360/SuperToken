from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tests._stubs import install_torch_stub

install_torch_stub()
main = importlib.import_module("main")


def _relative_paths(paths: list[Path]) -> list[str]:
    cwd = Path.cwd()
    return [str(path.relative_to(cwd)) for path in paths]


def _corpus_files() -> list[Path]:
    data_root = Path(__file__).resolve().parent / "data"
    corpus_dir = data_root / "evaluate_corpus"
    return sorted(corpus_dir.glob("*.txt"))


def test_evaluate_cli_generates_report(tmp_path: Path) -> None:
    data_root = Path(__file__).resolve().parent / "data"
    corpus_files = _corpus_files()
    artifacts = data_root / "models" / "bpe"
    output = tmp_path / "report.json"

    cwd = Path.cwd()
    artifacts_arg = str(artifacts.relative_to(cwd))

    main.main(
        [
            "evaluate",
            "--data",
            *_relative_paths(corpus_files),
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


def test_evaluate_cli_aborts_on_schema_violation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = Path(__file__).resolve().parent / "data"
    corpus_files = _corpus_files()
    artifacts = data_root / "models" / "bpe"
    output = tmp_path / "invalid.json"

    invalid_report = {
        "artifacts": {
            "vocab": "tests/data/models/bpe/vocab.json",
            "vocab_size": 10,
            "merges": None,
            "merge_rules": 0,
            "tokenizer": None,
            "model_type": "bpe",
        },
        "corpus": {
            "documents": 1,
            "total_bytes": 1,
            "total_tokens": 1,
            "average_bytes": 1.0,
            "average_tokens": 1.0,
        },
        "oov": {"instances": 0, "rate": 0.0, "unique": []},
        "morphology": {"enabled": False, "config": {"enabled": False}},
        "code_mode": {
            "mode": "plain",
            "documents": 1,
            "reduction": 0.0,
            "config": {
                "enabled": False,
                "languages_filter": None,
                "meta_compress": False,
                "meta_max_length": 8,
            },
        },
    }

    class _InvalidResult:
        def __init__(self) -> None:
            self.report = invalid_report
            self.summary: dict[str, object] = {}

    def _fake_evaluate_cli(options):  # pragma: no cover - exercised via CLI
        return _InvalidResult()

    monkeypatch.setattr(main.evaluate_module, "evaluate_cli", _fake_evaluate_cli)

    cwd = Path.cwd()
    with pytest.raises(SystemExit) as excinfo:
        main.main(
            [
                "evaluate",
                "--data",
                *_relative_paths(corpus_files),
                "--artifacts",
                str(artifacts.relative_to(cwd)),
                "--output",
                str(output),
            ]
        )

    assert "failed schema validation" in str(excinfo.value)
    assert not output.exists()


def test_evaluate_help_describes_new_flags(capsys: pytest.CaptureFixture[str]) -> None:
    parser = main._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--help"])
    help_text = capsys.readouterr().out
    assert "schema-validated JSON report" in help_text
    assert "Required inputs" in help_text
    assert "Artifact flavours" in help_text
    assert "table/json/none" in help_text
    assert "reports/evaluate.json" in help_text


def test_evaluate_cli_skip_flag(monkeypatch, tmp_path: Path) -> None:
    data_root = Path(__file__).resolve().parent / "data"
    corpus_files = _corpus_files()
    artifacts = data_root / "models" / "bpe"
    output = tmp_path / "skip.json"

    monkeypatch.setenv("SUPERTOKEN_SKIP_EVALUATION", "1")

    cwd = Path.cwd()
    main.main(
        [
            "evaluate",
            "--data",
            *_relative_paths(corpus_files),
            "--artifacts",
            str(artifacts.relative_to(cwd)),
            "--output",
            str(output),
        ]
    )

    assert not output.exists()


def test_evaluate_cli_force_overrides_skip(monkeypatch, tmp_path: Path) -> None:
    data_root = Path(__file__).resolve().parent / "data"
    corpus_files = _corpus_files()
    artifacts = data_root / "models" / "bpe"
    output = tmp_path / "forced.json"

    monkeypatch.setenv("SUPERTOKEN_SKIP_EVALUATION", "true")

    cwd = Path.cwd()
    main.main(
        [
            "evaluate",
            "--data",
            *_relative_paths(corpus_files),
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
    corpus_files = _corpus_files()
    unigram_vocab = data_root / "models" / "unigram" / "unigram.vocab"
    output = tmp_path / "unigram.json"

    cwd = Path.cwd()
    main.main(
        [
            "evaluate",
            "--data",
            *_relative_paths(corpus_files),
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
