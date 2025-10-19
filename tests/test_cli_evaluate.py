from __future__ import annotations

import importlib
import json
from pathlib import Path

from tests._stubs import install_torch_stub

install_torch_stub()
main = importlib.import_module("main")


def test_evaluate_cli_generates_report(tmp_path: Path) -> None:
    data_root = Path(__file__).resolve().parent / "data" / "evaluate"
    corpus = data_root / "corpus.txt"
    artifacts = data_root / "artifacts"
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
            "--deterministic",
            "--output",
            str(output),
        ]
    )

    assert output.exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    expected_path = data_root / "expected_report.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    assert report == expected
