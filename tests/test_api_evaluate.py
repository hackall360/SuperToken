"""API surface tests for the public evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._stubs import install_torch_stub

install_torch_stub()

from gpu_tokenizer import EvaluateCLIOptions, evaluate  # noqa: E402  pylint: disable=wrong-import-position
from gpu_tokenizer.evaluate import EvaluateCLIResult, evaluate_cli  # noqa: E402  pylint: disable=wrong-import-position
from gpu_tokenizer.morphology import create_plugin  # noqa: E402  pylint: disable=wrong-import-position


def _fixture_paths() -> dict[str, Path]:
    data_root = Path("tests/data")
    return {
        "corpus": data_root / "evaluate_corpus" / "plain.txt",
        "models": data_root / "models" / "bpe",
        "vocab": data_root / "models" / "bpe" / "vocab.json",
        "merges": data_root / "models" / "bpe" / "merges.json",
        "tokenizer": data_root / "models" / "bpe" / "tokenizer.json",
        "expected": data_root / "expected" / "evaluate_report.json",
    }


def test_top_level_evaluate_matches_fixture() -> None:
    paths = _fixture_paths()
    plugin = create_plugin("tr", case_markers=True, affix_tags=True)
    report = evaluate(
        [paths["corpus"]],
        vocab_path=paths["vocab"],
        merges_path=paths["merges"],
        tokenizer_path=paths["tokenizer"],
        morphology=plugin,
        deterministic=True,
    )

    # Mirror the CLI adjustments so we can compare against the golden file.
    report.setdefault("morphology", {})["config"] = {
        "enabled": True,
        "language": "tr",
        "case_markers": True,
        "affix_tags": True,
    }
    report.setdefault("code_mode", {})["config"] = {
        "enabled": False,
        "languages_filter": None,
        "meta_compress": False,
        "meta_max_length": 8,
    }

    expected = json.loads(paths["expected"].read_text(encoding="utf-8"))
    assert report == expected


def test_cli_wrapper_returns_structured_result() -> None:
    paths = _fixture_paths()
    plugin = create_plugin("tr", case_markers=True, affix_tags=True)
    options = EvaluateCLIOptions(
        data_files=[paths["corpus"]],
        vocab_path=paths["vocab"],
        merges_path=paths["merges"],
        tokenizer_path=paths["tokenizer"],
        model_type="bpe",
        morphology=plugin,
        morphology_config={
            "enabled": True,
            "language": "tr",
            "case_markers": True,
            "affix_tags": True,
        },
        deterministic=True,
    )

    result = evaluate_cli(options)
    assert isinstance(result, EvaluateCLIResult)

    expected = json.loads(paths["expected"].read_text(encoding="utf-8"))
    assert result.report == expected
    assert result.summary == {
        "documents": expected["corpus"]["documents"],
        "tokens": expected["corpus"]["total_tokens"],
        "tokens_per_byte": expected["compression"]["tokens_per_byte"],
        "oov_rate": expected["oov"]["rate"],
        "model_type": "bpe",
    }


def test_cli_wrapper_requires_data_files() -> None:
    with pytest.raises(ValueError):
        evaluate_cli(
            EvaluateCLIOptions(
                data_files=[],
                vocab_path=Path("missing.json"),
            )
        )
