"""Regression tests for the evaluation report helpers."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from tests._stubs import install_torch_stub

install_torch_stub()

from gpu_tokenizer import evaluate as eval_mod  # noqa: E402  pylint: disable=wrong-import-position
from gpu_tokenizer.morphology import (  # noqa: E402  pylint: disable=wrong-import-position
    MorphologyPlugin,
    MorphologySegment,
    create_plugin,
)


class _CountingPlugin(MorphologyPlugin):
    """Light-weight morphology plugin used to validate aggregation."""

    def presegment(self, sequence: bytes):  # type: ignore[override]
        surface = sequence.decode("utf-8")
        if surface.startswith("meta"):
            yield MorphologySegment(surface.encode("utf-8"), tags=("META",), role="prefix")
            yield MorphologySegment(surface[4:].encode("utf-8"), tags=(), role="suffix")
        else:
            yield MorphologySegment(sequence, tags=("ROOT",), role="root")


def _fixture_paths() -> dict[str, Path]:
    data_root = Path("tests/data/evaluate")
    return {
        "corpus": data_root / "corpus.txt",
        "artifacts": data_root / "artifacts",
        "vocab": data_root / "artifacts" / "vocab.json",
        "merges": data_root / "artifacts" / "merges.json",
        "tokenizer": data_root / "artifacts" / "tokenizer.json",
        "expected": data_root / "expected_report.json",
    }


def test_apply_merges_remains_stable() -> None:
    """Merging identical left/right pairs should remain deterministic."""

    rule_primary = eval_mod.MergeRule(left=10, right=11, new_id=20)
    rule_secondary = eval_mod.MergeRule(left=20, right=11, new_id=30)
    sequence = [10, 11, 11, 10]

    result = eval_mod._apply_merges(sequence, [rule_primary, rule_secondary])  # type: ignore[attr-defined]

    assert result == [30, 10]


def test_aggregate_morphology_counts_segments() -> None:
    """Verify that morphology aggregation keeps averages and tagging data stable."""

    documents = [b"metafoo", b"bar"]
    summary = eval_mod._aggregate_morphology(documents, plugin=_CountingPlugin())  # type: ignore[attr-defined]

    assert summary["enabled"] is True
    assert summary["documents"] == 2
    assert summary["total_segments"] == 3
    assert summary["tagged_segments"] == 2
    assert summary["roles"] == {"prefix": 1, "root": 1, "suffix": 1}
    assert summary["average_segments"] == pytest.approx(1.5)


def test_evaluate_report_matches_fixture(tmp_path: Path) -> None:
    """Call evaluate() directly and compare against the golden JSON report."""

    paths = _fixture_paths()
    report = eval_mod.evaluate(  # type: ignore[arg-type]
        [paths["corpus"]],
        vocab_path=paths["vocab"],
        merges_path=paths["merges"],
        tokenizer_path=paths["tokenizer"],
        morphology=create_plugin("tr", case_markers=False, affix_tags=False),
        deterministic=True,
    )

    report.setdefault("morphology", {})["config"] = {
        "enabled": True,
        "language": "tr",
        "case_markers": False,
        "affix_tags": False,
    }
    report.setdefault("code_mode", {})["config"] = {
        "enabled": False,
        "languages_filter": None,
        "meta_compress": False,
        "meta_max_length": 8,
    }

    destination = tmp_path / "report.json"
    destination.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")

    expected = json.loads(paths["expected"].read_text(encoding="utf-8"))

    assert report == expected


def test_cookbook_command_snippet_lists_required_flags() -> None:
    """Ensure the cookbook example stays aligned with the CLI requirements."""

    cookbook = Path("docs/cookbook/evaluate.md").read_text(encoding="utf-8")
    snippet: list[str] | None = None
    capture = False
    block: list[str] = []
    for line in cookbook.splitlines():
        stripped = line.strip()
        if stripped == "```bash":
            capture = True
            block = []
            continue
        if capture and stripped == "```":
            if any("python main.py evaluate" in piece for piece in block):
                snippet = block
                break
            capture = False
            continue
        if capture:
            block.append(line)

    assert snippet, "Expected to find evaluate command snippet in cookbook"

    command = " ".join(part.rstrip("\\").strip() for part in snippet if part.strip())
    tokens = shlex.split(command)

    assert tokens[0:3] == ["python", "main.py", "evaluate"]
    assert "--data" in tokens
    assert any(flag in tokens for flag in ["--artifacts", "--vocab"])
    assert "--deterministic" in tokens
    assert "--output" in tokens
