from __future__ import annotations

import copy
import json

import pytest

from gpu_tokenizer.evaluate_report import (
    EvaluateReportValidationError,
    serialize_evaluate_report,
    validate_evaluate_report,
)


@pytest.fixture
def sample_report() -> dict[str, object]:
    return {
        "artifacts": {
            "merge_rules": 10,
            "merges": None,
            "tokenizer": "tokenizer.json",
            "vocab": "vocab.json",
            "model_type": "bpe",
            "vocab_size": 42,
        },
        "corpus": {
            "average_bytes": 12.5,
            "average_tokens": 8.0,
            "documents": 2,
            "total_bytes": 25,
            "total_tokens": 16,
        },
        "compression": {
            "bytes_per_token": 1.5625,
            "tokens_per_byte": 0.64,
        },
        "oov": {
            "instances": 1,
            "rate": 0.0625,
            "unique": ["<unk>", 512],
        },
        "morphology": {
            "average_segments": 3.0,
            "config": {"enabled": False},
            "documents": 2,
            "enabled": False,
            "roles": {},
            "tagged_segments": 0,
            "total_segments": 6,
        },
        "code_mode": {
            "ast_samples": 1,
            "config": {
                "enabled": False,
                "languages_filter": None,
                "meta_compress": False,
                "meta_max_length": 8,
            },
            "documents": 2,
            "fallback_samples": 1,
            "languages": ["python"],
            "meta_compress": {"META_0": ["def", "foo"]},
            "meta_enabled": False,
            "meta_max_length": 8,
            "meta_token_count": 1,
            "mode": "code",
            "reduction": 0.25,
        },
    }


def test_serialize_report_is_deterministic(sample_report: dict[str, object]) -> None:
    expected = json.dumps(sample_report, indent=2, sort_keys=True)
    serialized = serialize_evaluate_report(sample_report)
    assert serialized == expected
    # A second call confirms that ordering is stable regardless of mapping iteration
    assert serialize_evaluate_report(sample_report) == expected


def test_missing_required_section_raises(sample_report: dict[str, object]) -> None:
    broken = copy.deepcopy(sample_report)
    del broken["compression"]
    with pytest.raises(
        EvaluateReportValidationError,
        match=r"\$: missing required property 'compression'",
    ):
        validate_evaluate_report(broken)


def test_invalid_item_type_is_reported(sample_report: dict[str, object]) -> None:
    broken = copy.deepcopy(sample_report)
    broken["oov"]["unique"] = [0.5]
    with pytest.raises(
        EvaluateReportValidationError,
        match=r"\$\.oov\.unique\[0\]: value 0\.5 does not satisfy anyOf constraints",
    ):
        validate_evaluate_report(broken)


def test_artifact_model_type_enum_enforced(sample_report: dict[str, object]) -> None:
    broken = copy.deepcopy(sample_report)
    broken["artifacts"]["model_type"] = "bytepair"
    with pytest.raises(
        EvaluateReportValidationError,
        match=r"\$\.artifacts\.model_type: value 'bytepair' not in enum \['bpe', 'unigram'\]",
    ):
        validate_evaluate_report(broken)


def test_morphology_config_required(sample_report: dict[str, object]) -> None:
    broken = copy.deepcopy(sample_report)
    del broken["morphology"]["config"]
    with pytest.raises(
        EvaluateReportValidationError,
        match=r"\$\.morphology: missing required property 'config'",
    ):
        validate_evaluate_report(broken)


def test_code_mode_languages_filter_type(sample_report: dict[str, object]) -> None:
    broken = copy.deepcopy(sample_report)
    broken["code_mode"]["config"]["languages_filter"] = "python"
    with pytest.raises(
        EvaluateReportValidationError,
        match=r"\$\.code_mode\.config\.languages_filter: value 'python' does not satisfy anyOf constraints",
    ):
        validate_evaluate_report(broken)


def test_serializer_materialises_sequences(sample_report: dict[str, object]) -> None:
    report = copy.deepcopy(sample_report)
    report["oov"]["unique"] = ("<unk>", 512)
    serialized = serialize_evaluate_report(report)
    payload = json.loads(serialized)
    assert payload["oov"]["unique"] == ["<unk>", 512]
