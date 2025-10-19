from __future__ import annotations

import copy

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
    expected = (
        "{\n"
        "  \"artifacts\": {\n"
        "    \"merge_rules\": 10,\n"
        "    \"merges\": null,\n"
        "    \"tokenizer\": \"tokenizer.json\",\n"
        "    \"vocab\": \"vocab.json\",\n"
        "    \"vocab_size\": 42\n"
        "  },\n"
        "  \"code_mode\": {\n"
        "    \"ast_samples\": 1,\n"
        "    \"config\": {\n"
        "      \"enabled\": false,\n"
        "      \"languages_filter\": null,\n"
        "      \"meta_compress\": false,\n"
        "      \"meta_max_length\": 8\n"
        "    },\n"
        "    \"documents\": 2,\n"
        "    \"fallback_samples\": 1,\n"
        "    \"languages\": [\n"
        "      \"python\"\n"
        "    ],\n"
        "    \"meta_compress\": {\n"
        "      \"META_0\": [\n"
        "        \"def\",\n"
        "        \"foo\"\n"
        "      ]\n"
        "    },\n"
        "    \"meta_enabled\": false,\n"
        "    \"meta_max_length\": 8,\n"
        "    \"meta_token_count\": 1,\n"
        "    \"mode\": \"code\",\n"
        "    \"reduction\": 0.25\n"
        "  },\n"
        "  \"compression\": {\n"
        "    \"bytes_per_token\": 1.5625,\n"
        "    \"tokens_per_byte\": 0.64\n"
        "  },\n"
        "  \"corpus\": {\n"
        "    \"average_bytes\": 12.5,\n"
        "    \"average_tokens\": 8.0,\n"
        "    \"documents\": 2,\n"
        "    \"total_bytes\": 25,\n"
        "    \"total_tokens\": 16\n"
        "  },\n"
        "  \"morphology\": {\n"
        "    \"average_segments\": 3.0,\n"
        "    \"config\": {\n"
        "      \"enabled\": false\n"
        "    },\n"
        "    \"documents\": 2,\n"
        "    \"enabled\": false,\n"
        "    \"roles\": {},\n"
        "    \"tagged_segments\": 0,\n"
        "    \"total_segments\": 6\n"
        "  },\n"
        "  \"oov\": {\n"
        "    \"instances\": 1,\n"
        "    \"rate\": 0.0625,\n"
        "    \"unique\": [\n"
        "      \"<unk>\",\n"
        "      512\n"
        "    ]\n"
        "  }\n"
        "}"
    )
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
