"""Serialization helpers for tokenizer evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping


_SCHEMA_CACHE: dict[str, Any] | None = None


class EvaluateReportValidationError(ValueError):
    """Raised when an evaluation payload fails schema validation."""


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "schemas" / "evaluate_report.schema.json"


def load_evaluate_report_schema() -> dict[str, Any]:
    """Load and cache the canonical evaluation report JSON schema."""

    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        path = _schema_path()
        try:
            _SCHEMA_CACHE = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:  # pragma: no cover - repository misconfiguration
            raise RuntimeError(f"Evaluation report schema not found: {path}") from exc
    return _SCHEMA_CACHE


def validate_evaluate_report(
    payload: Any, schema: Mapping[str, Any] | None = None, *, path: str = "$"
) -> None:
    """Validate ``payload`` against the evaluation report schema."""

    active_schema: Mapping[str, Any] = schema or load_evaluate_report_schema()
    _validate_node(payload, active_schema, path)


def serialize_evaluate_report(
    payload: Mapping[str, Any] | MutableMapping[str, Any], *, indent: int = 2
) -> str:
    """Return a deterministic JSON string for ``payload`` after validation."""

    validate_evaluate_report(dict(payload))
    return json.dumps(payload, indent=indent, sort_keys=True)


def _validate_node(instance: Any, schema: Mapping[str, Any], path: str) -> None:
    schema_type = schema.get("type")
    if schema_type is not None:
        _assert_type(instance, schema_type, path)
    if "enum" in schema:
        if instance not in schema["enum"]:
            raise EvaluateReportValidationError(
                f"{path}: value {instance!r} not in enum {schema['enum']!r}"
            )
    if "anyOf" in schema:
        errors = []
        for option in schema["anyOf"]:
            try:
                _validate_node(instance, option, path)
                errors = []
                break
            except EvaluateReportValidationError as exc:  # pragma: no cover - error accumulation
                errors.append(exc)
        if errors:
            raise EvaluateReportValidationError(
                f"{path}: value {instance!r} does not satisfy anyOf constraints"
            )
        return
    if schema.get("type") == "object":
        _validate_object(instance, schema, path)
    if schema.get("type") == "array":
        _validate_array(instance, schema, path)
    if "minimum" in schema:
        minimum = schema["minimum"]
        if isinstance(instance, (int, float)) and instance < minimum:
            raise EvaluateReportValidationError(
                f"{path}: value {instance!r} less than minimum {minimum}"
            )
    if "maximum" in schema:
        maximum = schema["maximum"]
        if isinstance(instance, (int, float)) and instance > maximum:
            raise EvaluateReportValidationError(
                f"{path}: value {instance!r} greater than maximum {maximum}"
            )


def _assert_type(instance: Any, expected: Any, path: str) -> None:
    if isinstance(expected, list):
        for option in expected:
            try:
                _assert_type(instance, option, path)
                return
            except EvaluateReportValidationError:
                continue
        raise EvaluateReportValidationError(
            f"{path}: value {instance!r} does not match any allowed type {expected!r}"
        )
    type_map: dict[str, Any] = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    py_type = type_map.get(expected)
    if py_type is None:
        return
    if expected == "integer" and isinstance(instance, bool):
        raise EvaluateReportValidationError(f"{path}: boolean value is not a valid integer")
    if not isinstance(instance, py_type):
        raise EvaluateReportValidationError(
            f"{path}: expected type {expected}, received {type(instance).__name__}"
        )


def _validate_object(instance: Any, schema: Mapping[str, Any], path: str) -> None:
    if not isinstance(instance, Mapping):
        raise EvaluateReportValidationError(f"{path}: expected object")
    required = schema.get("required", [])
    for key in required:
        if key not in instance:
            raise EvaluateReportValidationError(f"{path}: missing required property '{key}'")
    properties = schema.get("properties", {})
    for key, value in instance.items():
        if key in properties:
            _validate_node(value, properties[key], f"{path}.{key}")
        elif not schema.get("additionalProperties", True):
            raise EvaluateReportValidationError(f"{path}: unexpected property '{key}'")
    additional = schema.get("additionalProperties")
    if isinstance(additional, Mapping):
        for key, value in instance.items():
            if key not in properties:
                _validate_node(value, additional, f"{path}.{key}")


def _validate_array(instance: Any, schema: Mapping[str, Any], path: str) -> None:
    if not isinstance(instance, list):
        raise EvaluateReportValidationError(f"{path}: expected array")
    min_items = schema.get("minItems")
    if min_items is not None and len(instance) < min_items:
        raise EvaluateReportValidationError(f"{path}: expected at least {min_items} items")
    item_schema = schema.get("items")
    if isinstance(item_schema, Mapping):
        for index, value in enumerate(instance):
            _validate_node(value, item_schema, f"{path}[{index}]")


__all__ = [
    "EvaluateReportValidationError",
    "load_evaluate_report_schema",
    "serialize_evaluate_report",
    "validate_evaluate_report",
]
