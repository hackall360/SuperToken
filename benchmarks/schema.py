"""JSON schema utilities for benchmark serialization outputs."""

from __future__ import annotations

from typing import Any


BENCHMARK_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Benchmark output",
    "type": "object",
    "additionalProperties": False,
    "required": ["timestamp", "config", "corpus", "bpe", "unigram"],
    "properties": {
        "timestamp": {"type": "string"},
        "config": {"type": "object"},
        "corpus": {
            "type": "object",
            "additionalProperties": False,
            "required": ["sequences", "tokens", "max_length", "sources"],
            "properties": {
                "sequences": {"type": "integer", "minimum": 0},
                "tokens": {"type": "integer", "minimum": 0},
                "max_length": {"type": "integer", "minimum": 0},
                "sources": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
        },
        "bpe": {
            "type": "object",
            "additionalProperties": True,
            "required": [
                "config",
                "wall_time_s",
                "result",
                "overlap_enabled",
                "tokens_processed",
                "tokens_per_s",
                "autoscaler_window",
            ],
            "properties": {
                "config": {"type": "object"},
                "wall_time_s": {"type": "number"},
                "result": {"type": "object"},
                "overlap_enabled": {"type": "boolean"},
                "tokens_processed": {"type": "integer", "minimum": 0},
                "tokens_per_s": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "null"},
                    ]
                },
                "autoscaler_window": {"type": "array"},
            },
        },
        "unigram": {
            "type": "object",
            "additionalProperties": True,
            "required": ["config", "wall_time_s", "epochs"],
            "properties": {
                "config": {"type": "object"},
                "wall_time_s": {"type": "number"},
                "epochs": {"type": "array"},
            },
        },
        "bpe_runs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "required": ["name", "wall_time_s", "tokens_per_s"],
                "properties": {
                    "name": {"type": "string"},
                    "wall_time_s": {"type": "number"},
                    "tokens_per_s": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "null"},
                        ]
                    },
                    "scaling": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "reference": {"type": "string"},
                            "device_weights": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                            "expected_tokens_per_s": {
                                "anyOf": [
                                    {"type": "number"},
                                    {"type": "null"},
                                ]
                            },
                            "efficiency": {
                                "anyOf": [
                                    {"type": "number"},
                                    {"type": "null"},
                                ]
                            },
                            "meets_target": {
                                "anyOf": [
                                    {"type": "boolean"},
                                    {"type": "null"},
                                ]
                            },
                            "target_efficiency": {
                                "anyOf": [
                                    {"type": "number"},
                                    {"type": "null"},
                                ]
                            },
                        },
                    },
                },
            },
        },
    },
}


class SchemaValidationError(ValueError):
    """Raised when a benchmark payload fails schema validation."""


def validate_benchmark_output(
    payload: Any, schema: dict[str, Any] | None = None, *, path: str = "$"
) -> None:
    """Validate a benchmark payload against the JSON schema.

    Parameters
    ----------
    payload:
        Python data structure representing the benchmark JSON document.
    schema:
        Optional schema override. When omitted the canonical
        :data:`BENCHMARK_OUTPUT_SCHEMA` is used.
    path:
        Internal parameter used to provide precise error locations.

    Raises
    ------
    SchemaValidationError
        If ``payload`` does not conform to ``schema``.
    """

    active_schema = schema or BENCHMARK_OUTPUT_SCHEMA
    _validate_node(payload, active_schema, path)


def _validate_node(instance: Any, schema: dict[str, Any], path: str) -> None:
    schema_type = schema.get("type")
    if schema_type is not None:
        _assert_type(instance, schema_type, path)
    if "enum" in schema:
        if instance not in schema["enum"]:
            raise SchemaValidationError(f"{path}: value {instance!r} not in enum {schema['enum']!r}")
    if "anyOf" in schema:
        errors = []
        for option in schema["anyOf"]:
            try:
                _validate_node(instance, option, path)
                errors = []
                break
            except SchemaValidationError as exc:  # pragma: no cover - error accumulation
                errors.append(exc)
        if errors:
            raise SchemaValidationError(
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
            raise SchemaValidationError(f"{path}: value {instance!r} less than minimum {minimum}")


def _assert_type(instance: Any, expected: Any, path: str) -> None:
    if isinstance(expected, list):
        for option in expected:
            try:
                _assert_type(instance, option, path)
                return
            except SchemaValidationError:
                continue
        raise SchemaValidationError(f"{path}: value {instance!r} does not match any allowed type {expected!r}")
    type_map = {
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
        raise SchemaValidationError(f"{path}: boolean value is not a valid integer")
    if not isinstance(instance, py_type):
        raise SchemaValidationError(
            f"{path}: expected type {expected}, received {type(instance).__name__}"
        )


def _validate_object(instance: Any, schema: dict[str, Any], path: str) -> None:
    if not isinstance(instance, dict):
        raise SchemaValidationError(f"{path}: expected object")
    required = schema.get("required", [])
    for key in required:
        if key not in instance:
            raise SchemaValidationError(f"{path}: missing required property '{key}'")
    properties = schema.get("properties", {})
    for key, value in instance.items():
        if key in properties:
            _validate_node(value, properties[key], f"{path}.{key}")
        elif not schema.get("additionalProperties", True):
            raise SchemaValidationError(f"{path}: unexpected property '{key}'")
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        for key, value in instance.items():
            if key not in properties:
                _validate_node(value, additional, f"{path}.{key}")


def _validate_array(instance: Any, schema: dict[str, Any], path: str) -> None:
    if not isinstance(instance, list):
        raise SchemaValidationError(f"{path}: expected array")
    min_items = schema.get("minItems")
    if min_items is not None and len(instance) < min_items:
        raise SchemaValidationError(f"{path}: expected at least {min_items} items")
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for index, value in enumerate(instance):
            _validate_node(value, item_schema, f"{path}[{index}]")

