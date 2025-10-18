import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "gpu_tokenizer"

if "gpu_tokenizer" not in sys.modules:
    pkg = types.ModuleType("gpu_tokenizer")
    pkg.__path__ = [str(PKG_ROOT)]
    sys.modules["gpu_tokenizer"] = pkg

if "gpu_tokenizer.code_mode" not in sys.modules:
    sub_pkg = types.ModuleType("gpu_tokenizer.code_mode")
    sub_pkg.__path__ = [str(PKG_ROOT / "code_mode")]
    sys.modules["gpu_tokenizer.code_mode"] = sub_pkg


def _load_module(name: str):
    module_path = PKG_ROOT / "code_mode" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"gpu_tokenizer.code_mode.{name}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


linearizer = _load_module("linearizer")
py_frontend = _load_module("py_frontend")
ts_frontend = _load_module("ts_frontend")

linearize_python_source = py_frontend.linearize_python_source
linearize_typescript_source = ts_frontend.linearize_typescript_source


def _extract_identifier_sequence(tokens):
    return [token.value for token in tokens if token.kind == "identifier"]


def test_python_linearization_round_trip(tmp_path):
    source = """
from math import sqrt


def hypotenuse(a, b):
    length = sqrt(a ** 2 + b ** 2)
    return length
""".strip()

    result = linearize_python_source(source, filename="example.py")
    placeholders = _extract_identifier_sequence(result.tokens)
    assert placeholders, "expected identifiers to be normalized"
    reconstructed = [result.symbols[p] for p in placeholders]
    assert reconstructed.count("hypotenuse") == 1
    assert reconstructed.count("length") == 2
    assert set(result.symbols.values()) >= {"sqrt", "hypotenuse", "length", "a", "b"}

    sidecar_path = tmp_path / "example.py"
    sidecar_path.write_text(source)
    output = result.write_symbols_sidecar(sidecar_path)
    assert output.read_text() == json.dumps(result.symbols, indent=2, sort_keys=False)


@pytest.mark.skipif(
    not hasattr(linearize_typescript_source, "__call__"),
    reason="TypeScript linearizer unavailable",
)
def test_typescript_linearization_round_trip():
    source = """
export function greet(name) {
    const message = `${name}!`;
    return message;
}
""".strip()

    try:
        result = linearize_typescript_source(source, filename="example.ts")
    except RuntimeError as exc:  # pragma: no cover - dependency missing in CI.
        pytest.skip(str(exc))

    placeholders = _extract_identifier_sequence(result.tokens)
    assert placeholders, "expected TypeScript identifiers to be normalized"
    reconstructed = [result.symbols[p] for p in placeholders]
    assert reconstructed.count("greet") == 1
    assert reconstructed.count("name") >= 2
    assert "message" in reconstructed
    assert result.metadata["mode"] in {"tree_sitter", "esprima"}

    # Round trip the placeholders.
    mapping = result.symbols
    for placeholder in placeholders:
        assert mapping[placeholder]
