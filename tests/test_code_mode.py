import importlib.util
import json
import logging
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
meta_compress = _load_module("meta_compress")
pipeline = _load_module("pipeline")

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


def test_meta_token_compressor_discovers_patterns():
    compressor = meta_compress.MetaTokenCompressor(max_pattern_length=3, min_frequency=2)
    sequences = [
        ["A", "B", "C", "A", "B", "C"],
        ["X", "A", "B", "C", "Y"],
    ]
    result = compressor.compress(sequences)

    assert result.dictionary, "expected meta tokens to be discovered"
    assert result.dictionary["META0"] == ["A", "B", "C"]
    assert any("META0" in seq for seq in result.sequences)


def test_prepare_corpus_meta_and_fallback(caplog):
    entries = [
        {
            "language": "python",
            "source": "\n" "def alpha(x):\n" "    return x * 2\n" "\n" "def beta(x):\n" "    return x * 2\n",
            "filename": "alpha.py",
        },
        {
            "language": "python",
            "source": "def broken(: pass",
            "filename": "broken.py",
        },
        {
            "language": "typescript",
            "source": "export function hi(name) { return name.toUpperCase(); }",
            "filename": "hi.ts",
        },
    ]

    caplog.set_level(logging.WARNING)
    corpus = pipeline.prepare_corpus(entries, meta_enabled=True, meta_max_length=4)

    assert len(corpus.samples) == len(entries)
    fallbacks = corpus.byte_fallbacks()
    assert fallbacks, "expected at least one byte-level fallback"
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    for sample in fallbacks:
        assert sample.metadata["fallback"] is True
        assert all(isinstance(byte, int) for byte in sample.tokens)

    ast_samples = corpus.ast_samples()
    if corpus.meta_tokens:
        meta_keys = set(corpus.meta_tokens)
        assert meta_keys
        assert any(any(token in meta_keys for token in sample.tokens) for sample in ast_samples)
        assert all(1 < len(pattern) <= 4 for pattern in corpus.meta_tokens.values())


def test_prepare_corpus_disable_meta():
    entries = [
        {
            "language": "python",
            "source": "def gamma(y):\n    return y + 1\n",
            "filename": "gamma.py",
        }
    ]
    corpus = pipeline.prepare_corpus(entries, meta_enabled=False, meta_max_length=3)

    assert corpus.meta_tokens == {}
    ast_samples = corpus.ast_samples()
    assert ast_samples, "expected AST samples to be present"
    for sample in ast_samples:
        assert all(not token.startswith("META") for token in sample.tokens)
