"""Front-ends for language-aware code linearization utilities."""

from .linearizer import LinearizationResult, LinearizedToken, SymbolTable, write_symbol_sidecar
from .meta_compress import (
    MetaCompressionResult,
    MetaTokenCompressor,
    encode_linearized_sequence,
    encode_linearized_token,
)
from .pipeline import CodeModeCorpus, EncodedSample, prepare_corpus
from .py_frontend import linearize_python_file, linearize_python_source
from .ts_frontend import linearize_typescript_file, linearize_typescript_source

__all__ = [
    "LinearizationResult",
    "LinearizedToken",
    "SymbolTable",
    "MetaCompressionResult",
    "MetaTokenCompressor",
    "CodeModeCorpus",
    "EncodedSample",
    "encode_linearized_sequence",
    "encode_linearized_token",
    "linearize_python_file",
    "linearize_python_source",
    "linearize_typescript_file",
    "linearize_typescript_source",
    "prepare_corpus",
    "write_symbol_sidecar",
]
