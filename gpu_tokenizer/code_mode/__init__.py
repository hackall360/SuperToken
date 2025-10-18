"""Front-ends for language-aware code linearization utilities."""

from .linearizer import LinearizationResult, LinearizedToken, SymbolTable, write_symbol_sidecar
from .py_frontend import linearize_python_file, linearize_python_source
from .ts_frontend import linearize_typescript_file, linearize_typescript_source

__all__ = [
    "LinearizationResult",
    "LinearizedToken",
    "SymbolTable",
    "linearize_python_file",
    "linearize_python_source",
    "linearize_typescript_file",
    "linearize_typescript_source",
    "write_symbol_sidecar",
]
