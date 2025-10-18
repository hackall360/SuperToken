"""Python front-end for the code-mode linearizer."""

from __future__ import annotations

import ast
from typing import Any, List

from .linearizer import LinearizationResult, LinearizedToken, SymbolTable

_IDENTIFIER_FIELDS = {"id", "arg", "attr", "name"}


class _LinearizePython(ast.NodeVisitor):
    def __init__(self, symbol_table: SymbolTable) -> None:
        self.symbol_table = symbol_table
        self.tokens: List[LinearizedToken] = []

    def generic_visit(self, node: ast.AST) -> None:  # type: ignore[override]
        self.tokens.append(LinearizedToken("node", type(node).__name__))
        for field, value in ast.iter_fields(node):
            self.tokens.append(LinearizedToken("field", field))
            self._visit_field(field, value)

    def _visit_field(self, field: str, value: Any) -> None:
        if isinstance(value, ast.AST):
            self.visit(value)
        elif isinstance(value, list):
            for item in value:
                self.tokens.append(LinearizedToken("list-item", field))
                if isinstance(item, ast.AST):
                    self.visit(item)
                else:
                    self._emit_value(field, item)
        else:
            self._emit_value(field, value)

    def _emit_value(self, field: str, value: Any) -> None:
        if value is None:
            self.tokens.append(LinearizedToken("literal", "None", {"field": field}))
            return
        if field in _IDENTIFIER_FIELDS and isinstance(value, str):
            placeholder = self.symbol_table.register(value)
            self.tokens.append(
                LinearizedToken(
                    "identifier", placeholder, {"field": field, "original": value}
                )
            )
        elif isinstance(value, str):
            self.tokens.append(
                LinearizedToken("literal", value, {"field": field, "type": "str"})
            )
        elif isinstance(value, (int, float, complex, bool)):
            self.tokens.append(
                LinearizedToken(
                    "literal", repr(value), {"field": field, "type": type(value).__name__}
                )
            )
        else:
            self.tokens.append(
                LinearizedToken(
                    "literal", repr(value), {"field": field, "type": "unknown"}
                )
            )


def linearize_python_source(source: str, filename: str = "<unknown>") -> LinearizationResult:
    """Linearize *source* code using the standard Python :mod:`ast` module."""

    tree = ast.parse(source, filename=filename)
    symbol_table = SymbolTable()
    visitor = _LinearizePython(symbol_table)
    visitor.visit(tree)
    metadata = {"language": "python", "filename": filename, "mode": "ast"}
    return LinearizationResult(visitor.tokens, metadata, symbol_table.to_mapping())


def linearize_python_file(path: str) -> LinearizationResult:
    """Read *path* and linearize the contained source."""

    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    return linearize_python_source(source, filename=path)
