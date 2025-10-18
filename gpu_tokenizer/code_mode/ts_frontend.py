"""TypeScript front-end for the code-mode linearizer."""

from __future__ import annotations

from typing import Dict, List, Optional

from .linearizer import LinearizationResult, LinearizedToken, SymbolTable

try:  # pragma: no cover - exercised in tests when dependency exists.
    from tree_sitter import Parser
    from tree_sitter_languages import get_language

    _TS_LANGUAGE = get_language("typescript")
    _PARSER: Optional[Parser] = Parser()
    _PARSER.set_language(_TS_LANGUAGE)
    _TS_AVAILABLE = True
except Exception:  # pragma: no cover - handled gracefully during runtime.
    _PARSER = None
    _TS_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import esprima  # type: ignore

    _ESPRIMA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    esprima = None  # type: ignore
    _ESPRIMA_AVAILABLE = False


_IDENTIFIER_NODE_TYPES = {
    "identifier",
    "shorthand_property_identifier_pattern",
    "property_identifier",
    "type_identifier",
}

_LITERAL_NODE_TYPES = {
    "string",
    "number",
    "template_string",
    "true",
    "false",
    "null",
}

_IDENTIFIER_FIELDS = {"name"}
_LITERAL_FIELDS = {"raw", "value"}


class _TreeSitterLinearizer:
    def __init__(self, source: str, symbol_table: SymbolTable) -> None:
        self.source = source
        self.symbol_table = symbol_table
        self.tokens: List[LinearizedToken] = []

    def walk(self, node) -> None:
        self.tokens.append(LinearizedToken("node", node.type))
        for child in node.children:
            if child.is_named:
                if child.field_name:
                    self.tokens.append(LinearizedToken("field", child.field_name))
                self.walk(child)
            else:
                text = self.source[child.start_byte : child.end_byte]
                text = text.strip()
                if text:
                    self.tokens.append(LinearizedToken("punctuation", text))
        if node.child_count == 0:
            text = self.source[node.start_byte : node.end_byte]
            stripped = text.strip()
            if node.type in _IDENTIFIER_NODE_TYPES and stripped:
                placeholder = self.symbol_table.register(stripped)
                self.tokens.append(
                    LinearizedToken(
                        "identifier", placeholder, {"original": stripped, "node": node.type}
                    )
                )
            elif node.type in _LITERAL_NODE_TYPES and stripped:
                self.tokens.append(
                    LinearizedToken("literal", stripped, {"node": node.type})
                )


def _linearize_with_tree_sitter(source: str, symbol_table: SymbolTable) -> List[LinearizedToken]:
    if not _TS_AVAILABLE or _PARSER is None:
        raise RuntimeError("tree-sitter languages for TypeScript are unavailable")
    tree = _PARSER.parse(source.encode("utf-8"))
    linearizer = _TreeSitterLinearizer(source, symbol_table)
    linearizer.walk(tree.root_node)
    return linearizer.tokens


def _emit_esprima_value(
    tokens: List[LinearizedToken], symbol_table: SymbolTable, field: str, value
) -> None:
    if value is None:
        return
    if field in _IDENTIFIER_FIELDS and isinstance(value, str):
        placeholder = symbol_table.register(value)
        tokens.append(
            LinearizedToken("identifier", placeholder, {"field": field, "original": value})
        )
    elif isinstance(value, str) and field in _LITERAL_FIELDS:
        tokens.append(LinearizedToken("literal", value, {"field": field}))
    elif isinstance(value, (int, float, bool)):
        tokens.append(LinearizedToken("literal", repr(value), {"field": field}))
    elif isinstance(value, str):
        tokens.append(LinearizedToken("literal", value, {"field": field, "type": "str"}))


def _walk_esprima(
    node: Dict[str, object], tokens: List[LinearizedToken], symbol_table: SymbolTable
) -> None:
    node_type = node.get("type")
    if not node_type:
        return
    tokens.append(LinearizedToken("node", str(node_type)))
    for key, value in node.items():
        if key == "type":
            continue
        tokens.append(LinearizedToken("field", key))
        if isinstance(value, dict):
            _walk_esprima(value, tokens, symbol_table)
        elif isinstance(value, list):
            for item in value:
                tokens.append(LinearizedToken("list-item", key))
                if isinstance(item, dict):
                    _walk_esprima(item, tokens, symbol_table)
                else:
                    _emit_esprima_value(tokens, symbol_table, key, item)
        else:
            _emit_esprima_value(tokens, symbol_table, key, value)


def _linearize_with_esprima(source: str, symbol_table: SymbolTable) -> List[LinearizedToken]:
    if not _ESPRIMA_AVAILABLE or esprima is None:
        raise RuntimeError("esprima is unavailable")
    program = esprima.parseScript(source, tolerant=True)
    tokens: List[LinearizedToken] = []
    _walk_esprima(program.toDict(), tokens, symbol_table)  # type: ignore[operator]
    return tokens


def linearize_typescript_source(source: str, filename: str = "<unknown>") -> LinearizationResult:
    """Linearize *source* code using :mod:`tree_sitter` or :mod:`esprima`."""

    symbol_table = SymbolTable()
    tokens: List[LinearizedToken]
    mode = "unavailable"
    if _TS_AVAILABLE and _PARSER is not None:
        tokens = _linearize_with_tree_sitter(source, symbol_table)
        mode = "tree_sitter"
    elif _ESPRIMA_AVAILABLE:
        tokens = _linearize_with_esprima(source, symbol_table)
        mode = "esprima"
    else:
        raise RuntimeError(
            "Neither tree-sitter nor esprima are available for TypeScript parsing"
        )
    metadata = {"language": "typescript", "filename": filename, "mode": mode}
    return LinearizationResult(tokens, metadata, symbol_table.to_mapping())


def linearize_typescript_file(path: str) -> LinearizationResult:
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    return linearize_typescript_source(source, filename=path)
