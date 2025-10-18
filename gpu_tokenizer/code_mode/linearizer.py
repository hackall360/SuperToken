"""Shared helpers for producing normalized representations of source code."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class LinearizedToken:
    """A normalized token emitted while traversing a syntax tree."""

    kind: str
    value: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LinearizationResult:
    """Container for the normalized sequence and the associated symbol table."""

    tokens: List[LinearizedToken]
    metadata: Mapping[str, Any]
    symbols: Mapping[str, str]

    def write_symbols_sidecar(self, source_path: Path | str) -> Path:
        """Persist the symbol mapping beside *source_path*.

        The resulting file is stored as ``<original>.symbols.json`` and contains
        a JSON object with placeholders as keys and the original identifiers as
        values.
        """

        path = Path(source_path)
        sidecar = path.with_suffix(path.suffix + ".symbols.json")
        sidecar.write_text(json.dumps(self.symbols, indent=2, sort_keys=False))
        return sidecar


class SymbolTable:
    """Assigns deterministic placeholder identifiers for symbol names."""

    def __init__(self) -> None:
        self._name_to_symbol: Dict[str, str] = {}
        self._symbol_to_name: Dict[str, str] = {}

    def register(self, name: str) -> str:
        """Return the placeholder for *name*, creating it if necessary."""

        if name not in self._name_to_symbol:
            placeholder = f"SYM{len(self._name_to_symbol)}"
            self._name_to_symbol[name] = placeholder
            self._symbol_to_name[placeholder] = name
        return self._name_to_symbol[name]

    def to_mapping(self) -> Mapping[str, str]:
        return dict(self._symbol_to_name)

    def extend(self, names: Iterable[str]) -> None:
        for name in names:
            self.register(name)


def write_symbol_sidecar(source_path: Path | str, symbols: Mapping[str, str]) -> Path:
    """Persist *symbols* next to *source_path* using the standard suffix."""

    path = Path(source_path)
    sidecar = path.with_suffix(path.suffix + ".symbols.json")
    sidecar.write_text(json.dumps(symbols, indent=2, sort_keys=False))
    return sidecar
