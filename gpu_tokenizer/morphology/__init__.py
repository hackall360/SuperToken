"""Morphology preprocessing plugins used ahead of byte packing."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

BytesLike = bytes | bytearray | memoryview


@dataclass(frozen=True)
class MorphologySegment:
    """Container representing a morphologically annotated byte span."""

    surface: bytes
    tags: tuple[str, ...] = ()
    role: str | None = None

    def __post_init__(self) -> None:
        surface = self.surface
        if isinstance(surface, memoryview):
            view = surface
            if view.ndim != 1 or view.format != "B":
                view = view.cast("B")
            surface = view.tobytes()
        elif not isinstance(surface, (bytes, bytearray)):
            surface = bytes(surface)
        object.__setattr__(self, "surface", bytes(surface))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags if tag))
        if self.role is not None:
            object.__setattr__(self, "role", str(self.role))

    def to_bytes(self) -> bytes:
        """Return the surface representation as raw bytes."""

        return self.surface

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        tags = ",".join(self.tags) if self.tags else ""
        role = f" role={self.role!r}" if self.role else ""
        return f"MorphologySegment({self.surface!r}, tags={tags!r}{role})"


class MorphologyPlugin(abc.ABC):
    """Abstract base class for morphology preprocessing hooks."""

    def __init__(self, *, case_markers: bool = True, affix_tags: bool = True) -> None:
        self.case_markers = bool(case_markers)
        self.affix_tags = bool(affix_tags)

    @abc.abstractmethod
    def presegment(self, sequence: BytesLike) -> Iterable[MorphologySegment]:
        """Return an iterable of annotated segments covering *sequence*."""

    def iter_surfaces(self, sequence: BytesLike) -> Iterator[bytes]:
        """Yield surface byte spans for :class:`BytePacker` consumption."""

        for segment in self.presegment(sequence):
            if isinstance(segment, MorphologySegment):
                yield segment.surface
            else:  # pragma: no cover - defensive
                yield bytes(segment)

    def recompose(self, segments: Sequence[MorphologySegment]) -> bytes:
        """Reconstruct the original byte payload from *segments*."""

        return b"".join(segment.surface for segment in segments)


_REGISTRY: dict[str, type[MorphologyPlugin]] = {}


def register_plugin(name: str, cls: type[MorphologyPlugin]) -> None:
    """Register *cls* under *name* for discovery."""

    key = name.strip().lower()
    if not key:
        raise ValueError("Plugin names must be non-empty")
    if not issubclass(cls, MorphologyPlugin):
        raise TypeError("Registered class must subclass MorphologyPlugin")
    if key in _REGISTRY:
        raise ValueError(f"Morphology plugin '{name}' already registered")
    _REGISTRY[key] = cls


def available_plugins() -> tuple[str, ...]:
    """Return sorted plugin identifiers."""

    return tuple(sorted(_REGISTRY))


def create_plugin(name: str, **config: object) -> MorphologyPlugin:
    """Instantiate the plugin registered under *name* with *config*."""

    key = name.strip().lower()
    try:
        cls = _REGISTRY[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown morphology plugin: {name}") from exc
    return cls(**config)


__all__ = [
    "MorphologyPlugin",
    "MorphologySegment",
    "available_plugins",
    "create_plugin",
    "register_plugin",
]

# Ensure built-in plugins are registered on import.
try:  # pragma: no cover - import side effect
    from . import turkish as _turkish  # noqa: F401
except Exception:  # pragma: no cover - defensive fallback
    pass
