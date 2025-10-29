"""Japanese morphology plugin annotating script transitions."""

from __future__ import annotations

from typing import Iterable, List

from . import MorphologyPlugin, MorphologySegment, register_plugin


def _to_bytes(payload: bytes | bytearray | memoryview) -> bytes:
    if isinstance(payload, memoryview):
        view = payload
        if view.ndim != 1 or view.format != "B":
            view = view.cast("B")
        return bytes(view)
    return bytes(payload)


def _classify_char(ch: str) -> tuple[str | None, str]:
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return ("SCRIPT=KANJI", "lexeme")
    if 0x3040 <= cp <= 0x309F:
        return ("SCRIPT=HIRAGANA", "lexeme")
    if 0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF or 0xFF66 <= cp <= 0xFF9D:
        return ("SCRIPT=KATAKANA", "lexeme")
    if ch.isascii() and ch.isalpha():
        return ("SCRIPT=LATIN", "lexeme")
    if ch.isdigit():
        return ("SCRIPT=NUMERIC", "lexeme")
    if ch.isspace():
        return ("CATEGORY=SPACE", "separator")
    return ("CATEGORY=PUNCT", "separator")


class JapaneseMorphologyPlugin(MorphologyPlugin):
    """Segment Japanese text by script blocks while keeping surfaces intact."""

    def presegment(self, sequence: bytes | bytearray | memoryview) -> Iterable[MorphologySegment]:
        raw = _to_bytes(sequence)
        if not raw:
            return []
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return [MorphologySegment(raw)]

        segments: List[MorphologySegment] = []
        buffer: List[str] = []
        buffer_tag: str | None = None

        def flush() -> None:
            nonlocal buffer, buffer_tag
            if buffer:
                surface = "".join(buffer).encode("utf-8")
                segments.append(
                    MorphologySegment(surface, tags=(buffer_tag,) if buffer_tag else (), role="lexeme")
                )
                buffer = []
                buffer_tag = None

        for ch in text:
            tag, role = _classify_char(ch)
            if role == "lexeme" and tag:
                if buffer_tag == tag:
                    buffer.append(ch)
                else:
                    flush()
                    buffer.append(ch)
                    buffer_tag = tag
            else:
                flush()
                segments.append(
                    MorphologySegment(
                        ch.encode("utf-8"),
                        tags=(tag,) if tag else (),
                        role="separator" if role == "separator" else None,
                    )
                )
        flush()
        return segments


register_plugin("ja", JapaneseMorphologyPlugin)

__all__ = ["JapaneseMorphologyPlugin"]
