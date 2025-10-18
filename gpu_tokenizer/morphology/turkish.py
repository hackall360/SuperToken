"""Turkish morphology plugin providing heuristic affix annotation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from . import MorphologyPlugin, MorphologySegment, register_plugin


@dataclass(frozen=True)
class _SuffixPattern:
    suffix: str
    features: tuple[str, ...]
    category: str  # "case" or "affix"
    repeatable: bool = False


_WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû']+", re.UNICODE)


def _to_bytes(payload: bytes | bytearray | memoryview) -> bytes:
    if isinstance(payload, memoryview):
        view = payload
        if view.ndim != 1 or view.format != "B":
            view = view.cast("B")
        return bytes(view)
    return bytes(payload)


class TurkishMorphologyPlugin(MorphologyPlugin):
    """Annotate common Turkish suffixes without mutating the source text."""

    _SUFFIX_PATTERNS: tuple[_SuffixPattern, ...] = (
        _SuffixPattern("lardan", ("AFFIX=PL", "CASE=ABL"), "case"),
        _SuffixPattern("lerden", ("AFFIX=PL", "CASE=ABL"), "case"),
        _SuffixPattern("larda", ("AFFIX=PL", "CASE=LOC"), "case"),
        _SuffixPattern("lerde", ("AFFIX=PL", "CASE=LOC"), "case"),
        _SuffixPattern("lara", ("AFFIX=PL", "CASE=DAT"), "case"),
        _SuffixPattern("lere", ("AFFIX=PL", "CASE=DAT"), "case"),
        _SuffixPattern("dan", ("CASE=ABL",), "case"),
        _SuffixPattern("den", ("CASE=ABL",), "case"),
        _SuffixPattern("tan", ("CASE=ABL",), "case"),
        _SuffixPattern("ten", ("CASE=ABL",), "case"),
        _SuffixPattern("yla", ("CASE=INS",), "case"),
        _SuffixPattern("yle", ("CASE=INS",), "case"),
        _SuffixPattern("la", ("CASE=INS",), "case"),
        _SuffixPattern("le", ("CASE=INS",), "case"),
        _SuffixPattern("na", ("CASE=DAT",), "case"),
        _SuffixPattern("ne", ("CASE=DAT",), "case"),
        _SuffixPattern("ya", ("CASE=DAT",), "case"),
        _SuffixPattern("ye", ("CASE=DAT",), "case"),
        _SuffixPattern("a", ("CASE=DAT",), "case"),
        _SuffixPattern("e", ("CASE=DAT",), "case"),
        _SuffixPattern("nın", ("CASE=GEN",), "case"),
        _SuffixPattern("nin", ("CASE=GEN",), "case"),
        _SuffixPattern("nun", ("CASE=GEN",), "case"),
        _SuffixPattern("nün", ("CASE=GEN",), "case"),
        _SuffixPattern("ın", ("CASE=GEN",), "case"),
        _SuffixPattern("in", ("CASE=GEN",), "case"),
        _SuffixPattern("un", ("CASE=GEN",), "case"),
        _SuffixPattern("ün", ("CASE=GEN",), "case"),
        _SuffixPattern("ı", ("CASE=ACC",), "case"),
        _SuffixPattern("i", ("CASE=ACC",), "case"),
        _SuffixPattern("u", ("CASE=ACC",), "case"),
        _SuffixPattern("ü", ("CASE=ACC",), "case"),
        _SuffixPattern("lar", ("AFFIX=PL",), "affix", repeatable=False),
        _SuffixPattern("ler", ("AFFIX=PL",), "affix", repeatable=False),
        _SuffixPattern("cı", ("AFFIX=AGENT",), "affix"),
        _SuffixPattern("ci", ("AFFIX=AGENT",), "affix"),
        _SuffixPattern("cu", ("AFFIX=AGENT",), "affix"),
        _SuffixPattern("cü", ("AFFIX=AGENT",), "affix"),
        _SuffixPattern("siz", ("AFFIX=WITHOUT",), "affix"),
        _SuffixPattern("sız", ("AFFIX=WITHOUT",), "affix"),
        _SuffixPattern("suz", ("AFFIX=WITHOUT",), "affix"),
        _SuffixPattern("süz", ("AFFIX=WITHOUT",), "affix"),
        _SuffixPattern("miş", ("AFFIX=PAST",), "affix"),
        _SuffixPattern("mış", ("AFFIX=PAST",), "affix"),
        _SuffixPattern("muş", ("AFFIX=PAST",), "affix"),
        _SuffixPattern("müş", ("AFFIX=PAST",), "affix"),
    )

    def presegment(self, sequence: bytes | bytearray | memoryview) -> Iterable[MorphologySegment]:
        raw = _to_bytes(sequence)
        if not raw:
            return []
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return [MorphologySegment(raw)]

        segments: List[MorphologySegment] = []
        last = 0
        for match in _WORD_RE.finditer(text):
            start, end = match.span()
            if start > last:
                filler = text[last:start]
                if filler:
                    segments.append(MorphologySegment(filler.encode("utf-8")))
            segments.extend(self._segment_word(match.group(0)))
            last = end
        if last < len(text):
            tail = text[last:]
            if tail:
                segments.append(MorphologySegment(tail.encode("utf-8")))
        return segments

    def _segment_word(self, word: str) -> List[MorphologySegment]:
        lowered = word.lower()
        remainder = word
        remainder_lower = lowered
        suffixes: List[MorphologySegment] = []
        for pattern in self._SUFFIX_PATTERNS:
            if pattern.category == "case" and not self.case_markers:
                continue
            if pattern.category == "affix" and not self.affix_tags:
                continue
            while remainder_lower.endswith(pattern.suffix):
                if len(remainder_lower) <= len(pattern.suffix):
                    break
                surface = remainder[-len(pattern.suffix) :]
                remainder = remainder[: -len(pattern.suffix)]
                remainder_lower = remainder_lower[: -len(pattern.suffix)]
                suffixes.append(
                    MorphologySegment(
                        surface.encode("utf-8"),
                        tags=pattern.features,
                        role=pattern.category,
                    )
                )
                if not pattern.repeatable:
                    break
        if not suffixes:
            return [MorphologySegment(word.encode("utf-8"))]
        if not remainder:
            return [MorphologySegment(word.encode("utf-8"))]
        result = [MorphologySegment(remainder.encode("utf-8"))]
        result.extend(reversed(suffixes))
        return result


register_plugin("tr", TurkishMorphologyPlugin)

__all__ = ["TurkishMorphologyPlugin"]
