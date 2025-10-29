from __future__ import annotations

import pytest

from gpu_tokenizer.cpu_packer import BytePacker
from gpu_tokenizer.morphology import (
    MorphologyPlugin,
    MorphologySegment,
    available_plugins,
    create_plugin,
)


@pytest.fixture(scope="module")
def turkish_plugin() -> MorphologyPlugin:
    return create_plugin("tr", case_markers=True, affix_tags=True)


@pytest.fixture(scope="module")
def japanese_plugin() -> MorphologyPlugin:
    return create_plugin("ja")


@pytest.fixture(scope="module")
def korean_plugin() -> MorphologyPlugin:
    return create_plugin("ko")


def test_registry_exposes_turkish_plugin() -> None:
    assert "tr" in available_plugins()


def test_registry_exposes_japanese_and_korean_plugins() -> None:
    registry = available_plugins()
    assert "ja" in registry
    assert "ko" in registry


def test_turkish_segmentation_round_trip(turkish_plugin) -> None:
    plugin = turkish_plugin
    payload = "Ankara'ya evlerden hızla geldim ve kedilerle konuştum.".encode("utf-8")
    segments = list(plugin.presegment(payload))
    assert segments, "expected segments to be produced"
    recomposed = plugin.recompose(segments)
    assert recomposed == payload
    tag_lookup = {(seg.surface, seg.tags) for seg in segments if isinstance(seg, MorphologySegment)}
    assert any(b"ya" == surface and "CASE=DAT" in tags for surface, tags in tag_lookup)
    assert any(b"ler" == surface and "AFFIX=PL" in tags for surface, tags in tag_lookup)
    assert any(
        b"lerden" == surface
        and "CASE=ABL" in tags
        and "AFFIX=PL" in tags
        for surface, tags in tag_lookup
    )


def test_turkish_plugin_respects_disabled_flags() -> None:
    sample = "Ankara'ya".encode("utf-8")
    with_case = create_plugin("tr", case_markers=True, affix_tags=True)
    without_case = create_plugin("tr", case_markers=False, affix_tags=True)
    segments_with = list(with_case.presegment(sample))
    segments_without = list(without_case.presegment(sample))
    assert any(seg.tags for seg in segments_with)
    assert all(not seg.tags for seg in segments_without)


def test_japanese_segmentation_round_trip(japanese_plugin) -> None:
    plugin = japanese_plugin
    payload = "東京でAIを学ぶ。".encode("utf-8")
    segments = list(plugin.presegment(payload))
    assert segments, "expected segments to be produced"
    assert plugin.recompose(segments) == payload
    tag_lookup = {
        (seg.surface, seg.tags, seg.role)
        for seg in segments
        if isinstance(seg, MorphologySegment)
    }
    assert any(surface == "東京".encode("utf-8") and "SCRIPT=KANJI" in tags for surface, tags, _ in tag_lookup)
    assert any(surface == "で".encode("utf-8") and "SCRIPT=HIRAGANA" in tags for surface, tags, _ in tag_lookup)
    assert any(surface == b"AI" and "SCRIPT=LATIN" in tags for surface, tags, _ in tag_lookup)
    assert any(
        surface == "。".encode("utf-8")
        and "CATEGORY=PUNCT" in tags
        and role == "separator"
        for surface, tags, role in tag_lookup
    )


def test_korean_segmentation_round_trip(korean_plugin) -> None:
    plugin = korean_plugin
    payload = "서울에서 인공지능을 공부합니다!".encode("utf-8")
    segments = list(plugin.presegment(payload))
    assert segments, "expected segments to be produced"
    assert plugin.recompose(segments) == payload
    tag_lookup = {
        (seg.surface, seg.tags, seg.role)
        for seg in segments
        if isinstance(seg, MorphologySegment)
    }
    assert any(surface == "서울에서".encode("utf-8") and "SCRIPT=HANGUL" in tags for surface, tags, _ in tag_lookup)
    assert any(surface == "인공지능을".encode("utf-8") and "SCRIPT=HANGUL" in tags for surface, tags, _ in tag_lookup)
    assert any(
        surface == "!".encode("utf-8")
        and "CATEGORY=PUNCT" in tags
        and role == "separator"
        for surface, tags, role in tag_lookup
    )


def test_byte_packer_integration_preserves_bytes(turkish_plugin) -> None:
    plugin = turkish_plugin
    payload = "Ankara'ya evlerden hızla geldim.".encode("utf-8")
    baseline = list(BytePacker().encode_view(payload))
    with_morph = list(BytePacker(morphology=plugin).encode_view(payload))
    assert baseline == with_morph
