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


def test_registry_exposes_turkish_plugin() -> None:
    assert "tr" in available_plugins()


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


def test_byte_packer_integration_preserves_bytes(turkish_plugin) -> None:
    plugin = turkish_plugin
    payload = "Ankara'ya evlerden hızla geldim.".encode("utf-8")
    baseline = list(BytePacker().encode_view(payload))
    with_morph = list(BytePacker(morphology=plugin).encode_view(payload))
    assert baseline == with_morph
