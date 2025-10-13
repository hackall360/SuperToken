"""Adversarial corpora definitions for tokenizer regression testing.

This module provides a curated set of extremely small corpora that each exhibit
challenging byte-level behaviors.  The corpora are intentionally tiny so that
unit tests can iterate over them deterministically and reproducibly without any
random sampling.  The order of :data:`ADVERSARIAL_CORPORA` is fixed to guarantee
stable test parametrization.

Each corpus entry is described by :class:`AdversarialCorpus` and includes:

``name``
    Human readable identifier used for parametrized test IDs.
``description``
    Summary of the specific behavior that the corpus is meant to stress.
``corpus``
    List of UTF-8 encoded Python strings that make up the corpus.
``target_merge_operations``
    Suggested merge budget for BPE like trainers when building fixtures.
``target_vocab_size``
    Suggested vocabulary size for end-to-end tokenizer tests.

None of the corpora include leading or trailing whitespace except where such
characters are themselves part of the scenario under test.  Empty strings are
allowed and used to exercise byte-pair counters that must gracefully handle
zero-length inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pytest


@dataclass(frozen=True)
class AdversarialCorpus:
    """Container describing a deterministic adversarial corpus."""

    name: str
    description: str
    corpus: Sequence[str]
    target_merge_operations: int
    target_vocab_size: int

    def iter_corpus(self) -> Iterable[str]:
        """Return an iterator over the corpus preserving the fixed order."""

        return iter(self.corpus)


ADVERSARIAL_CORPORA: List[AdversarialCorpus] = [
    AdversarialCorpus(
        name="emoji_run",
        description=(
            "Dense emoji sequences that rely on multi-byte UTF-8 code points "
            "to ensure the trainer respects byte boundaries."
        ),
        corpus=[
            "😀😃😄😁😆😅😂🤣😊😇",
            "🤖⚙️🧪🔬🚀✨",
            "Mixed emoji and text ➕ latin",
            "",
        ],
        target_merge_operations=24,
        target_vocab_size=64,
    ),
    AdversarialCorpus(
        name="cjk_variants",
        description=(
            "CJK characters spanning simplified/traditional variants and "
            "half-width punctuation to validate byte-level normalization."
        ),
        corpus=[
            "漢字とかなのミックス",
            "繁體字與简体字的比較",
            "カタカナ・ひらがな・漢字",
            "全角スペース　と半角 space",
        ],
        target_merge_operations=32,
        target_vocab_size=96,
    ),
    AdversarialCorpus(
        name="long_repeat",
        description=(
            "Extremely long repeated substrings that create deep merge trees."
        ),
        corpus=[
            "ab" * 256,
            "xyz" * 128 + "\n" + "xyz" * 128,
            "A" * 1024,
        ],
        target_merge_operations=48,
        target_vocab_size=128,
    ),
    AdversarialCorpus(
        name="rare_pairs",
        description=(
            "Rare Unicode pairings, control characters, and spacing marks to "
            "stress byte-pair counting edge cases."
        ),
        corpus=[
            "̸̀ combining marks",
            "ZWSP​between",
            "Tab\tseparated",
            " line separator",
            "Latin–Greek–Русский mix",
        ],
        target_merge_operations=40,
        target_vocab_size=120,
    ),
]


def iter_adversarial_corpora() -> Iterable[AdversarialCorpus]:
    """Yield adversarial corpora in a deterministic order."""

    return iter(ADVERSARIAL_CORPORA)


def get_adversarial_corpora() -> Sequence[AdversarialCorpus]:
    """Return the full adversarial corpus collection.

    The returned sequence preserves the canonical ordering defined by
    :data:`ADVERSARIAL_CORPORA`.  Tests should not mutate the returned objects to
    avoid breaking reuse between parametrized cases.
    """

    return ADVERSARIAL_CORPORA


@pytest.fixture(scope="session")
def adversarial_corpora() -> Sequence[AdversarialCorpus]:
    """Pytest fixture exposing the adversarial corpora collection."""

    return get_adversarial_corpora()


@pytest.fixture(scope="session")
def adversarial_corpus_names() -> Sequence[str]:
    """Convenience fixture returning only the corpus names for parametrization."""

    return [corpus.name for corpus in get_adversarial_corpora()]
