"""Tests for canonical Unicode equivalence at the scoring boundary.

Spanish learner text contains both spellings of an accented vowel, and the
scorer compares strings, so without this the same word in two encodings reads
as a correction. Every case here is one a real COWS-L2H dev sentence produces.
"""

from __future__ import annotations

import unicodedata

from langlm.data.m2 import Edit, M2Sentence
from langlm.eval import canonical, overcorrection
from langlm.eval.m2_scorer import score

#: `tío` with a combining acute accent, exactly as COWS-L2H stores it for the
#: learners who typed it that way.
DECOMPOSED = unicodedata.normalize("NFD", "tío")
PRECOMPOSED = "tío"


def test_the_two_spellings_really_are_different_strings():
    """The premise. If this ever fails the rest of the module is unnecessary."""
    assert DECOMPOSED != PRECOMPOSED
    assert len(DECOMPOSED) == 4 and len(PRECOMPOSED) == 3
    assert canonical.nfc(DECOMPOSED) == PRECOMPOSED


def test_normalising_a_copied_sentence_is_not_scored_as_an_edit():
    """A model that writes precomposed accents copies a decomposed source
    through unchanged in every way a reader can see. It must not be charged a
    false positive for the encoding."""
    gold = M2Sentence(source_tokens=[f"mi {DECOMPOSED}".split()[0], DECOMPOSED])
    hypothesis = f"mi {PRECOMPOSED}"

    result = score([gold], [hypothesis])
    assert result.counts.fp == 0


def test_a_correction_still_counts_when_the_source_is_decomposed():
    """The edit is real; only the encoding of the untouched word differs."""
    gold = M2Sentence(
        source_tokens=["el", DECOMPOSED, "comio"],
        edits=[Edit(start=2, end=3, error_type="R:ORTH", correction="comió")],
    )

    result = score([gold], [f"el {PRECOMPOSED} comió"])
    assert (result.counts.tp, result.counts.fp, result.counts.fn) == (1, 0, 0)


def test_a_decomposed_gold_correction_is_matched_by_a_precomposed_one():
    """Nine dev sentences carry gold targets in decomposed form; a system
    writing normal Spanish must be able to match them."""
    gold = M2Sentence(
        source_tokens=["el", "tio"],
        edits=[Edit(start=1, end=2, error_type="R:ORTH", correction=DECOMPOSED)],
    )

    result = score([gold], [f"el {PRECOMPOSED}"])
    assert (result.counts.tp, result.counts.fn) == (1, 0)


def test_overcorrection_does_not_charge_for_normalising():
    result = overcorrection.measure([f"mi {DECOMPOSED}"], [f"mi {PRECOMPOSED}"], language="es")
    assert result.modified == 0
    assert result.edits == 0


def test_real_edits_are_still_counted_as_overcorrection():
    """The guard above must not blind the measure to an actual rewrite."""
    result = overcorrection.measure([f"mi {DECOMPOSED}"], ["su tío"], language="es")
    assert result.modified == 1


def test_spans_survive_normalisation():
    """Normalisation changes code points, never token counts, so edit offsets
    stay valid -- which is what makes normalising a gold sentence safe."""
    sentence = M2Sentence(
        source_tokens=["uno", DECOMPOSED, "tres"],
        edits=[Edit(start=1, end=2, error_type="R:OTHER", correction=DECOMPOSED)],
    )
    normalised = canonical.nfc_sentence(sentence)

    assert len(normalised.source_tokens) == len(sentence.source_tokens)
    assert normalised.source_tokens[1] == PRECOMPOSED
    assert normalised.edits[0].correction == PRECOMPOSED
    assert (normalised.edits[0].start, normalised.edits[0].end) == (1, 2)


def test_leaves_text_that_is_already_normalised_untouched():
    """German is entirely precomposed, so this path is every German score."""
    sentence = M2Sentence(
        source_tokens=["Die", "Universität", "schließt"],
        edits=[Edit(start=2, end=3, error_type="R:VERB", correction="öffnet")],
    )
    normalised = canonical.nfc_sentence(sentence)

    assert normalised.source_tokens == sentence.source_tokens
    assert normalised.edits[0].correction == sentence.edits[0].correction
