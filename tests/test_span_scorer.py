"""Tests for span-level scoring and the per-error-type breakdown.

The scorer's job is to say not just how well a system did but what it did well,
so the cases here check the bookkeeping that supports that: a hit is filed under
the annotator's type, a false positive under the system's own, and detection is
scored separately from correction.
"""

from __future__ import annotations

import pytest

from langlm.data.m2 import Edit, M2Sentence
from langlm.eval.m2_scorer import Counts
from langlm.eval.span_scorer import score


def sentence(source: str, *edits: tuple[int, int, str, str]) -> M2Sentence:
    return M2Sentence(
        source_tokens=source.split(),
        edits=[Edit(start=s, end=e, error_type=t, correction=c) for s, e, t, c in edits],
    )


GOLD = [
    sentence(
        "Ich sehe der Frau .",
        (2, 3, "R:DET:FORM", "die"),
        (1, 2, "R:VERB:FORM", "sah"),
    ),
    sentence("Alles richtig hier .", (-1, -1, "noop", "-NONE-")),
]


def test_a_perfect_hypothesis_scores_one():
    result = score(GOLD, GOLD)
    assert result.overall == Counts(tp=2, fp=0, fn=0)
    assert result.f_score == 1.0


def test_hits_are_filed_under_the_gold_type():
    hypothesis = [
        sentence("Ich sehe der Frau .", (2, 3, "R:DET", "die")),
        sentence("Alles richtig hier ."),
    ]
    result = score(GOLD, hypothesis)
    # The system called it R:DET and the annotator called it R:DET:FORM. It is
    # still the same correction, and it belongs in the annotator's bucket.
    assert result.by_type["R:DET:FORM"] == Counts(tp=1, fp=0, fn=0)
    assert "R:DET" not in result.by_type


def test_false_positives_are_filed_under_the_systems_own_type():
    hypothesis = [
        sentence("Ich sehe der Frau .", (2, 3, "R:DET:FORM", "die")),
        sentence("Alles richtig hier .", (1, 2, "R:ADJ", "falsch")),
    ]
    result = score(GOLD, hypothesis)
    assert result.by_type["R:ADJ"] == Counts(tp=0, fp=1, fn=0)


def test_detection_forgives_the_wrong_fix():
    hypothesis = [
        sentence("Ich sehe der Frau .", (2, 3, "R:DET:FORM", "das")),
        sentence("Alles richtig hier ."),
    ]
    correction = score(GOLD, hypothesis, mode="correction")
    detection = score(GOLD, hypothesis, mode="detection")
    # Right span, wrong article: found by detection, missed by correction.
    assert correction.overall.tp == 0
    assert detection.overall.tp == 1


def test_noop_annotations_are_not_scoreable():
    result = score(GOLD, [sentence("Ich sehe der Frau ."), sentence("Alles richtig hier .")])
    assert result.overall == Counts(tp=0, fp=0, fn=2)


def test_ranked_types_hides_the_long_tail():
    hypothesis = [sentence("Ich sehe der Frau ."), sentence("Alles richtig hier .")]
    result = score(GOLD, hypothesis)
    assert [t for t, _ in result.ranked_types(minimum=2)] == []
    assert len(result.ranked_types(minimum=1)) == 2


def test_annotations_of_different_sentences_are_rejected():
    with pytest.raises(ValueError, match="different sentences"):
        score(GOLD, [sentence("Etwas ganz anderes ."), sentence("Alles richtig hier .")])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown mode"):
        score(GOLD, GOLD, mode="vibes")
