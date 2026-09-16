"""Tests for the softened equality and for the scorers that accept it.

The risk being tested for is not that leniency is too strict but that it is too
generous: a relation that admits one word too many turns the correction score
into the detection score, and the whole point of reporting both is that they
differ. So most of these cases assert a refusal.
"""

from __future__ import annotations

import pytest

from langlm.data.m2 import Edit, M2Sentence
from langlm.eval import span_scorer
from langlm.eval.leniency import Leniency, same_compound, typographic
from langlm.eval.m2_scorer import Counts
from langlm.eval.m2_scorer import score as m2_score

#: A stand-in thesaurus, so the tests do not need the downloaded file.
SYNONYMS = {
    "ökonomie": frozenset({"wirtschaft"}),
    "wirtschaft": frozenset({"ökonomie"}),
    "druck": frozenset({"zwang"}),
    "zwang": frozenset({"druck"}),
}

LENIENT = Leniency(synonyms=SYNONYMS, compounds=True)
SHAPE_ONLY = Leniency(synonyms=None, compounds=True)


# --- the tiers --------------------------------------------------------------


def test_typographic_folds_what_a_writer_did_not_choose():
    assert typographic("„Zitat“") == typographic('"Zitat"')
    assert typographic("a\u00a0b") == "a b"


def test_compound_variants_are_the_same_word():
    assert same_compound("Chemiestudentin", "Chemie-Studentin")
    assert same_compound("Informatik-Lehrerin", "Informatiklehrerin")


def test_a_spaced_compound_is_the_error_not_a_variant():
    # `Kartoffel Suppe` is what the learner wrote; forgiving it would credit the
    # system for leaving the mistake in place.
    assert not same_compound("Kartoffelsuppe", "Kartoffel Suppe")


def test_two_closed_compounds_are_not_variants_of_each_other():
    assert not same_compound("Kinderzimmer", "Kindzimmer")


def test_synonyms_are_accepted_for_content_words():
    assert LENIENT.equivalent("Ökonomie", "Wirtschaft")
    assert LENIENT.equivalent("Druck", "Zwang")


def test_an_unrelated_word_is_not_a_synonym():
    assert not LENIENT.equivalent("Verbrechen", "Vergehen")
    assert not LENIENT.equivalent("lieben", "leben")


def test_the_synonym_tier_is_off_without_a_thesaurus():
    assert not SHAPE_ONLY.equivalent("Ökonomie", "Wirtschaft")


def test_a_different_inflection_is_still_an_error():
    # The whole task is getting the form right, so a form is never forgiven --
    # not even one built on a word the thesaurus lists.
    assert not LENIENT.equivalent("Druck", "Zwänge")


def test_strict_leniency_is_plain_equality():
    strict = Leniency.strict()
    assert strict.equivalent("die", "die")
    assert not strict.equivalent("Chemiestudentin", "Chemie-Studentin")


# --- the scorers ------------------------------------------------------------


def sentence(source: str, *edits: tuple[int, int, str, str]) -> M2Sentence:
    return M2Sentence(
        source_tokens=source.split(),
        edits=[Edit(start=s, end=e, error_type=t, correction=c) for s, e, t, c in edits],
    )


GOLD = [sentence("Ich mache Kartoffel Suppe .", (2, 4, "R:ORTH", "Kartoffelsuppe"))]


def test_m2_credits_a_compound_variant_only_when_lenient():
    hypothesis = ["Ich mache Kartoffel-Suppe ."]
    assert m2_score(GOLD, hypothesis).counts == Counts(tp=0, fp=1, fn=1)
    assert m2_score(GOLD, hypothesis, equivalent=SHAPE_ONLY).counts == Counts(tp=1, fp=0, fn=0)


def test_leniency_does_not_license_a_wrong_span():
    # The replacement is equivalent but sits somewhere else, so it is still a
    # false positive: softening the text must not soften the location.
    gold = [sentence("Ich mache Suppe und Salat .", (2, 3, "R:NOUN", "Ökonomie"))]
    hypothesis = ["Ich mache Suppe und Wirtschaft ."]
    assert m2_score(gold, hypothesis, equivalent=LENIENT).counts == Counts(tp=0, fp=1, fn=1)


def test_span_scorer_files_a_lenient_hit_under_the_gold_type():
    hyp = [sentence("Ich mache Kartoffel Suppe .", (2, 4, "R:OTHER", "Kartoffel-Suppe"))]
    result = span_scorer.score(GOLD, hyp, equivalent=SHAPE_ONLY)
    assert result.overall == Counts(tp=1, fp=0, fn=0)
    assert result.by_type["R:ORTH"] == Counts(tp=1)


def test_detection_mode_ignores_leniency():
    # Detection already disregards the replacement, so a lenient run must return
    # exactly the strict counts rather than double-crediting anything.
    hyp = [sentence("Ich mache Kartoffel Suppe .", (2, 4, "R:OTHER", "Kartoffel-Suppe"))]
    strict = span_scorer.score(GOLD, hyp, mode="detection")
    lenient = span_scorer.score(GOLD, hyp, mode="detection", equivalent=LENIENT)
    assert strict.overall == lenient.overall


def test_one_gold_edit_is_credited_once():
    # Two hypothesis edits on one span, both equivalent to the single gold edit.
    # Crediting both would push true positives past the number of gold edits.
    gold = [sentence("Ich sehe Frau .", (2, 2, "M:NOUN", "Druck"))]
    hyp = [
        M2Sentence(
            source_tokens=["Ich", "sehe", "Frau", "."],
            edits=[
                Edit(start=2, end=2, error_type="M:NOUN", correction="Zwang"),
                Edit(start=2, end=2, error_type="M:NOUN", correction="Druck"),
            ],
        )
    ]
    result = span_scorer.score(gold, hyp, equivalent=LENIENT)
    assert result.overall.tp == 1
    assert result.overall.fn == 0


@pytest.mark.parametrize("mode", ["correction", "detection"])
def test_leniency_never_lowers_a_score(mode: str):
    hyp = [sentence("Ich mache Kartoffel Suppe .", (2, 4, "R:OTHER", "Kartoffel-Suppe"))]
    strict = span_scorer.score(GOLD, hyp, mode=mode)
    lenient = span_scorer.score(GOLD, hyp, mode=mode, equivalent=LENIENT)
    assert lenient.f_score >= strict.f_score
