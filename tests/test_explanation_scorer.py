"""Scoring the `changes` list against the rewrite it came with.

These metrics exist because explanation quality used to be unmeasurable: the
model emitted prose, the prose was one of 69 templates, and nothing checked that
the template it chose had anything to do with the word it changed. Scoring the
list against ERRANT's reading of the model's own correction is what makes a
silent rewrite and an invented explanation two different, countable failures.
"""

from __future__ import annotations

from langlm.data.m2 import Edit, M2Sentence
from langlm.eval.explanation_scorer import ExplanationScore, score, score_sentence

SOURCE = "In meiner Erfahrung entscheidet sich die Reaktionen ."


def annotated(*edits: Edit) -> M2Sentence:
    return M2Sentence(source_tokens=SOURCE.split(), edits=list(edits))


def test_a_described_rewrite_scores_on_all_three() -> None:
    sentence = annotated(Edit(start=5, end=6, error_type="R:DET:FORM", correction="der"))
    result = score_sentence(SOURCE, sentence, [{"was": "die", "now": "der", "type": "R:DET:FORM"}])
    assert (result.coverage, result.precision, result.type_accuracy) == (1.0, 1.0, 1.0)


def test_a_silent_rewrite_costs_coverage_not_precision() -> None:
    """The dangerous failure: the learner sees a word change with no reason."""
    sentence = annotated(Edit(start=5, end=6, error_type="R:DET:FORM", correction="der"))
    result = score_sentence(SOURCE, sentence, [])
    assert result.derived == 1
    assert result.coverage == 0.0
    assert result.claimed == 0


def test_a_claim_about_a_word_it_did_not_touch_costs_precision() -> None:
    """An invented explanation. The sentence may still be perfectly corrected."""
    sentence = annotated(Edit(start=5, end=6, error_type="R:DET:FORM", correction="der"))
    result = score_sentence(
        SOURCE, sentence, [{"was": "Erfahrung", "now": "Erfahrungen", "type": "R:NOUN:FORM"}]
    )
    assert result.located == 1
    assert result.matched == 0
    assert result.precision == 0.0


def test_a_claim_quoting_text_that_is_not_there_is_not_even_located() -> None:
    sentence = annotated(Edit(start=5, end=6, error_type="R:DET:FORM", correction="der"))
    result = score_sentence(SOURCE, sentence, [{"was": "Fahrrad", "now": "Rad", "type": "R:OTHER"}])
    assert (result.located, result.matched) == (0, 0)


def test_a_widened_anchor_still_lands_on_its_edit() -> None:
    """Anchors widen to stay unique, so the window is usually bigger than the
    span. Matching on overlap rather than equality is what makes that free."""
    sentence = annotated(Edit(start=5, end=6, error_type="R:DET:FORM", correction="der"))
    result = score_sentence(
        SOURCE,
        sentence,
        [{"was": "sich die Reaktionen", "now": "sich der Reaktionen", "type": "R:DET:FORM"}],
    )
    assert result.matched == 1


def test_an_insertion_is_matched_through_the_gap_it_sits_in() -> None:
    """A zero-width edit has no token of its own to overlap."""
    sentence = annotated(Edit(start=3, end=3, error_type="M:PUNCT", correction=","))
    result = score_sentence(
        SOURCE,
        sentence,
        [{"was": "Erfahrung entscheidet", "now": "Erfahrung , entscheidet", "type": "M:PUNCT"}],
    )
    assert (result.matched, result.typed) == (1, 1)


def test_the_right_place_with_the_wrong_type_is_matched_but_not_typed() -> None:
    """This is the number the learner feels: the explanation shown is keyed on
    the type, so a matched claim with the wrong type is a wrong explanation."""
    sentence = annotated(Edit(start=5, end=6, error_type="R:DET:FORM", correction="der"))
    result = score_sentence(SOURCE, sentence, [{"was": "die", "now": "der", "type": "R:SPELL"}])
    assert (result.matched, result.typed) == (1, 0)
    assert result.type_accuracy == 0.0


def test_two_claims_on_one_edit_cover_it_once() -> None:
    """Coverage counts rewrites explained, not explanations offered."""
    sentence = annotated(Edit(start=5, end=6, error_type="R:DET:FORM", correction="der"))
    result = score_sentence(
        SOURCE,
        sentence,
        [
            {"was": "die", "now": "der", "type": "R:DET:FORM"},
            {"was": "sich die", "now": "sich der", "type": "R:DET:FORM"},
        ],
    )
    assert (result.covered, result.matched, result.claimed) == (1, 2, 2)
    assert result.coverage == 1.0


def test_scores_add_across_a_corpus() -> None:
    sentence = annotated(Edit(start=5, end=6, error_type="R:DET:FORM", correction="der"))
    total = score(
        [SOURCE, SOURCE],
        [sentence, sentence],
        [[{"was": "die", "now": "der", "type": "R:DET:FORM"}], []],
    )
    assert total == ExplanationScore(derived=2, claimed=1, located=1, matched=1, typed=1, covered=1)
    assert total.coverage == 0.5
    assert total.precision == 1.0
