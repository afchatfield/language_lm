"""Tests for the MaxMatch scorer.

The scorer decides whether every later claim in this project is true, so the
cases here are the ones that would silently inflate a score: a system that
describes a correct rewrite differently from the annotator, a system that
changes nothing, and a system that changes everything.
"""

from __future__ import annotations

import pytest

from langlm.data.m2 import Edit, M2Sentence
from langlm.eval.m2_scorer import Counts, extract_edits, score, sentence_counts


def toks(text: str) -> list[str]:
    """Whitespace tokens, the unit both M2 and the scorer count in."""
    return text.split()


def sentence(source: str, *edits: tuple[int, int, str, str]) -> M2Sentence:
    return M2Sentence(
        source_tokens=toks(source),
        edits=[Edit(start=s, end=e, error_type=t, correction=c) for s, e, t, c in edits],
    )


# --- counts -----------------------------------------------------------------


def test_f05_weights_precision_twice_as_heavily_as_recall():
    precise = Counts(tp=8, fp=2, fn=8)  # P=0.80 R=0.50
    lenient = Counts(tp=8, fp=8, fn=2)  # P=0.50 R=0.80
    assert precise.f(0.5) > lenient.f(0.5)
    assert precise.f(2.0) < lenient.f(2.0)
    assert precise.f(1.0) == pytest.approx(lenient.f(1.0))


def test_empty_counts_score_zero_rather_than_dividing_by_zero():
    assert Counts().f(0.5) == 0.0
    assert Counts().precision == 0.0


# --- edit extraction --------------------------------------------------------


def test_identical_sentence_yields_no_edits():
    assert extract_edits(toks("Das ist richtig ."), toks("Das ist richtig .")) == []


def test_extracts_deletion_and_insertion_the_annotator_described():
    gold = [
        Edit(start=2, end=3, error_type="U:ADV", correction=""),
        Edit(start=5, end=5, error_type="M:ADV", correction="heute"),
    ]
    edits = extract_edits(
        toks("Ich habe sehr viel gearbeitet ."),
        toks("Ich habe viel gearbeitet heute ."),
        gold,
    )
    assert (2, 3, "") in edits  # deleted "sehr"
    assert (5, 5, "heute") in edits  # inserted "heute"


def test_without_gold_the_fewest_edits_win():
    # Nothing to align to, so the same rewrite is described as one edit rather
    # than a deletion plus an insertion. This is why the scorer is always given
    # the gold edits: an unaligned description would count as a false positive
    # against an annotation that split it in two.
    edits = extract_edits(
        toks("Ich habe sehr viel gearbeitet ."),
        toks("Ich habe viel gearbeitet heute ."),
    )
    assert edits == [(2, 5, "viel gearbeitet heute")]


def test_description_bends_towards_the_gold_annotation():
    # "der Frau" -> "die Frau" can be described as a one-token or a two-token
    # edit. Both produce the same sentence, so the scorer must not punish the
    # system for the annotator's choice -- whichever choice that was.
    source = toks("Ich sehe der Frau .")
    hyp = toks("Ich sehe die Frau .")

    narrow = [Edit(start=2, end=3, error_type="R:DET:FORM", correction="die")]
    wide = [Edit(start=2, end=4, error_type="R:OTHER", correction="die Frau")]

    assert (2, 3, "die") in extract_edits(source, hyp, narrow)
    assert (2, 4, "die Frau") in extract_edits(source, hyp, wide)
    assert sentence_counts(source, hyp, narrow) == Counts(tp=1, fp=0, fn=0)
    assert sentence_counts(source, hyp, wide) == Counts(tp=1, fp=0, fn=0)


def test_an_edit_may_not_swallow_an_unbounded_run_of_matches():
    # Without a cap on unchanged tokens, a system could describe any rewrite as
    # one enormous edit and match any annotation.
    source = toks("a b c d e f g h")
    hyp = toks("x b c d e f g y")
    edits = extract_edits(source, hyp, max_unchanged=2)
    assert (0, 8, "x b c d e f g y") not in edits
    assert len(edits) == 2


# --- scoring ----------------------------------------------------------------


def test_a_system_that_changes_nothing_scores_zero():
    sentences = [sentence("Ich gehen nach Hause .", (1, 2, "R:VERB:FORM", "gehe"))]
    result = score(sentences, ["Ich gehen nach Hause ."])
    assert result.counts == Counts(tp=0, fp=0, fn=1)
    assert result.f_score == 0.0


def test_a_correct_system_scores_one():
    sentences = [sentence("Ich gehen nach Hause .", (1, 2, "R:VERB:FORM", "gehe"))]
    assert score(sentences, ["Ich gehe nach Hause ."]).f_score == 1.0


def test_gratuitous_changes_are_false_positives():
    # The overcorrection case in miniature: the sentence was already correct.
    sentences = [sentence("Alles richtig hier .")]
    result = score(sentences, ["Alles falsch hier ."])
    assert result.counts == Counts(tp=0, fp=1, fn=0)
    assert result.precision == 0.0


def test_noop_edits_are_not_scoreable_targets():
    sentences = [sentence("Alles richtig hier .", (-1, -1, "noop", "-NONE-"))]
    result = score(sentences, ["Alles richtig hier ."])
    assert result.counts == Counts(tp=0, fp=0, fn=0)


def test_the_best_annotator_is_chosen_per_sentence():
    # Annotator 1 left the sentence alone; annotator 0 corrected it. A system
    # that corrects it should be scored against the annotator who agrees.
    both = M2Sentence(
        source_tokens=toks("Ich gehen nach Hause ."),
        edits=[
            Edit(start=1, end=2, error_type="R:VERB:FORM", correction="gehe", annotator=0),
            Edit(start=-1, end=-1, error_type="noop", correction="-NONE-", annotator=1),
        ],
    )
    assert score([both], ["Ich gehe nach Hause ."]).f_score == 1.0


def test_mismatched_lengths_are_rejected_rather_than_silently_truncated():
    with pytest.raises(ValueError, match="one to one"):
        score([sentence("Alles richtig hier .")], [])
