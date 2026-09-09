"""Tests for the overcorrection set and the metric it supports.

The metric is one number -- the share of correct sentences a system changed --
so the tests are mostly about the filters that decide what counts as a correct
sentence, which is where the number could quietly become dishonest.
"""

from __future__ import annotations

from langlm.data.overcorrection import is_clean
from langlm.eval.overcorrection import measure

# --- what counts as clean prose ---------------------------------------------


def test_a_whole_sentence_is_kept():
    assert is_clean("Drei Tage lang wurde beim Musikverein in Reichenbach gefeiert.")


def test_a_fragment_cut_out_of_a_longer_sentence_is_rejected():
    assert not is_clean("und dann ging er nach Hause.")  # starts lower case
    assert not is_clean("Der Mann ging nach")  # no final punctuation


def test_web_furniture_is_rejected():
    assert not is_clean("Mehr dazu unter http://example.de lesen Sie hier.")
    assert not is_clean("Der Mann [sic] ging nach Hause.")


def test_unbalanced_quotes_mean_the_sentence_was_cut():
    assert not is_clean('Er sagte "Ich komme morgen und ging.')
    assert is_clean('Er sagte "Ich komme morgen" und ging.')


def test_headlines_in_capitals_are_not_prose():
    assert not is_clean("DIE LAGE IST ERNST.")


# --- the metric -------------------------------------------------------------


def test_a_system_that_changes_nothing_scores_zero():
    sources = ["Alles richtig hier .", "Auch das ist richtig ."]
    result = measure(sources, list(sources))
    assert result.rate == 0.0
    assert result.modified == 0
    assert result.changes == []


def test_every_altered_sentence_is_counted_and_kept():
    sources = ["Alles richtig hier .", "Auch das ist richtig ."]
    hypotheses = ["Alles falsch hier .", "Auch das ist richtig ."]
    result = measure(sources, hypotheses)
    assert result.modified == 1
    assert result.rate == 0.5
    assert result.changes[0].source == sources[0]
    assert result.changes[0].hypothesis == hypotheses[0]


def test_only_the_tokens_matter_not_the_spacing():
    # Systems are re-tokenised before they get here, so a difference in
    # whitespace alone is the harness talking, not the system.
    result = measure(["Alles richtig hier ."], ["Alles  richtig   hier ."])
    assert result.modified == 0
