"""Tests for the German ERRANT annotator.

These run the real spacy pipeline and the real dictionary, because the thing
worth testing is whether German error types come out right, and a mocked tagger
would only test the mock. The cases are the error classes the project is about:
noun capitalisation, article declension, eszett, and the das/dass confusion that
motivated the whole thing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("spacy", reason="the German annotator needs spacy and its de model")
pytest.importorskip("errant")

from langlm.eval.errant_de import annotate, tokenize
from langlm.eval.errant_de.spelling import is_known_word


def types(source: str, hypothesis: str) -> list[str]:
    return [edit.error_type for edit in annotate(source, hypothesis)]


def spans(source: str, hypothesis: str) -> list[tuple[int, int, str]]:
    return [(e.start, e.end, e.correction) for e in annotate(source, hypothesis)]


# --- spelling ---------------------------------------------------------------


def test_the_dictionary_knows_a_compound_it_has_never_seen():
    # The reason this is Hunspell and not a word list: German builds nouns on
    # demand, and a frequency list calls every one of them a typo.
    assert is_known_word("Universitätsabschlüsse")
    assert is_known_word("Studienprogrammen")


def test_a_lower_case_noun_is_not_a_word():
    # "Autos" is German; "autos" is a capitalisation error, and the classifier
    # relies on being told so.
    assert is_known_word("Autos")
    assert not is_known_word("autos")


# --- typing -----------------------------------------------------------------


def test_noun_capitalisation_is_an_orthography_error():
    assert types("Das ist ein falsches autos .", "Das ist ein falsches Autos .") == ["R:ORTH"]


def test_article_declension_is_a_determiner_form_error():
    assert types("Ich sehe der Frau .", "Ich sehe die Frau .") == ["R:DET:FORM"]


def test_eszett_is_a_spelling_error():
    assert types("Ich weiss es .", "Ich weiß es .") == ["R:SPELL"]


def test_das_dass_is_not_called_a_spelling_error():
    # They are two different words that happen to look alike, and the corpus
    # types them the same way. Calling it SPELL would put the commonest German
    # learner error in the wrong bucket for the whole of Phase 2.
    assert types("Ich weiß , das du kommst .", "Ich weiß , dass du kommst .") == ["R:OTHER"]


def test_a_missing_word_is_typed_by_what_was_missing():
    assert types("Wegen Feminismus haben alle .", "Wegen des Feminismus haben alle .") == ["M:DET"]


def test_a_superfluous_word_is_typed_by_what_was_removed():
    assert types("Ich habe sehr viel gearbeitet .", "Ich habe viel gearbeitet .") == ["U:ADV"]


def test_reordering_is_word_order():
    assert types("Obwohl es gibt Vorbilder .", "Obwohl es Vorbilder gibt .") == ["R:WO"]


def test_an_unchanged_sentence_has_no_edits():
    assert annotate("Alles ist richtig hier .", "Alles ist richtig hier .") == []


# --- spans ------------------------------------------------------------------


def test_edits_carry_source_token_offsets():
    assert spans("Ich sehe der Frau .", "Ich sehe die Frau .") == [(2, 3, "die")]


def test_an_insertion_has_an_empty_span():
    start, end, correction = spans(
        "Wegen Feminismus haben alle .", "Wegen des Feminismus haben alle ."
    )[0]
    assert start == end == 1
    assert correction == "des"


# --- tokenisation -----------------------------------------------------------


def test_free_text_is_retokenised_to_match_the_corpus():
    # A model answers in ordinary German. The corpus counts tokens.
    assert tokenize("Ich weiß, dass du kommst.") == "Ich weiß , dass du kommst ."


def test_retokenising_already_tokenised_text_leaves_it_alone():
    text = "Ich sehe die Frau ."
    assert tokenize(text) == text
