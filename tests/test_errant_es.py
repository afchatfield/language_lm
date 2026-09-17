"""Tests for the Spanish ERRANT annotator.

These run the real spacy pipeline and the real dictionary, mirroring
`test_errant_de.py`'s reasoning exactly: the thing worth testing is whether
Spanish error types come out right, and a mocked tagger would only test the
mock. The cases are the error classes Phase 5's port had to get right: the
stress mark as an orthography axis in place of German's capitalisation, and
gender/number agreement in place of case.
"""

from __future__ import annotations

import pytest

pytest.importorskip("spacy", reason="the Spanish annotator needs spacy and its es model")
pytest.importorskip("errant")

from langlm.eval.errant_es import annotate, tokenize
from langlm.eval.errant_es.spelling import is_known_word


def types(source: str, hypothesis: str) -> list[str]:
    return [edit.error_type for edit in annotate(source, hypothesis)]


def spans(source: str, hypothesis: str) -> list[tuple[int, int, str]]:
    return [(e.start, e.end, e.correction) for e in annotate(source, hypothesis)]


# --- spelling ---------------------------------------------------------------


def test_the_dictionary_knows_conjugated_forms_it_has_never_seen():
    # The reason this is Hunspell and not a word list: Spanish conjugates a
    # verb into dozens of forms, and a frequency list calls the rare ones typos.
    assert is_known_word("tenía")
    assert is_known_word("sabías")


def test_an_unaccented_word_needing_a_tilde_is_not_a_word():
    # "jardín" is Spanish; "jardin" is a missing stress mark, and the
    # classifier relies on being told so -- the same role case plays for German.
    assert is_known_word("jardín")
    assert not is_known_word("jardin")


# --- typing -----------------------------------------------------------------


def test_a_stress_mark_is_an_orthography_error():
    assert types("Esta muy cansado hoy .", "Está muy cansado hoy .") == ["R:ORTH"]


def test_tilde_is_not_folded_as_an_accent():
    # The failure this guards: Unicode decomposition (NFKD) strips the tilde
    # off "ñ" as if it were a stress mark, which would call "año" (year) and
    # "ano" (anus) the same spelling. They are different letters, not the same
    # word stressed differently, and must type as a real word substitution.
    assert types("El ano pasado fui a Peru .", "El año pasado fui a Perú .") == [
        "R:NOUN",
        "R:ORTH",  # Peru -> Perú *is* a missing stress mark, correctly ORTH
    ]


def test_subject_verb_agreement_is_a_verb_form_error():
    assert types("Yo tiene un perro .", "Yo tengo un perro .") == ["R:VERB:FORM"]


def test_gender_agreement_is_a_determiner_form_error():
    assert types("La problema es grave .", "El problema es grave .") == ["R:DET:FORM"]


def test_a_missing_word_is_typed_by_what_was_missing():
    assert types("Vamos a playa .", "Vamos a la playa .") == ["M:DET"]


def test_a_superfluous_word_is_typed_by_what_was_removed():
    assert types("Quiero mucho un perro .", "Quiero un perro .") == ["U:ADV"]


def test_reordering_is_word_order():
    assert types("Necesito comprar libros nuevos .", "Necesito comprar nuevos libros .") == ["R:WO"]


def test_an_unchanged_sentence_has_no_edits():
    assert annotate("Compre esta manana pan .", "Compre esta manana pan .") == []


# --- spans ------------------------------------------------------------------


def test_edits_carry_source_token_offsets():
    assert spans("Yo tiene un perro .", "Yo tengo un perro .") == [(1, 2, "tengo")]


def test_an_insertion_has_an_empty_span():
    start, end, correction = spans("Vamos a playa .", "Vamos a la playa .")[0]
    assert start == end == 2
    assert correction == "la"


# --- tokenisation -----------------------------------------------------------


def test_free_text_is_retokenised_to_match_the_corpus():
    # A model answers in ordinary Spanish, punctuation and all.
    assert tokenize("¿Cómo estás?") == "¿ Cómo estás ?"


def test_retokenising_already_tokenised_text_leaves_it_alone():
    text = "Yo tengo un perro ."
    assert tokenize(text) == text
