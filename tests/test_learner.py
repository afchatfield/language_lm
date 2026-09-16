"""Real learner errors as training data, and the guard on the evaluation splits."""

from __future__ import annotations

import pytest

from langlm.data import learner
from langlm.explanations import explain_type, load_types


@pytest.mark.parametrize("split", ["dev", "test", "validation"])
def test_evaluation_splits_are_refused(split: str) -> None:
    """Training on dev would make every number after it meaningless.

    The refusal is here rather than in the caller because a caller that passes
    the wrong string is exactly the failure this prevents.
    """
    with pytest.raises(learner.SplitMisuseError, match=split):
        list(learner.records(split))


def test_records_carry_type_level_explanations() -> None:
    records = list(learner.records(limit=50))
    assert records
    for record in records:
        for edit in record["edits"]:
            assert edit["explanation"]
            assert edit["rule"] in {"falko", "falko-move"}


def test_moved_words_are_explained_as_a_move() -> None:
    """ERRANT splits a move into a delete and an insert of the same word.

    Explained one at a time those read as "remove it" then "add it back", which
    is a contradiction rather than advice.
    """
    edits = [
        {
            "error_type": "U:VERB",
            "start": 3,
            "end": 4,
            "correction": "",
            "rule": "falko",
            "explanation": "remove",
        },
        {
            "error_type": "M:VERB",
            "start": 9,
            "end": 9,
            "correction": "gratulieren",
            "rule": "falko",
            "explanation": "add",
        },
    ]
    tokens = str.split("ich moechte dir gratulieren zur bestandenen Pruefung heute .")
    assert learner.mark_moves(edits, tokens) == 1
    for edit in edits:
        assert edit["rule"] == "falko-move"
        assert "wrong place" in edit["explanation"]


def test_unrelated_deletes_and_inserts_are_not_paired() -> None:
    """Only the same word in the same class is a move; everything else is two edits."""
    edits = [
        {
            "error_type": "U:VERB",
            "start": 0,
            "end": 1,
            "correction": "",
            "rule": "falko",
            "explanation": "remove",
        },
        {
            "error_type": "M:DET",
            "start": 3,
            "end": 3,
            "correction": "der",
            "rule": "falko",
            "explanation": "add",
        },
    ]
    assert learner.mark_moves(edits, str.split("gehen wir nach Hause .")) == 0


def test_records_round_trip_to_the_correction() -> None:
    """Applying the edits to the source must give the annotator's target back."""
    from langlm.train.format import apply_edits

    for record in list(learner.records(limit=200)):
        if not record["edits"]:
            continue
        rebuilt = apply_edits(record["source"].split(), record["edits"])
        assert rebuilt == record["target"]


def test_real_negatives_are_kept() -> None:
    """22% of Falko sentences need no correction; those are the negatives."""
    records = list(learner.records(limit=500))
    empty = [r for r in records if not r["edits"]]
    assert 0.15 < len(empty) / len(records) < 0.30


def test_every_falko_type_resolves_to_an_explanation() -> None:
    """A missing template would silently drop edits from the corpus."""
    from collections import Counter

    from langlm.data.splits import load_split

    types = Counter(
        edit.error_type
        for sentence in load_split("falko_merlin", "train")[:2000]
        for edit in sentence.edits_for()
        if not edit.is_noop
    )
    for error_type in types:
        assert explain_type(error_type, "alt", "neu").strip()


def test_composed_types_name_the_word_class() -> None:
    """The tail is composed rather than hand-written, and must still read well."""
    assert "adjective" in explain_type("U:ADJ", "alt", "")
    assert "pronoun" in explain_type("R:PRON:FORM", "ihm", "ihn")
    assert "wrong form" in explain_type("R:ADV:FORM", "schnell", "schneller")


def test_unclassifiable_types_fall_back_rather_than_guess() -> None:
    """ERRANT emits `R:*` for edits it cannot type; inventing a reason is worse."""
    text = explain_type("R:*", "alt", "neu")
    assert "neu" in text
    assert "None" not in text


def test_explicit_templates_cover_the_volume() -> None:
    """The hand-written ones should carry the common types, not the tail."""
    explicit = set(load_types()["explicit"])
    for error_type in ("R:SPELL", "R:DET:FORM", "M:PUNCT", "R:ORTH", "R:WO"):
        assert error_type in explicit


def test_explanations_are_derived_from_the_correction_not_from_a_claimed_type() -> None:
    """The learner-facing type comes from the two sentences, not from the model.

    A model that corrects well but names its edit wrongly would otherwise hand
    the learner an explanation of an error that was never made.
    """
    from langlm.explanations import explain_correction

    explained = explain_correction("Ich sehe der Frau .", "Ich sehe die Frau .")
    assert len(explained) == 1
    edit = explained[0]
    assert (edit.wrong, edit.right) == ("der", "die")
    assert edit.error_type.startswith("R:DET")
    assert edit.explanation.strip()


def test_an_unchanged_sentence_explains_nothing() -> None:
    from langlm.explanations import explain_correction

    assert explain_correction("Das ist richtig .", "Das ist richtig .") == []
