"""Error injection: the damage must be undoable, and it must be the right damage."""

from __future__ import annotations

import itertools
import random

import pytest

from langlm.corruptors import REGISTRY, apply, choose, propose
from langlm.corruptors.base import Corruption

SENTENCE = str.split("Der Lehrer hat gesagt , dass die Schüler die Prüfung bestanden haben .")


def test_every_rule_is_registered() -> None:
    assert "noun_case" in REGISTRY
    assert "drop_article" in REGISTRY
    assert len(REGISTRY) >= 9


@pytest.mark.parametrize("seed", range(25))
def test_corruption_round_trips(seed: int) -> None:
    """Applying the edits to the damaged text must give the original back.

    This is the invariant the whole corpus rests on. If it fails, the training
    data teaches a model to produce something other than the correct sentence,
    and nothing in the M2 format would object.
    """
    rng = random.Random(seed)
    picked = choose(propose(SENTENCE), 2, rng)
    sentence = apply(SENTENCE, picked)
    assert sentence.target() == " ".join(SENTENCE)


def test_damaged_text_actually_differs() -> None:
    picked = choose(propose(SENTENCE), 1, random.Random(0))
    assert apply(SENTENCE, picked).source != " ".join(SENTENCE)


def test_chosen_corruptions_never_overlap() -> None:
    """`apply` raises on overlap, so the sampler has to guarantee it does not."""
    rng = random.Random(5)
    for _ in range(50):
        picked = choose(propose(SENTENCE), 3, rng)
        spans = sorted((c.start, c.end) for c in picked)
        for (_, end), (start, _) in itertools.pairwise(spans):
            assert end <= start


def test_apply_rejects_overlaps() -> None:
    overlapping = [
        Corruption(0, 2, "Die", "R:DET:FORM", "x"),
        Corruption(1, 3, "Lehrerin", "R:NOUN", "y"),
    ]
    with pytest.raises(ValueError, match="Overlapping"):
        apply(SENTENCE, overlapping)


def test_weights_steer_the_error_mix() -> None:
    """A weighted type should dominate; an unweighted one should still appear."""
    rng = random.Random(1)
    weights = {"M:DET": 1.0}
    types = [
        c.error_type for _ in range(200) for c in choose(propose(SENTENCE), 1, rng, weights=weights)
    ]
    assert types.count("M:DET") > 0.8 * len(types)


def test_deletion_produces_a_missing_type_edit() -> None:
    """Dropping a token must leave an insertion edit, not a replacement."""
    tokens = str.split("Der Hund bellt .")
    picked = [Corruption(0, 1, "", "M:DET", "drop_article")]
    sentence = apply(tokens, picked)
    assert sentence.source == "Hund bellt ."
    edit = sentence.edits[0]
    assert edit.start == edit.end == 0
    assert edit.correction == "Der"
    assert sentence.target() == "Der Hund bellt ."


@pytest.fixture(scope="module")
def nlp():
    spacy = pytest.importorskip("spacy")
    return spacy.load("de_core_news_sm", disable=["ner"])


def test_noun_case_leaves_the_first_word_alone(nlp) -> None:
    """A sentence-initial capital says nothing about whether the word is a noun."""
    from langlm.corruptors.de import noun_case

    text = "Hunde bellen laut ."
    assert not [c for c in noun_case(text.split(), nlp(text)) if c.start == 0]


def test_noun_case_skips_proper_nouns(nlp) -> None:
    """English capitalises names too, so a learner does not get them wrong."""
    from langlm.corruptors.de import noun_case

    text = "Nach dem Krieg zog Josef nach Berlin ."
    corrupted = {c.replacement for c in noun_case(text.split(), nlp(text))}
    assert "krieg" in corrupted
    assert "josef" not in corrupted and "berlin" not in corrupted


def test_type_is_drawn_before_the_candidate() -> None:
    """A prolific rule must not out-vote a scarce one at equal weight.

    The keyboard rule offers a candidate per word and the word-order rule at
    most one per sentence. Weighting candidates rather than types would make the
    realised mix a function of how prolific each rule is.
    """
    many = [Corruption(i, i + 1, "x", "R:SPELL", "keyboard") for i in range(30)]
    one = [Corruption(40, 41, "y", "R:WO", "verb_final")]
    rng = random.Random(0)
    weights = {"R:SPELL": 0.5, "R:WO": 0.5}
    picked = [choose(many + one, 1, rng, weights=weights)[0].error_type for _ in range(200)]
    assert 0.3 < picked.count("R:WO") / len(picked) < 0.7


def test_quota_starves_an_over_represented_type() -> None:
    """Types already over their share should give way to ones below theirs."""
    from collections import Counter

    candidates = [
        Corruption(0, 1, "a", "R:SPELL", "keyboard"),
        Corruption(2, 3, "b", "R:WO", "verb_final"),
    ]
    weights = {"R:SPELL": 0.5, "R:WO": 0.5}
    emitted = Counter({"R:SPELL": 900, "R:WO": 100})
    rng = random.Random(0)
    picked = [
        choose(candidates, 1, rng, weights=weights, emitted=emitted)[0].error_type
        for _ in range(200)
    ]
    assert picked.count("R:WO") > picked.count("R:SPELL")


def test_only_real_adjectives_get_adjective_endings(nlp) -> None:
    """The endings are shared with articles, quantifiers and prepositions.

    Recognising them by suffix had this rule firing on `einem`, `gegen` and once
    on `koennen`; it asks the tagger now.
    """
    from langlm.corruptors.de import adjective_form

    text = "Er ging mit einem Freund gegen die große Mauer ."
    changed = {text.split()[c.start] for c in adjective_form(text.split(), nlp(text))}
    assert changed == {"große"}


# --- the new rules, and the labels they claim -------------------------------


def corrupt_one(rule: str, sentence: str):
    """Every corruption `rule` finds in `sentence`, parsed if the rule needs it."""
    import spacy

    from langlm.corruptors import NEEDS_PARSE, REGISTRY

    tokens = sentence.split()
    doc = spacy.load("de_core_news_sm", disable=["ner"])(sentence) if rule in NEEDS_PARSE else None
    return REGISTRY[rule](tokens, doc)


def test_the_declared_type_is_the_type_the_rule_emits() -> None:
    """`rules/de.yaml` names an error type for every corruptor. It has to be the
    one the code produces: the YAML is what a reader consults and the code is
    what the corpus is built from, and they were allowed to disagree once --
    `das_dass` was documented as `R:SPELL` while the corpus calls it `R:OTHER`.
    """
    import spacy

    from langlm.corruptors import REGISTRY, propose
    from langlm.explanations import load

    declared = {rule: entry["error_type"] for rule, entry in load()["corruptors"].items()}
    nlp = spacy.load("de_core_news_sm", disable=["ner"])
    probes = [
        "Der Lehrer hat gesagt , dass die Schüler mit großer Mühe die Aufgaben lösen .",
        "Gestern ging er mit den Kindern und Eltern ins Kino .",
        "Er interessiert sich für den Preis des Hauses und wartet auf mich .",
        "Sie ist gestern nach Hause gefahren , weil sie das Buch vergessen hat .",
    ]
    seen: dict[str, set[str]] = {}
    for probe in probes:
        for corruption in propose(probe.split(), doc=nlp(probe)):
            seen.setdefault(corruption.rule, set()).add(corruption.error_type)
    assert seen, "no rule fired on the probes"
    for rule, types in sorted(seen.items()):
        assert rule in declared, f"{rule} produces corruptions but has no template"
        assert types <= {declared[rule]} or declared[rule] in types, (
            f"{rule} emits {sorted(types)} but rules/de.yaml declares {declared[rule]}"
        )
    # `drop_comma` labels its corruptions with the comma rule that was broken,
    # so what shows up here are template names rather than registry names.
    assert set(seen) - set(REGISTRY) <= {"drop_comma_subordinate", "drop_comma_coordinating"}


def test_a_dropped_pronoun_is_never_the_first_word() -> None:
    # Deleting the first word leaves the next one lower-case, which is a second
    # error the edit does not claim and the explanation does not mention.
    found = corrupt_one("drop_pronoun", "Er interessiert sich für Musik .")
    assert found
    assert all(c.start > 0 for c in found)
    assert all(c.replacement == "" for c in found)


def test_the_dative_plural_n_only_comes_off_a_dative_plural() -> None:
    # `den` is an accusative singular here, so stripping the -n would produce a
    # non-word rather than a case error.
    assert not corrupt_one("dative_plural_n", "Wir warten auf den Bus .")
    assert corrupt_one("dative_plural_n", "Er hilft den Kindern in der Schule .")


def test_morphological_damage_leaves_a_real_word() -> None:
    # `Praxis` ends in -s without that -s being a genitive ending.
    assert not corrupt_one("genitive_s", "Die Ergebnisse der Praxis sind gut .")
    assert corrupt_one("genitive_s", "Der Preis des Hauses ist hoch .")


def test_a_comma_is_not_injected_before_und_joining_clauses() -> None:
    # A comma before an `und` that joins two main clauses is permitted German,
    # so injecting one would be labelling correct writing as an error.
    assert not corrupt_one("comma_before_und", "Er kam nach Hause und sie ging weg .")
    assert corrupt_one("comma_before_und", "Sie kaufte Brot und Butter .")


def test_the_infinitive_rule_skips_forms_that_are_already_infinitives() -> None:
    # First and third person plural present are identical to the infinitive, so
    # there is nothing to damage.
    assert not corrupt_one("verb_infinitive", "Wir warten auf den Bus .")
    assert corrupt_one("verb_infinitive", "Er wartet auf den Bus .")


def test_an_insertion_can_share_a_position_with_a_replacement() -> None:
    """A comma inserted before a word that is itself being changed.

    Both corruptions land on index 3 and they do not conflict -- the comma goes
    in front of the replaced token. Sorting on the start alone left the order to
    chance and raised on half of them, which is how the first build of the
    punctuation rules died.
    """
    from langlm.corruptors.base import Corruption, apply

    tokens = ["Gestern", "ging", "er", "ist", "nach", "Hause", "."]
    damaged = apply(
        tokens,
        [
            Corruption(3, 4, "sind", "R:AUX:FORM", "auxiliary_form"),
            Corruption(3, 3, ",", "U:PUNCT", "comma_after_fronted"),
        ],
    )
    assert damaged.source == "Gestern ging er , sind nach Hause ."
    assert [(e.start, e.end, e.correction) for e in damaged.edits] == [(3, 4, ""), (4, 5, "ist")]


def test_two_insertions_at_one_point_are_refused() -> None:
    from langlm.corruptors.base import Corruption, apply

    with pytest.raises(ValueError, match="same point"):
        apply(
            ["Brot", "und", "Butter", "."],
            [
                Corruption(1, 1, ",", "U:PUNCT", "comma_before_und"),
                Corruption(1, 1, ",", "U:PUNCT", "comma_after_fronted"),
            ],
        )
