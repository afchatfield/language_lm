"""Spanish error injection: the damage must be undoable, the right damage, and
typed the way `rules/es.yaml` says it will be.

Mirrors `test_corruptors.py` where the invariant is language-agnostic (round
tripping, no overlaps -- those exercise `corruptors/base.py`, already covered
there with German sentences) and adds targeted tests for the defects a
stress-test over 400+ real COWS-L2H sentences actually found while this file's
rules were built: five separate cases where a plausible-looking corruption
came back typed as something other than what the rule claimed.
"""

from __future__ import annotations

import random

import pytest

from langlm.corruptors import es
from langlm.corruptors.base import Corruption, apply, choose, propose

SENTENCE = str.split("El estudiante dijo que los profesores explicaron la lección muy bien .")


def test_every_rule_is_registered() -> None:
    assert "drop_accent" in es.REGISTRY
    assert "redundant_subject_pronoun" in es.REGISTRY
    assert len(es.REGISTRY) >= 12


def test_spanish_and_german_registries_do_not_collide() -> None:
    """`de.py` and `es.py` reuse rule names for rules that mean different
    things; each has to have its own table or importing both would raise.
    """
    from langlm.corruptors import REGISTRY as german

    shared_names = set(german) & set(es.REGISTRY)
    assert shared_names, "the test fixture assumes some names really do collide"
    assert es.REGISTRY["article_form"] is not german["article_form"]


@pytest.fixture(scope="module")
def nlp():
    spacy = pytest.importorskip("spacy")
    return spacy.load("es_core_news_sm", disable=["ner"])


@pytest.mark.parametrize("seed", range(15))
def test_corruption_round_trips(seed: int, nlp) -> None:
    """Applying the edits to the damaged text must give the original back."""
    doc = nlp(" ".join(SENTENCE))
    rng = random.Random(seed)
    picked = choose(propose(SENTENCE, doc=doc, registry=es.REGISTRY), 2, rng)
    sentence = apply(SENTENCE, picked)
    assert sentence.target() == " ".join(SENTENCE)


def corrupt_one(rule: str, sentence: str, nlp):
    """Every corruption `rule` finds in `sentence`, parsed if the rule needs it."""
    tokens = sentence.split()
    doc = nlp(sentence) if rule in es.NEEDS_PARSE else None
    return es.REGISTRY[rule](tokens, doc)


def test_the_declared_type_is_the_type_the_rule_emits(nlp) -> None:
    """`rules/es.yaml` names an error type for every corruptor, and it has to
    be the one `errant_es` actually assigns -- see `errant_es.classifier` for
    why a language's classifier is not always the type a rule's author would
    guess (`ser_estar` claims `R:AUX`, not `R:VERB`, because the copula is
    tagged AUX).
    """
    from langlm.eval.errant_es import annotate
    from langlm.explanations import load

    templates = load(language="es")["corruptors"]
    declared = {rule: entry["error_type"] for rule, entry in templates.items()}
    probes = [
        "El estudiante dijo que los profesores explicaron la lección muy bien .",
        "Mi hermana vive en la ciudad porque trabaja en un hospital grande .",
        "Ayer compramos comida para la fiesta y hablamos con nuestros amigos .",
        "Ella cree que su familia va a llegar antes de la cena .",
    ]
    seen: dict[str, set[str]] = {}
    for probe in probes:
        doc = nlp(probe)
        for corruption in propose(probe.split(), doc=doc, registry=es.REGISTRY):
            damaged = apply(probe.split(), [corruption])
            actual_types = {e.error_type for e in annotate(damaged.source, damaged.target())}
            seen.setdefault(corruption.rule, set()).update(actual_types)
    assert seen, "no rule fired on the probes"
    for rule, types in sorted(seen.items()):
        assert rule in declared, f"{rule} produces corruptions but has no template"
        assert declared[rule] in types, (
            f"{rule} actually emits {sorted(types)} but rules/es.yaml declares {declared[rule]}"
        )


# --- regressions: each of these is a real defect a stress-test found --------


def test_article_form_never_touches_an_object_clitic(nlp) -> None:
    # "la" is also the third-person direct-object clitic ("no la veo" -- I
    # don't see *her*), homographic with the article. An earlier, token-only
    # version of this rule swapped it and called the result a wrong article.
    found = corrupt_one("article_form", "No la veo tan a menudo .", nlp)
    assert not found


def test_drop_article_never_touches_an_object_clitic(nlp) -> None:
    found = corrupt_one("drop_article", "No la veo tan a menudo .", nlp)
    assert not found


def test_article_form_fires_on_a_real_article(nlp) -> None:
    found = corrupt_one("article_form", "Compré la casa ayer .", nlp)
    assert found
    assert {c.replacement.lower() for c in found} == {"el", "los", "las"}


def test_keyboard_skips_a_slip_that_lands_on_a_real_word(nlp) -> None:
    # "tías" (aunts) one key over is "rías" (estuaries) -- a real Spanish word.
    # Left unguarded, the explanation "not a Spanish word" would be false.
    found = corrupt_one("keyboard", "Mis tías favoritas viven lejos .", nlp)
    assert "rías" not in {c.replacement for c in found}


def test_adjective_form_skips_a_gender_invariant_adjective(nlp) -> None:
    # "indígena" has no "indígeno" -- some -a adjectives do not inflect for
    # gender, and spacy's morphology tags "indígena" Gender=Fem regardless of
    # whether it is one of them, so the tag alone cannot tell the classes apart.
    found = corrupt_one("adjective_form", "La gente indígena vive aquí .", nlp)
    assert "indígeno" not in {c.replacement for c in found}


def test_adjective_form_still_fires_on_a_regular_adjective(nlp) -> None:
    found = corrupt_one("adjective_form", "La casa roja es bonita .", nlp)
    assert found


def test_ser_estar_skips_an_identity_statement(nlp) -> None:
    # Spanish grammar requires "ser", not "estar", to say who someone is by
    # name -- "Su esposa es Michelle Obama" -- and forbids "estar" outright.
    # The swap is so ungrammatical that es_core_news_sm's own re-parse of the
    # damaged sentence stops recognising "está" as a copula at all.
    found = corrupt_one("ser_estar", "Su esposa es Michelle Obama .", nlp)
    assert not found


def test_ser_estar_still_fires_on_a_real_predicate(nlp) -> None:
    found = corrupt_one("ser_estar", "Ella es muy inteligente .", nlp)
    assert found and found[0].replacement == "está"


def test_redundant_subject_pronoun_never_fires_sentence_initial(nlp) -> None:
    # Inserting a pronoun before a sentence-initial verb would also have to
    # move the capital letter, a second edit the correction does not claim.
    found = corrupt_one("redundant_subject_pronoun", "Tengo un perro grande .", nlp)
    assert not found


def test_redundant_subject_pronoun_skips_third_person(nlp) -> None:
    # Third person splits by gender (él/ella) and the verb's own morphology
    # carries neither, so inserting one would be a guess.
    found = corrupt_one("redundant_subject_pronoun", "Creo que ella tiene razón .", nlp)
    assert not any(c.replacement in {"él", "ella"} for c in found)


def test_redundant_subject_pronoun_fires_on_first_person(nlp) -> None:
    found = corrupt_one("redundant_subject_pronoun", "Creo que tengo razón .", nlp)
    assert found and found[0].replacement == "yo"


def test_drop_preposition_leaves_a_contraction_alone(nlp) -> None:
    # Removing "al" (a + el) would take the article with it -- two errors
    # wearing the label of one, the same reason German's version excludes
    # "im"/"zum".
    sentence = "Fuimos al mercado ayer ."
    found = corrupt_one("drop_preposition", sentence, nlp)
    assert not any(c.start == sentence.split().index("al") for c in found)


def test_declared_types_have_no_unregistered_rule() -> None:
    """Every corruptor in `rules/es.yaml` must be a real, registered rule --
    the inverse of the type-fidelity test, catching a template for a rule
    that was renamed or removed.
    """
    from langlm.explanations import load

    declared = set(load(language="es")["corruptors"])
    assert declared <= set(es.REGISTRY)


def test_apply_rejects_overlaps_in_spanish_too() -> None:
    tokens = str.split("El perro come mucho .")
    overlapping = [
        Corruption(0, 1, "La", "R:DET:FORM", "article_form"),
        Corruption(0, 2, "gato come", "R:NOUN", "x"),
    ]
    with pytest.raises(ValueError, match="Overlapping"):
        apply(tokens, overlapping)
