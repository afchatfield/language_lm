"""Explanation templates: every error must be explicable, and correctly filled."""

from __future__ import annotations

import pytest

from langlm.corruptors import REGISTRY
from langlm.explanations import TemplateError, coverage, explain, load


def test_every_corruptor_has_a_template() -> None:
    """A generated example with no explanation is worse than no example."""
    assert coverage(sorted(REGISTRY)) == []


def test_placeholders_are_filled() -> None:
    text = explain("sharp_s", wrong="grosser", right="großer")
    assert "großer" in text
    assert "{" not in text and "}" not in text


def test_reference_is_appended_when_known() -> None:
    assert "Rat für deutsche Rechtschreibung, Amtliches Regelwerk 2024, § 55" in explain(
        "noun_case", "hund", "Hund"
    )


def test_reference_names_its_edition() -> None:
    """A bare paragraph number is not a citation.

    The 2024 edition renumbered part E -- the subordinate-clause comma is § 73
    there and § 74 in the 2018 text, where § 74 is now the semicolon. Three
    citations in this file were wrong for exactly that reason, so the edition
    travels with every number the model is taught to say.
    """
    text = explain("drop_comma_subordinate", "x", "y")
    assert "§ 73" in text
    assert load()["edition"] in text


def test_reference_is_omitted_when_none_exists() -> None:
    """No paragraph covers a dropped umlaut, so the explanation cites nothing."""
    assert "Rat für deutsche Rechtschreibung" not in explain("umlaut", "Bucher", "Bücher")


def test_every_uncited_rule_says_why_it_is_uncited() -> None:
    """`null` must mean "no citation exists", never "nobody got to it yet".

    Most of these rules are grammar -- agreement, case government, word order --
    and the Regelwerk covers orthography only, so no citation is possible. That
    is a finished state, and this test is what keeps it distinguishable from an
    unfinished one.
    """
    for rule, entry in load()["corruptors"].items():
        if not entry.get("reference"):
            assert entry.get("reference_note"), f"{rule}: null reference with no reason given"


def test_languagetool_refines_rather_than_replaces() -> None:
    base = explain("article_form", "den", "dem")
    refined = explain("article_form", "den", "dem", lt_rule_id="PRAEP_DAT")
    assert refined.startswith(base.split(" (")[0][:40])
    assert "dative" in refined


def test_unknown_languagetool_rule_is_ignored() -> None:
    """Most rule ids have no template; that is normal, not an error."""
    assert explain("umlaut", "Bucher", "Bücher", lt_rule_id="SOME_RULE_WE_NEVER_SAW") == explain(
        "umlaut", "Bucher", "Bücher"
    )


def test_missing_corruptor_template_raises() -> None:
    with pytest.raises(TemplateError, match="no_such_rule"):
        explain("no_such_rule", "a", "b")


def test_every_template_declares_its_error_type() -> None:
    """The type is what ties an explanation back to the evaluation's units."""
    for rule, entry in load()["corruptors"].items():
        assert entry.get("error_type"), f"{rule} has no error_type"
        assert "explanation" in entry
