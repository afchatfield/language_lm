"""The clean corpus, and the exclusion that keeps the eval set out of it."""

from __future__ import annotations

import pytest

from langlm.data.clean_de import MAX_TOKENS, MIN_TOKENS, Rejections, normalise
from langlm.data.quality import is_clean, spelled_correctly


def test_normalise_bridges_tokenisation() -> None:
    """The frozen set is tokenised and Leipzig is not; the key must not care."""
    assert normalise("Freilebende Kühe gebären .") == normalise("Freilebende Kühe gebären.")
    assert normalise("Er sagte : « Ja » .") == normalise("Er sagte: «Ja».")


def test_normalise_still_separates_different_sentences() -> None:
    assert normalise("Der Hund bellt.") != normalise("Die Katze bellt.")


def test_prose_filter_shared_with_phase_one() -> None:
    """`quality` is the single definition; a drift here is a drift in both sets."""
    assert is_clean("Drei Tage lang wurde beim Musikverein in Reichenbach gefeiert.")
    assert not is_clean("und dann ging er nach Hause.")
    assert not is_clean("Mehr dazu unter http://example.de lesen Sie hier.")


def test_spelling_filter_keeps_capitalised_unknowns() -> None:
    """Proper nouns are not typos, and rejecting them would bias the corpus."""
    assert spelled_correctly(["Selenskyj", "kam", "gestern", "an"])
    assert not spelled_correctly(["Er", "hat", "das", "Buch", "gelest"])


def test_length_window_is_wider_than_the_overcorrection_set() -> None:
    """A corruptor needs room; a comma error rarely fits in eight tokens."""
    from langlm.data import overcorrection

    assert MIN_TOKENS == overcorrection.MIN_TOKENS
    assert MAX_TOKENS > overcorrection.MAX_TOKENS


def test_rejection_rows_account_for_everything_read() -> None:
    """Every sentence read is either rejected for a named reason or kept.

    The report prints this funnel, so a stage that drops sentences without
    counting them would show up as a corpus that is smaller than its own
    arithmetic says -- which is exactly the failure that is hard to notice.
    """
    stats = Rejections(
        read=100, duplicate=5, unclean=10, length=4, misspelled=1, held_out=2, pooled=78
    )
    rows = dict(stats.as_rows())
    accounted = sum(count for label, count in rows.items() if label != "read from Leipzig")
    assert accounted == rows["read from Leipzig"]


def test_parser_rows_account_for_everything_parsed() -> None:
    """The parse stage has its own funnel, over the pool it actually reached."""
    stats = Rejections(pooled=78, examined=70, fragment=6, kept=64)
    rows = dict(stats.parser_rows())
    assert rows["kept"] + stats.fragment == rows["parsed"]


@pytest.mark.parametrize("sentence", ["Der Mann ging nach", "DIE LAGE IST ERNST."])
def test_prose_filter_rejects_damage(sentence: str) -> None:
    assert not is_clean(sentence)


@pytest.fixture(scope="module")
def nlp():
    spacy = pytest.importorskip("spacy")
    return spacy.load("de_core_news_sm", disable=["ner"])


@pytest.mark.parametrize(
    ("sentence", "complete"),
    [
        ("Die Tiere bewohnen angeschwemmtes Pflanzenmaterial an Gewässerufern .", True),
        # The conjunction governs a subordinate verb; the root is the main clause's.
        ("Obwohl es regnete , ging er spazieren .", True),
        # `Als Folge` is a prepositional phrase, not a subordinate clause.
        ("Als Folge starben die meisten Yana an Unterernährung .", True),
        # A subordinate clause printed on its own: the conjunction governs the root.
        ("Weil es immer länderspezifische Faktoren gibt , die berücksichtigt werden .", False),
        # Caption and headline: no finite verb anywhere.
        ("Schloss und Gut des Carl von Siemens bei St. Petersburg .", False),
        ("Eisdisco am Kirlteich , 14 bis 22 Uhr .", False),
    ],
)
def test_completeness(nlp, sentence: str, complete: bool) -> None:
    from langlm.data.clean_de import is_complete

    assert is_complete(nlp(sentence)) is complete


def test_completeness_is_phase_two_only() -> None:
    """The Phase 1 set is frozen; its published baselines depend on its membership."""
    import langlm.data.overcorrection as oc

    assert not hasattr(oc, "is_complete")
