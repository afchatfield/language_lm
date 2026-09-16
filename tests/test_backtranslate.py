"""The error generator's schema, and the filter on what it produces.

Two things here fail silently and are worth pinning.

The **loss mask** on the reverse task, for a sharper version of the forward
task's reason: a generator that trained on its prompts would learn to emit
correct German, and a generator that emits correct German produces a corpus of
identity pairs that the forward model learns to leave everything alone from.

The **filter**, because it is the only thing standing between a sampled
generator and a corpus of paraphrases. A pair whose "correction" is not a
correction teaches the forward model to rewrite sentences that are already fine,
which is iteration 1's failure arriving by a different road.
"""

from __future__ import annotations

import pytest

from langlm.data.backtranslate import (
    MAX_REPEAT,
    Pair,
    SplitMisuseError,
    divergence,
    rejection,
    reversed_records,
    token_distance,
)
from langlm.train.corrupter import IGNORE, build_prompt, encode


class FakeTokenizer:
    """Whitespace tokenisation with a stable vocabulary, so ids are predictable."""

    eos_token_id = 99

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        return {"input_ids": [len(word) for word in text.split()]}


@pytest.fixture
def record() -> dict:
    return {"correct": "Der Hund bellt .", "learner": "der hund bellt .", "count": 2}


# --- the prompt -------------------------------------------------------------


def test_the_error_budget_is_in_the_prompt() -> None:
    """Density is the statistic iteration 1 got wrong; it has to be steerable."""
    assert "3 mistakes" in build_prompt("Der Hund bellt .", 3)
    assert "1 mistake" in build_prompt("Der Hund bellt .", 1)
    assert "no mistakes" in build_prompt("Der Hund bellt .", 0)


def test_the_prompt_carries_the_correct_sentence(record: dict) -> None:
    prompt = build_prompt(record["correct"], record["count"])
    assert record["correct"] in prompt
    assert record["learner"] not in prompt


# --- the loss mask ----------------------------------------------------------


def test_only_the_learner_sentence_is_learned_from(record: dict) -> None:
    """The prompt is context. Training on it teaches the model to write correct German,
    which is the one thing an error generator must not do."""
    encoded = encode(record, FakeTokenizer(), max_length=256)
    prompt_ids = FakeTokenizer()(build_prompt(record["correct"], record["count"]))["input_ids"]
    assert encoded.labels[: len(prompt_ids)] == [IGNORE] * len(prompt_ids)
    assert IGNORE not in encoded.labels[len(prompt_ids) :]


def test_the_answer_ends_on_eos(record: dict) -> None:
    assert encode(record, FakeTokenizer(), max_length=256).labels[-1] == 99


def test_an_example_that_does_not_fit_is_dropped(record: dict) -> None:
    """Truncating would teach the generator to stop mid-sentence, and every pair it
    then made would carry a deletion no learner wrote."""
    assert encode(record, FakeTokenizer(), max_length=4) is None


# --- the reversed corpus ----------------------------------------------------


def test_evaluation_splits_are_refused() -> None:
    """A generator trained on dev would launder the evaluation set into the
    training corpus one corruption at a time."""
    with pytest.raises(SplitMisuseError):
        list(reversed_records("dev"))
    with pytest.raises(SplitMisuseError):
        list(reversed_records("test"))


# --- divergence -------------------------------------------------------------


def test_token_distance_counts_tokens_not_characters() -> None:
    assert token_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert token_distance(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert token_distance(["a", "b", "c"], ["a", "c"]) == 1


def test_divergence_is_a_fraction_of_the_longer_sentence() -> None:
    assert divergence("a b c d", "a b c d") == 0.0
    assert divergence("a b c d", "a b c x") == 0.25


# --- the filter -------------------------------------------------------------


def test_a_plausible_corruption_is_kept() -> None:
    pair = Pair(correct="Der Hund bellt laut .", learner="Der hund belt laut .", asked=2)
    assert rejection(pair, threshold=0.7) is None


def test_a_copy_is_rejected_when_errors_were_asked_for() -> None:
    pair = Pair(correct="Der Hund bellt .", learner="Der Hund bellt .", asked=2)
    assert rejection(pair, threshold=0.7) == "copied when asked for errors"


def test_a_copy_is_kept_when_no_errors_were_asked_for() -> None:
    """The zero bucket is how the 22% no-correction share arrives, so a copy here
    is the generator obeying rather than failing."""
    pair = Pair(correct="Der Hund bellt .", learner="Der Hund bellt .", asked=0)
    assert rejection(pair, threshold=0.7) is None


def test_damage_is_rejected_when_no_errors_were_asked_for() -> None:
    """A negative that is not its own input would teach the model to leave a real
    error alone."""
    pair = Pair(correct="Der Hund bellt .", learner="Der hund bellt .", asked=0)
    assert rejection(pair, threshold=0.7) == "asked for none, damaged anyway"


def test_a_paraphrase_is_rejected() -> None:
    pair = Pair(
        correct="Der Hund bellt laut im Garten .",
        learner="Eine Katze schlief still auf dem Sofa gestern .",
        asked=2,
    )
    assert rejection(pair, threshold=0.5) == "more damaged than real learner German"


def test_a_runaway_generation_is_rejected_on_length() -> None:
    pair = Pair(
        correct="Der Hund bellt .",
        learner="Der Hund bellt und bellt und läuft weg .",
        asked=1,
    )
    assert rejection(pair, threshold=0.9) == "length ratio out of range"


def test_degenerate_repetition_is_rejected() -> None:
    """A looping decoder produces the same token forever, and the length filter
    alone would let a short loop through."""
    loop = " ".join(["bellt"] * (MAX_REPEAT + 1))
    pair = Pair(correct=f"Der Hund {loop} .", learner=f"Der hund {loop} .", asked=1)
    assert rejection(pair, threshold=0.9) == "degenerate repetition"


def test_an_empty_generation_is_rejected() -> None:
    assert rejection(Pair(correct="Der Hund bellt .", learner="  ", asked=1), 0.9) == "empty"
