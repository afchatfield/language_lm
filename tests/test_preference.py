"""Preference pairs: only where the model is wrong about its own beam."""

from __future__ import annotations

import json

from langlm.data.m2 import Edit, M2Sentence
from langlm.train.format import parse_answer
from langlm.train.preference import build_pairs, complete_answer

SOURCE = "Ich sehe der Frau ."
RIGHT = "Ich sehe die Frau ."


def sentence(source: str, *edits: tuple[int, int, str, str]) -> M2Sentence:
    return M2Sentence(
        source_tokens=source.split(),
        edits=[Edit(start=s, end=e, error_type=t, correction=c) for s, e, t, c in edits],
    )


def beam(*corrections: str) -> list[dict]:
    """A beam in the model's own ranking, best-scored first."""
    return [{"text": text, "score": -0.1 * rank} for rank, text in enumerate(corrections)]


def answer_for(source: str, correction: str) -> str:
    """Stands in for the real one, which would load a parser to derive `changes`."""
    return f'{{"correction": "{correction}", "changes": []}}'


def pairs_for(sentence_, record, prompt="prompt"):
    return build_pairs([sentence_], [record], [prompt], answer_for=answer_for)


GOLD = sentence(SOURCE, (2, 3, "R:DET:FORM", "die"))


def test_pair_is_built_when_a_later_candidate_is_better() -> None:
    pairs = pairs_for(GOLD, beam(SOURCE, RIGHT))
    assert len(pairs) == 1
    assert RIGHT in pairs[0].chosen
    assert SOURCE in pairs[0].rejected
    assert pairs[0].chosen_rank == 1
    assert pairs[0].gain == 1


def test_no_pair_when_the_model_already_ranks_the_best_one_first() -> None:
    """Nothing to teach: the served answer is already the best in the beam."""
    assert pairs_for(GOLD, beam(RIGHT, SOURCE)) == []


def test_no_pair_when_candidates_tie() -> None:
    """Preferring the narrower of two equally correct answers is not German.

    Both candidates here score the same against the annotator, so the only thing
    a pair could teach is a decoding preference.
    """
    tie = beam(RIGHT, "Ich sehe die Frau!")
    pairs = pairs_for(GOLD, tie)
    assert pairs == [] or pairs[0].gain > 0


def test_identical_text_across_two_beams_is_not_a_pair() -> None:
    """Beam search can reach the same string twice with different scores."""
    duplicate = beam(SOURCE, SOURCE)
    assert pairs_for(GOLD, duplicate) == []


def test_a_candidate_that_fixes_more_than_it_breaks_wins() -> None:
    """The ordering is true positives minus false ones, not raw overlap.

    The second candidate makes the gold edit and one spurious one, so it nets
    zero against a top-1 that does nothing -- and must not produce a pair.
    """
    spoiled = beam(SOURCE, "Ich sah die Frau .")
    assert pairs_for(GOLD, spoiled) == []


def test_gain_reflects_how_much_better_the_chosen_side_is() -> None:
    two = sentence(SOURCE, (1, 2, "R:VERB:FORM", "sah"), (2, 3, "R:DET:FORM", "die"))
    pairs = pairs_for(two, beam(SOURCE, "Ich sah die Frau ."))
    assert len(pairs) == 1
    assert pairs[0].gain == 2


def test_a_single_candidate_beam_yields_nothing() -> None:
    assert pairs_for(GOLD, beam(SOURCE)) == []


def test_completion_is_a_whole_answer() -> None:
    """The guard on the EOS that TRL appends and offers no way to suppress.

    The beam stops at the `changes` marker to save width, so a pair built from
    the raw generation would carry a truncated answer and an EOS immediately
    after it -- which trains the model to stop mid-schema on every sentence.
    """
    answer = complete_answer(SOURCE, RIGHT)
    parsed = json.loads(answer)
    assert parsed["correction"] == RIGHT
    assert [change["now"] for change in parsed["changes"]] != []
    assert parse_answer(answer).correction == RIGHT
