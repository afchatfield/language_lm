"""Tests for the baseline systems' logic, without the systems.

LanguageTool's own suggestions arrive as character offsets into the submitted
text, and turning a list of overlapping suggestions into one corrected sentence
is where a baseline quietly becomes stronger or weaker than it should be. That
logic is tested here against synthetic matches; the client that fetches the real
ones is tested in `test_lt_client.py`.
"""

from __future__ import annotations

from langlm.baselines.identity import IdentityBaseline
from langlm.baselines.languagetool import apply_matches
from langlm.baselines.prompted import PromptedBaseline, sample_shots
from langlm.data.m2 import Edit, M2Sentence
from langlm.lt.client import Match


def match(offset: int, length: int, *replacements: str, rule_id: str = "RULE") -> Match:
    return Match(
        offset=offset,
        length=length,
        rule_id=rule_id,
        category_id="GRAMMAR",
        message="",
        short_message="",
        replacements=list(replacements),
        context="",
    )


# --- LanguageTool -----------------------------------------------------------


def test_applies_the_first_suggestion():
    text = "Ich weiss , das du kommst ."
    assert apply_matches(text, [match(4, 5, "weiß")]) == "Ich weiß , das du kommst ."


def test_later_overlapping_matches_yield_to_earlier_ones():
    # Two rules firing on the same span would otherwise corrupt each other's
    # offsets and produce a sentence neither rule proposed.
    text = "Ich weiss das ."
    result = apply_matches(text, [match(4, 5, "weiß"), match(4, 9, "weiß, dass")])
    assert result == "Ich weiß das ."


def test_matches_without_a_suggestion_are_left_alone():
    # LanguageTool raises these to say "something is wrong here" without knowing
    # what to do. Guessing on its behalf would put a false positive on the wrong
    # system's account.
    text = "Ich weiss das ."
    assert apply_matches(text, [match(4, 5)]) == text


def test_a_suggestion_that_changes_nothing_is_skipped():
    text = "Ich weiss das ."
    assert apply_matches(text, [match(4, 5, "weiss")]) == text


def test_several_matches_apply_in_one_pass():
    text = "der Frau gehen nach Hause"
    result = apply_matches(text, [match(0, 3, "die"), match(9, 5, "geht")])
    assert result == "die Frau geht nach Hause"


# --- identity ---------------------------------------------------------------


def test_identity_changes_nothing():
    assert IdentityBaseline().correct("Alles richtig hier .") == "Alles richtig hier ."


# --- prompting --------------------------------------------------------------


def test_prompt_ends_where_the_model_should_start():
    baseline = PromptedBaseline(repo_id="none", shots=[("falsch", "richtig")])
    prompt = baseline.build_prompt("Ich gehen .")
    assert prompt.endswith("Input: Ich gehen .\nOutput:")
    assert "Input: falsch\nOutput: richtig" in prompt


def test_only_the_first_line_of_the_continuation_is_the_answer():
    # A base model does not stop when it has answered; it invents the next
    # example. Everything after the first newline is that.
    completion = "Ich gehe .\n\nInput: Etwas anderes\nOutput: ..."
    assert PromptedBaseline.parse(completion, "Ich gehen .") == "Ich gehe ."


def test_an_empty_answer_counts_as_leaving_the_sentence_alone():
    assert PromptedBaseline.parse("   ", "Ich gehen .") == "Ich gehen ."


def test_shots_include_correct_sentences_in_the_corpus_proportion():
    sentences = [
        M2Sentence(
            source_tokens=[f"satz{i}"],
            edits=[] if i % 2 else [Edit(start=0, end=1, error_type="R:OTHER", correction="x")],
        )
        for i in range(20)
    ]
    shots = sample_shots(sentences, count=10, seed=1, correct_share=0.5)
    unchanged = sum(1 for source, target in shots if source == target)
    assert len(shots) == 10
    assert unchanged == 5
