"""The output schema, the text anchors, and the loss mask.

The mask is the one thing in Phase 3 that fails silently. A run that trains on
its prompts looks exactly like one that does not until the model starts writing
learner German instead of correcting it, days of GPU time later.

The anchors are the other. They replaced token offsets, which the model never
learned to count; if a widening window stops being unique the explanation lands
on the wrong word and nothing else in the pipeline notices.
"""

from __future__ import annotations

import json

import pytest

from langlm.train.format import (
    CHANGES_MARKER,
    IGNORE,
    TARGET_PREFIX,
    anchor,
    apply_edits,
    build_prompt,
    build_target,
    encode,
    parse_answer,
)


class FakeTokenizer:
    """Whitespace tokenisation with a stable vocabulary, so ids are predictable.

    Offsets are real character spans into the text, because `encode` splits the
    answer into its two fields on them: a fake that returned plausible-looking
    but wrong offsets would let a broken split pass.
    """

    eos_token_id = 99

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict:
        ids, offsets, cursor = [], [], 0
        for word in text.split():
            start = text.index(word, cursor)
            ids.append(len(word))
            offsets.append((start, start + len(word)))
            cursor = start + len(word)
        encoded = {"input_ids": ids}
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        return encoded


@pytest.fixture
def record() -> dict:
    return {
        "source": "Der hund bellt .",
        "target": "Der Hund bellt .",
        "edits": [
            {
                "start": 1,
                "end": 2,
                "error_type": "R:ORTH",
                "correction": "Hund",
                "explanation": "German capitalises every noun.",
            }
        ],
    }


def test_prompt_ends_on_the_answer_marker() -> None:
    """The model has to see where its turn starts."""
    assert build_prompt("der hund bellt .").endswith(TARGET_PREFIX)


# --- the schema -------------------------------------------------------------


def test_the_sentence_comes_first(record: dict) -> None:
    """Everything the score depends on has to survive a cut-off answer."""
    text = build_target(record)
    assert list(json.loads(text)) == ["correction", "changes"]
    assert text.index('"correction"') < text.index('"changes"')


def test_the_marker_that_ends_the_scored_part_is_really_there(record: dict) -> None:
    """Generation stops on this string; a change in separators would silently
    turn the stop into a no-op and cost the run its speed, not its answer."""
    text = build_target(record)
    assert CHANGES_MARKER in text
    assert json.dumps(record["target"], ensure_ascii=False) in text.split(CHANGES_MARKER)[0]


def test_changes_carry_text_and_a_type_but_no_offsets(record: dict) -> None:
    change = json.loads(build_target(record))["changes"][0]
    assert list(change) == ["was", "now", "type"]
    assert change == {"was": "hund", "now": "Hund", "type": "R:ORTH"}


def test_german_is_not_escaped() -> None:
    """Escaping umlauts would triple their token cost for nothing."""
    text = build_target(
        {
            "source": "Bucher lesen .",
            "target": "Bücher lesen .",
            "edits": [{"start": 0, "end": 1, "error_type": "R:SPELL", "correction": "Bücher"}],
        }
    )
    assert "Bücher" in text
    assert "\\u" not in text


def test_correct_sentences_supervise_an_empty_change_list() -> None:
    """22% of the corpus says the right answer is to change nothing."""
    record = {"source": "Der Hund bellt .", "target": "Der Hund bellt .", "edits": []}
    assert json.loads(build_target(record)) == {
        "correction": "Der Hund bellt .",
        "changes": [],
    }
    assert encode(record, FakeTokenizer(), max_length=512) is not None


# --- anchors ----------------------------------------------------------------


def test_a_unique_span_needs_no_context() -> None:
    tokens = ["der", "hund", "bellt", "."]
    assert anchor(tokens, 1, 2, "Hund") == ("hund", "Hund")


def test_a_repeated_span_widens_until_it_is_unique() -> None:
    """Otherwise the explanation attaches to whichever copy comes first."""
    tokens = ["die", "frau", "und", "die", "frau"]
    was, now = anchor(tokens, 0, 1, "Die")
    assert was == "die frau und"
    assert now == "Die frau und"
    assert tokens_count(tokens, was) == 1


def test_an_insertion_always_names_a_real_token() -> None:
    """A zero-width window quotes the empty string and points at nothing."""
    was, now = anchor(["ich", "gehe", "."], 1, 1, ",")
    assert was == "ich gehe"
    assert now == "ich , gehe"


def test_an_anchor_that_cannot_be_unique_still_names_something() -> None:
    """One edit in the corpus is genuinely ambiguous; it must not crash."""
    was, now = anchor(["ja"] * 8, 0, 1, "Ja", max_context=2)
    assert was and now.startswith("Ja")


def tokens_count(tokens: list[str], text: str) -> int:
    needle = text.split()
    width = len(needle)
    return sum(1 for i in range(len(tokens) - width + 1) if tokens[i : i + width] == needle)


# --- the loss mask ----------------------------------------------------------


def test_prompt_tokens_are_masked_out_of_the_loss(record: dict) -> None:
    """Everything before the answer must be IGNORE, and the answer must not be."""
    encoded = encode(record, FakeTokenizer(), max_length=512)
    assert encoded is not None

    prompt_length = len(FakeTokenizer()(build_prompt(record["source"]))["input_ids"])
    assert encoded.labels[:prompt_length] == [IGNORE] * prompt_length
    assert IGNORE not in encoded.labels[prompt_length:]


def test_supervised_labels_match_the_answer_ids(record: dict) -> None:
    """The unmasked labels must be exactly the answer, including its EOS."""
    tokenizer = FakeTokenizer()
    encoded = encode(record, tokenizer, max_length=512)
    supervised = [label for label in encoded.labels if label != IGNORE]
    expected = tokenizer(build_target(record), add_special_tokens=False)["input_ids"]
    assert supervised == [*expected, tokenizer.eos_token_id]


def test_answer_ends_with_eos(record: dict) -> None:
    """Without it the model never learns to stop, and generation runs to the cap."""
    encoded = encode(record, FakeTokenizer(), max_length=512)
    assert encoded.input_ids[-1] == FakeTokenizer.eos_token_id


def test_overlong_examples_are_dropped_not_truncated(record: dict) -> None:
    """Teaching the model to stop mid-answer is worse than dropping the example."""
    assert encode(record, FakeTokenizer(), max_length=4) is None


# --- reading the answer back ------------------------------------------------


def test_a_whole_answer_reads_back_complete(record: dict) -> None:
    answer = parse_answer(build_target(record))
    assert answer.correction == "Der Hund bellt ."
    assert answer.changes == [{"was": "hund", "now": "Hund", "type": "R:ORTH"}]
    assert answer.complete


def test_a_cut_off_answer_still_yields_its_sentence() -> None:
    """The failure that cost the last run a quarter of dev. The sentence is
    first in the schema precisely so that the cap takes the explanations."""
    answer = parse_answer('{"correction": "Der Hund bellt .", "changes": [{"was": "hu')
    assert answer is not None
    assert answer.correction == "Der Hund bellt ."
    assert answer.changes == []
    assert not answer.complete


def test_an_answer_with_no_sentence_is_a_failure() -> None:
    """Scoring one as "changed nothing" would hide it behind restraint."""
    assert parse_answer("not json at all") is None
    assert parse_answer('{"changes": []}') is None
    assert parse_answer('{"correction": 7}') is None


def test_quotes_inside_the_sentence_survive_recovery() -> None:
    """The recovery path reads escapes rather than stopping at the first quote."""
    answer = parse_answer('{"correction": "Er sagte \\" ja \\" .", "changes": [')
    assert answer.correction == 'Er sagte " ja " .'


# --- the corpus invariant ---------------------------------------------------


def test_edits_rebuild_the_target(record: dict) -> None:
    """`changes` describes a rewrite; the rewrite has to be the annotator's."""
    assert apply_edits(record["source"].split(), record["edits"]) == record["target"]


# --- the loss weights -------------------------------------------------------


def supervised_pairs(encoded, tokenizer, answer: str):
    """The (token text, weight) pairs over the answer, for readable assertions."""
    ids = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    words = answer.split()
    start = len(encoded.weights) - len(ids["input_ids"]) - 1  # -1 for the EOS
    return list(zip(words, encoded.weights[start : start + len(words)], strict=True))


def test_weights_default_to_the_uniform_loss_every_earlier_run_used(record: dict) -> None:
    """1.0 has to mean 'exactly what Trainer did before', or the control arm of
    an ablation is not a control."""
    encoded = encode(record, FakeTokenizer(), max_length=512)
    for label, weight in zip(encoded.labels, encoded.weights, strict=True):
        assert weight == (0.0 if label == IGNORE else 1.0)


def test_the_prompt_carries_no_weight(record: dict) -> None:
    """The mask and the weights have to agree, or one of them is decoration."""
    encoded = encode(record, FakeTokenizer(), max_length=512, changes_weight=0.5)
    for label, weight in zip(encoded.labels, encoded.weights, strict=True):
        if label == IGNORE:
            assert weight == 0.0
        else:
            assert weight > 0.0


def test_the_changes_list_can_be_down_weighted(record: dict) -> None:
    """The sentence keeps full weight; the list the score never reads does not."""
    tokenizer = FakeTokenizer()
    answer = build_target(record)
    encoded = encode(record, tokenizer, max_length=512, changes_weight=0.5)
    pairs = supervised_pairs(encoded, tokenizer, answer)

    correction = [w for word, w in pairs if answer.index(word) < answer.index(CHANGES_MARKER)]
    changes = [w for word, w in pairs if answer.index(word) > answer.index(CHANGES_MARKER)]
    assert correction and set(correction) == {1.0}
    assert changes and set(changes) == {0.5}


def test_eos_keeps_full_weight_whatever_the_dial_says(record: dict) -> None:
    """Knowing where to stop is how any answer ends, not part of the changes
    list; down-weighting it would make the dial secretly a length knob."""
    encoded = encode(record, FakeTokenizer(), max_length=512, changes_weight=0.1)
    assert encoded.input_ids[-1] == FakeTokenizer.eos_token_id
    assert encoded.weights[-1] == 1.0
