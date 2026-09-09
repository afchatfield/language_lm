"""The output schema and the loss mask.

The mask is the one thing in Phase 3 that fails silently. A run that trains on
its prompts looks exactly like one that does not until the model starts writing
learner German instead of correcting it, days of GPU time later.
"""

from __future__ import annotations

import json

import pytest

from langlm.train.format import (
    IGNORE,
    TARGET_PREFIX,
    build_prompt,
    build_target,
    encode,
    parse_answer,
)


class FakeTokenizer:
    """Whitespace tokenisation with a stable vocabulary, so ids are predictable."""

    eos_token_id = 99

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        return {"input_ids": [len(word) for word in text.split()]}


@pytest.fixture
def record() -> dict:
    return {
        "source": "der hund bellt .",
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


def test_target_is_valid_json_with_ordered_keys(record: dict) -> None:
    text = build_target(record["edits"])
    assert json.loads(text)["edits"][0]["type"] == "R:ORTH"
    # Fixed key order is what makes a decoding grammar simple to write.
    assert list(json.loads(text)["edits"][0]) == [
        "start",
        "end",
        "type",
        "correction",
        "explanation",
    ]


def test_german_is_not_escaped() -> None:
    """Escaping umlauts would triple their token cost for nothing."""
    edits = [
        {
            "start": 0,
            "end": 1,
            "error_type": "R:SPELL",
            "correction": "Bücher",
            "explanation": "Schön.",
        }
    ]
    assert "Bücher" in build_target(edits)
    assert "\\u" not in build_target(edits)


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
    expected = tokenizer(build_target(record["edits"]), add_special_tokens=False)["input_ids"]
    assert supervised == [*expected, tokenizer.eos_token_id]


def test_answer_ends_with_eos(record: dict) -> None:
    """Without it the model never learns to stop, and generation runs to the cap."""
    encoded = encode(record, FakeTokenizer(), max_length=512)
    assert encoded.input_ids[-1] == FakeTokenizer.eos_token_id


def test_overlong_examples_are_dropped_not_truncated(record: dict) -> None:
    """A truncated answer is invalid JSON; better no example than that one."""
    assert encode(record, FakeTokenizer(), max_length=4) is None


def test_correct_sentences_supervise_an_empty_edit_list() -> None:
    """22% of the corpus says the right answer is to change nothing."""
    encoded = encode(
        {"source": "Der Hund bellt .", "target": "Der Hund bellt .", "edits": []},
        FakeTokenizer(),
        max_length=512,
    )
    assert encoded is not None
    assert json.loads(build_target([])) == {"edits": []}


def test_invalid_json_is_a_failure_not_an_empty_answer() -> None:
    """Scoring a malformed answer as "no edits" would hide it behind restraint."""
    assert parse_answer('{"edits": [') is None
    assert parse_answer("not json at all") is None
    assert parse_answer('{"something_else": []}') is None
    assert parse_answer('{"edits": []}') == []
