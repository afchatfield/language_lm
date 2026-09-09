"""The prompt and target format, and the loss mask that makes training work.

Two decisions live here.

**The output schema.** The model answers with JSON: a list of span-based edits,
each carrying the correction, the ERRANT type, and the explanation. A span-based
answer is checkable -- it can be scored with the same M2 machinery as every
baseline in Phase 1 -- and it is what constrained decoding can be pointed at in
Phase 3. An empty list is a real answer, not a failure: 22% of the training data
says the right thing to do is nothing.

**The loss mask.** Only the answer is learned from. The prompt is context the
model is given, not text it should learn to produce, and training on it teaches
the model to generate German learner sentences -- which is the corpus's input
side, the opposite of the task. The mask is `-100` over the prompt, which is
torch's ignore index, and the test suite asserts it rather than trusting it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

#: The instruction. English, like the explanations, because the reader is
#: learning German. Kept short: a base model pays for every prompt token on
#: every example, and 30,000 examples make a long instruction expensive.
INSTRUCTION = "Correct the German sentence. List each edit with its reason."

SOURCE_PREFIX = "Input:"
TARGET_PREFIX = "Output:"

#: Torch's ignore index for cross-entropy.
IGNORE = -100


def build_prompt(source: str) -> str:
    """The part the model is given."""
    return f"{INSTRUCTION}\n\n{SOURCE_PREFIX} {source}\n{TARGET_PREFIX}"


def build_target(edits: list[dict[str, Any]]) -> str:
    """The part the model has to produce.

    Keys are ordered so the model always emits them in the same order, which is
    what makes a grammar for constrained decoding simple to write. `ensure_ascii`
    is off because German is full of characters that would otherwise be escaped,
    tripling their token cost for no gain.
    """
    payload = {
        "edits": [
            {
                "start": edit["start"],
                "end": edit["end"],
                "type": edit["error_type"],
                "correction": edit["correction"],
                "explanation": edit["explanation"],
            }
            for edit in edits
        ]
    }
    return json.dumps(payload, ensure_ascii=False)


@dataclass
class Encoded:
    """One tokenised example."""

    input_ids: list[int]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.input_ids)


def encode(
    record: dict[str, Any],
    tokenizer,
    max_length: int = 512,
) -> Encoded | None:
    """Tokenise one record into ids and a masked label sequence.

    Args:
        record: A line of the training file: `source`, `target`, `edits`.
        tokenizer: The model's tokenizer.
        max_length: Sequences longer than this are dropped rather than
            truncated. A truncated answer is a syntactically invalid JSON
            object, and teaching the model to emit those is worse than
            teaching it nothing from that example.

    Returns:
        The encoded example, or None if it does not fit.
    """
    prompt = build_prompt(record["source"])
    answer = build_target(record["edits"])

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = [*answer_ids, tokenizer.eos_token_id]

    if len(prompt_ids) + len(answer_ids) > max_length:
        return None

    return Encoded(
        input_ids=prompt_ids + answer_ids,
        # The prompt is context, not a target. Everything before the answer is
        # ignored by the loss.
        labels=[IGNORE] * len(prompt_ids) + answer_ids,
    )


def parse_answer(text: str) -> list[dict[str, Any]] | None:
    """Read the model's answer back, or None if it is not the schema.

    Used at evaluation time. A model that emits invalid JSON has failed the
    example, and that has to be visible as a failure rather than silently
    scored as "no edits" -- which would look identical to correctly declining
    to change a correct sentence.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    edits = payload.get("edits") if isinstance(payload, dict) else None
    if not isinstance(edits, list):
        return None
    return edits
