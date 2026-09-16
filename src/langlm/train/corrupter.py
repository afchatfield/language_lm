"""The reverse task: writing correct German the way a learner would get it wrong.

Ablation 5 ended on the finding that the hand-written corruptors are spent as a
lever, and the reason it gave was that they "were written for the types that are
easy to generate, and those are largely the frequent, mechanical ones". The
types they cannot reach are the ones the model is worst at -- `R:OTHER`, lexical
choice, is 350 missed edits and 185 wrong-fix attempts at a recall of 0.32, and
no rule will ever produce it, because producing it means knowing which wrong
word a learner would plausibly reach for.

A model can learn that, from the 19,237 Falko-MERLIN training sentences read
backwards. This module is the format for training it. `reports/phase3/headroom.md`
has the published evidence for the size of the effect.

**The schema is a sentence, not JSON.** The forward task answers in JSON because
the `changes` list is a second output that has to be machine-readable. The
reverse task has one output and nothing downstream parses it beyond taking the
line, so the schema is the line. Edits and types are not asked for and would not
be trusted if they were: they are derived afterwards by running German ERRANT
over (generated, clean), which is the same machinery that types the real half of
the corpus.

**The error count is in the prompt.** Iteration 1's failure was density -- a
corpus capped at two errors a sentence, against real learner text averaging 2.55
and 39% carrying three or more. Conditioning the generator on how many mistakes
to make turns that from something to hope for into something to sample: at
generation time the count is drawn from Falko train's own distribution, so the
corpus inherits the density it is meant to imitate, negatives included. A count
of zero is a copy, which is how the 22% no-correction share is kept without a
separate pass to add it.

The count is a steer rather than a contract. What ERRANT derives from the
generated pair is what the record carries; the number in the prompt only shapes
the draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langlm.train.format import IGNORE, Encoded

#: The instruction. English, like the forward task's, and for the same reason:
#: it is scaffolding, not material the learner reads.
INSTRUCTION = "Rewrite the German sentence as a learner would, making {count}."

SOURCE_PREFIX = "Input:"
TARGET_PREFIX = "Output:"


def _count_phrase(count: int) -> str:
    """How the instruction names the number of mistakes to make."""
    if count == 0:
        return "no mistakes"
    if count == 1:
        return "1 mistake"
    return f"{count} mistakes"


def build_prompt(correct: str, count: int) -> str:
    """The part the generator is given: a correct sentence and an error budget."""
    return (
        f"{INSTRUCTION.format(count=_count_phrase(count))}\n\n"
        f"{SOURCE_PREFIX} {correct}\n{TARGET_PREFIX}"
    )


def encode(
    record: dict[str, Any],
    tokenizer,
    max_length: int = 512,
) -> Encoded | None:
    """Tokenise one reversed record into ids and a masked label sequence.

    Args:
        record: `{correct, learner, count}` -- the annotator's sentence, the
            learner's, and how many edits separate them.
        tokenizer: The model's tokenizer.
        max_length: Sequences longer than this are dropped rather than
            truncated, for the reason given in `langlm.train.format`: a
            generator taught to stop mid-sentence produces truncated training
            data for the forward task, which is worse than producing less of it.

    Returns:
        The encoded example, or None if it does not fit.
    """
    prompt = build_prompt(record["correct"], record["count"])
    answer = " " + record["learner"]

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    if tokenizer.eos_token_id is not None:
        answer_ids.append(tokenizer.eos_token_id)

    if len(prompt_ids) + len(answer_ids) > max_length:
        return None

    return Encoded(
        input_ids=prompt_ids + answer_ids,
        labels=[IGNORE] * len(prompt_ids) + answer_ids,
    )


@dataclass
class Generated:
    """One generated pair, before it is judged fit to train on."""

    correct: str
    learner: str
    #: The count the prompt asked for, kept so a run can report how closely the
    #: generator obeyed it.
    asked: int
