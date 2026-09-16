"""The prompt and target format, and the loss mask that makes training work.

Three decisions live here.

**The output schema.** The model writes the corrected sentence, then lists what
it changed::

    {"correction": "Der Hund bellt .",
     "changes": [{"was": "der hund", "now": "Der Hund", "type": "R:ORTH"}]}

The corrected sentence comes first because it is the only field the score
depends on, and putting it first means a cut-off answer still yields one. The
schema it replaced asked for token offsets into the source, and the model never
learned them: after three epochs the index tokens were 3.7% of the answer and
43.9% of the remaining loss, at a mean cross-entropy of 0.83 against 0.02 for
the JSON around them. Text anchors cost nothing to learn and lose almost no
precision -- 71.8% of the corpus's edits are already unique in their sentence
and 99.8% become unique with one token of context either side.

**What is not in the schema.** No explanation prose. Every explanation in the
corpus is a pure function of ``(error type, wrong, right)`` -- 69 templates over
68,538 edits, one per corruptor on the synthetic half and one per ERRANT type on
the real half -- so a model generating them is reciting `rules/de_types.yaml`
rather than explaining anything. It cost 39% of the answer's length to do it.
The type is what the template is keyed on, so the model emits the type and
:mod:`langlm.explanations` renders the sentence at serving time. That makes the
explanation checkable, which prose never was: the type can be compared against
what ERRANT derives from the model's own correction.

**The loss mask.** Only the answer is learned from. The prompt is context the
model is given, not text it should learn to produce, and training on it teaches
the model to generate German learner sentences -- which is the corpus's input
side, the opposite of the task. The mask is `-100` over the prompt, which is
torch's ignore index, and the test suite asserts it rather than trusting it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: The instruction. English, like the rest of the scaffolding, because the
#: reader is learning German. Kept short: a base model pays for every prompt
#: token on every example, and 30,000 examples make a long instruction
#: expensive.
INSTRUCTION = "Correct the German sentence, then list what changed."

SOURCE_PREFIX = "Input:"
TARGET_PREFIX = "Output:"

#: Torch's ignore index for cross-entropy.
IGNORE = -100

#: The text that closes the ``correction`` field. Generation can stop here when
#: a run only needs the score, which is a median of 28 answer tokens instead of
#: 84. It cannot appear inside the correction itself: a quote in the sentence is
#: escaped, so the unescaped one that opens this marker is always the field's.
CHANGES_MARKER = ', "changes":'

#: How far an anchor may widen on each side before giving up on being unique.
#: Three covers all but one edit in the corpus.
MAX_ANCHOR_CONTEXT = 3


def build_prompt(source: str) -> str:
    """The part the model is given."""
    return f"{INSTRUCTION}\n\n{SOURCE_PREFIX} {source}\n{TARGET_PREFIX}"


def apply_edits(tokens: Sequence[str], edits: list[dict[str, Any]]) -> str:
    """Rebuild the corrected sentence from span edits, left to right.

    The corpus records ``source``, ``target`` and span-indexed ``edits``, and
    this is what makes the three agree. Nothing at inference time uses it -- the
    model hands back the sentence directly now -- but the corpus invariant is
    worth a function, because a record whose edits do not reach its target is a
    record whose ``changes`` list would describe a rewrite that never happened.
    """
    ordered = sorted(edits, key=lambda edit: edit["start"])
    result: list[str] = []
    cursor = 0
    for edit in ordered:
        start, end = edit["start"], edit["end"]
        if start < cursor:
            continue
        result.extend(tokens[cursor:start])
        result.extend(edit["correction"].split())
        cursor = end
    result.extend(tokens[cursor:])
    return " ".join(result)


def _occurrences(tokens: Sequence[str], window: Sequence[str]) -> int:
    width = len(window)
    return sum(1 for i in range(len(tokens) - width + 1) if tokens[i : i + width] == list(window))


def anchor(
    tokens: Sequence[str],
    start: int,
    end: int,
    correction: str,
    max_context: int = MAX_ANCHOR_CONTEXT,
) -> tuple[str, str]:
    """Describe one edit as a before/after pair of quoted text.

    The window is the edit's own span, widened by the fewest tokens that make it
    occur exactly once in the sentence. An insertion has an empty span and so
    always takes at least one token of context; without it ``was`` would be the
    empty string and point at nothing.

    Returns:
        ``(was, now)`` -- the text the window covers, and what it becomes.
    """
    a, b = start, end
    for pad in range(max_context + 1):
        a, b = max(0, start - pad), min(len(tokens), end + pad)
        if a == b:
            continue  # a zero-width window names no text; widen it
        if _occurrences(tokens, tokens[a:b]) == 1:
            break
    was = " ".join(tokens[a:b])
    now = " ".join([*tokens[a:start], *correction.split(), *tokens[end:b]])
    return was, now


def build_target(record: dict[str, Any]) -> str:
    """The part the model has to produce.

    Keys are ordered so the model always emits them in the same order, which is
    what makes a grammar for constrained decoding simple to write, and what lets
    a scoring run stop at :data:`CHANGES_MARKER`. ``ensure_ascii`` is off because
    German is full of characters that would otherwise be escaped, tripling their
    token cost for no gain.
    """
    tokens = record["source"].split()
    changes = []
    for edit in record["edits"]:
        was, now = anchor(tokens, edit["start"], edit["end"], edit["correction"])
        changes.append({"was": was, "now": now, "type": edit["error_type"]})
    return json.dumps({"correction": record["target"], "changes": changes}, ensure_ascii=False)


@dataclass
class Encoded:
    """One tokenised example."""

    input_ids: list[int]
    labels: list[int]
    #: Per-token multiplier on the loss. 0.0 wherever the label is masked. See
    #: :func:`encode` for why this is not simply 1.0 everywhere.
    weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = [0.0 if label == IGNORE else 1.0 for label in self.labels]

    def __len__(self) -> int:
        return len(self.input_ids)


def encode(
    record: dict[str, Any],
    tokenizer,
    max_length: int = 1024,
    changes_weight: float = 1.0,
) -> Encoded | None:
    """Tokenise one record into ids, a masked label sequence and token weights.

    Args:
        record: A line of the training file: `source`, `target`, `edits`.
        tokenizer: The model's tokenizer.
        max_length: Sequences longer than this are dropped rather than
            truncated. Teaching the model to stop mid-answer is worse than
            teaching it nothing from that example -- and the cost of a low cap
            is not how many records it drops but which: it takes them all from
            the many-edit tail.
        changes_weight: Multiplier on the loss over the ``changes`` list. The
            default of 1.0 is uniform cross-entropy over the answer, which is
            what every run before this one used.

            It is worth being deliberate about, because uniform is not neutral:
            measured over the corpus, the ``changes`` list is **70.1%** of the
            supervised tokens and the corrected sentence only 29.9%, so by
            default seven tenths of the gradient goes to a field the F0.5 score
            does not read. The list is not free to drop -- it is the diagnostic
            in :mod:`langlm.eval.explanation_scorer`, and generating it may well
            be what makes the model attend to its own edits -- so this is a dial
            rather than a switch.

    Returns:
        The encoded example, or None if it does not fit.
    """
    prompt = build_prompt(record["source"])
    answer = build_target(record)

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    answer_ids = list(encoded["input_ids"])

    # Split on character offsets rather than by tokenising the two halves
    # separately: BPE merges across the join, so a token can straddle the marker
    # and re-tokenising each half would silently produce a different sequence
    # from the one the model is trained on.
    split = answer.index(CHANGES_MARKER)
    weights = [1.0 if start < split else changes_weight for start, _ in encoded["offset_mapping"]]

    if tokenizer.eos_token_id is not None:
        answer_ids.append(tokenizer.eos_token_id)
        # Full weight regardless. Knowing where to stop is not part of the
        # changes list; it is how any answer ends, and down-weighting it would
        # make the dial quietly also a knob on generation length.
        weights.append(1.0)

    if len(prompt_ids) + len(answer_ids) > max_length:
        return None

    return Encoded(
        input_ids=prompt_ids + answer_ids,
        # The prompt is context, not a target. Everything before the answer is
        # ignored by the loss.
        labels=[IGNORE] * len(prompt_ids) + answer_ids,
        weights=[0.0] * len(prompt_ids) + weights,
    )


@dataclass
class Answer:
    """What the model said, as far as it can be read back."""

    #: The corrected sentence. This is the only field the score depends on.
    correction: str
    #: What the model says it changed, for the explanation metrics.
    changes: list[dict[str, Any]] = field(default_factory=list)
    #: False when the answer was cut off and only the correction survived.
    complete: bool = True


#: A ``"correction": "..."`` field, allowing for escapes inside the value. Used
#: when the answer is not parseable as a whole.
_CORRECTION = re.compile(r'"correction"\s*:\s*"((?:[^"\\]|\\.)*)"')


def parse_answer(text: str) -> Answer | None:
    """Read the model's answer back, or None if there is no sentence in it.

    Whole-answer JSON first, then the correction field on its own. The fallback
    is the point of the schema rather than a concession to it: the previous
    format put the sentence behind a list of edits, so an answer cut off by the
    generation cap parsed as nothing and was scored as leaving the sentence
    alone -- which is how a quarter of Falko-MERLIN dev came to be counted as
    restraint. Here the sentence is first, so a cut-off answer still has it, and
    only the explanations are lost.
    """
    text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict) and isinstance(payload.get("correction"), str):
        changes = payload.get("changes")
        return Answer(payload["correction"], changes if isinstance(changes, list) else [])

    found = _CORRECTION.search(text)
    if found is None:
        return None
    try:
        correction = json.loads(f'"{found.group(1)}"')
    except json.JSONDecodeError:
        return None
    return Answer(correction, [], complete=False)
