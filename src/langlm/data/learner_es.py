"""Real learner errors, from COWS-L2H, as training examples.

The Spanish counterpart of `learner.py` -- same shape, same reasoning for why
real data has to be the primary signal (see that module's docstring for the
measured German evidence: synthetic-only made the model *worse* than the
untrained baseline on 14 of 15 error types). What differs is only the source:
`cowsl2h`'s frozen M2 splits (`langlm.data.cowsl2h`, typed by `errant_es`) in
place of Falko-MERLIN's, and `explain_type(..., language="es")` for the
type-level explanations, since a COWS-L2H edit records what changed and not
why, the same limitation Falko-MERLIN edits have.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langlm.data.splits import load_split
from langlm.explanations import explain_type

#: The corpus key in the split manifest.
CORPUS = "cowsl2h"

#: The only split this module will read. Dev is the evaluation set and test is
#: frozen until Phase 5's final measurement -- same discipline `learner.py`
#: enforces for Falko-MERLIN, refused here rather than trusted to a caller.
TRAINING_SPLIT = "train"


class SplitMisuseError(RuntimeError):
    """Raised on an attempt to build training data from an evaluation split."""


def records(split: str = TRAINING_SPLIT, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield COWS-L2H sentences in the training-record format.

    Args:
        split: Must be `train`. Anything else raises.
        limit: Stop after this many sentences.

    Yields:
        `{source, target, edits}` where source is the learner's sentence, target
        the corrector's fix, and each edit carries a type-level explanation.

    Raises:
        SplitMisuseError: if asked for any split but train.
    """
    if split != TRAINING_SPLIT:
        raise SplitMisuseError(
            f"Refusing to build training data from the {split!r} split. "
            f"Dev is what the model is measured on and test is frozen; only "
            f"{TRAINING_SPLIT!r} may be trained on."
        )

    sentences = load_split(CORPUS, split)
    if limit:
        sentences = sentences[:limit]

    for sentence in sentences:
        tokens = sentence.source_tokens
        edits = []
        for edit in sentence.edits_for():
            if edit.is_noop:
                continue
            wrong = " ".join(tokens[edit.start : edit.end])
            edits.append(
                {
                    "start": edit.start,
                    "end": edit.end,
                    "error_type": edit.error_type,
                    "correction": edit.correction,
                    # Marks the provenance of the example, mirroring `learner.py`'s
                    # "falko" marker -- so the mix can be counted and a later
                    # iteration can weight the two sources differently.
                    "rule": "cowsl2h",
                    "explanation": explain_type(
                        edit.error_type, wrong=wrong, right=edit.correction, language="es"
                    ),
                }
            )
        mark_moves(edits, tokens)
        yield {"source": sentence.source, "target": sentence.target(), "edits": edits}


def mark_moves(edits: list[dict[str, Any]], tokens: list[str]) -> int:
    """Re-explain the edit pairs that are really one word moving.
    See `learner.mark_moves`: the reasoning and the mechanism are identical.
    """
    from langlm.explanations import load_types

    template = load_types(language="es")["move"]
    removed: dict[tuple[str, str], dict] = {}
    added: dict[tuple[str, str], dict] = {}
    for edit in edits:
        operation, _, rest = edit["error_type"].partition(":")
        category = rest.split(":")[0]
        if operation == "U":
            removed[(category, " ".join(tokens[edit["start"] : edit["end"]]))] = edit
        elif operation == "M":
            added[(category, edit["correction"])] = edit

    paired = 0
    for key in set(removed) & set(added):
        word = key[1]
        for edit in (removed[key], added[key]):
            edit["explanation"] = template.format(word=word).strip()
            edit["rule"] = "cowsl2h-move"
        paired += 1
    return paired
