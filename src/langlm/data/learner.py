"""Real learner errors, from Falko-MERLIN, as training examples.

This is the corpus the first iteration should have been built on. Phase 3's
evaluation is what makes that concrete: fine-tuning on synthetic corruptions
alone made the model *worse* than the untrained baseline on 14 of 15 error
types, including the ones it had seen most of. Injecting 25% spelling errors
took spelling recall from 0.49 to 0.12.

The reason is visible in the two corpora side by side. Falko sentences average
2.55 edits and 39% carry three or more; the synthetic corpus averaged 1.17 and
had none with three. Our inputs were pristine news German with one or two
surgical errors, and real learner text is learner text all the way through --
simpler vocabulary, several interacting mistakes, an L1 accent running through
the whole sentence. The model learned to find a mechanical corruption sitting in
otherwise-perfect prose, which is not the task.

What this corpus does not have is explanations. Falko records what the annotator
changed and its ERRANT type, not why. So the explanation here is type-level --
`langlm.explanations.explain_type` -- and the rule-level ones stay with the
synthetic data, which knows exactly which rule it broke. That split is the point
of mixing them: real data teaches what learner German looks like across all 54
types, synthetic data teaches how to explain 8 of them properly.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langlm.data.splits import load_split
from langlm.explanations import explain_type

#: The corpus key in the split manifest.
CORPUS = "falko_merlin"

#: The only split this module will read. Dev is the evaluation set and test is
#: frozen until Phase 3's final measurement; training on either would make every
#: number afterwards meaningless, so it is refused here rather than trusted to
#: a caller passing the right string.
TRAINING_SPLIT = "train"


class SplitMisuseError(RuntimeError):
    """Raised on an attempt to build training data from an evaluation split."""


def records(split: str = TRAINING_SPLIT, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield Falko-MERLIN sentences in the training-record format.

    Args:
        split: Must be `train`. Anything else raises.
        limit: Stop after this many sentences.

    Yields:
        `{source, target, edits}` where source is the learner's sentence, target
        the annotator's correction, and each edit carries a type-level
        explanation.

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
                    # Marks the provenance of the example, so the mix can be
                    # counted and a later iteration can weight the two sources
                    # differently without re-deriving where each came from.
                    "rule": "falko",
                    "explanation": explain_type(
                        edit.error_type, wrong=wrong, right=edit.correction
                    ),
                }
            )
        mark_moves(edits, tokens)
        yield {"source": sentence.source, "target": sentence.target(), "edits": edits}


def mark_moves(edits: list[dict[str, Any]], tokens: list[str]) -> int:
    """Re-explain the edit pairs that are really one word moving.

    ERRANT has no move operation, so a word that changed position arrives as a
    deletion and an insertion of the same text. Explained one at a time those
    become "remove «gratulieren»" and "add «gratulieren»", which tells a learner
    nothing and reads like a contradiction. Measured at 3.4% of examples.

    Returns:
        How many pairs were re-explained.
    """
    from langlm.explanations import load_types

    template = load_types()["move"]
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
            edit["rule"] = "falko-move"
        paired += 1
    return paired
