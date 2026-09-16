"""Scoring what the model said it changed, against what it actually changed.

The model hands back a corrected sentence and a `changes` list. Those are two
different claims and they are scored separately. The sentence is scored by
MaxMatch and ERRANT like every other system's output; this module scores the
list, and it does so against the edits ERRANT derives from *the model's own
correction* rather than against the gold annotation.

That is the useful comparison. Whether the model should have made an edit is the
correction score's question and it is already answered. The question here is
whether the model can say what it just did: a rewrite it cannot describe is one
it cannot explain to a learner, however right it was.

Three numbers come out, and they fail in different directions:

* **Coverage.** Of the rewrites the model made, how many did it list. Silent
  rewrites are the dangerous ones -- the learner sees a word change with no
  reason given.
* **Precision.** Of the changes it listed, how many correspond to a rewrite it
  really made. A claim matching nothing is an invented explanation.
* **Type accuracy.** Of the changes that match, how many carry the error type
  ERRANT assigns. The type is what `rules/de_types.yaml` is keyed on, so this is
  the explanation the learner would be shown, checked.

None of the three reaches 1.0 on a perfect system, and the gap is not the
model's. Pushing the gold correction and the gold changes through this scorer
over Falko-MERLIN dev gives coverage 0.9795, precision 1.0000 and type accuracy
0.8387: ERRANT segments a rewrite slightly differently from the annotator, and
re-derives the annotator's type on 86% of edits, the rest being coarsenings like
`M:PUNCT` to `M:OTHER`. Those are the numbers to read a run against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langlm.data.m2 import M2Sentence


@dataclass(frozen=True)
class ExplanationScore:
    """How well the `changes` list described the correction it came with."""

    #: Edits ERRANT found between the source and the model's correction.
    derived: int = 0
    #: Entries in the model's `changes` lists.
    claimed: int = 0
    #: Claims whose `was` text could be found in the source at all.
    located: int = 0
    #: Claims that land on a rewrite the model actually made.
    matched: int = 0
    #: Matching claims that also carry the type ERRANT assigns.
    typed: int = 0
    #: Derived edits that at least one claim described.
    covered: int = 0

    def __add__(self, other: ExplanationScore) -> ExplanationScore:
        return ExplanationScore(
            *(getattr(self, f) + getattr(other, f) for f in ExplanationScore.__annotations__)
        )

    @property
    def coverage(self) -> float:
        return self.covered / self.derived if self.derived else 0.0

    @property
    def precision(self) -> float:
        return self.matched / self.claimed if self.claimed else 0.0

    @property
    def type_accuracy(self) -> float:
        return self.typed / self.matched if self.matched else 0.0

    def __str__(self) -> str:
        return (
            f"coverage={self.coverage:.4f} precision={self.precision:.4f} "
            f"type={self.type_accuracy:.4f} "
            f"({self.derived} rewrites, {self.claimed} claims, {self.matched} matched)"
        )


def _windows(tokens: Sequence[str], text: str) -> list[tuple[int, int]]:
    """Every place `text` occurs in the sentence, as half-open token spans."""
    needle = text.split()
    if not needle:
        return []
    width = len(needle)
    return [
        (i, i + width)
        for i in range(len(tokens) - width + 1)
        if list(tokens[i : i + width]) == needle
    ]


def _overlaps(window: tuple[int, int], start: int, end: int) -> bool:
    """Does an anchor window cover an edit's span?

    An insertion has a zero-width span and sits *between* two tokens, so it is
    covered when the window's own bounds enclose the gap. Anything else needs a
    real token in common.
    """
    a, b = window
    if start == end:
        return a <= start <= b
    return a < end and start < b


def _fit(window: tuple[int, int], edit, claim_now: str) -> tuple[int, int]:
    """How well a claim fits one of the edits its window covers.

    An anchor widens until it is unique, so in a sentence with several errors a
    window routinely covers a neighbour as well as its own edit. Overlap alone
    then hands the claim to whichever edit comes first, and the type it is
    checked against is that neighbour's. Preferring the edit whose replacement
    text actually appears in the claim's `now` is what keeps the two apart; the
    tighter window breaks the remaining ties.
    """
    a, b = window
    replacement = edit.correction.split()
    said = claim_now.split()
    spoken = bool(replacement) and any(
        said[i : i + len(replacement)] == replacement
        for i in range(len(said) - len(replacement) + 1)
    )
    return (1 if spoken else 0, -(b - a))


def score_sentence(
    source: str,
    annotated: M2Sentence,
    claims: Sequence[dict[str, Any]],
) -> ExplanationScore:
    """Score one sentence's `changes` list against its derived edits."""
    tokens = source.split()
    edits = annotated.edits
    matched_edits: set[int] = set()
    located = matched = typed = 0

    for claim in claims:
        was = str(claim.get("was", ""))
        windows = _windows(tokens, was)
        if not windows:
            continue
        located += 1
        now = str(claim.get("now", ""))
        candidates = [
            (_fit(window, edit, now), index)
            for window in windows
            for index, edit in enumerate(edits)
            if _overlaps(window, edit.start, edit.end)
        ]
        if not candidates:
            continue
        hit = max(candidates)[1]
        matched += 1
        matched_edits.add(hit)
        if str(claim.get("type", "")) == edits[hit].error_type:
            typed += 1

    return ExplanationScore(
        derived=len(edits),
        claimed=len(claims),
        located=located,
        matched=matched,
        typed=typed,
        covered=len(matched_edits),
    )


def score(
    sources: Sequence[str],
    annotated: Sequence[M2Sentence],
    claims: Sequence[Sequence[dict[str, Any]]],
) -> ExplanationScore:
    """Score a corpus. `annotated` comes from `errant_de.annotate_corpus`."""
    if not len(sources) == len(annotated) == len(claims):
        raise ValueError(
            f"{len(sources)} sources, {len(annotated)} annotations, {len(claims)} claim lists"
        )
    total = ExplanationScore()
    for source, sentence, sentence_claims in zip(sources, annotated, claims, strict=True):
        total = total + score_sentence(source, sentence, sentence_claims)
    return total
