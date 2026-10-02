"""Measuring what a system does to text that was already correct.

F0.5 rewards finding errors and says nothing about the sentences a system should
have left alone -- a corpus of learner writing is almost entirely erroneous, so a
system that rewrites everything can score respectably on it. The failure that
actually drives people away from a grammar checker is the opposite one: it
flags a sentence that was fine.

So the metric is deliberately blunt. Take a few hundred sentences of published
German prose, run the system over them, and count how many it changes. Lower is
better and zero is achievable, which makes it the one number in this project
that can be read without context.

The set is built by :mod:`scripts.build_overcorrection_set` and frozen in the
split manifest like any other evaluation data. Its sentences are *assumed*
correct because they were published and edited, not because anyone verified
them, so a small residue of genuine errors is expected; that residue is the same
for every system, which is what makes the comparison fair even though the
absolute number is a slight over-estimate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from langlm.data.m2 import M2Sentence
from langlm.eval import canonical


@dataclass(frozen=True)
class Change:
    """One sentence a system altered, kept for inspection."""

    source: str
    hypothesis: str
    types: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OvercorrectionResult:
    """How much of a correct corpus a system rewrote."""

    sentences: int
    modified: int
    edits: int
    by_type: Counter[str] = field(default_factory=Counter)
    changes: list[Change] = field(default_factory=list)

    @property
    def rate(self) -> float:
        """Fraction of sentences the system changed. The headline number."""
        return self.modified / self.sentences if self.sentences else 0.0

    @property
    def edits_per_sentence(self) -> float:
        return self.edits / self.sentences if self.sentences else 0.0

    def __str__(self) -> str:
        return (
            f"{self.modified}/{self.sentences} sentences modified "
            f"({self.rate:.1%}), {self.edits} spurious edits"
        )


def measure(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    keep_examples: int = 25,
    language: str = "de",
) -> OvercorrectionResult:
    """Count the sentences a system changed, and what it thought it was fixing.

    Args:
        sources: Correct sentences, whitespace-tokenised.
        hypotheses: The system's output for each, tokenised the same way.
        keep_examples: How many altered sentences to keep for the report. The
            examples matter as much as the rate: they say whether the system is
            fighting the tokenisation or genuinely rewriting good German.
        language: Which ERRANT annotator types the spurious edits -- `"de"` or
            `"es"`. Defaults to German, so every existing caller is unchanged.

    Raises:
        ValueError: if the two sequences are of different lengths.
    """
    if len(sources) != len(hypotheses):
        raise ValueError(f"{len(hypotheses)} hypotheses for {len(sources)} sentences")

    # A system that rewrites a decomposed accent into its precomposed form has
    # not changed the sentence, and must not be charged an overcorrection for
    # it. See `langlm.eval.canonical`.
    sources = canonical.nfc_all(sources)
    hypotheses = canonical.nfc_all(hypotheses)

    # Imported here: annotating is what makes this expensive, and a caller that
    # only wants the rate should not pay for spacy at import time.
    if language == "de":
        from langlm.eval.errant_de import annotate
    elif language == "es":
        from langlm.eval.errant_es import annotate
    else:
        raise ValueError(f"No ERRANT annotator for language {language!r}.")

    modified = 0
    edits = 0
    by_type: Counter[str] = Counter()
    changes: list[Change] = []

    for source, hypothesis in zip(sources, hypotheses, strict=True):
        if source.split() == hypothesis.split():
            continue
        modified += 1
        found = annotate(source, hypothesis)
        edits += len(found)
        by_type.update(edit.error_type for edit in found)
        if len(changes) < keep_examples:
            changes.append(
                Change(source, hypothesis, [edit.error_type for edit in found]),
            )

    return OvercorrectionResult(
        sentences=len(sources),
        modified=modified,
        edits=edits,
        by_type=by_type,
        changes=changes,
    )


def measure_split(
    sentences: Sequence[M2Sentence], hypotheses: Sequence[str]
) -> OvercorrectionResult:
    """Convenience wrapper for a frozen overcorrection split."""
    return measure([s.source for s in sentences], hypotheses)
