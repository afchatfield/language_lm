"""ERRANT-style span scoring, and the per-error-type breakdown that comes with it.

Where the MaxMatch scorer re-describes a system's output to fit the gold
annotation, this one compares two annotations directly: both sides are run
through the same German ERRANT annotator, and an edit counts as correct when it
covers the same span and proposes the same replacement. The comparison is
stricter and less forgiving of a differently-drawn span, and in exchange it can
say *which kinds* of error a system finds -- the number Phase 2 needs to decide
which corruptions to generate more of, and Phase 3 needs to decide what to fix.

Two modes, both standard:

* **correction** -- the span and the replacement must both match. This is what
  a user experiences.
* **detection** -- only the span must match. The gap between the two is the
  system noticing something is wrong and proposing the wrong fix, which is a
  different failure with a different remedy.

Counts are bucketed by the *gold* type for hits and misses, and by the
*hypothesis* type for false positives: a spurious correction is filed under what
the system thought it was doing, because that is what has to be fixed.
"""

from __future__ import annotations

from collections import Counter as _Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from langlm.data.m2 import Edit, M2Sentence
from langlm.eval import canonical
from langlm.eval.m2_scorer import Counts

#: An edit reduced to what has to match: source span, and replacement text.
EditKey = tuple[int, int, str]


@dataclass(frozen=True)
class ErrantScore:
    """Overall counts plus a breakdown by error type."""

    overall: Counts
    by_type: dict[str, Counts] = field(default_factory=dict)
    beta: float = 0.5
    mode: str = "correction"
    sentences: int = 0

    @property
    def precision(self) -> float:
        return self.overall.precision

    @property
    def recall(self) -> float:
        return self.overall.recall

    @property
    def f_score(self) -> float:
        return self.overall.f(self.beta)

    def ranked_types(self, minimum: int = 1) -> list[tuple[str, Counts]]:
        """Types ordered by how much of the gold annotation they account for.

        Args:
            minimum: Drop types with fewer than this many gold edits, which
                otherwise fill the table with 0/1 scores that read as signal.
        """
        types = [(t, c) for t, c in self.by_type.items() if c.tp + c.fn >= minimum]
        return sorted(types, key=lambda item: (-(item[1].tp + item[1].fn), item[0]))

    def __str__(self) -> str:
        c = self.overall
        return (
            f"P={self.precision:.4f} R={self.recall:.4f} F{self.beta:g}={self.f_score:.4f} "
            f"(tp={c.tp} fp={c.fp} fn={c.fn}, {self.mode}, {self.sentences} sentences)"
        )


def _keys(edits: Sequence[Edit], mode: str) -> dict[EditKey, str]:
    """Map each edit to what must match, keeping its type."""
    if mode not in {"correction", "detection"}:
        raise ValueError(f"Unknown mode {mode!r}: choose 'correction' or 'detection'.")
    keyed: dict[EditKey, str] = {}
    for edit in edits:
        if edit.is_noop:
            continue
        key = (edit.start, edit.end, edit.correction if mode == "correction" else "")
        # Two edits reduced to the same key are the same claim; keep the first.
        keyed.setdefault(key, edit.error_type)
    return keyed


def _pair(
    gold_edits: dict[EditKey, str],
    hyp_edits: dict[EditKey, str],
    equivalent: Callable[[str, str], bool] | None,
) -> dict[EditKey, EditKey]:
    """Match hypothesis edits to gold edits, exactly first and leniently after.

    Exact matches are taken before any softened one so that a lenient tier can
    never steal a gold edit from the hypothesis edit that got it right, which
    would leave both scored as errors.
    """
    matched = {key: key for key in hyp_edits if key in gold_edits}
    if equivalent is None:
        return matched

    free = [key for key in gold_edits if key not in matched]
    for key in hyp_edits:
        if key in matched:
            continue
        start, end, replacement = key
        for candidate in free:
            if (candidate[0], candidate[1]) != (start, end):
                continue
            if equivalent(candidate[2], replacement):
                matched[key] = candidate
                free.remove(candidate)
                break
    return matched


def score(
    gold: Sequence[M2Sentence],
    hypothesis: Sequence[M2Sentence],
    beta: float = 0.5,
    mode: str = "correction",
    annotator: int = 0,
    equivalent: Callable[[str, str], bool] | None = None,
) -> ErrantScore:
    """Compare a hypothesis annotation with a gold one.

    Args:
        gold: Gold M2 sentences.
        hypothesis: The system's edits for the same sentences, in the same order,
            as produced by :func:`langlm.eval.errant_de.annotate_corpus`.
        beta: F-score beta.
        mode: ``"correction"`` or ``"detection"``.
        annotator: Which gold annotator to score against.
        equivalent: Optional softened equality on replacement text, from
            :mod:`langlm.eval.leniency`. Meaningless in detection mode, where
            the replacement is not compared at all.

    Raises:
        ValueError: if the two sequences differ in length or in their sources,
            which means the two annotations are not of the same text.
    """
    if len(gold) != len(hypothesis):
        raise ValueError(f"{len(hypothesis)} hypothesis sentences for {len(gold)} gold sentences")

    # Same normalisation as the M2 scorer, and needed for the same reason here:
    # the source-equality check below compares tokens, so a decomposed accent on
    # one side would read as "these are different sentences" rather than as the
    # same sentence spelled two ways. See `langlm.eval.canonical`.
    gold = canonical.nfc_sentences(gold)
    hypothesis = canonical.nfc_sentences(hypothesis)

    overall = Counts()
    by_type: dict[str, list[int]] = {}

    def bucket(error_type: str, index: int) -> None:
        by_type.setdefault(error_type, [0, 0, 0])[index] += 1

    for gold_sentence, hyp_sentence in zip(gold, hypothesis, strict=True):
        if gold_sentence.source_tokens != hyp_sentence.source_tokens:
            raise ValueError(
                f"Annotations are of different sentences:\n"
                f"  gold: {gold_sentence.source}\n"
                f"  hyp : {hyp_sentence.source}"
            )
        gold_edits = _keys(gold_sentence.edits_for(annotator), mode)
        hyp_edits = _keys(hyp_sentence.edits_for(), mode)
        matched = _pair(gold_edits, hyp_edits, equivalent if mode == "correction" else None)

        for key, hyp_type in hyp_edits.items():
            if key in matched:
                overall = overall + Counts(tp=1)
                bucket(gold_edits[matched[key]], 0)
            else:
                overall = overall + Counts(fp=1)
                bucket(hyp_type, 1)
        found = set(matched.values())
        for key, gold_type in gold_edits.items():
            if key not in found:
                overall = overall + Counts(fn=1)
                bucket(gold_type, 2)

    return ErrantScore(
        overall=overall,
        by_type={t: Counts(*counts) for t, counts in by_type.items()},
        beta=beta,
        mode=mode,
        sentences=len(gold),
    )


def gold_type_histogram(sentences: Sequence[M2Sentence], annotator: int = 0) -> _Counter[str]:
    """How many gold edits of each type the corpus holds; the table's denominator."""
    histogram: _Counter[str] = _Counter()
    for sentence in sentences:
        for edit in sentence.edits_for(annotator):
            histogram[edit.error_type] += 1
    return histogram
