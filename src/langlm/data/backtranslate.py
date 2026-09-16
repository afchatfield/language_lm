"""Learned error generation: Falko-MERLIN read backwards, then run over clean text.

The hand-written corruptors are a spent lever (`reports/phase3/ablations.md`,
section 5) and `reports/phase3/headroom.md` says why: they generate the
mechanical error types, which are the ones the model is already good at, and
they cannot reach lexical choice, which is the single largest remaining miss.
The published route past that is back-translation -- train a model to *make*
errors, then run it over correct German -- worth +2.4 F0.5 over gold-only for a
7B model, with most of that arriving by 100k generated sentences rather than 1M
([Luhtaru et al. 2024](https://aclanthology.org/2024.findings-emnlp.727/)).

This module is the two ends of that pipeline: the reversed corpus the generator
learns from, and the filter that decides which of its output is fit to train on.
The generator itself is `langlm.train.corrupter` plus
`scripts/train_corrupter.py`, and the run over clean text is
`scripts/generate_backtranslation.py`.

**Why a filter is needed at all.** The generator is sampled rather than decoded
greedily, because greedy decoding makes the same conservative mistake on every
sentence and the point of this is coverage. Sampling buys diversity and pays for
it in failures -- copies when it was asked for errors, paraphrases that rewrite
the sentence rather than damaging it, and the usual degenerate repetition. A
paraphrase is the dangerous one: it produces a pair whose "correction" is not a
correction, and the forward model trained on it learns to rewrite sentences that
are already fine. That is the iteration-1 failure with a new cause, so the
threshold is calibrated against real learner pairs rather than guessed.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langlm.config import INTERIM_DIR
from langlm.data.splits import load_split

#: The corpus key in the split manifest.
CORPUS = "falko_merlin"

#: The only split this reads. Same refusal as `langlm.data.learner`, for the
#: same reason: a generator trained on dev would launder the evaluation set into
#: the training corpus one corruption at a time.
TRAINING_SPLIT = "train"

#: Where a generation run writes its pairs.
PAIRS_DIR = INTERIM_DIR / "backtranslation"

#: Quantile of the real learner/correct divergence distribution that the
#: acceptance threshold is set at. A generated pair is kept if it damages its
#: input no more than 99% of real learner sentences damage theirs. Set below
#: 1.0 because the top of that distribution is Falko's near-unreadable A1
#: scripts, which are real but are not a target a generator should be held to.
DIVERGENCE_QUANTILE = 0.99

#: Longest run of one repeated token that is not treated as degeneration. Three
#: is above anything in Falko train and below what a looping decoder produces.
MAX_REPEAT = 3


class SplitMisuseError(RuntimeError):
    """Raised on an attempt to build a generator from an evaluation split."""


@dataclass
class Pair:
    """One (damaged, correct) pair, however it was made."""

    correct: str
    learner: str
    #: How many mistakes the prompt asked for. -1 for a real Falko pair, which
    #: was not asked for anything.
    asked: int = -1

    def as_json(self) -> str:
        return json.dumps(
            {"correct": self.correct, "learner": self.learner, "asked": self.asked},
            ensure_ascii=False,
        )


def reversed_records(
    split: str = TRAINING_SPLIT, limit: int | None = None
) -> Iterator[dict[str, Any]]:
    """Yield Falko-MERLIN with the arrow turned round.

    The annotator's corrected sentence becomes the input and the learner's
    original becomes the target, which is the whole trick: the corpus that
    teaches a model to correct German teaches a model to break it when read the
    other way, and it is the only corpus in this project that knows what a
    plausible mistake looks like.

    Noop sentences are kept. They are 4,256 of 19,237, and they are what teaches
    the generator to return its input unchanged when asked for no mistakes --
    which is how the corpus's 22% no-correction share survives into the
    generated half without a second pass to bolt it on.

    Args:
        split: Must be `train`. Anything else raises.
        limit: Stop after this many sentences.

    Yields:
        `{correct, learner, count}`.

    Raises:
        SplitMisuseError: if asked for any split but train.
    """
    if split != TRAINING_SPLIT:
        raise SplitMisuseError(
            f"Refusing to train an error generator on the {split!r} split. "
            f"Dev is what the forward model is measured on and test is frozen; "
            f"errors learned from either would end up in the training corpus."
        )

    sentences = load_split(CORPUS, split)
    if limit:
        sentences = sentences[:limit]

    for sentence in sentences:
        edits = [edit for edit in sentence.edits_for() if not edit.is_noop]
        yield {
            "correct": sentence.target(),
            "learner": sentence.source,
            "count": len(edits),
        }


def count_distribution(split: str = TRAINING_SPLIT) -> Counter[int]:
    """How many edits a real learner sentence carries, as a histogram.

    Sampled from at generation time so the generated corpus inherits Falko's
    density rather than whatever the generator happens to prefer. This is the
    statistic iteration 1 got wrong by capping it at two.
    """
    histogram: Counter[int] = Counter()
    for record in reversed_records(split):
        histogram[record["count"]] += 1
    return histogram


def token_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance over tokens."""
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            if left == right:
                current.append(previous[j - 1])
            else:
                current.append(1 + min(previous[j - 1], previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def divergence(correct: str, learner: str) -> float:
    """How far a damaged sentence is from its source, as a fraction of length.

    Token Levenshtein over the longer of the two, so it is comparable across
    sentence lengths and bounded above by 1.0. This is deliberately not the
    ERRANT edit count: ERRANT has to be run to type the edits anyway, but it is
    slow, and this filter exists to throw work away before that point.
    """
    left, right = correct.split(), learner.split()
    longest = max(len(left), len(right))
    return token_distance(left, right) / longest if longest else 0.0


def divergence_threshold(
    quantile: float = DIVERGENCE_QUANTILE, split: str = TRAINING_SPLIT
) -> float:
    """The acceptance threshold, read off real learner data.

    A number picked by hand here would be a guess about how badly a learner
    writes, and the corpus that answers that question is already loaded. The
    threshold is the `quantile`-th percentile of the divergence between real
    learner sentences and their corrections, so "too damaged to be plausible"
    means "more damaged than all but 1% of real learner German".
    """
    values = sorted(
        divergence(record["correct"], record["learner"])
        for record in reversed_records(split)
        if record["count"] > 0
    )
    if not values:
        raise ValueError("No damaged sentences to calibrate against.")
    index = min(len(values) - 1, int(quantile * len(values)))
    return values[index]


def _has_repetition(tokens: Sequence[str], limit: int = MAX_REPEAT) -> bool:
    """True if one token repeats more than `limit` times in a row."""
    run, previous = 0, None
    for token in tokens:
        run = run + 1 if token == previous else 1
        if run > limit:
            return True
        previous = token
    return False


def rejection(pair: Pair, threshold: float) -> str | None:
    """Why this generated pair is unfit to train on, or None if it is fit.

    Returns a reason string rather than a bool so a run can print the funnel.
    A filter that silently drops 90% of a generation run looks exactly like one
    that drops 5%, right up until the corpus is too small -- the same argument
    `langlm.data.clean_de.Rejections` is built on.
    """
    learner, correct = pair.learner.strip(), pair.correct.strip()
    if not learner:
        return "empty"
    if _has_repetition(learner.split()):
        return "degenerate repetition"
    if pair.asked == 0:
        # Asked for no mistakes. Anything but a copy is the generator
        # disobeying, and a "negative" that is not actually its own input would
        # teach the forward model to leave a real error alone.
        return None if learner == correct else "asked for none, damaged anyway"
    if learner == correct:
        return "copied when asked for errors"
    ratio = len(learner.split()) / max(len(correct.split()), 1)
    if not 0.5 <= ratio <= 1.6:
        return "length ratio out of range"
    if divergence(correct, learner) > threshold:
        return "more damaged than real learner German"
    return None


def write_pairs(path: Path, pairs: Sequence[Pair]) -> None:
    """Write generated pairs as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(pair.as_json() + "\n" for pair in pairs), encoding="utf-8")


def read_pairs(path: Path) -> list[Pair]:
    """Read back what a generation run wrote."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No generated pairs at {path}. Make them with "
            f"`python scripts/generate_backtranslation.py`."
        )
    return [
        Pair(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def training_records(pairs: Sequence[Pair]) -> Iterator[dict[str, Any]]:
    """Turn kept pairs into training records, typed and explained.

    The generator knows it made a mistake but not what kind, so the types come
    from running German ERRANT over (damaged, clean) -- the same machinery that
    types the real half of the corpus, and the same reason: what a pair records
    is what changed, not why. Explanations are therefore type-level, which is
    why this does not replace the rule-based corruptors. Those remain the only
    source of rule-level explanations, and `rules/de.yaml` is what they key.

    The damaged side is re-tokenised before it is annotated. The generator
    imitates Falko's whitespace tokenisation closely because that is what it was
    trained on, but "closely" is not "exactly", and a record whose tokenisation
    differs from the evaluation set's is one the forward model learns the wrong
    spacing from.

    Yields:
        `{source, target, edits}` in the shape `scripts/build_training_set.py`
        writes, with `rule` marking the provenance.
    """
    from langlm.data.learner import mark_moves
    from langlm.eval import errant_de
    from langlm.explanations import explain_type

    for pair in pairs:
        source = errant_de.tokenize(pair.learner)
        target = pair.correct
        tokens = source.split()
        edits = []
        for edit in errant_de.annotate(source, target):
            wrong = " ".join(tokens[edit.start : edit.end])
            edits.append(
                {
                    "start": edit.start,
                    "end": edit.end,
                    "error_type": edit.error_type,
                    "correction": edit.correction,
                    "rule": "backtranslation",
                    "explanation": explain_type(
                        edit.error_type, wrong=wrong, right=edit.correction
                    ),
                }
            )
        # `mark_moves` re-explains a delete/insert pair as one word moving, and
        # stamps the provenance it knows: Falko's. Correct it, so the corpus
        # report counts these against the source that actually made them.
        mark_moves(edits, tokens)
        for edit in edits:
            if edit["rule"] == "falko-move":
                edit["rule"] = "backtranslation-move"
        yield {"source": source, "target": target, "edits": edits}
