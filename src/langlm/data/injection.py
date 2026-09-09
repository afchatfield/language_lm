"""Turning the clean corpus into training examples.

One clean sentence becomes one example: either damaged, with edits that repair
it, or left exactly as it is with an empty edit list. The second kind is not
filler. Falko-MERLIN says 22% of real learner sentences need no correction, and
a model trained only on sentences that contain an error learns that every
sentence contains one -- which is the overcorrection Phase 1 measured
LanguageTool doing to 28.7% of correct published prose.

How many errors a damaged sentence gets, and of which types, is decided by
`configs/phase2.yaml`; see `reports/phase2/injection_weights.md` for where those
weights came from.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from langlm.corruptors import apply, choose, propose
from langlm.data.m2 import NONE, NOOP_TYPE, Edit, M2Sentence


def noop(tokens: Sequence[str]) -> M2Sentence:
    """A correct sentence as a training example: nothing to change."""
    return M2Sentence(
        source_tokens=list(tokens),
        edits=[Edit(start=-1, end=-1, error_type=NOOP_TYPE, correction=NONE)],
    )


@dataclass
class InjectionStats:
    """What the generator actually produced, as opposed to what it was asked for."""

    total: int = 0
    negative: int = 0
    damaged: int = 0
    #: Sentences meant to be damaged that offered no corruption to apply. A
    #: short sentence with no article, comma, umlaut or capitalised noun in it
    #: is not a failure, but a lot of them would mean the rules are too narrow.
    barren: int = 0

    @property
    def negative_share(self) -> float:
        return self.negative / self.total if self.total else 0.0


def corrupt(
    tokens: Sequence[str],
    weights: dict[str, float],
    rng: random.Random,
    errors: tuple[int, int] = (1, 2),
    doc: object | None = None,
    emitted: Counter | None = None,
) -> M2Sentence | None:
    """Damage one sentence, or return None if no rule can find purchase on it."""
    available = propose(tokens, doc=doc)
    if not available:
        return None
    wanted = rng.randint(*errors)
    picked = choose(available, wanted, rng, weights=weights, emitted=emitted)
    if not picked:
        return None
    return apply(tokens, picked)


def generate(
    sentences: Sequence[Sequence[str]],
    weights: dict[str, float],
    seed: int,
    negative_share: float = 0.22,
    errors: tuple[int, int] = (1, 2),
    stats: InjectionStats | None = None,
    docs: Sequence[object] | None = None,
) -> Iterator[M2Sentence]:
    """Yield one training example per clean sentence.

    Args:
        sentences: Tokenised correct sentences.
        weights: Error-type weights, from the phase 2 config.
        seed: Seed for every random choice, so the corpus is reproducible.
        negative_share: Fraction left uncorrupted. The default is the measured
            Falko-MERLIN rate rather than a round number.
        errors: Inclusive range of errors to inject into a damaged sentence.
        stats: Filled in as a side effect, if given.
        docs: Parses, one per sentence, in the same order. Without them the
            word-order and separable-prefix rules produce nothing, and the
            corpus quietly loses the two types the Phase 1 baseline was worst
            at -- so callers that can afford a parse should pass one.
    """
    rng = random.Random(seed)
    stats = stats if stats is not None else InjectionStats()
    emitted: Counter[str] = Counter()

    for index, tokens in enumerate(sentences):
        stats.total += 1
        if rng.random() < negative_share:
            stats.negative += 1
            yield noop(tokens)
            continue
        damaged = corrupt(
            tokens,
            weights,
            rng,
            errors,
            doc=docs[index] if docs is not None else None,
            emitted=emitted,
        )
        if damaged is None:
            # Nothing to break: keep it, as a negative rather than dropping it.
            stats.barren += 1
            stats.negative += 1
            yield noop(tokens)
            continue
        stats.damaged += 1
        emitted.update(edit.error_type for edit in damaged.edits)
        yield damaged


def token_spans(tokens: Sequence[str]) -> list[tuple[int, int]]:
    """Character offsets of each token in the whitespace-joined sentence.

    LanguageTool reports character offsets and the corpus is token-indexed, so
    something has to translate, and doing it here means neither side has to know
    about the other.
    """
    spans = []
    cursor = 0
    for token in tokens:
        spans.append((cursor, cursor + len(token)))
        cursor += len(token) + 1
    return spans
