"""A small, fixed slice of Falko-MERLIN dev, for screening runs.

Scoring the full 2,503-sentence dev set costs a generation pass per variant,
which is the wrong price for a question like "is dropout 0.05 better than 0.0".
This is the cheap screen that goes in front of it: 600 sentences, stratified by
how many edits the annotator made, so the mix of easy and hard sentences is the
full set's mix rather than a sample's.

**It is a screen, not a result.** Calibrated against the two cached systems
already scored on full dev, a paired delta measured here has a standard
deviation of 0.0046, so it separates effects of about 0.01 and nothing finer --
and the published number stays the full-dev one. At 400 sentences the same
figure is 0.0079 and at 800 it is unchanged, which is what fixes the size.

The indices are deterministic, and every variant is scored on the same
sentences: the sampling error is then common to both arms of a comparison and
cancels out of the difference, which is the only quantity this is for.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

#: Sentences in the screen. See the module docstring for why this number.
SIZE = 600

#: Frozen. Changing it invalidates every comparison already made against it.
SEED = 1000


def _bucket(count: int) -> int:
    """Which stratum a sentence with `count` gold edits belongs to."""
    if count <= 2:
        return count
    return 3 if count <= 4 else 4


def indices(sentences: Sequence, size: int = SIZE, seed: int = SEED) -> list[int]:
    """Positions of the screening subset within `sentences`.

    Args:
        sentences: The dev split, as `M2Sentence`.
        size: Target size; each stratum contributes in proportion to its share.
        seed: Frozen by default.

    Returns:
        Sorted positions into `sentences`.
    """
    strata: dict[int, list[int]] = {}
    for position, sentence in enumerate(sentences):
        edits = len([e for e in sentence.edits_for() if not e.is_noop])
        strata.setdefault(_bucket(edits), []).append(position)

    rng = random.Random(seed)
    chosen: list[int] = []
    for _, pool in sorted(strata.items()):
        chosen += rng.sample(pool, round(len(pool) * size / len(sentences)))
    return sorted(chosen)


def take(sentences: Sequence, size: int = SIZE, seed: int = SEED) -> tuple[list, list[int]]:
    """The screening subset and the positions it came from."""
    chosen = indices(sentences, size=size, seed=seed)
    return [sentences[i] for i in chosen], chosen
