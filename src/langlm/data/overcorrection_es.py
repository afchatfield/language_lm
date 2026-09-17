"""Building the Spanish overcorrection set: sentences that need no correction.

The Spanish counterpart of `overcorrection.py` -- same reasoning, same filters
from `langlm.data.quality` (now parameterised by `language` so the two sets
share one implementation rather than drifting apart the way `overcorrection.py`
itself warns against), same shape of output. What differs is the source corpus
(`spa_news_2023_100K`, not the German Leipzig pull) and the tokenizer
(`errant_es`, not `errant_de`).

The result is written as an M2 file whose every sentence carries a noop
annotation, exactly as the German set's is -- loadable, freezable and
scoreable with everything else in the project, including a Spanish
LanguageTool baseline once one exists.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from langlm.config import PROCESSED_DIR
from langlm.data.leipzig import read_sentences
from langlm.data.m2 import NONE, NOOP_TYPE, Edit, M2Sentence
from langlm.data.quality import is_clean, spelled_correctly

__all__ = ["CORPUS", "build", "candidates"]

#: Where the built set lives.
OVERCORRECTION_DIR = PROCESSED_DIR / "overcorrection_es"

#: Corpus key under which the set is frozen in the split manifest. Suffixed
#: the same way the German one is (`overcorrection_de`), not left generic --
#: the two sets are drawn from different corpora and scored against different
#: annotators, so nothing should ever ask for "the" overcorrection set.
CORPUS = "overcorrection_es"

#: Same window as German's, for the same reason: short sentences give a system
#: too little to object to, very long ones are usually scrape damage.
MIN_TOKENS = 8
MAX_TOKENS = 30


def candidates(corpora: Sequence[str], seed: int) -> Iterator[str]:
    """Yield clean, deduplicated sentences from the given Leipzig corpora, shuffled."""
    seen: set[str] = set()
    pool: list[str] = []
    for corpus in corpora:
        for sentence in read_sentences(corpus):
            key = sentence.lower()
            if key in seen or not is_clean(sentence, language="es"):
                continue
            seen.add(key)
            pool.append(sentence)
    random.Random(seed).shuffle(pool)
    yield from pool


def build(
    corpora: Sequence[str],
    size: int,
    seed: int = 20260909,
) -> list[M2Sentence]:
    """Sample `size` correct-looking sentences and return them as noop M2 blocks.

    Args:
        corpora: Leipzig corpus names to draw from.
        size: How many sentences to keep.
        seed: Shuffle seed, so the same corpora always give the same set.

    Raises:
        RuntimeError: if the filters leave fewer than `size` sentences, which
            means the filters are wrong rather than the corpus being small.
    """
    from langlm.eval.errant_es import tokenize

    kept: list[M2Sentence] = []
    for sentence in candidates(corpora, seed):
        tokens = tokenize(sentence).split()
        if not MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
            continue
        if not spelled_correctly(tokens, language="es"):
            continue
        kept.append(
            M2Sentence(
                source_tokens=tokens,
                edits=[Edit(start=-1, end=-1, error_type=NOOP_TYPE, correction=NONE)],
            )
        )
        if len(kept) == size:
            return kept

    raise RuntimeError(
        f"Only {len(kept)} sentences survived the filters, wanted {size}. "
        f"Widen the corpora or loosen `is_clean`."
    )
