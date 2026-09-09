"""Building the overcorrection set: German sentences that need no correction.

The sentences come from the Leipzig news and Wikipedia corpora already pinned
for the tokenizer analysis -- published, edited prose, which is as close to
"known correct" as an unlabelled corpus gets. The filters here are not about
grammar; they are about removing the scrape artefacts that would otherwise let a
system look bad for objecting to a broken sentence that no editor ever wrote.

The filters themselves live in `langlm.data.quality`, because Phase 2's training
corpus asks the same question of the same scrapes and the two sets must not
drift apart. Note what is deliberately absent there: any filter that asks a
grammar checker whether the sentence is correct. Screening this set with
LanguageTool would guarantee LanguageTool an overcorrection rate of zero, and
the baseline it is measured against would be its own opinion.

The result is written as an M2 file whose every sentence carries a noop
annotation: a corpus in which the correct answer is to change nothing. That
makes it loadable, freezable and scoreable with everything else in the project.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from langlm.config import PROCESSED_DIR
from langlm.data.leipzig import read_sentences
from langlm.data.m2 import NONE, NOOP_TYPE, Edit, M2Sentence
from langlm.data.quality import is_clean, spelled_correctly

__all__ = ["CORPUS", "build", "candidates", "is_clean", "spelled_correctly"]

#: Where the built set lives.
OVERCORRECTION_DIR = PROCESSED_DIR / "overcorrection"

#: Corpus key under which the set is frozen in the split manifest.
CORPUS = "overcorrection_de"

#: Sentence length window, in tokens. Short sentences give a system too little
#: to object to; very long ones are usually scrape damage rather than prose.
MIN_TOKENS = 8
MAX_TOKENS = 30


def candidates(corpora: Sequence[str], seed: int) -> Iterator[str]:
    """Yield clean, deduplicated sentences from the given Leipzig corpora, shuffled."""
    seen: set[str] = set()
    pool: list[str] = []
    for corpus in corpora:
        for sentence in read_sentences(corpus):
            key = sentence.lower()
            if key in seen or not is_clean(sentence):
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
    from langlm.eval.errant_de import tokenize

    kept: list[M2Sentence] = []
    for sentence in candidates(corpora, seed):
        tokens = tokenize(sentence).split()
        if not MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
            continue
        if not spelled_correctly(tokens):
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
