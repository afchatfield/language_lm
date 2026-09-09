"""Building the overcorrection set: German sentences that need no correction.

The sentences come from the Leipzig news and Wikipedia corpora already pinned
for the tokenizer analysis -- published, edited prose, which is as close to
"known correct" as an unlabelled corpus gets. The filters here are not about
grammar; they are about removing the scrape artefacts that would otherwise let a
system look bad for objecting to a broken sentence that no editor ever wrote.

Deliberately absent: any filter that asks a grammar checker whether the sentence
is correct. Screening the set with LanguageTool would guarantee LanguageTool an
overcorrection rate of zero, and the baseline it is measured against would be its
own opinion. The one lexical filter that is applied -- every lower-case word must
be in the German dictionary -- targets typos and OCR damage, and leaves the
grammar untouched. Capitalised unknowns are kept because German news is full of
perfectly correct proper nouns no dictionary has heard of.

The result is written as an M2 file whose every sentence carries a noop
annotation: a corpus in which the correct answer is to change nothing. That
makes it loadable, freezable and scoreable with everything else in the project.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator, Sequence

from langlm.config import PROCESSED_DIR
from langlm.data.leipzig import read_sentences
from langlm.data.m2 import NONE, NOOP_TYPE, Edit, M2Sentence

#: Where the built set lives.
OVERCORRECTION_DIR = PROCESSED_DIR / "overcorrection"

#: Corpus key under which the set is frozen in the split manifest.
CORPUS = "overcorrection_de"

#: Sentence length window, in tokens. Short sentences give a system too little
#: to object to; very long ones are usually scrape damage rather than prose.
MIN_TOKENS = 8
MAX_TOKENS = 30

#: Characters a clean German sentence is made of. Anything else -- angle
#: brackets, pipes, stray control characters -- marks a scraping artefact. The
#: dashes and quotation marks are the German typographic ones on purpose, which
#: is what the ambiguous-character lint below is being told.
ALLOWED = re.compile(r"^[\w \-–—.,;:!?'\"„“”‚‘’()§%&/+°ÄÖÜäöüß€$]+$")  # noqa: RUF001

#: Substrings that mark a sentence as web furniture rather than prose.
JUNK = ("http", "www.", "@", "|", "©", "...", "…", "[", "]", "<", ">")

#: Paired characters that must balance. An unclosed quote usually means the
#: sentence was cut out of a longer one.
PAIRS = (('"', '"'), ("(", ")"), ("„", "“"))


def is_clean(sentence: str) -> bool:
    """True if the sentence looks like a whole, undamaged sentence of prose."""
    if not sentence or sentence[0].islower() or sentence[-1] not in ".!?":
        return False
    if any(junk in sentence for junk in JUNK):
        return False
    if not ALLOWED.match(sentence):
        return False
    if sentence.isupper():
        return False
    for left, right in PAIRS:
        if left == right:
            if sentence.count(left) % 2:
                return False
        elif sentence.count(left) != sentence.count(right):
            return False
    return True


def spelled_correctly(tokens: Sequence[str]) -> bool:
    """True if every lower-case word in the sentence is in the dictionary.

    Only lower-case words are checked. A capitalised unknown is far more likely
    to be a name -- ``Selenskyj``, ``Bundesverfassungsgericht`` -- than a typo,
    and rejecting those would quietly bias the set towards sentences with no
    proper nouns in them.
    """
    from langlm.eval.errant_de.spelling import is_known_word

    return all(
        is_known_word(token)
        for token in tokens
        if token.isalpha() and token.islower() and len(token) > 1
    )


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
