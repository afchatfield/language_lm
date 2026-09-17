"""The clean Spanish text that Phase 2's synthetic half corrupts on purpose.

The Spanish counterpart of `clean_de.py` -- same job, same filters from
`langlm.data.quality`, same held-out exclusion against the overcorrection set
so the training corpus and the metric that watches for overcorrection never
overlap. What differs is the source corpus, the tokenizer, and how
"a whole main clause" is recognised: German's `is_complete` reads a TIGER tag
(`VVFIN`/`VAFIN`/`VMFIN`) that names finiteness directly; `es_core_news_sm`
gives Spanish's `tag_` field the same universal value as `pos_` (`VERB`,
`AUX`, ...), with no finer-grained tag to ask, so this reads `VerbForm=Fin`
out of the morphology instead. Verified against real spacy output before
being written, not assumed from German's shape.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from langlm.config import PROCESSED_DIR
from langlm.data.leipzig import read_sentences
from langlm.data.quality import is_clean, spelled_correctly

#: Where the built corpus lives.
CLEAN_DIR = PROCESSED_DIR / "clean_es"

#: Corpus key under which the corpus is frozen in the split manifest.
CORPUS = "clean_es"

#: Same window as German's, for the same reason: a corruptor needs room to
#: work, and a four-word sentence teaches a model almost nothing about
#: agreement or word order.
MIN_TOKENS = 8
MAX_TOKENS = 60


@dataclass
class Rejections:
    """Why sentences were dropped, so the report can show the funnel. Mirrors
    `clean_de.Rejections` field for field."""

    read: int = 0
    duplicate: int = 0
    unclean: int = 0
    length: int = 0
    misspelled: int = 0
    held_out: int = 0
    pooled: int = 0
    examined: int = 0
    fragment: int = 0
    kept: int = 0

    def as_rows(self) -> list[tuple[str, int]]:
        return [
            ("read from Leipzig", self.read),
            ("duplicate", self.duplicate),
            ("failed the prose filter", self.unclean),
            ("outside the length window", self.length),
            ("unknown lower-case word", self.misspelled),
            ("held out for overcorrection", self.held_out),
            ("pooled for parsing", self.pooled),
        ]

    def parser_rows(self) -> list[tuple[str, int]]:
        return [
            ("parsed", self.examined),
            ("fragment: no finite verb, or a stranded subordinate clause", self.fragment),
            ("kept", self.kept),
        ]


def normalise(sentence: str) -> str:
    """A key that identifies a sentence regardless of how it was tokenised.
    See `clean_de.normalise`: the reasoning is identical."""
    return "".join(sentence.split()).lower()


def held_out_keys() -> set[str]:
    """Normalised keys of every sentence reserved for the Spanish
    overcorrection set.

    Raises:
        FileNotFoundError: if that set has not been built. Training on an
            unknown exclusion list is the failure this is meant to prevent.
    """
    from langlm.data import overcorrection_es
    from langlm.data.splits import load_split

    sentences = load_split(overcorrection_es.CORPUS, "all")
    return {normalise(sentence.source) for sentence in sentences}


def is_complete(doc) -> bool:
    """True if the sentence is a whole main clause.

    Two ways to fail, the same two `clean_de.is_complete` checks. A sentence
    with no finite verb is a headline or a caption rather than a sentence --
    read here as `VerbForm=Fin` in the morphology, since `es_core_news_sm`'s
    `tag_` is the universal POS repeated, not a TIGER-style tag that names
    finiteness on its own the way German's does. And a sentence whose opening
    subordinating conjunction governs the root is a subordinate clause printed
    on its own, the identical check German's version makes -- `SCONJ` is a
    universal tag, so that half ports without change.
    """
    if not any(token.morph.get("VerbForm") == ["Fin"] for token in doc):
        return False
    first = doc[0]
    return not (first.pos_ == "SCONJ" and first.head.dep_ == "ROOT")


def _pipeline():
    """The spacy pipeline, shared with the error annotator. `is_complete`
    reads morphology and one dependency label; the lemmatizer is dead weight.
    """
    import spacy

    return spacy.load("es_core_news_sm", disable=["ner", "lemmatizer"])


#: Same tuning as German's -- see `clean_de.PARSE_JOBS`/`PARSE_CHUNK` for why.
PARSE_JOBS = 8
PARSE_CHUNK = 10_000


def candidates(
    corpora: Sequence[str],
    seed: int,
    stats: Rejections | None = None,
    jobs: int = PARSE_JOBS,
) -> Iterator[str]:
    """Yield clean, deduplicated, non-held-out, syntactically whole sentences.
    See `clean_de.candidates`: the funnel is identical, language swapped."""
    from langlm.eval.errant_es import tokenize

    stats = stats if stats is not None else Rejections()
    excluded = held_out_keys()
    seen: set[str] = set()
    pool: list[str] = []

    for corpus in corpora:
        for sentence in read_sentences(corpus):
            stats.read += 1
            key = normalise(sentence)
            if key in seen:
                stats.duplicate += 1
                continue
            seen.add(key)
            if key in excluded:
                stats.held_out += 1
                continue
            if not is_clean(sentence, language="es"):
                stats.unclean += 1
                continue
            tokens = tokenize(sentence).split()
            if not MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
                stats.length += 1
                continue
            if not spelled_correctly(tokens, language="es"):
                stats.misspelled += 1
                continue
            pool.append(sentence)

    stats.pooled = len(pool)
    random.Random(seed).shuffle(pool)

    nlp = _pipeline()
    for start in range(0, len(pool), PARSE_CHUNK):
        chunk = pool[start : start + PARSE_CHUNK]
        docs = (
            list(nlp.pipe(chunk, batch_size=250, n_process=jobs))
            if jobs > 1
            else list(nlp.pipe(chunk, batch_size=64))
        )
        for sentence, doc in zip(chunk, docs, strict=True):
            stats.examined += 1
            if not is_complete(doc):
                stats.fragment += 1
                continue
            stats.kept += 1
            yield sentence


def build(
    corpora: Sequence[str],
    size: int | None = None,
    seed: int = 20260909,
    jobs: int = PARSE_JOBS,
) -> tuple[list[str], Rejections]:
    """Collect the clean corpus. See `clean_de.build`: identical contract."""
    stats = Rejections()
    stream = candidates(corpora, seed, stats, jobs=jobs)
    sentences = list(stream if size is None else itertools.islice(stream, size))
    return sentences, stats
