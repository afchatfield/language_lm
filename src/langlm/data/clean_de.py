"""The clean German text that Phase 2 corrupts on purpose.

Phase 2 builds training data by damaging correct sentences and labelling the
damage, so everything downstream rests on this corpus actually being correct.
The filters are `langlm.data.quality`, shared with the Phase 1 overcorrection
set: same question, same scrapes, one answer.

The one thing this module adds over that shared filter is an exclusion. The
overcorrection set was drawn from these very corpora, and it is the instrument
that measures whether a trained model leaves correct German alone. Train on the
sentences it is scored against and the measurement stops meaning anything --
the model will have seen them, and its restraint will be memory rather than
judgement. So those 400 sentences are removed here, and removed on a key that
survives tokenisation, because the frozen set stores them tokenised and the
Leipzig files do not.

Held-out by construction, in other words, rather than by remembering to.

The second thing it adds is a completeness test. Leipzig is news and Wikipedia,
so a few percent of it is headlines and captions -- `Eisdisco am Kirlteich , 14
bis 22 Uhr .` -- and subordinate clauses printed as whole sentences. None of
that is ungrammatical; German headlines are legitimate. It is simply poor
material to damage on purpose: a word-order corruptor has no main clause to
reorder, and a model asked to restore a verbless caption is learning something
other than grammar.

That test lives here rather than in `quality`, and so applies to Phase 2 only.
The Phase 1 overcorrection set is frozen and the published baselines are scored
against it; quietly changing its membership now would move numbers that have
already been reported.
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
CLEAN_DIR = PROCESSED_DIR / "clean_de"

#: Corpus key under which the corpus is frozen in the split manifest.
CORPUS = "clean_de"

#: Sentence length window, in tokens. Wider at the top than the overcorrection
#: set's 8-30: a corruptor needs room to work, and a subordinate clause with a
#: comma error in it is rarely eight tokens long. The floor is the same, because
#: a four-word sentence teaches a model almost nothing about word order.
MIN_TOKENS = 8
MAX_TOKENS = 60


@dataclass
class Rejections:
    """Why sentences were dropped, so the report can show the funnel.

    A filter that silently removes 90% of the corpus is indistinguishable from
    one that removes 5%, right up until the training run is too small.
    """

    read: int = 0
    duplicate: int = 0
    unclean: int = 0
    length: int = 0
    misspelled: int = 0
    held_out: int = 0
    #: Survivors of the cheap filters -- the pool the parser draws from.
    pooled: int = 0
    #: How many of that pool were actually parsed. The syntactic filter runs
    #: lazily, so this is not the whole pool once a size limit is reached.
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

    The overcorrection set is stored tokenised -- ``Lebensjahr .`` -- and the
    Leipzig files are not, so comparing them literally would let all 400
    sentences straight back into the training data. Dropping whitespace and case
    makes the two representations agree.
    """
    return "".join(sentence.split()).lower()


def held_out_keys() -> set[str]:
    """Normalised keys of every sentence reserved for the overcorrection set.

    Raises:
        FileNotFoundError: if that set has not been built. Training on an
            unknown exclusion list is the failure this is meant to prevent, so
            it is louder than an empty set.
    """
    from langlm.data import overcorrection
    from langlm.data.splits import load_split

    sentences = load_split(overcorrection.CORPUS, "all")
    return {normalise(sentence.source) for sentence in sentences}


#: German finite-verb tags in the TIGER tagset spacy uses: full, auxiliary and
#: modal. Read off the tag rather than the morphology, which is less reliably
#: populated by the small model.
FINITE_TAGS = frozenset({"VVFIN", "VAFIN", "VMFIN"})


def is_complete(doc) -> bool:
    """True if the sentence is a whole main clause.

    Two ways to fail. A sentence with no finite verb is a headline or a caption
    rather than a sentence. And a sentence whose opening subordinating
    conjunction governs the root -- `Weil es Faktoren gibt , die ... werden .`
    -- is a subordinate clause printed on its own; contrast `Obwohl es regnete ,
    ging er spazieren .`, where the conjunction governs a subordinate verb and
    the root is the main clause's.

    The finite-verb test inherits the parser's mistakes: `Ausgehend vom 550
    fertigte Klie ein Modell .` is dropped because the small model does not tag
    `fertigte` as finite. That costs real sentences, which the corpus can
    afford -- there are far more candidates than the corpus needs.
    """
    if not any(token.tag_ in FINITE_TAGS for token in doc):
        return False
    first = doc[0]
    return not (first.pos_ == "SCONJ" and first.head.dep_ == "ROOT")


def _pipeline():
    """The spacy pipeline, shared with the error annotator.

    ``is_complete`` reads a tag, a coarse part of speech and one dependency
    label, so the lemmatizer is dead weight here and is left out.
    """
    import spacy

    return spacy.load("de_core_news_sm", disable=["ner", "lemmatizer"])


#: Worker processes for the parse. The parse is the only expensive stage left
#: and it is embarrassingly parallel, so it is worth spending cores on; more
#: than a handful buys little, because spacy pays to pickle every doc back.
PARSE_JOBS = 8

#: How many sentences to parse before checking whether the caller still wants
#: more. The parse is lazy so that a size limit does not pay for the whole
#: pool, but spacy's workers deadlock if a `pipe` is abandoned half-drained --
#: they block writing results nobody reads. Draining a chunk at a time keeps
#: both: the caller stops within a chunk of where it asked to, and every pipe
#: this opens is finished before the next one starts.
PARSE_CHUNK = 10_000


def candidates(
    corpora: Sequence[str],
    seed: int,
    stats: Rejections | None = None,
    jobs: int = PARSE_JOBS,
) -> Iterator[str]:
    """Yield clean, deduplicated, non-held-out, syntactically whole sentences.

    The cheap filters run over everything; the parse runs a chunk at a time over
    a shuffled pool, so a caller that stops early parses roughly what it needed
    and not 167,000 sentences to throw most of them away.
    """
    from langlm.eval.errant_de import tokenize

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
            if not is_clean(sentence):
                stats.unclean += 1
                continue
            tokens = tokenize(sentence).split()
            if not MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
                stats.length += 1
                continue
            if not spelled_correctly(tokens):
                stats.misspelled += 1
                continue
            pool.append(sentence)

    stats.pooled = len(pool)
    random.Random(seed).shuffle(pool)

    nlp = _pipeline()
    for start in range(0, len(pool), PARSE_CHUNK):
        chunk = pool[start : start + PARSE_CHUNK]
        # Bigger batches with workers, because each batch costs a round trip.
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
    """Collect the clean corpus.

    Args:
        corpora: Leipzig corpus names to draw from.
        size: Stop after this many sentences, or take everything if None.
        seed: Shuffle seed, so the same corpora always give the same corpus.
        jobs: Worker processes for the parse. Affects speed only; the pool is
            shuffled before it is parsed and the docs come back in order, so
            the corpus is the same whatever this is set to.

    Returns:
        The sentences, and the funnel that produced them.
    """
    stats = Rejections()
    stream = candidates(corpora, seed, stats, jobs=jobs)
    sentences = list(stream if size is None else itertools.islice(stream, size))
    return sentences, stats
