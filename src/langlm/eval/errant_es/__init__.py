"""Spanish ERRANT: turning two sentences into a list of typed edits.

The M2 scorer answers "how many corrections were right". This answers "right
about what" -- it aligns a system's output with the original, decides where
one error ends and the next begins, and labels each with an error type. That
is what turns a single F0.5 into the per-type table Phase 2's German build
used to decide which corruptions to generate more of, and the same table
Phase 5 uses for `corruptors/es.py`.

Unlike Falko-MERLIN, COWS-L2H does not ship pre-typed edits: `cowsl2h.py`
aligns essays to their corrections as plain (source, target) pairs, and this
module is what turns those into typed M2. No public Spanish ERRANT existed to
install when Phase 5 started (the multilingual reimplementation covers
en/de/cs/ko/zh, not es, and runs on Stanza rather than spacy), so this ports
`errant_de` instead: same alignment plumbing and merger (`errant_core`, moved
there unchanged), a Spanish spacy model, and a Spanish classifier and spelling
backend (`errant_es.classifier`, `errant_es.spelling`).

Example:
    >>> from langlm.eval.errant_es import annotate                # doctest: +SKIP
    >>> for edit in annotate("Yo tiene un perro .", "Yo tengo un perro ."):
    ...     print(edit.start, edit.end, edit.error_type, edit.correction)
    1 2 R:VERB:FORM tengo
"""

from __future__ import annotations

from functools import cache

from errant.annotator import Annotator

from langlm.data.m2 import Edit, M2Sentence
from langlm.eval import errant_core
from langlm.eval.errant_core import merger
from langlm.eval.errant_es import classifier

#: The spacy pipeline. The small model is enough, for the same reason
#: `errant_de.SPACY_MODEL` gives for German: the classifier reads part of
#: speech, lemma and a handful of dependency labels only.
SPACY_MODEL = "es_core_news_sm"


@cache
def load_annotator(model: str = SPACY_MODEL) -> Annotator:
    """Build the Spanish annotator, loading spacy on first use.

    Raises:
        OSError: with an actionable message if the spacy model is not installed.
    """
    return errant_core.build_annotator("es", model, merger, classifier)


@cache
def _nlp(model: str = SPACY_MODEL):
    return load_annotator(model).nlp


def tokenize(text: str) -> str:
    """Re-tokenise free text the way the corpus is tokenised.

    See `errant_de.tokenize`: the same reasoning applies verbatim, with
    ``es_core_news_sm`` in place of the German model.
    """
    return errant_core.tokenize(text, _nlp())


@cache
def parse(text: str):
    """Tag and lemmatise a fragment, for callers that need more than tokens."""
    return [token for token in _nlp()(text) if not token.is_space]


def annotate(source: str, hypothesis: str) -> list[Edit]:
    """Extract and type the edits that turn `source` into `hypothesis`.

    Args:
        source: The original sentence, whitespace-tokenised.
        hypothesis: The corrected sentence, tokenised the same way.

    Returns:
        Edits in source order, in this project's M2 representation.
    """
    return errant_core.annotate(source, hypothesis, load_annotator)


def annotate_corpus(sources: list[str], hypotheses: list[str]) -> list[M2Sentence]:
    """Annotate a whole corpus into M2 sentences, ready to score or write out."""
    return errant_core.annotate_corpus(sources, hypotheses, load_annotator)
