"""German ERRANT: turning two sentences into a list of typed edits.

The M2 scorer answers "how many corrections were right". This answers "right
about what" -- it aligns a system's output with the original, decides where one
error ends and the next begins, and labels each with an error type. That is what
turns a single F0.5 into the per-type table Phase 2 uses to decide which
corruptions to generate more of.

The corpus this project scores against was itself annotated with a German
adaptation of ERRANT (Boyd, 2018), so the labels produced here are meant to be
comparable with the corpus's own. ``scripts/check_errant_de.py`` measures how
comparable by re-annotating the gold corrections and diffing the result against
the gold labels.

ERRANT supplies the alignment and the Edit type; the merging rules and the
classifier are German and live in this package.

Example:
    >>> from langlm.eval.errant_de import annotate                # doctest: +SKIP
    >>> for edit in annotate("Ich sehe der Frau .", "Ich sehe die Frau ."):
    ...     print(edit.start, edit.end, edit.error_type, edit.correction)
    2 3 R:DET:FORM die
"""

from __future__ import annotations

from functools import cache

import spacy
from errant.annotator import Annotator

from langlm.data.m2 import Edit, M2Sentence
from langlm.eval.errant_de import classifier, merger

#: The spacy pipeline. The small model is enough: the classifier reads part of
#: speech, lemma and a handful of dependency labels, none of which the larger
#: models change often enough to justify their cost on an M1.
SPACY_MODEL = "de_core_news_sm"


@cache
def load_annotator(model: str = SPACY_MODEL) -> Annotator:
    """Build the German annotator, loading spacy on first use.

    Raises:
        OSError: with an actionable message if the spacy model is not installed.
    """
    try:
        nlp = spacy.load(model, disable=["ner"])
    except OSError as exc:  # pragma: no cover - depends on the environment
        raise OSError(
            f"The spacy model {model!r} is not installed. Run `python -m spacy download {model}`."
        ) from exc
    return Annotator("de", nlp, merger, classifier)


@cache
def _nlp(model: str = SPACY_MODEL):
    return load_annotator(model).nlp


def tokenize(text: str) -> str:
    """Re-tokenise free text the way the corpus is tokenised.

    A model asked to correct a sentence answers in ordinary German, with the
    full stop attached to the last word. The corpus counts tokens, so an output
    that is not tokenised the same way scores as a whole extra edit on every
    sentence that ends in punctuation.

    Only the tokenizer runs, not the pipeline behind it. Nothing in
    ``de_core_news_sm`` retokenises -- the tagger and parser annotate the
    tokenizer\'s output rather than revising it -- so the tokens are the ones
    ``annotator.parse`` would produce, at a hundredth of the cost. That matters
    because the Phase 2 corpus build tokenises every sentence it reads.
    """
    return " ".join(tok.text for tok in _nlp().tokenizer(text) if not tok.is_space)


@cache
def parse(text: str):
    """Tag and lemmatise a fragment, for callers that need more than tokens.

    Cached because the callers are scorers, which ask about the same handful of
    replacement strings over and over as they compare two annotations.
    """
    return [token for token in _nlp()(text) if not token.is_space]


def annotate(source: str, hypothesis: str) -> list[Edit]:
    """Extract and type the edits that turn `source` into `hypothesis`.

    Args:
        source: The original sentence, whitespace-tokenised.
        hypothesis: The corrected sentence, tokenised the same way.

    Returns:
        Edits in source order, in this project's M2 representation.
    """
    annotator = load_annotator()
    orig = annotator.parse(source)
    cor = annotator.parse(hypothesis)
    return [
        Edit(
            start=edit.o_start,
            end=edit.o_end,
            error_type=edit.type,
            correction=" ".join(tok.text for tok in edit.c_toks),
        )
        for edit in annotator.annotate(orig, cor)
        # "UNK" marks a span ERRANT could not describe as a change; keeping it
        # would count a non-edit against the system.
        if edit.type != "UNK"
    ]


def annotate_corpus(sources: list[str], hypotheses: list[str]) -> list[M2Sentence]:
    """Annotate a whole corpus into M2 sentences, ready to score or write out."""
    if len(sources) != len(hypotheses):
        raise ValueError(f"{len(hypotheses)} hypotheses for {len(sources)} sources")
    return [
        M2Sentence(source_tokens=source.split(), edits=annotate(source, hypothesis))
        for source, hypothesis in zip(sources, hypotheses, strict=True)
    ]
