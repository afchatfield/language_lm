"""Shared ERRANT plumbing, factored out of `errant_de` while porting Spanish.

Everything in this package operates on universal POS tags and dependency
labels, not on any one language's lexicon, and so is identical whichever
language calls it: building an ERRANT `Annotator` from a spacy pipeline plus a
merger and classifier (`build_annotator`), re-tokenising free text to match a
corpus's own tokenisation (`tokenize`), and turning two sentences into typed
edits (`annotate`, `annotate_corpus`).

`errant_de` moved its plumbing here without changing its behaviour -- verified
by `tests/test_errant_de.py`, which is unchanged and still green -- so that
`errant_es` costs a classifier, a spelling backend and a spacy model, not a
second copy of this file. `merger.py` made the same move for the same reason:
it merges alignment operations by universal POS tag (`ADJ`, `AUX`, `DET`, ...),
and the one language-specific case German needed -- separable-prefix verbs --
is itself keyed on a universal tag (`PART`) already, so nothing in it reads
German at all.
"""

from __future__ import annotations

from functools import partial

import spacy
from errant.annotator import Annotator

from langlm.data.m2 import Edit, M2Sentence


def build_annotator(lang: str, spacy_model: str, merger, classifier) -> Annotator:
    """Load `spacy_model` and wire it into an ERRANT `Annotator`.

    Args:
        lang: ERRANT's own language tag. Stored on the `Annotator` but never
            read by it once `nlp`, `merger` and `classifier` are all supplied
            explicitly -- confirmed by reading `errant.annotator.Annotator`
            itself, not assumed -- so this only has to be distinct, not correct
            in any deeper sense.
        spacy_model: A spacy pipeline name, e.g. ``de_core_news_sm``. NER is
            disabled: nothing here reads entities, and it is the slowest
            component in the pipeline for what this measures.
        merger: A module exposing `get_rule_edits(alignment)`.
        classifier: A module exposing `classify(edit)`.

    Raises:
        OSError: with an actionable message if the spacy model is not
            installed.
    """
    try:
        nlp = spacy.load(spacy_model, disable=["ner"])
    except OSError as exc:  # pragma: no cover - depends on the environment
        raise OSError(
            f"The spacy model {spacy_model!r} is not installed. "
            f"Run `python -m spacy download {spacy_model}`."
        ) from exc
    return Annotator(lang, nlp, merger, classifier)


def tokenize(text: str, nlp) -> str:
    """Re-tokenise free text the way a corpus tokenises it.

    A model asked to correct a sentence answers in ordinary text, with the
    full stop attached to the last word. The corpus counts tokens, so an
    output that is not tokenised the same way scores as a whole extra edit on
    every sentence that ends in punctuation.

    Only the tokenizer runs, not the pipeline behind it: nothing that follows
    the tokenizer in a spacy pipeline retokenises, so the tokens are the ones
    parsing the text in full would produce, at a fraction of the cost.
    """
    return " ".join(tok.text for tok in nlp.tokenizer(text) if not tok.is_space)


def annotate(source: str, hypothesis: str, load_annotator) -> list[Edit]:
    """Extract and type the edits that turn `source` into `hypothesis`.

    Args:
        source: The original sentence, whitespace-tokenised.
        hypothesis: The corrected sentence, tokenised the same way.
        load_annotator: A zero-argument callable returning this language's
            `Annotator` -- passed in rather than built here so each language
            keeps its own `@cache`d singleton.

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


def annotate_corpus(sources: list[str], hypotheses: list[str], load_annotator) -> list[M2Sentence]:
    """Annotate a whole corpus into M2 sentences, ready to score or write out."""
    if len(sources) != len(hypotheses):
        raise ValueError(f"{len(hypotheses)} hypotheses for {len(sources)} sources")
    annotate_one = partial(annotate, load_annotator=load_annotator)
    return [
        M2Sentence(source_tokens=source.split(), edits=annotate_one(source, hypothesis))
        for source, hypothesis in zip(sources, hypotheses, strict=True)
    ]
