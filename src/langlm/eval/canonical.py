"""Canonical Unicode equivalence, applied where systems are compared.

Spanish accents have two encodings: precomposed (``í``, U+00ED) and decomposed
(``i`` + U+0301). They render identically, mean identically, and a reader
cannot tell them apart -- but they are different strings, so a scorer that
compares text will call one a correction of the other.

COWS-L2H contains both, because learners type on whatever keyboard they have.
The distribution is uneven in a way that matters: 2 of 37,473 train sentences
are decomposed against 14 of 4,716 in dev, sixty times the rate. That is the
learner-id split working as designed -- one student's dead-key habit stays with
that student, and that student is in dev.

The cost lands on exactly one kind of system. A model trained on text that is
99.998% precomposed writes precomposed, so copying a decomposed word through
unchanged still reads as an edit: 18 false positives on dev for changing
nothing. LanguageTool, which splices replacements into the original bytes,
pays none of it. Left alone this quietly discounts the fine-tuned model in the
one comparison Phase 5 exists to make.

Normalising here rather than in `langlm.data.m2` is deliberate. The parser
round-trips the corpus byte for byte and that property is load-bearing; this is
a statement about *comparison*, not about storage. It sits with the tokeniser
in the same seam every system's output already passes through, which is the
seam the identity baseline exists to price -- and German is unaffected either
way: Falko-MERLIN, `clean_de` and the German overcorrection set are already
entirely precomposed, so every German number ever reported is unchanged.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import replace

from langlm.data.m2 import M2Sentence


def nfc(text: str) -> str:
    """Return ``text`` in Normalization Form C."""
    return unicodedata.normalize("NFC", text)


def nfc_all(texts: Sequence[str]) -> list[str]:
    """Normalise a sequence of sentences."""
    return [nfc(text) for text in texts]


def nfc_sentence(sentence: M2Sentence) -> M2Sentence:
    """Normalise a gold sentence: its source tokens and every correction.

    Spans are indices into the token list, and normalisation never changes how
    many tokens there are, so the edits stay valid.
    """
    return replace(
        sentence,
        source_tokens=[nfc(token) for token in sentence.source_tokens],
        edits=[replace(edit, correction=nfc(edit.correction)) for edit in sentence.edits],
    )


def nfc_sentences(sentences: Sequence[M2Sentence]) -> list[M2Sentence]:
    """Normalise a corpus of gold sentences."""
    return [nfc_sentence(sentence) for sentence in sentences]
