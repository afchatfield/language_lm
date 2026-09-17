"""Is this an undamaged sentence of published prose?

One filter, two callers per language. Phase 1's overcorrection set needs
sentences that a grammar checker has no business changing; Phase 2's training
corpus needs sentences that are safe to corrupt on purpose. Both are asking the
same question of the same Leipzig scrapes, and asking it twice in two places is
how the two sets quietly drift apart -- true within German, and true again
between German and Spanish, which is why this stayed one file with a
`language` parameter rather than becoming `quality_es.py`: `ALLOWED` and
`PAIRS` are the only things that actually differ, and they differ in the
characters a language's own punctuation and diacritics use, not in the
question being asked.

Deliberately absent: any filter that asks a grammar checker whether the
sentence is correct. Screening with LanguageTool would guarantee LanguageTool an
overcorrection rate of zero on the Phase 1 set, and it would define "correct
German" (or Spanish) in Phase 2's training data as "the language LanguageTool
has no opinion about" -- which is precisely the standard the project measured
itself against and beat. The one lexical filter here asks a dictionary, not a
checker: every lower-case word must be a known word of the target language.
That catches typos and OCR damage and leaves the grammar alone.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

#: Characters a clean sentence is made of, per language. Anything else --
#: angle brackets, pipes, stray control characters, a stray CJK character from
#: a scrape gone wrong -- marks a scraping artefact rather than prose. Each
#: language's own diacritics and typographic quotation marks are on purpose,
#: which is what the ambiguous-character lint below is being told.
#:
#: Spanish's set was read off a real sample of `spa_news_2023_100K` (30,000
#: sentences, every non-ASCII character counted) rather than guessed: the
#: accented vowels and `ñ` are the highest-volume characters after `,`/`.`, as
#: expected, but so are `«»` and curly `""''` alongside the straight ASCII
#: ones -- Spanish news prose uses both quoting conventions -- and `¿¡` open
#: what `?!` close. `ü` is real (`pingüino`) and rare enough that a scraping
#: survey would not have found it on its own; it is here because it is a
#: letter, not a guess.
ALLOWED = {
    "de": re.compile(r"^[\w \-–—.,;:!?'\"„“”‚‘’()§%&/+°ÄÖÜäöüß€$]+$"),  # noqa: RUF001
    "es": re.compile(
        r"^[\w \-–—.,;:!?¿¡'\"“”‘’«»()%&/+°ºªÁÉÍÓÚÑÜáéíóúñü€$]+$"  # noqa: RUF001
    ),
}

#: Substrings that mark a sentence as web furniture rather than prose. Shared
#: across languages: these are markup and URL artefacts, not language content.
JUNK = ("http", "www.", "@", "|", "©", "...", "…", "[", "]", "<", ">")

#: Paired characters that must balance, per language. An unclosed quote usually
#: means the sentence was cut out of a longer one.
PAIRS = {
    "de": (('"', '"'), ("(", ")"), ("„", "“")),
    "es": (('"', '"'), ("(", ")"), ("«", "»"), ("“", "”"), ("‘", "’")),  # noqa: RUF001
}


def is_clean(sentence: str, language: str = "de") -> bool:
    """True if the sentence looks like a whole, undamaged sentence of prose."""
    if not sentence or sentence[0].islower() or sentence[-1] not in ".!?":
        return False
    if any(junk in sentence for junk in JUNK):
        return False
    if not ALLOWED[language].match(sentence):
        return False
    if sentence.isupper():
        return False
    for left, right in PAIRS[language]:
        if left == right:
            if sentence.count(left) % 2:
                return False
        elif sentence.count(left) != sentence.count(right):
            return False
    return True


#: Where each language's `is_known_word` lives. Lazily imported inside
#: `spelled_correctly` -- each import loads a spacy model and a Hunspell
#: dictionary, so only the language actually asked for pays for it.
_SPELLING_MODULE = {
    "de": "langlm.eval.errant_de.spelling",
    "es": "langlm.eval.errant_es.spelling",
}


def spelled_correctly(tokens: Sequence[str], language: str = "de") -> bool:
    """True if every lower-case word in the sentence is in the dictionary.

    Only lower-case words are checked. A capitalised unknown is far more likely
    to be a name -- ``Selenskyj``, ``Bundesverfassungsgericht`` -- than a typo,
    and rejecting those would quietly bias the corpus towards sentences with no
    proper nouns in them.
    """
    import importlib

    is_known_word = importlib.import_module(_SPELLING_MODULE[language]).is_known_word
    return all(
        is_known_word(token)
        for token in tokens
        if token.isalpha() and token.islower() and len(token) > 1
    )
