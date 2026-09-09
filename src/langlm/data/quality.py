"""Is this an undamaged sentence of published prose?

One filter, two callers. Phase 1's overcorrection set needs sentences that a
grammar checker has no business changing; Phase 2's training corpus needs
sentences that are safe to corrupt on purpose. Both are asking the same
question of the same Leipzig scrapes, and asking it twice in two places is how
the two sets quietly drift apart.

Deliberately absent: any filter that asks a grammar checker whether the
sentence is correct. Screening with LanguageTool would guarantee LanguageTool an
overcorrection rate of zero on the Phase 1 set, and it would define "correct
German" in Phase 2's training data as "German LanguageTool has no opinion
about" -- which is precisely the standard the project measured itself against
and beat. The one lexical filter here asks a dictionary, not a checker: every
lower-case word must be a known German word. That catches typos and OCR damage
and leaves the grammar alone.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

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
    and rejecting those would quietly bias the corpus towards sentences with no
    proper nouns in them.
    """
    from langlm.eval.errant_de.spelling import is_known_word

    return all(
        is_known_word(token)
        for token in tokens
        if token.isalpha() and token.islower() and len(token) > 1
    )
