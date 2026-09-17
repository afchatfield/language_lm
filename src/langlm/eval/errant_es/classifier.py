"""Error typing for Spanish edits.

The scheme and the machinery are `errant_core.classifier_kit`, shared with
German. What is Spanish here is `is_known_word`, from the Spanish Hunspell
dictionary in `errant_es.spelling`, and `same_spelling`: the one branch point
the kit does not default for.

German's default is case- and spacing-insensitive equality, because German
capitalises every noun and that is the large, distinct cosmetic class it
produces. Spanish does not capitalise common nouns, but it carries a stress
mark on almost the same axis: ``esta -> está``, ``que -> qué``, ``mas -> más``
are the same word with an accent added, restored, or moved, and without this
override the kit's default would call each a real replacement -- typically
`R:VERB` or `R:ADV` -- rather than the orthography error it actually is.

Only the five stressed vowels are folded together, and specifically not by
Unicode decomposition: `unicodedata.normalize("NFKD", "año")` strips the tilde
off ``ñ`` as if it were an accent on ``n``, which would call ``año`` (year) and
``ano`` (anus) the same spelling -- a wrong answer with an unfortunate example
sentence, caught by testing the letter before trusting the library rather than
after. ``ñ`` is not in the translation table, and neither is the diaeresis on
``ü`` (``pingüino``, ``averigüe``): both are letters that make a different
word, not accents that decorate the same one.
"""

from __future__ import annotations

from typing import Any

from langlm.eval.errant_core.classifier_kit import make_classifier
from langlm.eval.errant_es.spelling import is_known_word

#: Stressed vowel -> plain vowel. Deliberately not ``ñ`` or ``ü``: see the
#: module docstring for why folding those would be wrong, not just incomplete.
_UNSTRESS = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")


def _unstressed(tokens: Any) -> str:
    return "".join(tok.lower_ for tok in tokens).translate(_UNSTRESS)


def _same_spelling(o_toks: Any, c_toks: Any) -> bool:
    """Same word, with a stress mark added, removed, or moved."""
    return _unstressed(o_toks) == _unstressed(c_toks)


classify = make_classifier(is_known_word, same_spelling=_same_spelling)
