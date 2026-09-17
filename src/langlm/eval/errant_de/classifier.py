"""Error typing for German edits.

The scheme and the machinery are `errant_core.classifier_kit`, shared with
every language now. What is German here is two things: `is_known_word`, which
tells a misspelling from a wrong word choice using the German Hunspell
dictionary in `errant_de.spelling`; and *implicitly*, the default notion of
"same word, cosmetically different" the kit falls back to when a language
passes no `same_spelling` of its own -- case and spacing, because German
capitalises every noun and that is the large, distinct class it produces.
Spanish's cosmetic axis is diacritics instead, so `errant_es.classifier` passes
its own.
"""

from __future__ import annotations

from langlm.eval.errant_core.classifier_kit import make_classifier
from langlm.eval.errant_de.spelling import is_known_word

#: German passes no `same_spelling`: the kit's default (case- and
#: spacing-insensitive equality) already is the "capitalises every noun" rule.
classify = make_classifier(is_known_word)
