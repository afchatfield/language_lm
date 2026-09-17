"""Spanish spell checking, used to tell a misspelling from a wrong word choice.

The single most useful question when typing an edit is whether the token the
learner wrote is a Spanish word at all: ``Pesticidas -> Pesticidos`` is a
spelling error, ``mit -> in``'s Spanish shape -- ``con -> en`` -- is a
preposition error, and the strings look about equally similar. This mirrors
`errant_de.spelling` exactly, down to the source: a Hunspell dictionary rather
than a frequency list, because a flat word list cannot tell a real but rare
conjugation from a typo.

``es_ES`` is the LibreOffice dictionary. Peninsular Spanish, not a Latin
American variant, chosen for the same reason `errant_de` uses `de_DE`: it is
the variant LanguageTool's own `es` module and the Leipzig `spa_news` corpus
default to, so the baseline this project compares against and the spelling
model agree on which words exist. COWS-L2H is written by learners across many
first languages and dialect exposures, so some genuinely correct Latin
American forms will be scored as spelling errors -- worth revisiting if the
per-type breakdown ever shows a suspicious concentration in `R:SPELL`.

It is downloaded on first use rather than vendored: triple-licensed
(GPL-3.0 / LGPL-3.0 / MPL-1.1) and this project only reads it locally.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from spylls.hunspell import Dictionary

from langlm.config import RAW_DIR
from langlm.data.download import download

#: Where the dictionary pair lives once fetched.
HUNSPELL_DIR = RAW_DIR / "hunspell"

#: Base name of the dictionary; Hunspell always comes as ``<name>.aff`` plus
#: ``<name>.dic``.
DICTIONARY = "es_ES"

#: Pinned to a commit rather than to ``master``, for the same reason
#: `errant_de.spelling.DICTIONARY_URL` is: a rerun a year from now should score
#: against the same dictionary this project's numbers were produced with.
DICTIONARY_URL = (
    "https://raw.githubusercontent.com/LibreOffice/dictionaries/"
    "762abe74008b94b2ff06db6f4024b59a8254c467/es/"
)


def ensure_dictionary(directory: Path = HUNSPELL_DIR) -> Path:
    """Download the dictionary if it is not already on disk.

    Returns:
        The base path, without the ``.aff`` / ``.dic`` extension.
    """
    base = directory / DICTIONARY
    for suffix in (".aff", ".dic"):
        path = base.with_suffix(suffix)
        if not path.exists():
            download(f"{DICTIONARY_URL}{DICTIONARY}{suffix}", path)
    return base


@cache
def load_dictionary(directory: Path = HUNSPELL_DIR) -> Dictionary:
    """Load the Spanish Hunspell dictionary, fetching it on first use."""
    return Dictionary.from_files(str(ensure_dictionary(directory)))


@cache
def is_known_word(text: str) -> bool:
    """True if `text` is a Spanish word form.

    Unlike German, Spanish does not capitalise common nouns, so there is no
    ``autos``-vs-``Autos`` asymmetry to warn callers about here: the lookup can
    be trusted at face value for ordinary words. Proper nouns and the
    anonymisation tokens COWS-L2H substitutes (``*CITY*``, ``*NUMBER*``) are
    still not in the dictionary and read as unknown, same as any name would.
    """
    return load_dictionary().lookup(text)
