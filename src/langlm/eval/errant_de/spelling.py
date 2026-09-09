"""German spell checking, used to tell a misspelling from a wrong word choice.

The single most useful question when typing an edit is whether the token the
learner wrote is a German word at all. ``Pestizides -> Pestiziden`` is a spelling
error; ``mit -> in`` is a preposition error; the strings look equally similar.
ERRANT's English classifier answers that with a Hunspell word list, and this does
the same with the Hunspell dictionary German actually needs -- a flat word list
cannot work for a language that builds ``Universitätsabschlüsse`` on demand, and
a frequency list scraped from news text calls every such compound a typo.

``de_DE_frami`` is the igerman98-derived dictionary shipped with LibreOffice. It
is downloaded on first use rather than vendored: it is triple-licensed
(GPL-2.0 / LGPL-2.1 / MPL-1.1) and this project only reads it locally.
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
DICTIONARY = "de_DE_frami"

#: Pinned to a commit rather than to ``master`` so that a rerun a year from now
#: scores against the same dictionary this project's numbers were produced with.
DICTIONARY_URL = (
    "https://raw.githubusercontent.com/LibreOffice/dictionaries/"
    "31bc2a1104a1cd175f900902f994c76dea35c763/de/"
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
    """Load the German Hunspell dictionary, fetching it on first use."""
    return Dictionary.from_files(str(ensure_dictionary(directory)))


@cache
def is_known_word(text: str) -> bool:
    """True if `text` is a German word form.

    Cased exactly as written: German capitalises nouns and nothing else, so
    ``autos`` is genuinely not a word even though ``Autos`` is. Callers that care
    about case alone should have classified the edit as an orthography error
    before reaching this.
    """
    return load_dictionary().lookup(text)
