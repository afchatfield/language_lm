"""Loader for the Leipzig Corpora Collection.

Leipzig publishes fixed-size, sentence-per-line samples of news and Wikipedia
text for a long list of languages, which makes it a convenient and reproducible
basis for the tokenizer fertility analysis: the same named archive gives the
same sentences on every machine.

Each archive contains a ``<corpus>-sentences.txt`` file whose lines are
``<id>\\t<sentence>``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from langlm.config import LEIPZIG_DIR

BASE_URL = "https://downloads.wortschatz-leipzig.de/corpora"


def archive_url(corpus: str) -> str:
    """URL of a Leipzig corpus archive, e.g. ``deu_news_2024_100K``."""
    return f"{BASE_URL}/{corpus}.tar.gz"


def sentences_file(corpus: str, root: Path = LEIPZIG_DIR) -> Path:
    """Locate the ``*-sentences.txt`` file for an extracted corpus.

    Raises:
        FileNotFoundError: if the corpus has not been downloaded.
    """
    directory = root / corpus
    if directory.is_dir():
        matches = sorted(directory.glob("*-sentences.txt"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No sentences file for {corpus!r} under {root}. Run `make corpora` first."
    )


def read_sentences(corpus: str, root: Path = LEIPZIG_DIR) -> Iterator[str]:
    """Yield the sentences of a Leipzig corpus, stripped of their id column."""
    with sentences_file(corpus, root).open(encoding="utf-8") as fh:
        for line in fh:
            _, _, sentence = line.partition("\t")
            sentence = sentence.strip()
            if sentence:
                yield sentence
