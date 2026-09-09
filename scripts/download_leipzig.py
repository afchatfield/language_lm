#!/usr/bin/env python
"""Download the Leipzig corpora used for the tokenizer fertility analysis.

python scripts/download_leipzig.py [--corpus deu_news_2024_100K ...]
"""

from __future__ import annotations

import argparse

from langlm.config import LEIPZIG_DIR, load_config
from langlm.data.download import download, extract
from langlm.data.leipzig import archive_url, read_sentences


def main() -> None:
    configured = [entry["leipzig"] for entry in load_config()["corpora"]]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        nargs="*",
        default=configured,
        help=f"Leipzig corpus names (default: {', '.join(configured)})",
    )
    parser.add_argument("--force", action="store_true", help="Re-download")
    args = parser.parse_args()

    for corpus in args.corpus:
        print(f"{corpus} <- {archive_url(corpus)}")
        archive = download(archive_url(corpus), LEIPZIG_DIR / f"{corpus}.tar.gz", args.force)
        extract(archive, LEIPZIG_DIR / corpus, strip_if_single_root=True)
        count = sum(1 for _ in read_sentences(corpus))
        print(f"  {count:,} sentences\n")


if __name__ == "__main__":
    main()
