#!/usr/bin/env python
"""Download LanguageTool's n-gram language model data.

Without this data LanguageTool silently degrades to dictionary spellchecking:
the confusion-set rules that catch das/dass and similar homophone confusions
never fire. Since those are precisely the errors this project cares about, the
n-grams are not optional.

The archives are large (~1.7GB compressed, ~6GB extracted per language), so only
the language you are working on should be downloaded. German is the default;
Spanish is needed for Phase 5.

    python scripts/download_ngrams.py --lang de
"""

from __future__ import annotations

import argparse
import shutil

from langlm.config import NGRAMS_DIR
from langlm.data.download import download, extract

#: LanguageTool publishes one dated archive per language.
ARCHIVES = {
    "de": "ngrams-de-20150819.zip",
    "es": "ngrams-es-20150915.zip",
    "en": "ngrams-en-20150817.zip",
    "fr": "ngrams-fr-20150913.zip",
    "nl": "ngrams-nl-20181229.zip",
}

BASE_URL = "https://languagetool.org/download/ngram-data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="de", choices=sorted(ARCHIVES), help="Language code")
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the zip after extraction (it is ~1.7GB)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract")
    args = parser.parse_args()

    target = NGRAMS_DIR / args.lang
    if target.exists() and not args.force:
        print(f"{target} already exists. Pass --force to redo, or `make clean-ngrams` first.")
        return

    url = f"{BASE_URL}/{ARCHIVES[args.lang]}"
    print(f"n-grams ({args.lang}) <- {url}")
    print("This is roughly 1.7GB compressed and 6GB extracted. Sit tight.\n")

    archive = download(url, NGRAMS_DIR / ARCHIVES[args.lang], force=args.force)
    # The archive already contains a top-level language directory, so extract
    # into the parent and let it land at data/raw/ngrams/<lang>/.
    extract(archive, NGRAMS_DIR)

    if not target.is_dir():
        raise SystemExit(
            f"Expected {target} after extraction. Contents of {NGRAMS_DIR}: "
            f"{sorted(p.name for p in NGRAMS_DIR.iterdir())}"
        )

    if not args.keep_archive:
        archive.unlink()
        print(f"  removed {archive.name}")

    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    subdirs = sorted(p.name for p in target.iterdir() if p.is_dir())
    print(f"\nReady: {target} ({size / 1e9:.1f}GB) containing {subdirs}")
    print("Next: make lt-up && make lt-check")

    free = shutil.disk_usage(NGRAMS_DIR).free
    if free < 10e9:
        print(f"\nHeads up: only {free / 1e9:.1f}GB of disk left.")


if __name__ == "__main__":
    main()
