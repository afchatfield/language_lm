#!/usr/bin/env python
"""Download OpenThesaurus, the synonym resource behind the lenient scorer.

`langlm.eval.leniency` uses it to decide when the model's word choice is a
defensible alternative to the annotator's rather than a mistake -- `Wirtschaft`
for `Ökonomie`. Only the lenient score needs it; the strict scorer, which is
what the headline numbers are reported at, does not read it at all.

The download is small (~1.6MB) and the data is dual-licensed LGPL / CC-BY-SA 4.0,
so the licence ships next to it.

    python scripts/download_thesaurus.py
"""

from __future__ import annotations

import argparse

from langlm.config import RAW_DIR
from langlm.data.download import download, extract

URL = "https://www.openthesaurus.de/export/OpenThesaurus-Textversion.zip"
DESTINATION = RAW_DIR / "openthesaurus"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract")
    args = parser.parse_args()

    archive = download(URL, DESTINATION / "OpenThesaurus-Textversion.zip", force=args.force)
    extract(archive, DESTINATION)
    archive.unlink(missing_ok=True)

    text = DESTINATION / "openthesaurus.txt"
    if not text.exists():
        raise SystemExit(f"Extracted, but {text} is missing: the archive layout has changed.")
    entries = sum(1 for line in text.open(encoding="utf-8") if not line.startswith("#"))
    print(f"{text}: {entries:,} synonym sets")


if __name__ == "__main__":
    main()
