#!/usr/bin/env python
"""Download the Falko-MERLIN GEC corpus and normalise its layout.

python scripts/download_falko_merlin.py [--force]
"""

from __future__ import annotations

import argparse

from langlm.data import falko_merlin
from langlm.data.m2 import parse_m2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract")
    args = parser.parse_args()

    print(f"Falko-MERLIN <- {falko_merlin.RELEASE_URL}")
    splits = falko_merlin.download_and_prepare(force=args.force)

    print("\nPrepared splits:")
    for name, path in splits.items():
        sentences = list(parse_m2(path))
        edits = sum(len(s.edits_for()) for s in sentences)
        correct = sum(1 for s in sentences if s.is_correct())
        print(
            f"  {name:<6} {len(sentences):>6} sentences  "
            f"{edits:>6} edits  "
            f"{correct / len(sentences):>5.1%} already correct  {path}"
        )

    print("\nNext: python scripts/freeze_splits.py")


if __name__ == "__main__":
    main()
