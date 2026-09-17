#!/usr/bin/env python
"""Download COWS-L2H, sentence-align it against its corrections, and write
the aligned pairs to `data/interim/cowsl2h/`.

This is as far as the Spanish corpus goes until `errant_es` exists: the pairs
here are plain (source, target) text, not M2, because M2 needs typed edits and
there is no annotator yet to type them with. `freeze_splits_es.py` runs after
`errant_es` turns these into M2.

    python scripts/download_cowsl2h.py [--force]
"""

from __future__ import annotations

import argparse

from langlm.config import ensure_dirs
from langlm.data import cowsl2h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download and re-align")
    args = parser.parse_args()

    ensure_dirs()

    print(f"COWS-L2H <- {cowsl2h.RAW_BASE_URL} ({len(cowsl2h.CSV_FILES)} files)")
    cowsl2h.download_csvs(force=args.force)

    print("Sentence-aligning essays against corrected1 (spacy: es_core_news_sm)...")
    splits, stats = cowsl2h.build_pairs()
    written = cowsl2h.write_pairs(splits)

    print("\nEssays:")
    print(f"  seen                {stats['essays']:>7,}")
    print(f"  with corrected1     {stats['corrected']:>7,}")
    print(f"  sentence counts matched  {stats['aligned']:>7,}")
    print(f"  dropped (mismatch)  {stats['dropped_mismatch']:>7,}")

    print("\nAligned pairs by split (train/dev/test, by learner id):")
    for name, path in written.items():
        pairs = splits[name]
        learners = len({p.learner_id for p in pairs})
        changed = sum(1 for p in pairs if p.source != p.target)
        print(
            f"  {name:<6} {len(pairs):>6} pairs  {changed:>6} changed  "
            f"{learners:>4} learners  {path}"
        )

    print("\nNext: build errant_es, then convert these pairs to M2 and run freeze_splits_es.py")


if __name__ == "__main__":
    main()
