#!/usr/bin/env python
"""Type the aligned COWS-L2H pairs with `errant_es` and write M2.

Falko-MERLIN arrived pre-typed: Boyd (2018) annotated it with German ERRANT
before this project ever saw it. COWS-L2H arrived as plain corrected text, so
this project's own annotator has to do that step -- this script is the
Spanish equivalent of the typing half of Falko-MERLIN's release, not of
`download_falko_merlin.py`, which only copies files.

Run after `scripts/download_cowsl2h.py`, before `scripts/freeze_splits_es.py`:

    python scripts/download_cowsl2h.py
    python scripts/build_cowsl2h_m2.py
    python scripts/freeze_splits_es.py
"""

from __future__ import annotations

from langlm.config import ensure_dirs
from langlm.data import cowsl2h
from langlm.data.m2 import write_m2
from langlm.eval import errant_es


def main() -> None:
    ensure_dirs()
    cowsl2h.PROCESSED_DIR_FOR_CORPUS.mkdir(parents=True, exist_ok=True)

    print("Typing COWS-L2H pairs with errant_es (spacy: es_core_news_sm)...")
    for name, _ in cowsl2h.SPLIT_RATIOS:
        pairs = list(cowsl2h.read_pairs(name))
        sources = [p.source for p in pairs]
        targets = [p.target for p in pairs]

        sentences = errant_es.annotate_corpus(sources, targets)
        path = cowsl2h.PROCESSED_DIR_FOR_CORPUS / f"{name}.m2"
        write_m2(sentences, path)

        edits = sum(len(s.edits_for()) for s in sentences)
        noop = sum(1 for s in sentences if s.is_correct())
        print(
            f"  {name:<6} {len(sentences):>6} sentences  {edits:>6} edits  "
            f"{noop / max(len(sentences), 1):>5.1%} already correct  {path}"
        )

    print("\nNext: python scripts/freeze_splits_es.py")


if __name__ == "__main__":
    main()
