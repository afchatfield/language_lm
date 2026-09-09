#!/usr/bin/env python
"""Build and freeze the overcorrection set.

A few hundred sentences of published German prose that need no correction. The
metric they support is blunt on purpose: the fraction of them a system changes.

The set is written as an M2 file in which every sentence carries a noop
annotation -- a corpus whose correct answer is to change nothing -- and frozen in
the split manifest, so it is version-controlled and hash-checked like the
learner corpus.

    python scripts/build_overcorrection_set.py [--size 400] [--refreeze]
"""

from __future__ import annotations

import argparse

from langlm.config import SPLIT_MANIFEST, ensure_dirs, load_config
from langlm.data import overcorrection
from langlm.data.m2 import write_m2
from langlm.data.splits import ManifestError, freeze_splits, verify_manifest


def main() -> None:
    settings = load_config("phase1")["overcorrection"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=settings["size"])
    parser.add_argument("--seed", type=int, default=settings["seed"])
    parser.add_argument(
        "--refreeze",
        action="store_true",
        help="Drop the existing entry and freeze again (changes the evaluation data!)",
    )
    args = parser.parse_args()

    ensure_dirs()
    print(f"Sampling {args.size} sentences from {', '.join(settings['corpora'])} ...")
    sentences = overcorrection.build(settings["corpora"], args.size, args.seed)

    path = overcorrection.OVERCORRECTION_DIR / "de.m2"
    write_m2(sentences, path)
    tokens = sum(len(s.source_tokens) for s in sentences)
    print(f"Wrote {path} ({len(sentences)} sentences, {tokens / len(sentences):.1f} tokens each)")

    if args.refreeze:
        _drop_entry(overcorrection.CORPUS)
    try:
        freeze_splits(
            corpus=overcorrection.CORPUS,
            split_paths={"all": path},
            source_url="Leipzig Corpora Collection; see configs/phase1.yaml",
        )
        print(f"Froze {overcorrection.CORPUS} in {SPLIT_MANIFEST}")
    except ManifestError as exc:
        print(f"Already frozen: {exc}")
    verify_manifest()
    print("Manifest verified.")

    print("\nFirst five sentences:")
    for sentence in sentences[:5]:
        print(f"  {sentence.source}")


def _drop_entry(corpus: str) -> None:
    import json

    with SPLIT_MANIFEST.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["corpora"].pop(corpus, None)
    with SPLIT_MANIFEST.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"Dropped the existing {corpus} entry (--refreeze).")


if __name__ == "__main__":
    main()
