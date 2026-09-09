#!/usr/bin/env python
"""Freeze the evaluation splits and write the per-error-type histogram.

Freezing records a SHA-256 of every split in `data/splits/manifest.json`, which
is committed. From this point on the test split must not influence any decision
until final evaluation: `langlm.data.splits.load_split` enforces that.

The histogram is computed over train and dev only, never test. Phase 2 uses it
to weight synthetic error injection so the training data resembles the errors
real learners actually make.

    python scripts/freeze_splits.py [--refreeze]
"""

from __future__ import annotations

import argparse

from langlm.config import PHASE0_REPORT_DIR, SPLIT_MANIFEST, ensure_dirs
from langlm.data import falko_merlin
from langlm.data.m2 import error_type_histogram, parse_m2
from langlm.data.splits import ManifestError, freeze_splits, verify_manifest

#: Splits the histogram may read. `test` is deliberately absent.
HISTOGRAM_SPLITS = ("train", "dev")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refreeze",
        action="store_true",
        help="Drop the existing entry and freeze again (changes the evaluation data!)",
    )
    args = parser.parse_args()

    ensure_dirs()
    paths = falko_merlin.split_files()

    if args.refreeze and SPLIT_MANIFEST.exists():
        _drop_entry(falko_merlin.CORPUS)

    try:
        entry = freeze_splits(
            corpus=falko_merlin.CORPUS,
            split_paths=paths,
            source_url=falko_merlin.RELEASE_URL,
        )
        print(f"Froze {falko_merlin.CORPUS} in {SPLIT_MANIFEST}")
        for split, record in entry["splits"].items():
            print(
                f"  {split:<6} {record['sentences']:>6} sentences  "
                f"{record['edits']:>6} edits  sha256 {record['sha256'][:12]}..."
            )
    except ManifestError as exc:
        print(f"Already frozen: {exc}")

    verify_manifest()
    print("Manifest verified.")

    write_histogram()


def write_histogram() -> None:
    """Write the train+dev error-type histogram to the Phase 0 report directory."""
    paths = falko_merlin.split_files()
    histogram = error_type_histogram(
        sentence for split in HISTOGRAM_SPLITS for sentence in parse_m2(paths[split])
    )
    total = sum(histogram.values())

    lines = [
        "# Falko-MERLIN error-type distribution",
        "",
        f"Computed over the **{' + '.join(HISTOGRAM_SPLITS)}** splits only "
        f"({total:,} edits). The test split is held out and was not read.",
        "",
        "Phase 2 weights synthetic error injection to match this distribution, so that "
        "the training data resembles the mistakes real learners make rather than the "
        "mistakes that are easiest to generate.",
        "",
        "| Error type | Count | Share |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| `{error_type}` | {count:,} | {count / total:.2%} |"
        for error_type, count in histogram.most_common()
    ]

    PHASE0_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = PHASE0_REPORT_DIR / "error_type_histogram.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({len(histogram)} error types, {total:,} edits)")


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
