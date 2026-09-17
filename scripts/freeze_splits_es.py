#!/usr/bin/env python
"""Freeze the Spanish evaluation splits and write the per-error-type histogram.

The Spanish counterpart of `freeze_splits.py`. Run after `download_cowsl2h.py`
and `build_cowsl2h_m2.py` have produced `data/processed/cowsl2h/*.m2`: this is
where the test split stops being safe to read casually, the same way German's
test split has been off-limits since Phase 0.

    python scripts/freeze_splits_es.py [--refreeze]
"""

from __future__ import annotations

import argparse

from langlm.config import REPORTS_DIR, SPLIT_MANIFEST, ensure_dirs
from langlm.data import cowsl2h
from langlm.data.m2 import error_type_histogram, parse_m2
from langlm.data.splits import ManifestError, freeze_splits, verify_manifest

#: Splits the histogram may read. `test` is deliberately absent.
HISTOGRAM_SPLITS = ("train", "dev")

REPORT_DIR = REPORTS_DIR / "phase5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refreeze",
        action="store_true",
        help="Drop the existing entry and freeze again (changes the evaluation data!)",
    )
    args = parser.parse_args()

    ensure_dirs()
    paths = cowsl2h.split_files()

    if args.refreeze and SPLIT_MANIFEST.exists():
        _drop_entry(cowsl2h.CORPUS)

    try:
        entry = freeze_splits(
            corpus=cowsl2h.CORPUS,
            split_paths=paths,
            source_url=cowsl2h.RAW_BASE_URL,
        )
        print(f"Froze {cowsl2h.CORPUS} in {SPLIT_MANIFEST}")
        for split, record in entry["splits"].items():
            print(
                f"  {split:<6} {record['sentences']:>6} sentences  "
                f"{record['edits']:>6} edits  sha256 {record['sha256'][:12]}..."
            )
    except ManifestError as exc:
        print(f"Already frozen: {exc}")

    verify_manifest(corpora=[cowsl2h.CORPUS])
    print("Manifest verified.")

    write_histogram()


def write_histogram() -> None:
    """Write the train+dev error-type histogram to the Phase 5 report directory."""
    paths = cowsl2h.split_files()
    histogram = error_type_histogram(
        sentence for split in HISTOGRAM_SPLITS for sentence in parse_m2(paths[split])
    )
    total = sum(histogram.values())

    lines = [
        "# COWS-L2H error-type distribution",
        "",
        f"Computed over the **{' + '.join(HISTOGRAM_SPLITS)}** splits only "
        f"({total:,} edits), typed by `errant_es`. The test split is held out and "
        f"was not read.",
        "",
        "Mirrors `error_type_histogram.md` from Phase 0, which Phase 2 used to weight "
        "German synthetic error injection to match real learner errors. This is the "
        "same table for `corruptors/es.py`.",
        "",
        "| Error type | Count | Share |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| `{error_type}` | {count:,} | {count / total:.2%} |"
        for error_type, count in histogram.most_common()
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "error_type_histogram_es.md"
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
