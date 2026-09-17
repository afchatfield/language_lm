#!/usr/bin/env python
"""Build and freeze the clean Spanish text that Phase 2's synthetic half corrupts.

The Spanish counterpart of `build_clean_corpus.py`. Filters the pinned Leipzig
samples down to sentences that look like whole, undamaged, correctly spelled
prose, drops the 400 sentences reserved for the Spanish overcorrection set, and
writes the survivors as an M2 file whose every sentence carries a noop
annotation -- the same shape `corruptors/es.py` will later fill with real edits.

    python scripts/build_clean_corpus_es.py
    python scripts/build_clean_corpus_es.py --size 5000   # smaller, for a trial

Frozen in the split manifest with its SHA-256, same as `clean_de`.
"""

from __future__ import annotations

import argparse

from langlm.config import REPORTS_DIR, load_config
from langlm.data import clean_es
from langlm.data.m2 import NONE, NOOP_TYPE, Edit, M2Sentence, write_m2
from langlm.data.splits import ManifestError, freeze_splits, verify_manifest

REPORT_DIR = REPORTS_DIR / "phase5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, help="Override the configured corpus size")
    parser.add_argument("--seed", type=int, help="Override the configured shuffle seed")
    parser.add_argument(
        "--jobs",
        type=int,
        default=clean_es.PARSE_JOBS,
        help="Worker processes for the parse. Speed only; the corpus is unaffected.",
    )
    args = parser.parse_args()

    settings = load_config("phase2_es")["clean_corpus"]
    size = args.size or settings["size"]
    seed = args.seed or settings["seed"]
    corpora = settings["corpora"]

    print(f"Filtering {', '.join(corpora)} ...")
    sentences, stats = clean_es.build(corpora, size=size, seed=seed, jobs=args.jobs)
    print(f"{len(sentences):,} sentences kept of {stats.read:,} read")

    from langlm.eval.errant_es import tokenize

    blocks = [
        M2Sentence(
            source_tokens=tokenize(sentence).split(),
            edits=[Edit(start=-1, end=-1, error_type=NOOP_TYPE, correction=NONE)],
        )
        for sentence in sentences
    ]

    clean_es.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    path = clean_es.CLEAN_DIR / "es.m2"
    write_m2(blocks, path)
    print(f"Wrote {path}")

    try:
        freeze_splits(
            clean_es.CORPUS,
            {"all": path},
            source_url="Leipzig Corpora Collection; see configs/phase2_es.yaml",
        )
        print("Frozen in the split manifest.")
    except ManifestError:
        print("Already frozen; verifying the rebuild matches ...")
        verify_manifest(corpora=[clean_es.CORPUS])
        print("Manifest verified: this corpus is byte-identical to the frozen one.")

    write_report(stats, len(sentences), corpora, seed)


def write_report(stats, kept: int, corpora: list[str], seed: int) -> None:
    lines = [
        "# Phase 5: the Spanish clean corpus",
        "",
        f"Source: {', '.join(f'`{c}`' for c in corpora)}, shuffled with seed {seed}.",
        "Mirrors `reports/phase2/clean_corpus.md`'s funnel for German.",
        "",
        "## The funnel",
        "",
        "| Stage | Sentences |",
        "|---|---:|",
    ]
    for label, count in stats.as_rows():
        share = f" ({count / stats.read:.1%})" if stats.read else ""
        lines.append(f"| {label} | {count:,}{share} |")

    lines += [
        "",
        "The pool is then shuffled and parsed until the corpus is full, so the stage "
        "below counts the sentences the parser actually saw rather than the whole pool.",
        "",
        "| Stage | Sentences |",
        "|---|---:|",
    ]
    for label, count in stats.parser_rows():
        share = f" ({count / stats.examined:.1%})" if stats.examined else ""
        lines.append(f"| {label} | {count:,}{share} |")

    lines += [
        "",
        f"**{kept:,}** sentences in the corpus.",
        "",
        "## Why fragments are dropped",
        "",
        "Leipzig is news and Wikipedia, so some of it is headlines and captions, and "
        "subordinate clauses printed as whole sentences. None of that is ungrammatical; "
        "it is poor material to damage on purpose -- a word-order or agreement corruptor "
        "has no main clause to work on, and a model asked to restore a verbless caption "
        "is learning something other than grammar.",
        "",
        "A sentence is kept if it has a finite verb (`VerbForm=Fin` in the morphology -- "
        "`es_core_news_sm` gives Spanish's `tag_` the same universal value as `pos_`, with "
        "no TIGER-style tag that names finiteness the way German's `VVFIN`/`VAFIN`/`VMFIN` "
        "do, so this reads the feature German's version reads off the tag directly) and is "
        "not a stranded subordinate clause -- an opening subordinating conjunction that "
        "governs the root. That second check is the same one German's version makes, "
        "unchanged: `SCONJ` is a universal tag.",
        "",
        "This filter is **not** applied to the Phase 1 overcorrection set, for the same "
        "reason German's is not: that set is frozen once built, and changing its "
        "membership after the fact would move numbers already reported.",
        "",
        "## Held out",
        "",
        f"The {stats.held_out} sentences of the Spanish overcorrection set "
        "(`overcorrection_es`) were removed here rather than remembered later. They come "
        "from these same corpora, and training on them would make the overcorrection "
        "measurement a memory test rather than a test of restraint. Matched on a "
        "whitespace- and case-insensitive key, because the frozen set stores its "
        "sentences tokenised and the Leipzig files do not.",
        "",
        "## What the filters do not ask",
        "",
        "No filter here asks LanguageTool whether a sentence is correct, for the same "
        "reason German's does not: doing so would define correct Spanish, in the "
        "training data, as Spanish LanguageTool has no opinion about -- the standard "
        "Phase 1's baseline is measured against. Spelling is checked against the Spanish "
        "Hunspell dictionary instead, lower-case words only, so a capitalised unknown "
        "(usually a name) does not bias the corpus towards sentences with no proper "
        "nouns in them.",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "clean_corpus_es.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
