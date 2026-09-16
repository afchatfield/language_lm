#!/usr/bin/env python
"""Build and freeze the clean German text that Phase 2 corrupts.

Filters the pinned Leipzig samples down to sentences that look like whole,
undamaged, correctly spelled prose, drops the 400 sentences reserved for the
overcorrection set, and writes the survivors as an M2 file whose every sentence
carries a noop annotation -- the same shape the corruptors will later fill with
real edits.

    python scripts/build_clean_corpus.py
    python scripts/build_clean_corpus.py --size 5000   # smaller, for a trial

The corpus is frozen in the split manifest with its SHA-256, so a training run
can say which text it learned from.
"""

from __future__ import annotations

import argparse

from langlm.config import REPORTS_DIR, load_config
from langlm.data import clean_de
from langlm.data.m2 import NONE, NOOP_TYPE, Edit, M2Sentence, write_m2
from langlm.data.splits import ManifestError, freeze_splits, verify_manifest

REPORT_DIR = REPORTS_DIR / "phase2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, help="Override the configured corpus size")
    parser.add_argument("--seed", type=int, help="Override the configured shuffle seed")
    parser.add_argument(
        "--jobs",
        type=int,
        default=clean_de.PARSE_JOBS,
        help="Worker processes for the parse. Speed only; the corpus is unaffected.",
    )
    args = parser.parse_args()

    settings = load_config("phase2")["clean_corpus"]
    size = args.size or settings["size"]
    seed = args.seed or settings["seed"]
    corpora = settings["corpora"]

    print(f"Filtering {', '.join(corpora)} ...")
    sentences, stats = clean_de.build(corpora, size=size, seed=seed, jobs=args.jobs)
    print(f"{len(sentences):,} sentences kept of {stats.read:,} read")

    from langlm.eval.errant_de import tokenize

    blocks = [
        M2Sentence(
            source_tokens=tokenize(sentence).split(),
            edits=[Edit(start=-1, end=-1, error_type=NOOP_TYPE, correction=NONE)],
        )
        for sentence in sentences
    ]

    clean_de.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    path = clean_de.CLEAN_DIR / "de.m2"
    write_m2(blocks, path)
    print(f"Wrote {path}")

    try:
        freeze_splits(
            clean_de.CORPUS,
            {"all": path},
            source_url="Leipzig Corpora Collection; see configs/phase2.yaml",
        )
        print("Frozen in the split manifest.")
    except ManifestError:
        # Already frozen, which is the normal case on a second machine: the
        # manifest is committed and this rebuild should reproduce it exactly.
        # Verifying rather than re-freezing is what turns that expectation into
        # a check -- a different spacy version or a changed Leipzig archive
        # would otherwise train on a quietly different corpus.
        print("Already frozen; verifying the rebuild matches ...")
        verify_manifest(corpora=[clean_de.CORPUS])
        print("Manifest verified: this corpus is byte-identical to the frozen one.")

    write_report(stats, len(sentences), corpora, seed)


def write_report(stats, kept: int, corpora: list[str], seed: int) -> None:
    lines = [
        "# Phase 2: the clean corpus",
        "",
        f"Source: {', '.join(f'`{c}`' for c in corpora)}, shuffled with seed {seed}.",
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
        "Leipzig is news and Wikipedia, so some of it is headlines and captions -- "
        "`Eisdisco am Kirlteich , 14 bis 22 Uhr .` -- and subordinate clauses printed as "
        "whole sentences. None of that is ungrammatical; German headlines are legitimate. "
        "It is poor material to damage on purpose: a word-order corruptor has no main "
        "clause to reorder, and a model asked to restore a verbless caption is learning "
        "something other than grammar.",
        "",
        "A sentence is kept if it has a finite verb and is not a stranded subordinate "
        "clause -- an opening subordinating conjunction that governs the root. "
        "`Obwohl es regnete , ging er spazieren .` is kept, because there the conjunction "
        "governs a subordinate verb and the root belongs to the main clause. The "
        "finite-verb half inherits the parser's mistakes and drops some real sentences "
        "with them, which the corpus can afford.",
        "",
        "This filter is **not** applied to the Phase 1 overcorrection set. That set is "
        "frozen and the published baselines are scored against it; changing its "
        "membership now would move numbers that have already been reported.",
        "",
        "## Held out",
        "",
        f"The {stats.held_out} sentences of the Phase 1 overcorrection set were removed "
        "here rather than remembered later. They are drawn from these same corpora, and "
        "they are the instrument that measures whether a trained model leaves correct "
        "German alone -- training on them would make that measurement a memory test. "
        "The match is made on a whitespace- and case-insensitive key, because the frozen "
        "set stores its sentences tokenised and the Leipzig files do not.",
        "",
        "## What the filters do not ask",
        "",
        "No filter here asks LanguageTool whether a sentence is correct. Doing so would "
        "define correct German, in the training data, as German LanguageTool has no "
        "opinion about -- the standard Phase 1 measured against and beat. Spelling is "
        "checked against a Hunspell dictionary instead, and only for lower-case words: a "
        "capitalised unknown is usually a proper noun, and rejecting those would bias the "
        "corpus towards sentences with no names in them.",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "clean_corpus.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
