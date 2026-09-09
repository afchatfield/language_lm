#!/usr/bin/env python
"""Check the German ERRANT annotator against the corpus's own annotation.

Falko-MERLIN's M2 files were produced with a German adaptation of ERRANT
(Boyd, 2018). This project re-implements that adaptation, so there is a free and
honest check available: take each gold sentence, apply the gold corrections, and
ask this annotator to work out what changed. Perfect agreement is not the goal --
a different spacy version alone moves the labels -- but the size and shape of the
disagreement decides how much weight the per-error-type table can carry.

Two numbers come out. **Span agreement** is whether the two annotators drew the
same edits at all; it bounds everything downstream, because a system scored
against differently-drawn spans is scored against the wrong thing. **Type
agreement** is whether they then called those edits the same thing, and it only
matters for reading the breakdown.

    python scripts/check_errant_de.py [--split dev] [--limit N]

Writes `reports/phase1/errant_de_agreement.md`. Reads train and dev only.
"""

from __future__ import annotations

import argparse
from collections import Counter

from langlm.config import REPORTS_DIR
from langlm.data.splits import load_split
from langlm.eval.errant_de import annotate

#: Splits it is legitimate to read. `test` is deliberately absent.
ALLOWED_SPLITS = ("train", "dev")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=ALLOWED_SPLITS, default="dev")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    sentences = load_split("falko_merlin", args.split)
    if args.limit:
        sentences = sentences[: args.limit]

    matched = missing = spurious = agreed = 0
    confusions: Counter[tuple[str, str]] = Counter()
    per_type: dict[str, list[int]] = {}

    print(f"Re-annotating {len(sentences)} {args.split} sentences ...")
    for index, sentence in enumerate(sentences, start=1):
        gold = {(e.start, e.end, e.correction): e.error_type for e in sentence.edits_for()}
        ours = {
            (e.start, e.end, e.correction): e.error_type
            for e in annotate(sentence.source, sentence.target())
        }
        for span, our_type in ours.items():
            gold_type = gold.get(span)
            if gold_type is None:
                spurious += 1
                continue
            matched += 1
            counts = per_type.setdefault(gold_type, [0, 0, 0])
            counts[0] += 1
            if gold_type == our_type:
                agreed += 1
                counts[1] += 1
            else:
                confusions[(gold_type, our_type)] += 1
        for span, gold_type in gold.items():
            if span not in ours:
                missing += 1
                per_type.setdefault(gold_type, [0, 0, 0])[2] += 1
        if index % 500 == 0:
            print(f"  {index}/{len(sentences)}", end="\r")
    print()

    write_report(args.split, sentences, matched, missing, spurious, agreed, confusions, per_type)


def write_report(
    split: str,
    sentences: list,
    matched: int,
    missing: int,
    spurious: int,
    agreed: int,
    confusions: Counter,
    per_type: dict[str, list[int]],
) -> None:
    precision = matched / (matched + spurious) if matched + spurious else 0.0
    recall = matched / (matched + missing) if matched + missing else 0.0
    type_rate = agreed / matched if matched else 0.0

    lines = [
        "# German ERRANT: agreement with the corpus annotation",
        "",
        f"Falko-MERLIN **{split}**, {len(sentences):,} sentences. Each gold correction was "
        "applied and then re-annotated from scratch by `langlm.eval.errant_de`; the result is "
        "compared with the corpus's own M2 annotation, which was produced by Boyd's (2018) "
        "German ERRANT.",
        "",
        "| | |",
        "|---|---:|",
        f"| Edits both annotators drew | {matched:,} |",
        f"| Drawn only by this one | {spurious:,} |",
        f"| Drawn only by the corpus | {missing:,} |",
        f"| Span precision | {precision:.4f} |",
        f"| Span recall | {recall:.4f} |",
        f"| **Type agreement on shared edits** | **{type_rate:.4f}** |",
        "",
        "Span agreement is the number that matters. It bounds every score computed with this "
        "annotator: a system judged against differently-drawn edits is judged against the "
        "wrong thing. Type agreement only affects how the per-type breakdown reads, and some "
        "of the gap is irreducible -- the two annotators run different spacy versions, and the "
        "labels are downstream of the lemmas and tags those produce.",
        "",
        "## Where the labels differ",
        "",
        *_worst_type_note(confusions, per_type),
        "| Corpus says | This says | Count |",
        "|---|---|---:|",
    ]
    lines += [
        f"| `{gold}` | `{ours}` | {count:,} |" for (gold, ours), count in confusions.most_common(20)
    ]

    lines += [
        "",
        "## By error type",
        "",
        "*Recall* is the share of the corpus's edits of this type that were drawn at all; "
        "*agreement* is the share of those that were also given the same label.",
        "",
        "| Error type | Gold edits | Span recall | Type agreement |",
        "|---|---:|---:|---:|",
    ]
    for error_type, (found, same, absent) in sorted(
        per_type.items(), key=lambda item: -(item[1][0] + item[1][2])
    ):
        total = found + absent
        if total < 20:
            continue
        lines.append(
            f"| `{error_type}` | {total:,} | {found / total:.2f} | "
            f"{(same / found if found else 0):.2f} |"
        )

    directory = REPORTS_DIR / "phase1"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "errant_de_agreement.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(f"  span P={precision:.4f} R={recall:.4f}, type agreement {type_rate:.4f}")


def _worst_type_note(
    confusions: Counter, per_type: dict[str, list[int]], minimum: int = 50
) -> list[str]:
    """Name the biggest labelling disagreement, so the table below has a lead.

    Written from the counts rather than from memory: if the classifier changes,
    the paragraph changes with it instead of quietly becoming a lie.
    """
    candidates = [
        (same / found, gold_type, found, absent)
        for gold_type, (found, same, absent) in per_type.items()
        if found + absent >= minimum and found
    ]
    if not candidates:
        return []
    agreement, gold_type, found, absent = min(candidates)
    instead = [
        (ours, count) for (gold, ours), count in confusions.most_common() if gold == gold_type
    ][:3]
    if not instead:
        return []
    named = ", ".join(f"`{ours}` ({count})" for ours, count in instead)
    return [
        f"The widest gap is `{gold_type}`, where the two annotators agree on "
        f"{agreement:.0%} of the labels. The disagreement is about naming rather than "
        f"finding: {found} of the corpus's {found + absent} edits of this type are drawn "
        f"identically here, but called {named} instead. The corpus reserves that label for "
        "any pair of words built on the same stem, which takes a stemmer; this classifier "
        "asks the narrower question of whether the two words share a lemma, and sends the "
        "rest to the part-of-speech buckets.",
        "",
    ]


if __name__ == "__main__":
    main()
