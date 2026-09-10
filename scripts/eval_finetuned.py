#!/usr/bin/env python
"""Score the fine-tuned model against the Phase 1 baselines.

Runs the adapter over Falko-MERLIN dev and the overcorrection set, then scores it
with exactly the machinery the baselines were scored with -- the MaxMatch scorer,
the German ERRANT span scorer, and the overcorrection metric. The comparison
column is read from `reports/phase1/baselines.json`, so the numbers in the table
are the ones already published rather than a fresh run of the baselines.

    python scripts/eval_finetuned.py --adapter checkpoints/phase3-de
    python scripts/eval_finetuned.py --adapter checkpoints/phase3-de --limit 200
    python scripts/eval_finetuned.py --report-only

Generation is slow and scoring is fast, so they are separate: the corrected
sentences are written to `data/interim/phase3/` and reused on a rerun.

The test split is not touched.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path

from langlm.baselines.finetuned import FineTunedBaseline
from langlm.config import INTERIM_DIR, PROJECT_ROOT, REPORTS_DIR, load_config
from langlm.data import overcorrection as overcorrection_data
from langlm.data.m2 import M2Sentence
from langlm.data.splits import load_split
from langlm.eval import errant_de, overcorrection, span_scorer
from langlm.eval import m2_scorer as m2

HYPOTHESIS_DIR = INTERIM_DIR / "phase3"
REPORT_DIR = REPORTS_DIR / "phase3"
BASELINES_JSON = REPORTS_DIR / "phase1" / "baselines.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="checkpoints/phase3-de", help="Adapter directory")
    parser.add_argument("--limit", type=int, help="Use only the first N sentences")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--report-only", action="store_true", help="Score what already exists")
    parser.add_argument("--regenerate", action="store_true", help="Ignore cached output")
    parser.add_argument("--name", default="fine-tuned", help="Row label in the table")
    args = parser.parse_args()

    phase1, phase3 = load_config("phase1"), load_config("phase3")
    split = phase1["evaluation"]["split"]
    beta = phase1["evaluation"]["beta"]

    dev = load_split("falko_merlin", split)
    clean = load_split(overcorrection_data.CORPUS, "all")
    if args.limit:
        dev, clean = dev[: args.limit], clean[: args.limit]
    print(f"{len(dev):,} learner sentences ({split}), {len(clean):,} correct sentences")

    faults, answers = None, []
    if not args.report_only:
        adapter = PROJECT_ROOT / args.adapter
        if not adapter.exists():
            raise SystemExit(f"No adapter at {adapter}. Train one with `make train` first.")
        system = FineTunedBaseline(
            repo_id=phase3["model"]["repo_id"],
            adapter_path=str(adapter),
            batch_size=args.batch_size,
            name=args.name,
        )
        for key, sentences in (("dev", dev), ("overcorrection", clean)):
            path = path_for(args.name, key, len(sentences))
            if path.exists() and not args.regenerate:
                print(f"{key}: already generated")
                continue
            print(f"{key}: correcting {len(sentences):,} sentences")
            corrected = system.correct_many([s.source for s in sentences])
            write_lines(path, [errant_de.tokenize(text) for text in corrected])
        faults, answers = system.faults, system.answers

    write_report(args.name, dev, clean, beta, split, faults, answers)


def path_for(name: str, key: str, count: int) -> Path:
    return HYPOTHESIS_DIR / f"{name}.{key}.{count}.txt"


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write_report(
    name: str,
    dev: Sequence[M2Sentence],
    clean: Sequence[M2Sentence],
    beta: float,
    split: str,
    faults,
    answers: list[str],
) -> None:
    dev_path = path_for(name, "dev", len(dev))
    clean_path = path_for(name, "overcorrection", len(clean))
    if not dev_path.exists() or not clean_path.exists():
        raise SystemExit("Nothing generated yet; run without --report-only first.")

    hypotheses = read_lines(dev_path)
    maxmatch = m2.score(dev, hypotheses, beta=beta)
    annotated = errant_de.annotate_corpus([s.source for s in dev], hypotheses)
    correction = span_scorer.score(dev, annotated, beta=beta, mode="correction")
    detection = span_scorer.score(dev, annotated, beta=beta, mode="detection")
    over = overcorrection.measure([s.source for s in clean], read_lines(clean_path))

    baselines = {}
    if BASELINES_JSON.exists():
        baselines = json.loads(BASELINES_JSON.read_text(encoding="utf-8"))
    rows = baselines.get("baselines", {})

    lines = [
        "# Phase 3: the fine-tuned model",
        "",
        f"Falko-MERLIN **{split}**: {len(dev):,} sentences. Overcorrection set: "
        f"{len(clean):,} sentences of published German prose. Scored with the same "
        f"MaxMatch and ERRANT machinery as Phase 1, so these numbers sit beside those "
        f"ones honestly. The test split has not been read.",
        "",
        "## The table",
        "",
        f"| System | P | R | F{beta:g} | Overcorrection |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in rows.items():
        lines.append(
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f0.5']:.4f} | {row['overcorrection_rate']:.1%} |"
        )
    lines.append(
        f"| **{name}** | {maxmatch.precision:.4f} | {maxmatch.recall:.4f} | "
        f"**{maxmatch.f_score:.4f}** | {over.rate:.1%} |"
    )

    best = max((row["f0.5"] for row in rows.values()), default=0.0)
    verdict = "beats" if maxmatch.f_score > best else "does not beat"
    lines += [
        "",
        f"The bar is the best Phase 1 baseline, **F{beta:g} = {best:.4f}** (few-shot), not "
        f"LanguageTool's 0.4134. This model **{verdict}** it.",
        "",
        "## Detection versus correction",
        "",
        f"| | Detection F{beta:g} | Correction F{beta:g} | Gap |",
        "|---|---:|---:|---:|",
        f"| {name} | {detection.f_score:.4f} | {correction.f_score:.4f} | "
        f"{detection.f_score - correction.f_score:+.4f} |",
        "",
        "## Where it helps",
        "",
        "Per-error-type recall. This is the table Phase 2 is meant to be fixed from: a type "
        "that is still weak here is a type to inject more of, or to inject better.",
        "",
        "| Error type | Gold | Recall |",
        "|---|---:|---:|",
    ]
    for error_type, counts in correction.ranked_types(minimum=20)[:15]:
        lines.append(f"| `{error_type}` | {counts.tp + counts.fn} | {counts.recall:.2f} |")

    if faults is not None:
        lines += [
            "",
            "## Did it answer in the schema?",
            "",
            f"{faults}, out of {faults.total:,} answers.",
            "",
            "Invalid JSON is scored as an unchanged sentence, which is why it is counted "
            "here rather than left to look like restraint. A high number means constrained "
            "decoding is worth doing before the next iteration rather than after.",
        ]

    if answers:
        lines += ["", "## Sample answers", ""]
        for answer in random.Random(20260909).sample(answers, min(5, len(answers))):
            lines += ["```json", answer[:600], "```", ""]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "evaluation.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    print(f"  F{beta:g} {maxmatch.f_score:.4f} | overcorrection {over.rate:.1%} | bar {best:.4f}")

    (REPORT_DIR / "evaluation.json").write_text(
        json.dumps(
            {
                "f0.5": round(maxmatch.f_score, 4),
                "precision": round(maxmatch.precision, 4),
                "recall": round(maxmatch.recall, 4),
                "overcorrection_rate": round(over.rate, 4),
                "detection_f": round(detection.f_score, 4),
                "correction_f": round(correction.f_score, 4),
                "baseline_bar": best,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
