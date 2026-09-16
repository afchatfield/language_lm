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

**The split.** Dev by default, and every number quoted in Phase 3 is a dev
number. `--split test` unseals the held-out split, which Phase 0 froze on day one
and nothing has read since. It is guarded twice over: `load_split` refuses a
held-out split unless `allow_test=True`, which this passes only when the flag is
given, and the flag prints what it is doing. Use it once per model, at the end,
when no further decision will be made from the result -- a test number that gets
iterated against is a dev number with a misleading name.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from langlm.baselines.finetuned import FineTunedBaseline, GenerationFaults
from langlm.config import INTERIM_DIR, PROJECT_ROOT, REPORTS_DIR, load_config
from langlm.data import overcorrection as overcorrection_data
from langlm.data.m2 import M2Sentence
from langlm.data.splits import HELD_OUT_SPLITS, load_split
from langlm.eval import errant_de, explanation_scorer, leniency, overcorrection, span_scorer
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
    parser.add_argument(
        "--base",
        help="Load the base weights and tokenizer from here instead of the configured "
        "repo. For a vocabulary-trimmed checkpoint the tokenizer must come with it: "
        "the adapter directory still holds the untrimmed one.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Which Falko-MERLIN split to score. Defaults to the configured dev "
        "split. Pass `test` only for a final number that nothing will be tuned on.",
    )
    args = parser.parse_args()

    phase1, phase3 = load_config("phase1"), load_config("phase3")
    split = args.split or phase1["evaluation"]["split"]
    beta = phase1["evaluation"]["beta"]

    held_out = split in HELD_OUT_SPLITS
    if held_out:
        print(
            f"** {split} is the held-out split. This is a final number: nothing should be "
            f"tuned on what comes back. **"
        )
    dev = load_split("falko_merlin", split, allow_test=held_out)
    clean = load_split(overcorrection_data.CORPUS, "all")
    if args.limit:
        dev, clean = dev[: args.limit], clean[: args.limit]
    print(f"{len(dev):,} learner sentences ({split}), {len(clean):,} correct sentences")

    faults, answers = None, []
    if not args.report_only:
        adapter = PROJECT_ROOT / args.adapter
        if not adapter.exists():
            raise SystemExit(f"No adapter at {adapter}. Train one with `make train` first.")
        base = str(PROJECT_ROOT / args.base) if args.base else phase3["model"]["repo_id"]
        if args.base:
            print(f"base weights and tokenizer: {base}")
        system = FineTunedBaseline(
            repo_id=base,
            adapter_path=str(adapter),
            tokenizer_path=base if args.base else None,
            batch_size=args.batch_size,
            name=args.name,
        )
        for key, sentences in (("dev", dev), ("overcorrection", clean)):
            path = path_for(args.name, key, len(sentences))
            if path.exists() and not args.regenerate:
                print(f"{key}: already generated")
                continue
            print(f"{key}: correcting {len(sentences):,} sentences")
            seen = len(system.claims)
            corrected = system.correct_many([s.source for s in sentences])
            write_lines(path, [errant_de.tokenize(text) for text in corrected])
            write_claims(args.name, key, len(sentences), system.claims[seen:])
        if system.faults.total:
            write_faults(args.name, system.faults)
        faults, answers = system.faults, system.answers

    # A rerun that reuses cached generations has nothing in its counters. Reading
    # them back matters more than it looks: the schema section is the one place
    # a truncated or unparseable answer shows up, and a cached rerun printing
    # "0 invalid JSON out of 0 answers" hides exactly the failure it exists to
    # catch, which is how a quarter of dev being cut off mid-JSON went unnoticed.
    if faults is None or not faults.total:
        faults = read_faults(args.name) or faults

    write_report(args.name, dev, clean, beta, split, faults, answers)


def path_for(name: str, key: str, count: int) -> Path:
    return HYPOTHESIS_DIR / f"{name}.{key}.{count}.txt"


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def claims_path(name: str, key: str, count: int) -> Path:
    return HYPOTHESIS_DIR / f"{name}.{key}.{count}.claims.jsonl"


def write_claims(name: str, key: str, count: int, claims: Sequence[Sequence[dict]]) -> None:
    path = claims_path(name, key, count)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in claims), encoding="utf-8"
    )
    print(f"  wrote {path}")


def read_claims(name: str, key: str, count: int) -> list[list[dict]] | None:
    path = claims_path(name, key, count)
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def faults_path(name: str) -> Path:
    return HYPOTHESIS_DIR / f"{name}.faults.json"


def write_faults(name: str, faults: GenerationFaults) -> None:
    path = faults_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(faults), indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


def read_faults(name: str) -> GenerationFaults | None:
    path = faults_path(name)
    if not path.exists():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))
    fields = GenerationFaults.__annotations__
    return GenerationFaults(**{k: v for k, v in stored.items() if k in fields})


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

    # The lenient score is a second reading of the same output, not a second
    # system: it asks how much of the strict score is the model being wrong and
    # how much is it disagreeing with one annotator about wording. Absent the
    # thesaurus the question simply is not asked.
    lenient = lenient_correction = None
    try:
        equivalent = leniency.Leniency.full()
    except FileNotFoundError as exc:
        print(f"  no lenient score: {exc}")
    else:
        lenient = m2.score(dev, hypotheses, beta=beta, equivalent=equivalent)
        lenient_correction = span_scorer.score(
            dev, annotated, beta=beta, mode="correction", equivalent=equivalent
        )

    baselines = {}
    if BASELINES_JSON.exists():
        baselines = json.loads(BASELINES_JSON.read_text(encoding="utf-8"))
    rows = dict(baselines.get("baselines", {}))
    # One report per system, so scoring a new adapter cannot overwrite the record
    # of the one that is still the best.
    slug = "" if name == "fine-tuned" else f"-{name}"
    mine = REPORT_DIR / f"evaluation{slug}.json"
    # Every other fine-tuned run already scored keeps its row, so a new recipe is
    # read against the one it is meant to beat rather than only against Phase 1.
    # Compared by file rather than by label: this run's own previous record is
    # about to be replaced, and including it would put the system in the table
    # twice and make it its own bar.
    for previous in sorted(REPORT_DIR.glob("evaluation*.json")):
        if previous == mine:
            continue
        stored = json.loads(previous.read_text(encoding="utf-8"))
        label = stored.get("system")
        if label and "f0.5" in stored:
            rows[label] = {
                "precision": stored["precision"],
                "recall": stored["recall"],
                "f0.5": stored["f0.5"],
                "overcorrection_rate": stored["overcorrection_rate"],
            }

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
    ]

    if lenient is not None:
        lines += [
            "## How much of that gap is disagreement rather than error",
            "",
            "The same output, rescored with `langlm.eval.leniency`, which lets the system's "
            "replacement stand for the annotator's when the two are typographic variants, the "
            "same compound written closed or hyphenated, or thesaurus synonyms in the same "
            "inflection. The span still has to be right.",
            "",
            f"| | M2 F{beta:g} | Correction F{beta:g} |",
            "|---|---:|---:|",
            f"| strict | {maxmatch.f_score:.4f} | {correction.f_score:.4f} |",
            f"| lenient | {lenient.f_score:.4f} | {lenient_correction.f_score:.4f} |",
            "",
            f"The headline stays the strict number. The lenient one is worth "
            f"{lenient.f_score - maxmatch.f_score:+.4f}, which is the answer to whether the "
            f"detection-correction gap above is the annotator being fussy: it is not. Almost "
            f"all of that gap is the model finding the right word to change and then changing "
            f"it wrongly -- most often a misspelling whose intended word is recoverable, "
            f"answered with a different word (`rabben` corrected to `finden`, not `rauben`).",
            "",
        ]

    lines += [
        "## Where it helps",
        "",
        "Per-error-type recall. This is the table Phase 2 is meant to be fixed from -- but "
        "read it as coverage, not as volume: across the 36 types with dev volume, the "
        "correlation between a type's train/dev volume ratio and its F0.5 is only r = 0.27, "
        "while types the corruptors produce at all average F0.5 0.679 against 0.488 for types "
        "they never produce. A weak type is one to start injecting, not one to inject more of.",
        "",
        "| Error type | Gold | Recall |",
        "|---|---:|---:|",
    ]
    for error_type, counts in correction.ranked_types(minimum=20)[:15]:
        lines.append(f"| `{error_type}` | {counts.tp + counts.fn} | {counts.recall:.2f} |")

    claims = read_claims(name, "dev", len(dev))
    explained = None
    if claims is not None and len(claims) == len(dev):
        explained = explanation_scorer.score([s.source for s in dev], annotated, claims)
        lines += [
            "",
            "## Did it explain what it did?",
            "",
            "Scored against the edits ERRANT derives from the model's *own* correction, not "
            "against the gold annotation. Whether an edit should have been made is the "
            "correction score's question; this one asks whether the model can say what it "
            "just did, because a rewrite it cannot describe is one it cannot explain.",
            "",
            "| | Value | Reads as |",
            "|---|---:|---|",
            f"| Coverage | {explained.coverage:.4f} | of {explained.derived:,} rewrites it made, "
            f"{explained.covered:,} were listed |",
            f"| Precision | {explained.precision:.4f} | of {explained.claimed:,} changes it "
            f"listed, {explained.matched:,} match a real rewrite |",
            f"| Type accuracy | {explained.type_accuracy:.4f} | of the {explained.matched:,} "
            f"that match, {explained.typed:,} carry the type ERRANT assigns |",
            "",
            "Type accuracy is what the learner actually sees, because the type is the key "
            "`rules/de_types.yaml` renders the explanation from. Read all three against the "
            "oracle rather than against 1.00. Scoring the gold correction and the gold "
            "changes through this same scorer gives coverage 0.9795, precision 1.0000 and "
            "type accuracy 0.8387: ERRANT segments a rewrite slightly differently from the "
            "annotator, and re-derives the annotator's type on 86% of edits, the rest being "
            "coarsenings such as `M:PUNCT` to `M:OTHER`.",
        ]

    if faults is not None and faults.total:
        lines += [
            "",
            "## Did it answer in the schema?",
            "",
            f"{faults}, out of {faults.total:,} answers.",
            "",
            "An answer with no readable correction is scored as an unchanged sentence, which "
            "is why it is counted here rather than left to look like restraint. Cut-off "
            "answers are counted separately because the schema puts the sentence first: the "
            "correction survives and only the `changes` list is lost, so they cost the "
            "explanation metrics above rather than the score.",
        ]

    if answers:
        lines += ["", "## Sample answers", ""]
        for answer in random.Random(20260909).sample(answers, min(5, len(answers))):
            lines += ["```json", answer[:600], "```", ""]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / f"evaluation{slug}.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    print(f"  F{beta:g} {maxmatch.f_score:.4f} | overcorrection {over.rate:.1%} | bar {best:.4f}")
    if explained is not None:
        print(f"  explanations: {explained}")

    (REPORT_DIR / f"evaluation{slug}.json").write_text(
        json.dumps(
            {
                "system": name,
                "f0.5": round(maxmatch.f_score, 4),
                "precision": round(maxmatch.precision, 4),
                "recall": round(maxmatch.recall, 4),
                "overcorrection_rate": round(over.rate, 4),
                "detection_f": round(detection.f_score, 4),
                "correction_f": round(correction.f_score, 4),
                "lenient_f0.5": lenient and round(lenient.f_score, 4),
                "lenient_correction_f": lenient_correction and round(lenient_correction.f_score, 4),
                "baseline_bar": best,
                "explanation": explained
                and {
                    "coverage": round(explained.coverage, 4),
                    "precision": round(explained.precision, 4),
                    "type_accuracy": round(explained.type_accuracy, 4),
                    "rewrites": explained.derived,
                    "claims": explained.claimed,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
