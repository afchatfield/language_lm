#!/usr/bin/env python
"""Run the Phase 1 baselines and write the results table.

Three systems, plus the identity system that prices the harness itself:

    identity        changes nothing; measures what re-tokenisation costs
    languagetool    a mature rule-based checker, applying its own suggestions
    zero-shot       EuroLLM-1.7B prompted to correct German
    few-shot        the same model, with worked examples from the train split

Each system is run over the development split and over the overcorrection set,
and its output is written to `data/interim/phase1/` before anything is scored.
Generating is slow and scoring is fast, so they are separate: rerunning the
script rebuilds the report from whatever outputs already exist and only
generates the ones that are missing.

    python scripts/run_baselines.py                    # everything
    python scripts/run_baselines.py --only languagetool
    python scripts/run_baselines.py --report-only
    python scripts/run_baselines.py --limit 100        # smoke test
    python scripts/run_baselines.py --only few-shot --batch-size 16

The test split is not touched. It stays frozen until Phase 3.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from langlm.baselines.identity import IdentityBaseline
from langlm.baselines.languagetool import LanguageToolBaseline
from langlm.baselines.prompted import PromptedBaseline, sample_shots
from langlm.config import INTERIM_DIR, REPORTS_DIR, load_config
from langlm.data import overcorrection as overcorrection_data
from langlm.data.m2 import M2Sentence
from langlm.data.splits import load_split
from langlm.eval import errant_de, overcorrection, span_scorer
from langlm.eval import m2_scorer as m2

#: Where generated corrections are kept between runs.
HYPOTHESIS_DIR = INTERIM_DIR / "phase1"

#: Where the report goes.
REPORT_DIR = REPORTS_DIR / "phase1"

#: The order the table reads in: cheapest and least interesting first.
BASELINE_ORDER = ("identity", "languagetool", "zero-shot", "few-shot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=BASELINE_ORDER, default=list(BASELINE_ORDER))
    parser.add_argument("--report-only", action="store_true", help="Score what already exists")
    parser.add_argument("--regenerate", action="store_true", help="Ignore cached outputs")
    parser.add_argument("--limit", type=int, help="Use only the first N sentences (smoke test)")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the model batch size. Few-shot prompts are several hundred "
        "tokens longer than zero-shot ones, so the batch that fits one may not fit "
        "the other on a 16GB machine.",
    )
    args = parser.parse_args()

    config = load_config("phase1")
    split = config["evaluation"]["split"]

    dev = load_split("falko_merlin", split)
    clean = load_split(overcorrection_data.CORPUS, "all")
    if args.limit:
        dev, clean = dev[: args.limit], clean[: args.limit]

    print(f"{len(dev)} learner sentences ({split}), {len(clean)} correct sentences")

    if not args.report_only:
        for name in args.only:
            generate(
                name, dev, clean, config, regenerate=args.regenerate, batch_size=args.batch_size
            )

    write_report(dev, clean, config)


# --- generation -------------------------------------------------------------


def build(name: str, config: dict, batch_size: int | None = None):
    """Construct a baseline by name."""
    if name == "identity":
        return IdentityBaseline()
    if name == "languagetool":
        return LanguageToolBaseline.from_config()
    settings = config["baselines"]["model"]
    shots = []
    if name == "few-shot":
        shots = sample_shots(
            load_split("falko_merlin", "train"),
            settings["few_shot_examples"],
            settings["few_shot_seed"],
        )
    overrides = {"batch_size": batch_size} if batch_size else {}
    return PromptedBaseline.from_config(shots=shots, **overrides)


def generate(
    name: str,
    dev: Sequence[M2Sentence],
    clean: Sequence[M2Sentence],
    config: dict,
    regenerate: bool = False,
    batch_size: int | None = None,
) -> None:
    """Run one baseline over both evaluation sets, unless it has been run already."""
    targets = {"dev": dev, "overcorrection": clean}
    missing = {
        key: sentences
        for key, sentences in targets.items()
        if regenerate or not path_for(name, key, len(sentences)).exists()
    }
    if not missing:
        print(f"{name}: already generated")
        return

    system = build(name, config, batch_size)
    try:
        for key, sentences in missing.items():
            print(f"{name}: correcting {len(sentences)} sentences ({key})")
            sources = [sentence.source for sentence in sentences]
            raw = (
                system.correct_many(sources)
                if hasattr(system, "correct_many")
                else [system.correct(source) for source in sources]
            )
            # One tokeniser for every system, so that a model writing ordinary
            # German is not charged for where it puts the full stop.
            hypotheses = [errant_de.tokenize(text) for text in raw]
            write_lines(path_for(name, key, len(sentences)), hypotheses)
    finally:
        release(system)


def release(system: object) -> None:
    """Drop a baseline's model before the next one loads its own.

    Torch's MPS allocator keeps freed blocks in its own pool, so letting the
    reference fall out of scope returns the memory to torch and not to the
    machine. Two 1.7B models resident at once is what the 16GB M1 does not
    have, and the second one dies while the first is still nominally cached.
    """
    if getattr(system, "_model", None) is None:
        return
    system._model = None
    system._tokenizer = None
    gc.collect()
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def path_for(name: str, key: str, count: int) -> Path:
    return HYPOTHESIS_DIR / f"{name}.{key}.{count}.txt"


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


# --- scoring ----------------------------------------------------------------


def score_baseline(
    name: str,
    dev: Sequence[M2Sentence],
    clean: Sequence[M2Sentence],
    beta: float,
) -> dict | None:
    """Score one baseline, or return None if it has not been generated."""
    dev_path = path_for(name, "dev", len(dev))
    clean_path = path_for(name, "overcorrection", len(clean))
    if not dev_path.exists() or not clean_path.exists():
        return None

    hypotheses = read_lines(dev_path)
    maxmatch = m2.score(dev, hypotheses, beta=beta)
    annotated = errant_de.annotate_corpus([s.source for s in dev], hypotheses)
    correction = span_scorer.score(dev, annotated, beta=beta, mode="correction")
    detection = span_scorer.score(dev, annotated, beta=beta, mode="detection")
    over = overcorrection.measure([s.source for s in clean], read_lines(clean_path))

    return {
        "name": name,
        "m2": maxmatch,
        "correction": correction,
        "detection": detection,
        "overcorrection": over,
    }


def write_report(dev: Sequence[M2Sentence], clean: Sequence[M2Sentence], config: dict) -> None:
    """Score every generated baseline and write `reports/phase1/baselines.md`."""
    beta = config["evaluation"]["beta"]
    results = [r for r in (score_baseline(n, dev, clean, beta) for n in BASELINE_ORDER) if r]
    if not results:
        print("Nothing generated yet; no report written.")
        return

    lines = _report_lines(results, dev, clean, config)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "baselines.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")

    meta = {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": config["evaluation"]["split"],
        "sentences": len(dev),
        "overcorrection_sentences": len(clean),
        "baselines": {
            r["name"]: {
                "f0.5": round(r["m2"].f_score, 4),
                "precision": round(r["m2"].precision, 4),
                "recall": round(r["m2"].recall, 4),
                "overcorrection_rate": round(r["overcorrection"].rate, 4),
            }
            for r in results
        },
    }
    (REPORT_DIR / "baselines.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _report_lines(results: list[dict], dev, clean, config: dict) -> list[str]:
    split = config["evaluation"]["split"]
    beta = config["evaluation"]["beta"]
    model = config["baselines"]["model"]["repo_id"]
    gold_edits = sum(len(s.edits_for()) for s in dev)

    lines = [
        "# Phase 1 baselines",
        "",
        f"Falko-MERLIN **{split}**: {len(dev):,} sentences, {gold_edits:,} gold edits. "
        f"Overcorrection set: {len(clean):,} sentences of published German prose that "
        f"need no correction. The test split has not been read.",
        "",
        "## The table",
        "",
        f"| System | P | R | F{beta:g} | Overcorrection | Spurious edits |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        score, over = result["m2"], result["overcorrection"]
        lines.append(
            f"| {result['name']} | {score.precision:.4f} | {score.recall:.4f} | "
            f"**{score.f_score:.4f}** | {over.rate:.1%} | {over.edits} |"
        )

    ceiling = m2.score(dev, [sentence.target() for sentence in dev], beta=beta)
    lines += [
        "",
        f"F{beta:g} is MaxMatch (Dahlmeier and Ng, 2012) over the corrected sentences. "
        "*Overcorrection* is the share of already-correct sentences the system changed at "
        "all; *spurious edits* counts the individual changes it made to them.",
        "",
        f"The metric's own ceiling on this split is **F{beta:g} = {ceiling.f_score:.4f}**, not "
        f"1.0: that is what a system scores when it reproduces the annotator's corrections "
        f"exactly. MaxMatch recovers a system's edits from an optimal alignment between the "
        f"original and the correction, and {ceiling.counts.fn} of the corpus's "
        f"{ceiling.counts.tp + ceiling.counts.fn:,} edits are drawn in a way that no optimal "
        f"alignment produces. Read every score below against that number rather than against 1.",
        "",
        "## Detection versus correction",
        "",
        "Span-level ERRANT scoring. *Detection* asks only whether the system flagged the "
        "right span, *correction* whether it also proposed the annotator's fix. The gap "
        "between them is the system seeing the error and getting the repair wrong.",
        "",
        f"| System | Detection F{beta:g} | Correction F{beta:g} | Gap |",
        "|---|---:|---:|---:|",
    ]
    for result in results:
        detection, correction = result["detection"], result["correction"]
        lines.append(
            f"| {result['name']} | {detection.f_score:.4f} | {correction.f_score:.4f} | "
            f"{detection.f_score - correction.f_score:+.4f} |"
        )

    lines += ["", "## Where each system helps", ""]
    lines += _type_table(results)

    lines += [
        "",
        "## What the systems change in correct prose",
        "",
    ]
    for result in results:
        over = result["overcorrection"]
        if not over.changes:
            lines += [f"**{result['name']}** changed nothing.", ""]
            continue
        lines += [
            f"**{result['name']}** -- {over}. Most common: "
            + ", ".join(f"`{t}` ({c})" for t, c in over.by_type.most_common(5))
            + ".",
            "",
        ]
        for change in over.changes[:3]:
            lines += [f"> {change.source}", f">> {change.hypothesis}", ""]

    lines += [
        "## Notes",
        "",
        f"* Model: `{model}`, greedy decoding.",
        "* Every system's output goes through the same German tokeniser before scoring. "
        "The identity row prices that step: whatever false positives it shows are the "
        "harness disagreeing with the corpus about hyphens and apostrophes, not a system "
        "making mistakes.",
        "* LanguageTool applies the first suggestion of every non-overlapping match that "
        "offers one. Matches with no suggestion are left alone.",
        "* The overcorrection set was **not** screened with LanguageTool, which would have "
        "handed LanguageTool a zero by construction.",
    ]
    return lines


def _type_table(results: list[dict], top: int = 15) -> list[str]:
    """A per-error-type recall table across systems, ordered by how common the type is."""
    reference = results[0]["correction"]
    types = [t for t, _ in reference.ranked_types(minimum=20)][:top]
    if not types:
        return ["_Not enough gold edits per type to break down._"]

    header = "| Error type | Gold | " + " | ".join(f"{r['name']} R" for r in results) + " |"
    divider = "|---|---:|" + "---:|" * len(results)
    lines = [header, divider]
    for error_type in types:
        gold = reference.by_type[error_type]
        cells = []
        for result in results:
            counts = result["correction"].by_type.get(error_type)
            cells.append(f"{counts.recall:.2f}" if counts else "0.00")
        lines.append(f"| `{error_type}` | {gold.tp + gold.fn} | " + " | ".join(cells) + " |")
    return lines


if __name__ == "__main__":
    main()
