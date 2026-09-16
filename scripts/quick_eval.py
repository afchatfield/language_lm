#!/usr/bin/env python
"""Screen a variant on the 600-sentence dev subset, in minutes rather than hours.

    python scripts/quick_eval.py --adapter checkpoints/phase3-de-iter3 --name iter3
    python scripts/quick_eval.py --adapter checkpoints/phase3-de-iter3 --name iter3-r2 --rounds 2
    python scripts/quick_eval.py --table          # everything screened so far

This is the front of the funnel, not the result. `langlm.eval.subset` explains
the size: a paired delta measured here has a standard deviation of 0.0046, so it
separates effects of about 0.01 and nothing finer. Anything that looks promising
goes to `scripts/eval_finetuned.py` on full dev, and that is the number that gets
published.

Generations are cached under `data/interim/phase3/screen/`, so rescoring after a
scorer change costs nothing. The test split is not touched.
"""

from __future__ import annotations

import argparse
import json

from langlm.baselines.finetuned import FineTunedBaseline
from langlm.config import INTERIM_DIR, PROJECT_ROOT, REPORTS_DIR, load_config
from langlm.data import overcorrection as overcorrection_data
from langlm.data.splits import load_split
from langlm.eval import errant_de, overcorrection, subset
from langlm.eval import m2_scorer as m2

SCREEN_DIR = INTERIM_DIR / "phase3" / "screen"
RESULTS = REPORTS_DIR / "phase3" / "screen.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", help="Adapter directory to screen")
    parser.add_argument("--name", help="Row label; also the cache key")
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Run the model over its own output this many times",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--overcorrection",
        action="store_true",
        help="Also run the 400 correct sentences. Worth it for anything that "
        "makes the model more willing to edit; skip it for a checkpoint sweep.",
    )
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--table", action="store_true", help="Print the results so far and stop")
    args = parser.parse_args()

    if args.table:
        print_table()
        return
    if not (args.adapter and args.name):
        raise SystemExit("--adapter and --name are required (or pass --table)")

    phase1, phase3 = load_config("phase1"), load_config("phase3")
    beta = phase1["evaluation"]["beta"]

    dev = load_split("falko_merlin", phase1["evaluation"]["split"])
    screen, _ = subset.take(dev)
    print(f"screening {args.name} on {len(screen)} of {len(dev):,} dev sentences")

    hypotheses = generate(args, screen, phase3, "dev")
    scored = m2.score(screen, hypotheses, beta=beta)
    print(f"  {scored}")

    over_rate = None
    if args.overcorrection:
        clean = load_split(overcorrection_data.CORPUS, "all")
        produced = generate(args, clean, phase3, "overcorrection")
        over_rate = overcorrection.measure([s.source for s in clean], produced).rate
        print(f"  overcorrection {over_rate:.1%}")

    record(
        args.name,
        {
            "adapter": args.adapter,
            "rounds": args.rounds,
            "f0.5": round(scored.f_score, 4),
            "precision": round(scored.precision, 4),
            "recall": round(scored.recall, 4),
            "tp": scored.counts.tp,
            "fp": scored.counts.fp,
            "fn": scored.counts.fn,
            "proposed": scored.counts.tp + scored.counts.fp,
            "overcorrection_rate": over_rate and round(over_rate, 4),
        },
    )
    print_table()


def generate(args, sentences, phase3: dict, key: str) -> list[str]:
    """Correct `sentences`, reusing a cached run when there is one."""
    path = SCREEN_DIR / f"{args.name}.{key}.{len(sentences)}.txt"
    if path.exists() and not args.regenerate:
        print(f"  {key}: cached")
        return path.read_text(encoding="utf-8").splitlines()

    adapter = PROJECT_ROOT / args.adapter
    if not adapter.exists():
        raise SystemExit(f"No adapter at {adapter}")
    system = FineTunedBaseline(
        repo_id=phase3["model"]["repo_id"],
        adapter_path=str(adapter),
        batch_size=args.batch_size,
        rounds=args.rounds,
        name=args.name,
    )
    corrected = system.correct_many([s.source for s in sentences])
    if system.rewrites_per_round:
        rounds = ", ".join(
            f"round {i + 1} rewrote {n:,}" for i, n in enumerate(system.rewrites_per_round)
        )
        print(f"  {rounds}")
    lines = [errant_de.tokenize(text) for text in corrected]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def record(name: str, row: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    stored = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    stored[name] = row
    RESULTS.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")


#: The row a variant is read against, by prefix. An ablation arm trained on a
#: subsample has to be compared with the arm trained the same way and not with
#: the full-corpus model, or the subsample's own cost is read as the effect.
CONTROLS = (("abl-", "abl-control"),)

#: The row everything else is read against.
CONTROL = "iter3"

#: Below this many edits of difference, a marginal hit rate is noise.
MIN_MARGIN = 20


def control_for(name: str, stored: dict) -> tuple[str, dict] | tuple[None, None]:
    """Which row `name` should be compared with, and that row."""
    for prefix, control in CONTROLS:
        if name.startswith(prefix):
            return (control, stored[control]) if control in stored else (None, None)
    return (CONTROL, stored[CONTROL]) if CONTROL in stored else (None, None)


def break_even(row: dict) -> float:
    """The hit rate a marginal edit needs to be worth proposing.

    MaxMatch F0.5 reduces to ``1.25 * TP / (proposed + 0.25 * gold)``, and gold
    is a constant of the corpus, so one more proposed edit raises the score
    exactly when its chance of being right beats ``TP / (proposed + 0.25 gold)``.
    On the current model that threshold is about 0.56 while the model itself
    runs at 0.73 precision -- which is the whole reason these experiments push
    towards editing more rather than filtering.
    """
    gold = row["tp"] + row["fn"]
    return row["tp"] / (row["proposed"] + 0.25 * gold)


def print_table() -> None:
    if not RESULTS.exists():
        print("nothing screened yet")
        return
    stored = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = sorted(stored.items(), key=lambda kv: -kv[1]["f0.5"])
    width = max(len(name) for name, _ in rows)

    print(
        f"\n{'variant'.ljust(width)}  {'F0.5':>7} {'P':>7} {'R':>7} {'edits':>7}  {'over':>6}"
        f"  {'dF0.5':>7}  marginal edits, against the row named"
    )
    for name, row in rows:
        over = f"{row['overcorrection_rate']:.1%}" if row.get("overcorrection_rate") else "--"
        delta = margin = ""
        control_name, control = control_for(name, stored)
        if control and name != control_name:
            delta = f"{row['f0.5'] - control['f0.5']:+7.4f}"
            added = row["proposed"] - control["proposed"]
            # A hit rate over a handful of edits is a ratio of two small
            # differences and says nothing -- one edit either way swings it by
            # a hundred points. Report the count and stop.
            if abs(added) < MIN_MARGIN:
                margin = f"{added:+,} edits: too few to rate"
            else:
                hit = (row["tp"] - control["tp"]) / added
                verdict = "above" if hit > break_even(control) else "below"
                margin = (
                    f"{added:+,} edits at {hit:.0%} hit rate, {verdict} the "
                    f"{break_even(control):.0%} it has to clear"
                )
            margin = f"[vs {control_name}] {margin}"
        print(
            f"{name.ljust(width)}  {row['f0.5']:7.4f} {row['precision']:7.4f} "
            f"{row['recall']:7.4f} {row['proposed']:7,}  {over:>6}  {delta:>7}  {margin}"
        )
    if CONTROL in stored:
        print(
            f"\nA variant is worth having when its extra edits beat "
            f"{break_even(stored[CONTROL]):.1%}; the screen separates about 0.009 of F0.5 "
            f"and nothing finer."
        )


if __name__ == "__main__":
    main()
