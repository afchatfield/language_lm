#!/usr/bin/env python
"""Run the trained error generator over clean German, and keep what survives.

    python scripts/generate_backtranslation.py --size 2000      # a trial
    python scripts/generate_backtranslation.py --size 100000    # the real run

Each clean sentence is given an error budget drawn from Falko-MERLIN train's own
edit-count histogram, so the generated corpus inherits real learner density
rather than whatever the generator prefers -- including the zero bucket, which
is how the 22% no-correction share arrives without a separate pass.

Decoding is sampled, not greedy. A greedy generator makes the same conservative
mistake on every sentence, and coverage of the hard error types is the entire
reason this exists. The price is failures, so everything is filtered:
`langlm.data.backtranslate.rejection` says why each rejected pair went, and the
report prints the funnel.

Output is appended as it goes, so an interrupted run keeps what it made and a
rerun continues from there rather than starting again.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from langlm.config import PROJECT_ROOT, REPORTS_DIR, load_config
from langlm.data import backtranslate, clean_de
from langlm.data.splits import load_split
from langlm.train import corrupter

REPORT_DIR = REPORTS_DIR / "phase2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="checkpoints/corrupter-de")
    parser.add_argument("--size", type=int, default=100000, help="Clean sentences to damage")
    parser.add_argument(
        "--skip",
        type=int,
        help="Clean sentences to step over first. Defaults to the rule-based "
        "synthetic size, so the two synthetic halves damage disjoint text and no "
        "target sentence is learned twice under two different corruptions.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, help="Defaults to the project seed")
    parser.add_argument("--out", default=str(backtranslate.PAIRS_DIR / "de.jsonl"))
    parser.add_argument("--regenerate", action="store_true", help="Ignore what is already there")
    parser.add_argument("--report-only", action="store_true", help="Re-read and re-report")
    args = parser.parse_args()

    path = Path(args.out)
    if args.regenerate and path.exists():
        path.unlink()

    config = load_config("phase2")
    seed = args.seed or config["clean_corpus"]["seed"]
    skip = args.skip if args.skip is not None else config["dataset"]["synthetic"]["size"]
    clean = [sentence.source for sentence in load_split(clean_de.CORPUS, "all")]
    available = max(len(clean) - skip, 0)
    if args.size > available:
        print(
            f"Only {available:,} clean sentences are free after the {skip:,} the "
            f"corruptors use; asked for {args.size:,}. Rebuild a bigger corpus with "
            f"`python scripts/build_clean_corpus.py --size {skip + args.size}` -- the "
            f"pool holds about 167,000, and a larger build keeps the current corpus as "
            f"its exact prefix, so nothing already generated is invalidated."
        )
    wanted = clean[skip : skip + args.size]

    histogram = backtranslate.count_distribution()
    counts, weights = zip(*sorted(histogram.items()), strict=True)
    rng = random.Random(seed)
    budgets = rng.choices(counts, weights=weights, k=len(wanted))

    done = already_done(path) if not args.report_only else set()
    pending = [
        (text, budget) for text, budget in zip(wanted, budgets, strict=True) if text not in done
    ]
    if done:
        print(f"{len(done):,} already generated; {len(pending):,} to go")

    if pending and not args.report_only:
        generate(pending, path, args, seed)

    threshold = backtranslate.divergence_threshold()
    pairs = backtranslate.read_pairs(path)
    kept, funnel = filter_pairs(pairs, threshold)
    backtranslate.write_pairs(path.with_suffix(".kept.jsonl"), kept)
    print(f"\n{len(kept):,} of {len(pairs):,} pairs kept -> {path.with_suffix('.kept.jsonl')}")
    write_report(kept, pairs, funnel, threshold, args, seed)


def already_done(path: Path) -> set[str]:
    """Which clean sentences a previous run has already damaged."""
    if not path.exists():
        return set()
    return {
        json.loads(line)["correct"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def generate(pending: list[tuple[str, int]], path: Path, args, seed: int) -> None:
    """Sample one damaged sentence per clean one, appending as it goes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = PROJECT_ROOT / args.adapter
    if not adapter.exists():
        raise SystemExit(
            f"No generator at {adapter}. Train one with `python scripts/train_corrupter.py`."
        )
    config = load_config("phase3")

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32
    print(f"device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["repo_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(config["model"]["repo_id"], dtype=dtype)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.to(device).eval()
    torch.manual_seed(seed)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [corrupter.build_prompt(text, budget) for text, budget in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    # Generous relative to the input: a learner sentence is
                    # rarely longer than its correction, and the length filter
                    # catches the ones that run away. A cap that truncated would
                    # manufacture deletions that no learner made.
                    max_new_tokens=encoded["input_ids"].shape[1] + 32,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            width = encoded["input_ids"].shape[1]
            for (text, budget), row in zip(batch, generated[:, width:], strict=True):
                answer = tokenizer.decode(row, skip_special_tokens=True).strip()
                fh.write(
                    backtranslate.Pair(correct=text, learner=answer, asked=budget).as_json() + "\n"
                )
            fh.flush()
            print(f"  {min(start + args.batch_size, len(pending)):,} / {len(pending):,}", end="\r")
    print()


def filter_pairs(pairs, threshold: float):
    """Split the generated pairs into the fit and the funnel that explains the rest."""
    kept, funnel = [], Counter()
    for pair in pairs:
        reason = backtranslate.rejection(pair, threshold)
        if reason is None:
            kept.append(pair)
            funnel["kept"] += 1
        else:
            funnel[reason] += 1
    return kept, funnel


def write_report(kept, pairs, funnel, threshold: float, args, seed: int) -> None:
    """The funnel, the density it produced, and samples to read."""
    asked = Counter(pair.asked for pair in kept)
    made = Counter()
    for pair in kept:
        left, right = pair.correct.split(), pair.learner.split()
        made[backtranslate.token_distance(left, right)] += 1
    negatives = sum(1 for pair in kept if pair.learner.strip() == pair.correct.strip())

    lines = [
        "# Phase 2: back-translated errors",
        "",
        f"{len(kept):,} pairs kept of {len(pairs):,} generated, from a generator trained on "
        f"Falko-MERLIN train read backwards. The argument for this over the hand-written "
        f"corruptors is in `reports/phase3/headroom.md`.",
        "",
        "## Settings",
        "",
        "| | |",
        "|---|---|",
        f"| adapter | `{args.adapter}` |",
        f"| sampling | temperature {args.temperature}, top-p {args.top_p} |",
        f"| seed | {seed} |",
        f"| clean sentences | {args.size:,}, skipping the first {args.skip} |"
        if args.skip is not None
        else f"| clean sentences | {args.size:,} |",
        f"| divergence threshold | {threshold:.4f} "
        f"({backtranslate.DIVERGENCE_QUANTILE:.0%} of real learner pairs) |",
        "",
        "## The funnel",
        "",
        "| Outcome | Pairs | Share |",
        "|---|---:|---:|",
    ]
    for reason, count in funnel.most_common():
        lines.append(f"| {reason} | {count:,} | {count / max(len(pairs), 1):.1%} |")

    lines += [
        "",
        "## Density",
        "",
        f"{negatives:,} of the kept pairs need no correction "
        f"({negatives / max(len(kept), 1):.1%}); Falko-MERLIN train is 22.1%.",
        "",
        "| Edits | Asked for | Actually made |",
        "|---:|---:|---:|",
    ]
    for n in range(0, 10):
        lines.append(f"| {n} | {asked.get(n, 0):,} | {made.get(n, 0):,} |")
    total_made = sum(n * c for n, c in made.items())
    lines += [
        f"| 10+ | {sum(c for n, c in asked.items() if n >= 10):,} | "
        f"{sum(c for n, c in made.items() if n >= 10):,} |",
        "",
        f"Mean edits a sentence: **{total_made / max(len(kept), 1):.2f}** "
        f"(Falko train 2.55, the rule-based corpus 2.34).",
        "",
        "## Samples",
        "",
    ]
    for pair in random.Random(seed).sample(kept, min(30, len(kept))):
        lines += [
            f"- asked for {pair.asked}",
            f"  - clean: `{pair.correct}`",
            f"  - damaged: `{pair.learner}`",
        ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "backtranslation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
