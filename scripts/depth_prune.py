#!/usr/bin/env python
"""Rank decoder layers by redundancy and write a pruned checkpoint per sweep point.

    python scripts/depth_prune.py --analyse                 # just the ranking
    python scripts/depth_prune.py --drop 2 4 6              # ranking + checkpoints

The vocabulary trim turned out to be free -- 21.9% of parameters for 0.0011 of
F0.5 -- and it stops 155 MB short of the 600 MB target. Everything remaining has
to come from depth, and depth will not be free: every layer is on the path of
every token, where a vocabulary row the language never emits is not.

The roadmap asks for a *curve*, not a point, so this writes one checkpoint per
sweep value and leaves the scoring to `scripts/eval_finetuned.py`. The ranking is
a heuristic that proposes candidates; the curve decides.

Measured on clean German, because German at inference is what the layers have to
survive doing.
"""

from __future__ import annotations

import argparse
import json

from langlm.compress import depth
from langlm.config import PROCESSED_DIR, PROJECT_ROOT
from langlm.data.m2 import parse_m2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="checkpoints/base-trimmed")
    parser.add_argument("--drop", type=int, nargs="*", default=[], help="Sweep points")
    parser.add_argument("--analyse", action="store_true", help="Rank layers and stop")
    parser.add_argument("--sentences", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--protect-edges",
        type=int,
        default=1,
        help="Layers at each end never dropped. The first block turns embeddings "
        "into what the stack expects and the last prepares what the head reads; "
        "both score as redundant on this metric and are not.",
    )
    parser.add_argument("--out-prefix", default="checkpoints/base-trimmed-pruned")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = PROJECT_ROOT / args.base
    if not base.exists():
        raise SystemExit(f"No checkpoint at {base}. Run scripts/trim_vocab.py first.")

    corpus = PROCESSED_DIR / "clean_de" / "de.m2"
    texts = [s.source for s in parse_m2(corpus)][: args.sentences]
    print(f"{len(texts):,} German sentences for the measurement")

    tokenizer = AutoTokenizer.from_pretrained(str(base))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.bfloat16).to(device)

    scores = depth.measure(model, tokenizer, texts, device=device, batch_size=args.batch_size)
    print(f"\n{len(scores)} layers, input-output cosine similarity (higher = more redundant)\n")
    ranked = sorted(scores, key=lambda s: s.similarity, reverse=True)
    for score in scores:
        rank = ranked.index(score) + 1
        bar = "#" * int(max(0.0, score.similarity) * 40)
        print(f"  layer {score.index:2d}  {score.similarity:+.4f}  rank {rank:2d}  {bar}")

    report = {
        "base": args.base,
        "sentences": len(texts),
        "layers": [{"index": s.index, "similarity": s.similarity} for s in scores],
        "windows": {},
    }
    for count in sorted({*args.drop, 2, 4, 6}):
        window = depth.best_window(scores, count, protect_edges=args.protect_edges)
        report["windows"][str(count)] = {
            "start": window.start,
            "layers": window.layers,
            "similarity": window.similarity,
        }
        print(
            f"\nbest contiguous window of {count}: layers {window.layers} "
            f"(mean similarity {window.similarity:.4f})"
        )

    out = PROJECT_ROOT / "reports" / "phase4" / "layer_similarity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")

    if args.analyse or not args.drop:
        return

    for count in args.drop:
        window = depth.best_window(scores, count, protect_edges=args.protect_edges)
        destination = PROJECT_ROOT / f"{args.out_prefix}{count}"
        if destination.exists() and any(destination.iterdir()):
            print(f"  {destination} exists, skipping")
            continue

        # Reloaded each time: pruning is in place, so reusing the model would
        # drop layers from an already-pruned stack and the sweep points would
        # not be independent.
        fresh = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.bfloat16)
        depth.prune(fresh, window)
        before = sum(p.numel() for p in model.parameters())
        after = sum(p.numel() for p in fresh.parameters())

        destination.mkdir(parents=True, exist_ok=True)
        fresh.save_pretrained(destination)
        tokenizer.save_pretrained(destination)
        for name in ("tokenizer.model", "tokenizer.json"):
            source = base / name
            if source.exists():
                (destination / name).write_bytes(source.read_bytes())
        (destination / "prune_plan.json").write_text(
            json.dumps(
                {
                    "base": args.base,
                    "dropped_layers": window.layers,
                    "mean_similarity": window.similarity,
                    "layers_before": model.config.num_hidden_layers,
                    "layers_after": fresh.config.num_hidden_layers,
                    "parameters_before": before,
                    "parameters_after": after,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"  dropped {count} -> {destination.name}: {before:,} -> {after:,} params "
            f"({(before - after) / before:.1%}, {(before - after) * 2 / 1e6:.0f} MB bf16)"
        )


if __name__ == "__main__":
    main()
