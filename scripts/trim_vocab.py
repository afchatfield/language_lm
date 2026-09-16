#!/usr/bin/env python
"""Trim the base model's vocabulary to the rows German and English actually use.

    python scripts/trim_vocab.py --threshold 0.999
    python scripts/trim_vocab.py --threshold 0.999 --sentences 20000  # quicker

No training happens here, which is the point: the roadmap's Phase 4 gate is to
confirm a trimmed checkpoint still converts before spending GPU hours healing
one. What comes out is a standard Hugging Face checkpoint with `vocab_size`
updated and both tokenizer artefacts rebuilt -- see `langlm.compress.vocab` for
why "both" is load-bearing.

The kept set is chosen by coverage over the Leipzig German and English corpora,
using the same counting Phase 0 used to *predict* this trim, so the realised
saving can be checked against `reports/phase0/tokenizer_fertility.md` rather
than merely reported.

**Natural language is not the whole distribution.** The model answers in JSON, and
the tokens that structure a JSON answer -- `▁{`, `▁[`, `}]`, `:"` -- are vanishing
rare in news and Wikipedia prose. A trim counted on prose alone deletes them, and
the model stops being able to emit its own schema: the first run of this script
produced a checkpoint that corrected German correctly and returned it as bare
text. So the training corpus is counted too, prompts and answers both, which is
the only place the output format appears. Any future task with a new output
vocabulary has to be added here as well.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from langlm.compress import vocab as compress_vocab
from langlm.config import PROCESSED_DIR, PROJECT_ROOT, load_config
from langlm.data.leipzig import read_sentences
from langlm.tokenizers import fertility


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.999, help="Coverage target")
    parser.add_argument(
        "--sentences",
        type=int,
        default=100_000,
        help="Sentences per corpus to count tokens over",
    )
    parser.add_argument("--out", default="checkpoints/base-trimmed", help="Where to write")
    parser.add_argument(
        "--no-task-data",
        action="store_true",
        help="Count prose only. The trim will then drop the JSON schema tokens; "
        "kept as a flag because that failure is worth being able to reproduce.",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=["de", "en"],
        help="Language codes whose usage protects a vocabulary row",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    phase0 = load_config("phase0")
    repo_id = load_config("phase3")["model"]["repo_id"]
    print(f"base model: {repo_id}")

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    counts = token_counts(tokenizer, args.languages, args.sentences, phase0)
    if not args.no_task_data:
        counts.update(task_counts(tokenizer))

    plan = compress_vocab.plan(tokenizer, counts, threshold=args.threshold)
    print(
        f"\nkeeping {plan.new_size:,} of {plan.original_size:,} rows "
        f"({plan.share_removed:.1%} removed, {plan.protected:,} protected)"
    )

    model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.bfloat16)
    before = sum(p.numel() for p in model.parameters())
    compress_vocab.trim_weights(model, plan)
    after = sum(p.numel() for p in model.parameters())
    print(
        f"parameters: {before:,} -> {after:,} "
        f"({(before - after) / before:.1%} removed, {(before - after) * 2 / 1e6:.0f} MB at bf16)"
    )

    destination = PROJECT_ROOT / args.out
    source = Path(tokenizer.name_or_path)
    if not source.exists():
        from transformers.utils import cached_file

        source = Path(cached_file(repo_id, "tokenizer.model")).parent

    compress_vocab.write(source, destination, plan, model, tokenizer)
    (destination / "trim_plan.json").write_text(
        json.dumps(
            {
                "base": repo_id,
                "threshold": args.threshold,
                "languages": args.languages,
                "sentences_per_corpus": args.sentences,
                "original_vocab": plan.original_size,
                "new_vocab": plan.new_size,
                "protected": plan.protected,
                "parameters_before": before,
                "parameters_after": after,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {destination}")


def task_counts(tokenizer) -> Counter[int]:
    """Token occurrences over the Phase 3 corpus, prompts and answers both.

    This is where the output schema lives. Counting it is what keeps `▁{` and
    friends in the vocabulary, and it also protects the ERRANT type tags the
    model has to emit.
    """
    from langlm.train.format import build_prompt, build_target

    path = PROCESSED_DIR / "training" / "de.jsonl"
    if not path.exists():
        raise SystemExit(
            f"No training corpus at {path}. Run `make training-set`, or pass "
            f"--no-task-data to trim on prose alone and lose the schema tokens."
        )

    counts: Counter[int] = Counter()
    records = 0
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        for text in (build_prompt(record["source"]), build_target(record)):
            counts.update(tokenizer.encode(text, add_special_tokens=False))
        records += 1
    print(f"  {'task corpus':<14} {records:>7,} examples (prompts and answers)")
    return counts


def token_counts(tokenizer, languages, sentences: int, phase0: dict) -> Counter[int]:
    """Count token occurrences over the Leipzig corpora for the kept languages."""
    counts: Counter[int] = Counter()
    for entry in phase0["corpora"]:
        if entry["lang"] not in languages:
            continue
        texts = list(read_sentences(entry["leipzig"]))[:sentences]
        print(f"  {entry['key']:<14} {len(texts):>7,} sentences")
        stats = fertility.analyse_corpus(
            tokenizer=tokenizer,
            sentences=texts,
            corpus=entry["key"],
            lang=entry["lang"],
        )
        counts.update(stats.token_counts)
    if not counts:
        raise SystemExit(f"No corpora matched {languages}. Run `make corpora` first.")
    return counts


if __name__ == "__main__":
    main()
