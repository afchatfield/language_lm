#!/usr/bin/env python
"""Generate a beam over Falko-MERLIN train and pair its best candidate against its first.

The last thing `reports/phase3/nbest.md` left standing. The beam holds a much
better correction than the model ranks first, no feature computed outside the
model finds it, and preference training is the one channel that is not bounded by
what those features separate.

    python scripts/build_preference_pairs.py --adapter checkpoints/phase3-de-iter4-cw025 \
        --name iter4-cw025

Generations are cached, so re-deriving pairs after a change to the selection rule
costs nothing. This reads Falko-MERLIN *train*; the test split is not touched.

The pairs are labelled by the gold annotation and both sides are the model's own
output, so nothing is distilled from a larger model.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from langlm.config import INTERIM_DIR, PROJECT_ROOT, load_config
from langlm.data.splits import load_split
from langlm.eval import errant_de
from langlm.train.format import CHANGES_MARKER, build_prompt, parse_answer
from langlm.train.preference import build_pairs

CACHE_DIR = INTERIM_DIR / "phase3" / "preference"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="checkpoints/phase3-de-iter4-cw025")
    parser.add_argument("--name", default="iter4-cw025")
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, help="First N train sentences")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument(
        "--min-gain",
        type=int,
        default=1,
        help="Drop pairs whose chosen side beats the rejected one by less than this "
        "in true positives minus false ones. 1 keeps everything that is strictly better.",
    )
    args = parser.parse_args()

    train = load_split("falko_merlin", "train")
    if args.limit:
        train = train[: args.limit]
    print(f"{len(train):,} train sentences, {args.beams} beams")

    prompts = [build_prompt(sentence.source) for sentence in train]
    records = generate(args, train, prompts)
    pairs = [p for p in build_pairs(train, records, prompts) if p.gain >= args.min_gain]

    yielded = len(pairs) / len(train)
    print(f"\n{len(pairs):,} pairs from {len(train):,} sentences ({yielded:.1%})")

    ranks = Counter(pair.chosen_rank for pair in pairs)
    print("chosen rank: " + ", ".join(f"{rank}: {ranks[rank]:,}" for rank in sorted(ranks)))
    gains = Counter(min(pair.gain, 4) for pair in pairs)
    print(
        "gain (tp-fp): "
        + ", ".join(f"{'4+' if g == 4 else g}: {gains[g]:,}" for g in sorted(gains))
    )

    path = CACHE_DIR / f"{args.name}.pairs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "chosen_rank": pair.chosen_rank,
                    "gain": pair.gain,
                },
                ensure_ascii=False,
            )
            + "\n"
            for pair in pairs
        ),
        encoding="utf-8",
    )
    print(f"wrote {path}")


def generate(args, sentences, prompts: list[str]) -> list[list[dict]]:
    """n-best per sentence as ``{'answer', 'text', 'score'}``.

    ``answer`` is the raw completion, which is what DPO trains on; ``text`` is
    the correction tokenised the way the scorer expects, which is what decides
    which side of the pair it lands on. The n-best cache from
    `scripts/nbest_rerank.py` keeps only the second, so this generates its own.
    """
    path = CACHE_DIR / f"{args.name}.beams{args.beams}.{len(sentences)}.jsonl"
    if path.exists() and not args.regenerate:
        print(f"  cached: {path}")
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = PROJECT_ROOT / args.adapter
    if not adapter.exists():
        raise SystemExit(f"No adapter at {adapter}")

    phase3 = load_config("phase3")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(phase3["model"]["repo_id"], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    records: list[list[dict]] = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        sources = [s.source for s in sentences[start : start + args.batch_size]]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.beams,
                num_return_sequences=args.beams,
                do_sample=False,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # The preference is over the correction, so the beam stops where
                # the correction does -- and the completion the pair carries is
                # the one the beam actually scored.
                stop_strings=[CHANGES_MARKER],
                tokenizer=tokenizer,
            )
        width = encoded["input_ids"].shape[1]
        answers = tokenizer.batch_decode(out.sequences[:, width:], skip_special_tokens=True)
        scores = out.sequences_scores.tolist()
        for position, source in enumerate(sources):
            window = slice(position * args.beams, (position + 1) * args.beams)
            record = []
            for answer, score in zip(answers[window], scores[window], strict=True):
                answer = answer.strip()
                parsed = parse_answer(answer)
                text = parsed.correction if parsed else source
                record.append({"answer": answer, "text": errant_de.tokenize(text), "score": score})
            records.append(record)
        done = min(start + args.batch_size, len(prompts))
        print(f"  {done}/{len(prompts)}", end="\r")
    print()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    print(f"  wrote {path}")
    return records


if __name__ == "__main__":
    main()
