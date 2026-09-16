#!/usr/bin/env python
"""Is there a better correction inside the beam than the one greedy picks?

`reports/phase3/headroom.md` proposes reranking n-best candidates by edit
distance to the source, on the grounds that 558 of the 843 harmful false
positives are *wider* than the gold edit against 3 that are narrower. That is a
one-directional bias, which is the kind a prior can correct and a better search
cannot.

The proposal has a precondition nobody has measured: a rerank can only reach a
correction that is somewhere in the beam. This measures that precondition before
anything is built.

    python scripts/nbest_rerank.py --adapter checkpoints/phase3-de-iter4-cw025 --beams 5

Three numbers matter, on the 600-sentence screen subset:

* **beam top-1** -- what beam search alone buys over greedy, which is the
  serving cost without the prior.
* **min-edit rerank** -- the same candidates, selected by
  `sequence_score - penalty x edit_distance(candidate, source)`, swept over the
  penalty so the result is a curve rather than one arbitrary setting. At
  penalty 0 this is beam top-1 by construction.
* **oracle** -- per sentence, the candidate that scores best against the gold.
  No reranker of any kind can beat this, so if it sits on top of greedy the
  idea is dead however clever the reranker.

Generations are cached, so re-scoring after a change to the selection rule costs
nothing. The test split is not touched.
"""

from __future__ import annotations

import argparse
import json
import random

from langlm.config import INTERIM_DIR, PROJECT_ROOT, load_config
from langlm.data.splits import load_split
from langlm.eval import errant_de, subset
from langlm.eval import m2_scorer as m2
from langlm.train.format import CHANGES_MARKER, build_prompt, parse_answer

CACHE_DIR = INTERIM_DIR / "phase3" / "nbest"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="checkpoints/phase3-de-iter4-cw025")
    parser.add_argument("--name", default="iter4-cw025")
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--limit", type=int, help="First N screen sentences, for a smoke test")
    parser.add_argument(
        "--complement",
        action="store_true",
        help="Score dev *minus* the screen subset. The penalty was chosen on the "
        "screen, so this is the held-out half and the number that can be quoted.",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=0.02,
        help="The pre-registered penalty, chosen on the screen. Reported as the "
        "headline alongside the sweep.",
    )
    args = parser.parse_args()

    phase1, phase3 = load_config("phase1"), load_config("phase3")
    beta = phase1["evaluation"]["beta"]

    dev = load_split("falko_merlin", phase1["evaluation"]["split"])
    screen, chosen = subset.take(dev)
    positions = chosen
    if args.complement:
        held_out = set(chosen)
        positions = [i for i in range(len(dev)) if i not in held_out]
        screen = [dev[i] for i in positions]
        args.name = f"{args.name}-heldout"
    if args.limit:
        screen, positions = screen[: args.limit], positions[: args.limit]
    print(f"{len(screen)} sentences of {len(dev):,} dev, {args.beams} beams")

    records = generate(args, screen, phase3)

    # Candidates come back tokenised the way the scorer expects.
    golds = [s.edits_for() for s in screen]
    sources = [s.source_tokens for s in screen]

    print("\n--- what is actually in the beam ---")
    distinct = [len({c["text"] for c in r}) for r in records]
    print(
        f"distinct corrections per sentence: mean {sum(distinct) / len(distinct):.2f}, "
        f"all identical in {sum(1 for d in distinct if d == 1):,} of {len(distinct):,}"
    )

    widths = []
    for record, source in zip(records, sources, strict=True):
        widths.append(min(distance(source, c["text"].split()) for c in record))
    top1 = [distance(s, r[0]["text"].split()) for r, s in zip(records, sources, strict=True)]
    print(
        f"edit distance to source: beam top-1 mean {sum(top1) / len(top1):.2f}, "
        f"narrowest candidate mean {sum(widths) / len(widths):.2f}"
    )

    print("\n--- scores ---")
    rows: list[tuple[str, float, float, float]] = []

    greedy = cached_greedy(args, screen, positions)
    if greedy:
        rows.append(("greedy (published path)", *scored(screen, greedy, beta)))

    rows.append(("beam top-1", *scored(screen, [r[0]["text"] for r in records], beta)))

    # A control on how much of the oracle is real signal. The oracle is a
    # maximum over five candidates, and a maximum over noise is above the mean
    # whether or not anything is there to find; if random selection sits at
    # top-1 rather than well below it, the candidates differ in ways the metric
    # rewards at random.
    rng = random.Random(subset.SEED)
    picked = [rng.choice(r)["text"] for r in records]
    rows.append(("random pick from beam", *scored(screen, picked, beta)))

    # Minimum Bayes risk: the candidate most like the *other* candidates, rather
    # than the one the model scores highest. No training, no gold, and it uses
    # the diversity directly instead of through a hand-set prior.
    picked = [mbr(r) for r in records]
    rows.append(("MBR (consensus of the beam)", *scored(screen, picked, beta)))

    headline = args.penalty
    for penalty in sorted({0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, headline}):
        picked = [rerank(r, s, penalty) for r, s in zip(records, sources, strict=True)]
        mark = "  <- pre-registered" if penalty == headline else ""
        rows.append((f"min-edit rerank, penalty {penalty:g}{mark}", *scored(screen, picked, beta)))

    picked, changed = oracle(records, sources, golds)
    rows.append(("oracle (per-sentence best)", *scored(screen, picked, beta)))

    print(f"{'selection':34s} {'F0.5':>7s} {'P':>7s} {'R':>7s}")
    for label, f, p, r in rows:
        print(f"{label:34s} {f:7.4f} {p:7.4f} {r:7.4f}")
    print(f"\noracle picks a non-top-1 candidate on {changed:,} of {len(records):,} sentences")


def scored(sentences, hypotheses, beta: float) -> tuple[float, float, float]:
    result = m2.score(sentences, hypotheses, beta=beta)
    return result.f_score, result.precision, result.recall


def distance(a: list[str], b: list[str]) -> int:
    """Token-level Levenshtein distance."""
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            current.append(
                previous[j - 1]
                if left == right
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


def rerank(record: list[dict], source: list[str], penalty: float) -> str:
    """Pick by likelihood discounted for how far the candidate moves from the source."""
    best, best_key = None, None
    for candidate in record:
        key = candidate["score"] - penalty * distance(source, candidate["text"].split())
        if best_key is None or key > best_key:
            best, best_key = candidate["text"], key
    return best or record[0]["text"]


def mbr(record: list[dict]) -> str:
    """The candidate closest to all the others, by token edit distance."""
    best, best_key = None, None
    for candidate in record:
        tokens = candidate["text"].split()
        risk = sum(distance(tokens, other["text"].split()) for other in record)
        key = -risk
        if best_key is None or key > best_key:
            best, best_key = candidate["text"], key
    return best or record[0]["text"]


def oracle(records, sources, golds) -> tuple[list[str], int]:
    """Per sentence, the candidate that scores best against the annotator.

    Greedy per sentence rather than a joint optimum over the corpus, so it is a
    close approximation to the ceiling rather than exactly it. Ties go to the
    narrower edit, which is the tie-break a real reranker could also make.
    """
    picked: list[str] = []
    changed = 0
    for record, source, gold in zip(records, sources, golds, strict=True):
        best, best_key, best_index = None, None, 0
        for index, candidate in enumerate(record):
            counts = m2.sentence_counts(source, candidate["text"].split(), gold)
            key = (counts.tp - counts.fp, -distance(source, candidate["text"].split()))
            if best_key is None or key > best_key:
                best, best_key, best_index = candidate["text"], key, index
        picked.append(best or record[0]["text"])
        changed += best_index != 0
    return picked, changed


def cached_greedy(args, screen, positions) -> list[str] | None:
    """The greedy generations for these sentences, from whichever run has them.

    The screen has its own cached run; any other slice of dev is taken out of
    the full-dev run by position, which is the same generations the published
    0.7072 was scored from rather than a fresh pass that would differ from it.
    """
    base = args.name.removesuffix("-heldout")
    screened = INTERIM_DIR / "phase3" / "screen" / f"{base}.dev.{len(screen)}.txt"
    if screened.exists():
        return screened.read_text(encoding="utf-8").splitlines()

    full = INTERIM_DIR / "phase3" / f"{base}.dev.2503.txt"
    if full.exists():
        lines = full.read_text(encoding="utf-8").splitlines()
        if max(positions) < len(lines):
            print(f"  greedy row sliced from {full.name} by position")
            return [lines[i] for i in positions]

    print("  (no cached greedy run for these sentences; skipping that row)")
    return None


def generate(args, screen, phase3: dict) -> list[list[dict]]:
    """n-best per sentence, as {'text': tokenised correction, 'score': logprob}."""
    path = CACHE_DIR / f"{args.name}.beams{args.beams}.{len(screen)}.jsonl"
    if path.exists() and not args.regenerate:
        print(f"  cached: {path}")
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = PROJECT_ROOT / args.adapter
    if not adapter.exists():
        raise SystemExit(f"No adapter at {adapter}")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(phase3["model"]["repo_id"], dtype=torch.bfloat16)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    sources = [s.source for s in screen]
    records: list[list[dict]] = []
    for start in range(0, len(sources), args.batch_size):
        batch = sources[start : start + args.batch_size]
        encoded = tokenizer(
            [build_prompt(source) for source in batch], return_tensors="pt", padding=True
        ).to(model.device)
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
                # The correction is the only field scored, and it comes first in
                # the schema, so there is no reason to spend beam width on the
                # `changes` list this experiment never reads.
                stop_strings=[CHANGES_MARKER],
                tokenizer=tokenizer,
            )
        width = encoded["input_ids"].shape[1]
        answers = tokenizer.batch_decode(out.sequences[:, width:], skip_special_tokens=True)
        scores = out.sequences_scores.tolist()
        for position, source in enumerate(batch):
            window = slice(position * args.beams, (position + 1) * args.beams)
            record = []
            for answer, score in zip(answers[window], scores[window], strict=True):
                parsed = parse_answer(answer.strip())
                text = parsed.correction if parsed else source
                record.append({"text": errant_de.tokenize(text), "score": score})
            records.append(record)
        print(f"  {min(start + args.batch_size, len(sources))}/{len(sources)}", end="\r")
    print()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    print(f"  wrote {path}")
    return records


if __name__ == "__main__":
    main()
