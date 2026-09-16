#!/usr/bin/env python
"""Build the Phase 3 training set: real learner errors, plus synthetic ones.

Iteration 2 of the recipe. The first was synthetic-only and it made the model
worse than the untrained baseline on 14 of 15 error types -- see
`reports/phase3/evaluation.md`. Real learner errors are the primary signal now.

Three sources, mixed:

    learner     Falko-MERLIN train. 54 error types, real error density, real
                negatives. Explanations are type-level, because a Falko edit
                records what changed and not why.
    synthetic   Clean text damaged by `langlm.corruptors`. 17 error types, but it
                knows which rule it broke, so its explanations are specific.
    backtranslation
                Clean text damaged by a model trained on Falko read backwards
                (`langlm.data.backtranslate`). It reaches the error types no
                rule can write -- lexical choice above all -- and its
                explanations are type-level for the same reason the learner
                half's are. Iteration 5; see `reports/phase3/headroom.md`.

Each line of the output is one training example:

    {"source": "...", "target": "...",
     "edits": [{"start": 3, "end": 4, "error_type": "R:ORTH",
                "correction": "Hund", "rule": "noun_case", "source_kind":
                "synthetic", "explanation": "..."}]}

    python scripts/build_training_set.py
    python scripts/build_training_set.py --size 500     # trial
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from langlm.config import PROCESSED_DIR, REPORTS_DIR, load_config
from langlm.data import backtranslate, clean_de, injection, learner
from langlm.data.splits import load_split
from langlm.explanations import TemplateError, explain

OUTPUT_DIR = PROCESSED_DIR / "training"
REPORT_DIR = REPORTS_DIR / "phase2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, help="Cap both sources, for a trial run")
    parser.add_argument("--samples", type=int, default=100, help="Examples to write out")
    parser.add_argument(
        "--backtranslation",
        help="Kept pairs from `scripts/generate_backtranslation.py`. Defaults to the "
        "configured path; pass an empty string to build the corpus without them.",
    )
    args = parser.parse_args()

    config = load_config("phase2")
    dataset_cfg = config["dataset"]
    seed = dataset_cfg["seed"]

    learner_limit = dataset_cfg["learner"]["limit"]
    synthetic_size = dataset_cfg["synthetic"]["size"]
    if args.size:
        learner_limit = args.size
        synthetic_size = args.size

    print("Reading real learner errors ...")
    real = [dict(record, source_kind="learner") for record in learner.records(limit=learner_limit)]
    print(f"  {len(real):,} from Falko-MERLIN {dataset_cfg['learner']['split']}")

    print(f"Generating {synthetic_size:,} synthetic examples ...")
    synthetic, stats, unexplained = build_synthetic(config, synthetic_size, seed)
    print(f"  {len(synthetic):,} generated, {unexplained} edits without a template")

    generated = build_backtranslated(dataset_cfg, args.backtranslation)

    records = real + synthetic + generated
    random.Random(seed).shuffle(records)
    for record in records:
        for edit in record["edits"]:
            edit["source_kind"] = record["source_kind"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "de.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {path} ({len(records):,} examples)")

    write_report(records, real, synthetic, generated, stats, unexplained, args.samples, seed)


def build_backtranslated(dataset_cfg: dict, override: str | None) -> list[dict]:
    """Read the kept back-translation pairs and type them, if there are any.

    Absent pairs are not an error. The corpus has been built twice without them
    and the recipe has to stay runnable from a fresh clone, where training the
    generator is a separate and much longer step.
    """
    settings = dataset_cfg.get("backtranslation") or {}
    if override == "":
        print("Skipping back-translated errors (--backtranslation '')")
        return []
    path = Path(override or settings.get("pairs") or backtranslate.PAIRS_DIR / "de.kept.jsonl")
    if not path.exists():
        print(
            f"No back-translated pairs at {path}; building without them. "
            f"Make them with `make backtranslation`."
        )
        return []

    pairs = backtranslate.read_pairs(path)
    size = settings.get("size")
    if size:
        pairs = pairs[:size]
    print(f"Typing {len(pairs):,} back-translated pairs ...")
    records = [
        dict(record, source_kind="backtranslation")
        for record in backtranslate.training_records(pairs)
    ]
    print(f"  {len(records):,} typed")
    return records


def build_synthetic(config: dict, size: int, seed: int):
    """Corrupt clean sentences and explain the damage."""
    import spacy

    settings = config["injection"]
    clean = load_split(clean_de.CORPUS, "all")[:size]
    nlp = spacy.load("de_core_news_sm", disable=["ner"])
    docs = list(nlp.pipe([s.source for s in clean], batch_size=64))

    stats = injection.InjectionStats()
    generated = injection.generate(
        [s.source_tokens for s in clean],
        weights=settings["weights"],
        seed=seed,
        errors=tuple(settings["errors_per_sentence"]),
        stats=stats,
        docs=docs,
    )

    records, unexplained = [], 0
    for example, source in zip(generated, clean, strict=True):
        record = {
            "source": example.source,
            "target": source.source,
            "edits": [],
            "source_kind": "synthetic",
        }
        for edit in example.edits:
            if edit.is_noop:
                continue
            rule = edit.comment
            try:
                text = explain(
                    rule,
                    wrong=" ".join(example.source_tokens[edit.start : edit.end]),
                    right=edit.correction,
                )
            except TemplateError:
                unexplained += 1
                continue
            record["edits"].append(
                {
                    "start": edit.start,
                    "end": edit.end,
                    "error_type": edit.error_type,
                    "correction": edit.correction,
                    "rule": rule,
                    "explanation": text,
                }
            )
        records.append(record)
    return records, stats, unexplained


def write_report(
    records, real, synthetic, generated, stats, unexplained, samples: int, seed: int
) -> None:
    """Verify the invariants and write the sample the exit criterion asks for."""
    damaged = [r for r in records if r["edits"]]
    correct = [r for r in records if not r["edits"]]
    broken = [r for r in damaged if r["source"] == r["target"]]
    missing = [e for r in damaged for e in r["edits"] if not e["explanation"]]

    per_sentence = Counter(len(r["edits"]) for r in records)
    total = len(records)
    types = Counter(e["error_type"] for r in records for e in r["edits"])
    kinds = Counter(r["source_kind"] for r in records)
    edit_kinds = Counter(e["source_kind"] for r in records for e in r["edits"])

    lines = [
        "# Phase 2: the training set",
        "",
        f"{total:,} examples: **{kinds['learner']:,} real** learner sentences from "
        f"Falko-MERLIN train, **{kinds['synthetic']:,} rule-generated** ones, and "
        f"**{kinds['backtranslation']:,} back-translated** ones. "
        f"{len(correct):,} need no correction ({len(correct) / total:.1%}).",
        "",
        "Iteration 1 was synthetic-only and made the model worse than the untrained "
        "baseline on 14 of 15 error types (`reports/phase3/evaluation.md`), so real "
        "learner errors became the primary signal and the corruptors the supporting one. "
        "Iteration 3 widened the corruptors from 8 error types to 17 and bought +0.0016, "
        "which is nothing (`reports/phase3/ablations.md`, section 5). The back-translated "
        "share is iteration 5's answer to why: a rule can only write the error types that "
        "are easy to write, and those are the ones the model was already good at. "
        "See `reports/phase3/headroom.md`.",
        "",
        "## Invariants",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Damaged examples whose source equals their target | {len(broken)} |",
        f"| Edits with no explanation | {len(missing)} |",
        f"| Synthetic edits with no template | {unexplained} |",
        f"| Sentences no corruptor could damage | {stats.barren:,} |",
        "",
        "## Error density",
        "",
        "The number iteration 1 got wrong: it produced nothing with three or more errors, "
        "and 39% of real learner sentences have them.",
        "",
        "| Edits in a sentence | Share | Falko train |",
        "|---|---:|---:|",
    ]
    reference = {0: 0.221, 1: 0.212, 2: 0.175}
    for count in (0, 1, 2):
        lines.append(f"| {count} | {per_sentence[count] / total:.1%} | {reference[count]:.1%} |")
    three_plus = sum(v for k, v in per_sentence.items() if k >= 3)
    lines += [
        f"| 3 or more | {three_plus / total:.1%} | 39.2% |",
        "",
        f"Mean edits per sentence: **{sum(types.values()) / total:.2f}** "
        f"(Falko train: 2.55, iteration 1: 1.17).",
        "",
        "## Error types",
        "",
        f"**{len(types)}** distinct types, against 8 in iteration 1 and 54 in Falko train.",
        "",
        "| Error type | Count | Share |",
        "|---|---:|---:|",
    ]
    edit_total = sum(types.values())
    for error_type, count in types.most_common(20):
        lines.append(f"| `{error_type}` | {count:,} | {count / edit_total:.1%} |")

    lines += [
        "",
        "## Explanations",
        "",
        f"{edit_kinds['synthetic']:,} edits carry a rule-level explanation, from the "
        f"corruptor that made them. {edit_kinds['learner']:,} carry a type-level one, "
        "because a Falko edit records what the annotator changed and not why.",
        "",
        "That split is the point of the mix: the real data covers all the error types and "
        "teaches what learner German looks like, and the synthetic data teaches how to "
        "explain the eight it can produce properly.",
        "",
        "## 100 samples",
        "",
        "The Phase 2 exit criterion: read these, and if more than five have a wrong "
        "correction or a wrong explanation, fix the pipeline before training on it.",
        "",
    ]
    for index, record in enumerate(random.Random(seed).sample(damaged, min(samples, len(damaged)))):
        lines += [
            f"**{index + 1}.** *({record['source_kind']})* `{record['source']}`",
            f"→ `{record['target']}`",
        ]
        for edit in record["edits"]:
            lines.append(f"  - *{edit['error_type']}*: {edit['explanation']}")
        lines.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "training_set.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(
        f"  {kinds['learner']:,} real + {kinds['synthetic']:,} synthetic | "
        f"{len(types)} types | mean {edit_total / total:.2f} edits | "
        f"{len(correct) / total:.1%} correct | broken {len(broken)}"
    )


if __name__ == "__main__":
    main()
