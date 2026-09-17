#!/usr/bin/env python
"""Build the Spanish training set: real learner errors, plus synthetic ones.

The Spanish counterpart of `build_training_set.py`, built to the state German
was in when it first cleared the exit criterion -- real data primary,
synthetic supporting -- not to every later German iteration. Two things
follow from that scoping call, made explicitly rather than silently:

* **No back-translation section.** German's own iteration 5 added it after a
  successful first run, predicted +0.02-0.03 F0.5, and measured -0.0016 to
  -0.0017 instead (`reports/phase3/headroom.md`) -- it did not earn its keep
  even for German. Building the generator it needs (a model trained on
  COWS-L2H read backwards) is real, separate work with no evidence yet that
  Spanish would want it before a first real+synthetic run exists to compare
  against.
* **`injection.negative_share` is measured, not copied.** COWS-L2H train is
  33.0% already-correct sentences (`load_split("cowsl2h", "train")`), not
  Falko-MERLIN's 22% -- a real, different number, used as the real number.

Two sources, mixed:

    learner     COWS-L2H train. Real error density, real negatives.
                Explanations are type-level -- a COWS-L2H edit records what
                changed and not why, the same limitation Falko-MERLIN's
                annotation has.
    synthetic   Clean Spanish text (`clean_es`) damaged by `corruptors.es`.
                13 error types, but it knows which rule it broke, so its
                explanations are specific.

Each line of the output is one training example, the same shape German's is:

    {"source": "...", "target": "...",
     "edits": [{"start": 3, "end": 4, "error_type": "R:ORTH",
                "correction": "año", "rule": "drop_accent", "source_kind":
                "synthetic", "explanation": "..."}]}

    python scripts/build_training_set_es.py
    python scripts/build_training_set_es.py --size 500     # trial
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from langlm.config import PROCESSED_DIR, REPORTS_DIR, load_config
from langlm.data import clean_es, injection, learner_es
from langlm.data.splits import load_split
from langlm.explanations import TemplateError, explain

OUTPUT_DIR = PROCESSED_DIR / "training"
REPORT_DIR = REPORTS_DIR / "phase5"

#: Measured from `load_split("cowsl2h", "train")`: 12,363 of 37,473 sentences
#: (33.0%) carry no edit. Falko-MERLIN's is 22% -- a real difference in how
#: the two corpora were collected, not a value either language should borrow
#: from the other.
NEGATIVE_SHARE = 0.330


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, help="Cap both sources, for a trial run")
    parser.add_argument("--samples", type=int, default=100, help="Examples to write out")
    args = parser.parse_args()

    config = load_config("phase2_es")
    dataset_cfg = config.get("dataset", {})
    seed = dataset_cfg.get("seed", 20260909)

    learner_limit = args.size
    synthetic_size = args.size or config["clean_corpus"]["size"]

    print("Reading real learner errors ...")
    real = [
        dict(record, source_kind="learner") for record in learner_es.records(limit=learner_limit)
    ]
    print(f"  {len(real):,} from COWS-L2H train")

    print(f"Generating {synthetic_size:,} synthetic examples ...")
    synthetic, stats, unexplained = build_synthetic(config, synthetic_size, seed)
    print(f"  {len(synthetic):,} generated, {unexplained} edits without a template")

    records = real + synthetic
    random.Random(seed).shuffle(records)
    for record in records:
        for edit in record["edits"]:
            edit["source_kind"] = record["source_kind"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "es.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {path} ({len(records):,} examples)")

    write_report(records, real, synthetic, stats, unexplained, args.samples, seed)


def build_synthetic(config: dict, size: int, seed: int):
    """Corrupt clean Spanish sentences and explain the damage."""
    import spacy

    from langlm.corruptors import es as es_corruptors

    settings = config["injection"]
    clean = load_split(clean_es.CORPUS, "all")[:size]
    nlp = spacy.load("es_core_news_sm", disable=["ner"])
    docs = list(nlp.pipe([s.source for s in clean], batch_size=64))

    stats = injection.InjectionStats()
    generated = injection.generate(
        [s.source_tokens for s in clean],
        weights=settings["weights"],
        seed=seed,
        negative_share=NEGATIVE_SHARE,
        errors=tuple(settings["errors_per_sentence"]),
        stats=stats,
        docs=docs,
        registry=es_corruptors.REGISTRY,
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
                    language="es",
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


def write_report(records, real, synthetic, stats, unexplained, samples: int, seed: int) -> None:
    """Verify the invariants and write the sample the exit criterion asks for.

    Mirrors `build_training_set.py`'s report exactly, minus the
    back-translation columns this build does not have.
    """
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
        "# Phase 5: the Spanish training set",
        "",
        f"{total:,} examples: **{kinds['learner']:,} real** learner sentences from "
        f"COWS-L2H train, and **{kinds['synthetic']:,} rule-generated** ones. "
        f"{len(correct):,} need no correction ({len(correct) / total:.1%}).",
        "",
        "Built to German's iteration-2 state -- real data primary, synthetic "
        "supporting -- not to every later German iteration. In particular there is "
        "no back-translated third source: German's own (iteration 5) measured "
        "*negative* against the bar its first successful run set "
        "(`reports/phase3/headroom.md`), so it is not assumed Spanish wants one "
        "before a first real+synthetic run exists to compare against.",
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
        "| Edits in a sentence | Share | COWS-L2H train |",
        "|---|---:|---:|",
    ]
    # Measured directly from `load_split("cowsl2h", "train")` rather than
    # copied from Falko-MERLIN's reference row.
    reference = {0: 0.330, 1: 0.250, 2: 0.161}
    for count in (0, 1, 2):
        lines.append(f"| {count} | {per_sentence[count] / total:.1%} | {reference[count]:.1%} |")
    three_plus = sum(v for k, v in per_sentence.items() if k >= 3)
    lines += [
        f"| 3 or more | {three_plus / total:.1%} | 25.9% |",
        "",
        f"Mean edits per sentence: **{sum(types.values()) / total:.2f}**.",
        "",
        "## Error types",
        "",
        f"**{len(types)}** distinct types, against 13 the synthetic rules alone can "
        "make and 51 in the real COWS-L2H histogram "
        "(`reports/phase5/error_type_histogram_es.md`).",
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
        "because a COWS-L2H edit records what changed and not why.",
        "",
        "## 100 samples",
        "",
        "The same exit criterion Phase 2 used: read these, and if more than five have "
        "a wrong correction or a wrong explanation, fix the pipeline before training "
        "on it.",
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
    output = REPORT_DIR / "training_set_es.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(
        f"  {kinds['learner']:,} real + {kinds['synthetic']:,} synthetic | "
        f"{len(types)} types | mean {edit_total / total:.2f} edits | "
        f"{len(correct) / total:.1%} correct | broken {len(broken)}"
    )


if __name__ == "__main__":
    main()
