#!/usr/bin/env python
"""Build the Phase 2 training set.

Takes the frozen clean corpus, damages the configured share of it with the
configured error mix, explains every edit, and writes the result as JSON lines.

Each line is one training example:

    {"source": "...",            # what the learner wrote
     "target": "...",            # what they should have written
     "edits": [{"start": 3, "end": 4, "error_type": "R:ORTH",
                "correction": "Hund", "rule": "noun_case",
                "explanation": "German capitalises every noun ..."}]}

A correct example has an empty edit list, which is the point of it: 22% of them
say that the right answer is to change nothing.

    python scripts/build_training_set.py
    python scripts/build_training_set.py --size 500     # trial

The output is not frozen in the split manifest. It is derived data that a
config change is supposed to invalidate, not evaluation data whose stability
matters -- and the manifest exists to make the second kind hard to change by
accident.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from langlm.config import PROCESSED_DIR, REPORTS_DIR, load_config
from langlm.data import clean_de, injection
from langlm.data.splits import load_split
from langlm.explanations import TemplateError, explain

OUTPUT_DIR = PROCESSED_DIR / "training"
REPORT_DIR = REPORTS_DIR / "phase2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=30000, help="Examples to build")
    parser.add_argument("--samples", type=int, default=100, help="Examples to write out")
    args = parser.parse_args()

    config = load_config("phase2")
    settings = config["injection"]
    seed = config["clean_corpus"]["seed"]

    clean = load_split(clean_de.CORPUS, "all")[: args.size]
    print(f"Parsing {len(clean):,} sentences ...")
    docs = list(parse([s.source for s in clean]))

    stats = injection.InjectionStats()
    examples = list(
        injection.generate(
            [s.source_tokens for s in clean],
            weights=settings["weights"],
            seed=seed,
            errors=tuple(settings["errors_per_sentence"]),
            stats=stats,
            docs=docs,
        )
    )

    records, types, rules, unexplained = [], Counter(), Counter(), 0
    for example, source in zip(examples, clean, strict=True):
        record = {"source": example.source, "target": source.source, "edits": []}
        for edit in example.edits:
            if edit.is_noop:
                continue
            # The corruptor's own name, put here by `apply`.
            corruption_rule = edit.comment
            types[edit.error_type] += 1
            rules[corruption_rule] += 1
            try:
                text = explain(
                    corruption_rule,
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
                    "rule": corruption_rule,
                    "explanation": text,
                }
            )
        records.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "de.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {path} ({len(records):,} examples)")

    check(records, stats, types, rules, unexplained, args.samples, seed)


def parse(texts: list[str]):
    """Parse every sentence, so the word-order rules have something to work with."""
    import spacy

    nlp = spacy.load("de_core_news_sm", disable=["ner"])
    return nlp.pipe(texts, batch_size=64)


def check(records, stats, types, rules, unexplained, samples: int, seed: int) -> None:
    """Verify the invariants and write the sample the exit criterion asks for."""
    damaged = [r for r in records if r["edits"]]
    correct = [r for r in records if not r["edits"]]

    broken = [r for r in damaged if r["source"] == r["target"]]
    missing = [r for r in damaged for e in r["edits"] if not e["explanation"]]

    lines = [
        "# Phase 2: the training set",
        "",
        f"{len(records):,} examples: {len(damaged):,} carrying errors and "
        f"{len(correct):,} already correct ({len(correct) / len(records):.1%}).",
        "",
        "## Invariants",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Damaged examples whose source equals their target | {len(broken)} |",
        f"| Edits with no explanation | {len(missing)} |",
        f"| Edits whose corruptor could not be identified | {unexplained} |",
        f"| Sentences no rule could damage (kept as correct) | {stats.barren:,} |",
        "",
        "The first two must be zero. A damaged example identical to its target "
        "teaches the model to change nothing when something is wrong; an edit with no "
        "explanation is the half of this project that is not a grammar checker.",
        "",
        "## Injected mix",
        "",
        "| Error type | Count | Share |",
        "|---|---:|---:|",
    ]
    total = sum(types.values())
    for error_type, count in types.most_common():
        lines.append(f"| `{error_type}` | {count:,} | {count / total:.1%} |")

    lines += [
        "",
        "## 100 samples",
        "",
        "The Phase 2 exit criterion: read these, and if more than five have a wrong "
        "correction or a wrong explanation, fix the pipeline before training on it.",
        "",
    ]
    for index, record in enumerate(random.Random(seed).sample(damaged, min(samples, len(damaged)))):
        lines += [f"**{index + 1}.** `{record['source']}`", f"→ `{record['target']}`"]
        for edit in record["edits"]:
            lines.append(f"  - *{edit['error_type']}* ({edit['rule']}): {edit['explanation']}")
        lines.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "training_set.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(
        f"  correct share {len(correct) / len(records):.1%}, broken {len(broken)}, "
        f"unexplained {unexplained}"
    )


if __name__ == "__main__":
    main()
