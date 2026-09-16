#!/usr/bin/env python
"""Work out how often to inject each error type.

Two inputs, blended.

The *natural* rate is how often each type occurs in Falko-MERLIN: training data
shaped like this produces a model that has seen German learner errors in the
proportions learners make them.

The *boost* comes from Phase 1. Matching the histogram alone spends most of the
data on what a prompted model can already do -- few-shot recall is 0.49 on
spelling and 0.42 on orthography, but 0.04 on prepositions and 0.05 on missing
determiners. Types the baseline is blind to are worth more per example, so they
are weighted up, capped at twice their natural rate so the corpus still looks
like learner German rather than like a list of the model's weaknesses.

    python scripts/derive_injection_weights.py

Writes `reports/phase2/injection_weights.md` and the YAML block to paste into
`configs/phase2.yaml`.
"""

from __future__ import annotations

from collections import Counter

from langlm.config import REPORTS_DIR, load_config
from langlm.corruptors import REGISTRY, propose
from langlm.data.splits import load_split
from langlm.eval import errant_de, span_scorer

#: Recall at or below this counts as blind, and earns the boost.
BLIND = 0.10

#: The cap. Two is a judgement: enough to matter, not enough to turn the corpus
#: into something that no longer resembles the errors learners actually make.
BOOST = 2.0

REPORT_DIR = REPORTS_DIR / "phase2"


def injectable_types() -> set[str]:
    """Error types the German corruptors can actually produce.

    Derived by running every rule over sentences built to offer them something,
    rather than listed by hand -- a list would go stale the first time a rule was
    added. The probes are the part that can still go stale, so a rule that none
    of them exercises is an error rather than a silent omission: a type missing
    from here gets no weight, and a corruptor with no weight is one that was
    written and then never fired.
    """
    import spacy

    from langlm.corruptors import REGISTRY

    # Parsed, because most rules produce nothing without one and a probe that
    # skipped them would silently leave word order out of the weights -- the
    # very type Phase 1 found the baseline worst at.
    nlp = spacy.load("de_core_news_sm", disable=["ner"])
    probes = [
        "Der Lehrer hat gesagt , dass die Schüler mit großer Mühe die Aufgaben lösen .",
        "Der Zug fährt um acht Uhr ab .",
        "Ich sehe den großen Hund in dem Garten .",
        "Gestern ging er mit den Kindern und Eltern ins Kino .",
        "Er interessiert sich für den Preis des Hauses und wartet auf mich .",
        "Sie ist gestern nach Hause gefahren , weil sie das Buch vergessen hat .",
        "Wir kaufen Brot und Butter , und sie bezahlt mit einer Karte .",
    ]
    types: set[str] = set()
    fired: set[str] = set()
    for probe in probes:
        tokens = probe.split()
        for corruption in propose(tokens, doc=nlp(probe)):
            types.add(corruption.error_type)
            fired.add(corruption.rule)
    # `drop_comma` names its corruptions after the comma rule that was broken.
    silent = set(REGISTRY) - {rule.split("_subordinate")[0] for rule in fired} - fired
    silent -= {"drop_comma"} if any(f.startswith("drop_comma") for f in fired) else set()
    if silent:
        raise SystemExit(
            f"No probe sentence exercises {', '.join(sorted(silent))}. Add one to "
            f"`injectable_types`, or these rules will be left out of the weights and "
            f"so out of the corpus."
        )
    return types


def natural_rates() -> dict[str, float]:
    """How often each error type occurs in Falko-MERLIN train and dev."""
    counts: Counter[str] = Counter()
    for split in ("train", "dev"):
        for sentence in load_split("falko_merlin", split):
            for edit in sentence.edits_for():
                if not edit.is_noop:
                    counts[edit.error_type] += 1
    total = sum(counts.values())
    return {error_type: count / total for error_type, count in counts.items()}


def baseline_recall() -> tuple[dict[str, float], str]:
    """Per-type recall of the best system on disk, recomputed from its output.

    The boost exists to spend the corpus on what the model cannot do, so it has
    to be read off the model that is actually being improved. That was the
    Phase 1 few-shot baseline when there was nothing else; once an adapter has
    been evaluated, its own dev output is the honest source, and using the
    prompted baseline instead would keep boosting types the fine-tuned model has
    since learned. Whichever is used is named in the report.

    Returns:
        Per-type recall, and the name of the system it came from.
    """
    from langlm.config import INTERIM_DIR

    dev = load_split("falko_merlin", "dev")
    candidates = [
        ("fine-tuned", INTERIM_DIR / "phase3" / f"fine-tuned.dev.{len(dev)}.txt"),
        ("few-shot", INTERIM_DIR / "phase1" / f"few-shot.dev.{len(dev)}.txt"),
    ]
    found = next(((name, path) for name, path in candidates if path.exists()), None)
    if found is None:
        raise FileNotFoundError(
            f"No system output to read the boost from. Looked for "
            f"{', '.join(str(path) for _, path in candidates)}. Run `make baselines` "
            f"or `make evaluate` first: the boost is derived from what the model "
            f"could not do."
        )
    name, path = found
    hypotheses = path.read_text(encoding="utf-8").splitlines()
    annotated = errant_de.annotate_corpus([s.source for s in dev], hypotheses)
    scored = span_scorer.score(dev, annotated, beta=0.5, mode="correction")
    return {error_type: counts.recall for error_type, counts in scored.by_type.items()}, name


def main() -> None:
    config = load_config("phase2")
    types = injectable_types()
    natural = natural_rates()
    recall, measured_on = baseline_recall()

    rows = []
    for error_type in sorted(types):
        rate = natural.get(error_type, 0.0)
        got = recall.get(error_type)
        boost = BOOST if got is not None and got <= BLIND else 1.0
        rows.append((error_type, rate, got, boost, rate * boost))

    total = sum(row[4] for row in rows)
    weights = {row[0]: row[4] / total for row in rows}

    lines = [
        "# Phase 2: the injected error distribution",
        "",
        f"Natural rates are Falko-MERLIN train + dev. Recall is the **{measured_on}** "
        f"system, recomputed from its saved dev output. A type it recovers at "
        f"{BLIND:.2f} or less is weighted up {BOOST:g}x.",
        "",
        f"| Error type | Natural | {measured_on} recall | Boost | Injected |",
        "|---|---:|---:|---:|---:|",
    ]
    for error_type, rate, got, boost, _ in sorted(rows, key=lambda r: -weights[r[0]]):
        shown = f"{got:.2f}" if got is not None else "--"
        mark = "**x2**" if boost > 1 else "1x"
        lines.append(
            f"| `{error_type}` | {rate:.1%} | {shown} | {mark} | **{weights[error_type]:.1%}** |"
        )

    boosted = [row[0] for row in rows if row[3] > 1]
    lines += [
        "",
        (
            f"Boosted: {', '.join(f'`{t}`' for t in boosted)}. These are the types the "
            f"{measured_on} system barely touches, so an example of one buys more than an "
            f"example of a type it already handles."
            if boosted
            else f"Boosted: none. Nothing the corruptors produce falls under the "
            f"{BLIND:.2f} recall the boost is for -- the {measured_on} system's weakest "
            f"injectable type is at {min(r[2] for r in rows if r[2] is not None):.2f}. The "
            f"boost was calibrated against a prompted baseline that scored 0.04 on "
            f"prepositions; a model that is no longer blind anywhere should be trained on "
            f"the distribution learners actually produce, which is what these weights are."
        ),
        "",
        "Rates are renormalised over the types the corruptors can produce, so they do "
        "not match the corpus-wide histogram directly: types no rule injects are absent "
        "here and their share is redistributed.",
        "",
        "## The block for `configs/phase2.yaml`",
        "",
        "```yaml",
        "injection:",
        f"  errors_per_sentence: {config['injection']['errors_per_sentence']}"
        if "injection" in config
        else "  errors_per_sentence: [1, 2]",
        "  weights:",
    ]
    for error_type, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {error_type}: {weight:.4f}")
    lines.append("```")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "injection_weights.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}\n")
    print(f"{len(REGISTRY)} rules produce {len(types)} error types")
    for error_type, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {error_type:<12} {weight:.1%}")


if __name__ == "__main__":
    main()
