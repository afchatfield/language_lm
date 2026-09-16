#!/usr/bin/env python
"""Check that each corruptor produces the error it says it produces.

A corruptor claims an ERRANT tag for the damage it does. Nothing forces that
claim to be true, and a wrong one is invisible: the corpus builds, the
invariants hold, the sentence round-trips, and the training data quietly teaches
the model to call an error something the evaluation calls something else. The
injected weights are expressed in the same tags, so a mislabelled rule also
spends the corpus on a type it is not producing.

The check is to damage real sentences and re-annotate the pair with the same
German ERRANT the evaluation uses, then compare. `das_dass` was documented and
labelled `R:SPELL` and is `R:OTHER` in 220 of the 221 das/dass edits in
Falko-MERLIN; this is the script that found it.

Disagreement is not automatically a bug. Where ERRANT differs from the corpus
itself the corpus wins, because the corpus is what the model is scored against:
`mir` -> `mich` is `R:PRON:FORM` on 154 of 156 gold edits while ERRANT calls it
`R:PRON`, because spacy does not lemmatise the two pronouns alike. Those rules
are listed as known disagreements rather than failures.

    python scripts/check_corruptors.py
    python scripts/check_corruptors.py --sentences 1000
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from langlm.config import PROCESSED_DIR, REPORTS_DIR
from langlm.corruptors import REGISTRY
from langlm.corruptors.base import apply
from langlm.data.m2 import parse_m2
from langlm.eval import errant_de

REPORT = REPORTS_DIR / "phase2" / "corruptor_types.md"

#: Rules whose claimed type disagrees with ERRANT for a known and accepted
#: reason, with that reason. Listed so a *new* disagreement stands out instead of
#: being lost among the old ones.
KNOWN = {
    "pronoun_form": "spacy gives `mir` and `mich` different lemmas; the corpus "
    "calls the pair `R:PRON:FORM` on 154 of 156 edits",
    "dative_plural_n": "spacy lemmatises long compounds inconsistently, so ERRANT "
    "reads the number change as a different word",
    "verb_final": "ERRANT has no move operation and describes a reordering as a "
    "deletion plus an insertion",
    "separable_prefix": "same: the prefix moving reads as two edits, not one",
    "umlaut": "a stripped umlaut sometimes lands on a real word, which ERRANT "
    "types by part of speech rather than as a misspelling",
    "adjective_form": "an ending swap sometimes produces another real word",
    "sharp_s": "`ss` for `ß` is a real spelling in Switzerland and ERRANT types "
    "some of them as orthography",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", type=int, default=600, help="Clean sentences to damage")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import spacy

    nlp = spacy.load("de_core_news_sm", disable=["ner"])
    sentences = [s.source for s in parse_m2(PROCESSED_DIR / "clean_de" / "de.m2")]
    random.Random(args.seed).shuffle(sentences)
    sentences = sentences[: args.sentences]

    rng = random.Random(args.seed)
    fired: Counter[str] = Counter()
    agreed: Counter[str] = Counter()
    instead: defaultdict[str, Counter[str]] = defaultdict(Counter)
    claimed: dict[str, str] = {}

    for text in sentences:
        tokens = text.split()
        doc = nlp(text)
        if len(doc) != len(tokens):
            continue
        for name, rule in REGISTRY.items():
            found = rule(tokens, doc)
            if not found:
                continue
            corruption = rng.choice(found)
            damaged = apply(tokens, [corruption])
            derived = [edit.error_type for edit in errant_de.annotate(damaged.source, text)]
            fired[name] += 1
            claimed[name] = corruption.error_type
            if corruption.error_type in derived:
                agreed[name] += 1
            else:
                instead[name]["+".join(derived) or "(nothing)"] += 1

    rows = sorted(fired, key=lambda name: (agreed[name] / fired[name], name))
    lines = [
        "# Phase 2: do the corruptors produce the errors they claim?",
        "",
        f"Each rule was run over {len(sentences):,} clean sentences; one of the corruptions "
        "it offered was applied, and the damaged sentence re-annotated with the same German "
        "ERRANT the evaluation uses. The column is how often the tag the rule claims is "
        "among the tags ERRANT derives.",
        "",
        "| Rule | Claims | Fired | ERRANT agrees |",
        "|---|---|---:|---:|",
    ]
    for name in rows:
        share = agreed[name] / fired[name]
        note = " *" if name in KNOWN and share < 0.9 else ""
        lines.append(f"| `{name}` | `{claimed[name]}` | {fired[name]:,} | {share:.1%}{note} |")

    unexplained = [name for name in rows if agreed[name] / fired[name] < 0.9 and name not in KNOWN]
    lines += [
        "",
        "\\* a known disagreement, where ERRANT and the corpus differ and the corpus wins:",
        "",
    ]
    lines += [f"* `{name}` -- {reason}" for name, reason in sorted(KNOWN.items())]
    lines += ["", "## What ERRANT says instead", ""]
    for name in rows:
        if agreed[name] / fired[name] >= 0.9:
            continue
        top = ", ".join(f"`{tag}` ({n})" for tag, n in instead[name].most_common(3))
        lines.append(f"* `{name}` claims `{claimed[name]}`, gets {top}")
    lines += [
        "",
        (
            "Every rule below 90% is a known disagreement."
            if not unexplained
            else "**Unexplained disagreements: "
            + ", ".join(f"`{name}`" for name in unexplained)
            + ".** A rule that claims a tag ERRANT does not derive, for a reason not listed "
            "above, is mislabelling its damage."
        ),
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")
    for name in rows:
        print(f"  {name:22s} {claimed[name]:14s} {agreed[name] / fired[name]:6.1%}")
    if unexplained:
        raise SystemExit(f"Unexplained type disagreements: {', '.join(unexplained)}")


if __name__ == "__main__":
    main()
