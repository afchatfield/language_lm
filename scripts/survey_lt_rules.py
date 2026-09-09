#!/usr/bin/env python
"""Run LanguageTool over injected errors, and see what it calls them.

Phase 2 explains its corrections, and hand-writing an explanation for every
possible error is not a plan. LanguageTool already has a message for most of
them, keyed by a rule id, so the templates in `rules/de.yaml` can be written
against the ids that actually fire rather than against a guess at which ones
will. This script finds out which those are.

It corrupts clean sentences with the configured weights, checks the damaged
text, and attributes each match to the injected error whose span it overlaps.
Two things come out: how much of our own injected damage LanguageTool detects
at all, and the frequency-ordered list of rule ids to write templates for.

    python scripts/survey_lt_rules.py --limit 3000

Needs the LanguageTool server: `make lt-up && make lt-check`.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from langlm.config import REPORTS_DIR, load_config
from langlm.data import clean_de, injection
from langlm.data.splits import load_split
from langlm.lt.cache import CachedLanguageToolClient

REPORT_DIR = REPORTS_DIR / "phase2"

#: How far either side of an injected span a match may sit and still be counted
#: as being about it. A dropped comma is flagged on the words around the gap,
#: not on the gap, so a zero-tolerance match would score that as undetected.
TOLERANCE_TOKENS = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3000, help="Sentences to survey")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent LT requests")
    args = parser.parse_args()

    config = load_config("phase2")
    weights = config["injection"]["weights"]
    errors = tuple(config["injection"]["errors_per_sentence"])

    clean = load_split(clean_de.CORPUS, "all")[: args.limit]
    stats = injection.InjectionStats()
    examples = list(
        injection.generate(
            [s.source_tokens for s in clean],
            weights=weights,
            seed=config["clean_corpus"]["seed"],
            errors=errors,
            stats=stats,
        )
    )
    damaged = [e for e in examples if not e.is_correct()]
    print(f"{len(damaged):,} damaged sentences of {len(examples):,} ({stats.negative:,} negative)")

    lt = CachedLanguageToolClient()
    # Phase 2's own list, not Phase 1's: see the note in configs/phase2.yaml for
    # why the published baseline's config is left alone.
    params = {"disabledRules": ",".join(config["languagetool"]["disabled_rules"])}

    print(f"Checking with LanguageTool ({args.workers} workers) ...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda text: lt.check(text, **params), [e.source for e in damaged]))
    print(f"cache: {lt.hits:,} hits, {lt.misses:,} misses")

    detected: Counter[str] = Counter()
    injected: Counter[str] = Counter()
    rules: Counter[str] = Counter()
    rules_by_type: defaultdict[str, Counter[str]] = defaultdict(Counter)
    messages: dict[str, str] = {}

    for example, matches in zip(damaged, results, strict=True):
        spans = injection.token_spans(example.source_tokens)
        windows = []
        for edit in example.edits:
            injected[edit.error_type] += 1
            low = max(edit.start - TOLERANCE_TOKENS, 0)
            high = min(edit.end + TOLERANCE_TOKENS, len(spans))
            if low >= len(spans):
                windows.append(None)
                continue
            windows.append((spans[low][0], spans[high - 1][1] if high else spans[low][1]))

        # Each match belongs to one injected error: the one it overlaps most.
        # A sentence carries one or two errors and the windows are padded, so
        # they can touch -- and crediting a spelling match to the preposition
        # next to it would make the coverage table quietly wrong.
        hits: dict[int, bool] = {}
        for match in matches:
            best, best_overlap = None, 0
            for index, window in enumerate(windows):
                if window is None:
                    continue
                overlap = min(match.offset + match.length, window[1]) - max(match.offset, window[0])
                if overlap > best_overlap:
                    best, best_overlap = index, overlap
            if best is None:
                continue
            error_type = example.edits[best].error_type
            rules[match.rule_id] += 1
            rules_by_type[error_type][match.rule_id] += 1
            messages.setdefault(match.rule_id, match.message)
            hits[best] = True

        for index, edit in enumerate(example.edits):
            detected[edit.error_type] += hits.get(index, False)

    write_report(injected, detected, rules, rules_by_type, messages, len(damaged))


def write_report(injected, detected, rules, rules_by_type, messages, sentences: int) -> None:
    total_injected = sum(injected.values())
    total_detected = sum(detected.values())
    lines = [
        "# Phase 2: what LanguageTool calls our injected errors",
        "",
        f"{sentences:,} damaged sentences carrying {total_injected:,} injected errors. "
        f"A match counts as being about an injected error if it overlaps that error's "
        f"span, give or take {TOLERANCE_TOKENS} token -- a dropped comma is flagged on "
        f"the words either side of the gap, not on the gap.",
        "",
        "## Detection by injected type",
        "",
        f"LanguageTool finds **{total_detected:,} of {total_injected:,}** "
        f"({total_detected / total_injected:.1%}) of the errors we planted.",
        "",
        "| Injected type | Planted | Detected | Rate | Most common rule |",
        "|---|---:|---:|---:|---|",
    ]
    for error_type, planted in injected.most_common():
        found = detected[error_type]
        top = rules_by_type[error_type].most_common(1)
        top_rule = f"`{top[0][0]}`" if top else "--"
        lines.append(
            f"| `{error_type}` | {planted:,} | {found:,} | {found / planted:.1%} | {top_rule} |"
        )

    lines += [
        "",
        "A low rate is not a failure of the corruptor. It means LanguageTool cannot see "
        "that error, which is exactly where a trained model has something to add -- and "
        "it also means no template can be derived from a rule id there, so those "
        "explanations have to be written from the error type instead.",
        "",
        "## Rule ids to write templates for",
        "",
        f"{len(rules):,} distinct rule ids fired. The top of this list is where the "
        "explanation coverage is: hand-writing the first hundred covers most of the "
        "volume, and the tail can be generated.",
        "",
        "| Rule id | Fired | Message |",
        "|---|---:|---|",
    ]
    for rule_id, count in rules.most_common(100):
        message = messages.get(rule_id, "").replace("|", "\\|")[:90]
        lines.append(f"| `{rule_id}` | {count:,} | {message} |")

    covered = sum(count for _, count in rules.most_common(100))
    lines += [
        "",
        f"Those hundred account for {covered:,} of {sum(rules.values()):,} matches "
        f"({covered / sum(rules.values()):.1%}).",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "lt_rule_survey.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
