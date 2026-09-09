#!/usr/bin/env python
"""Tokenizer fertility analysis and vocabulary-trim projection (Phase 0).

Measures how many tokens each candidate tokenizer spends per German word, and
how much of its vocabulary German and English text never touch. Writes the
tables and the base-model decision into `reports/phase0/`.

Only tokenizers and configs are downloaded -- a few MB per candidate, not the
weights.

    python scripts/run_fertility.py
"""

from __future__ import annotations

import argparse
from collections import Counter

from langlm.config import PHASE0_REPORT_DIR, load_config
from langlm.data.leipzig import read_sentences
from langlm.tokenizers.decision import render_decision
from langlm.tokenizers.fertility import FertilityResult, analyse_corpus, project_trim, protected_ids
from langlm.tokenizers.hub import ModelUnavailableError, load_candidate
from langlm.tokenizers.report import render_markdown, write_csv


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=config["analysis"]["max_sentences"],
        help="Cap sentences per corpus (lower is faster; the default is the full sample)",
    )
    args = parser.parse_args()

    analysis = config["analysis"]
    corpora = config["corpora"]
    thresholds = analysis["coverage_thresholds"]
    trim_languages = set(analysis["trim_languages"])

    # Read each corpus once into memory. Four 100K-sentence samples is well
    # under 100MB, and re-reading per candidate would dominate the runtime.
    print("Loading corpora...")
    texts: dict[str, list[str]] = {}
    for entry in corpora:
        sentences = list(read_sentences(entry["leipzig"]))[: args.max_sentences]
        texts[entry["key"]] = sentences
        print(f"  {entry['key']:<14} {len(sentences):>7,} sentences ({entry['leipzig']})")

    results: list[FertilityResult] = []
    skipped: dict[str, str] = {}

    for candidate in config["candidates"]:
        name = candidate["name"]
        print(f"\n{name}")
        try:
            loaded = load_candidate(name, candidate["repo_id"], candidate.get("note", ""))
        except ModelUnavailableError as exc:
            print(f"  SKIPPED: {exc}")
            skipped[name] = str(exc)
            continue

        spec, tokenizer = loaded.spec, loaded.tokenizer
        print(
            f"  {spec.params / 1e9:.2f}B params, vocab {spec.vocab_size:,}, "
            f"hidden {spec.hidden_size}, "
            f"{'tied' if spec.tie_word_embeddings else 'UNTIED'} embeddings "
            f"({spec.embedding_share:.1%} of the model)"
        )

        result = FertilityResult(spec=spec)
        for entry in corpora:
            stats = analyse_corpus(
                tokenizer=tokenizer,
                sentences=texts[entry["key"]],
                corpus=entry["key"],
                lang=entry["lang"],
            )
            result.corpora[entry["key"]] = stats
            print(
                f"    {stats.corpus:<14} fertility {stats.fertility:.3f}  "
                f"bytes/token {stats.bytes_per_token:.2f}  "
                f"split words {stats.split_word_rate:.1%}  "
                f"unique ids {stats.unique_tokens:,}"
            )

        # Merge the retained languages' token usage: a Phase 4 trim keeps the
        # union, not any single corpus.
        merged: Counter[int] = Counter()
        for entry in corpora:
            if entry["lang"] in trim_languages:
                merged.update(result.corpora[entry["key"]].token_counts)

        keep = protected_ids(tokenizer)
        result.projections = [
            project_trim(spec, merged, t, keep, analysis["bytes_per_param"]) for t in thresholds
        ]
        for projection in result.projections:
            print(
                f"    trim @{projection.threshold:<7g} keep {projection.kept_vocab:>7,} "
                f"-> save {projection.removed_mb:>5.0f} MB "
                f"({projection.model_fraction:.1%} of the model)"
            )
        results.append(result)

    if not results:
        raise SystemExit("No candidate could be analysed. Check network access and gating.")

    PHASE0_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_order = [entry["key"] for entry in corpora]

    markdown = render_markdown(
        results=results,
        corpus_order=corpus_order,
        trim_languages=analysis["trim_languages"],
        skipped=skipped,
        max_sentences=args.max_sentences,
    )
    (PHASE0_REPORT_DIR / "tokenizer_fertility.md").write_text(markdown, encoding="utf-8")
    write_csv(results, PHASE0_REPORT_DIR / "tokenizer_fertility.csv")

    decision, winner = render_decision(results, skipped, analysis["bytes_per_param"])
    (PHASE0_REPORT_DIR / "base_model_decision.md").write_text(decision, encoding="utf-8")

    for filename in ("tokenizer_fertility.md", "tokenizer_fertility.csv", "base_model_decision.md"):
        print(
            f"\nWrote {PHASE0_REPORT_DIR / filename}"
            if filename.endswith(".md")
            else f"Wrote {PHASE0_REPORT_DIR / filename}"
        )

    print("\nGerman fertility ranking (lower is better):")
    for result in sorted(results, key=lambda r: r.fertility_for("de")):
        print(f"  {result.spec.name:<16} {result.fertility_for('de'):.3f}")
    print(f"\nBase model: {winner.spec.name} ({winner.spec.repo_id})")


if __name__ == "__main__":
    main()
