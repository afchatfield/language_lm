"""Render the Phase 0 base-model decision.

The decision rule is fixed here, in code, ahead of the numbers, so that the
choice is a consequence of the measurement rather than a rationalisation of a
preference formed earlier.
"""

from __future__ import annotations

from collections.abc import Sequence

from langlm.tokenizers.fertility import FertilityResult

#: Coverage threshold used for the headline trim figure. 99.9% of token
#: occurrences keeps the tail that matters without paying for ids seen once.
HEADLINE_THRESHOLD = 0.999

#: Bits per weight of a realistic 4-bit quantisation (llama.cpp Q4_K_M and MLX
#: 4-bit both land near here once scales and zero-points are counted).
QUANT_BITS_PER_WEIGHT = 4.8

DECISION_RULE = """\
1. **Primary: German fertility.** Tokens per German word, averaged over the news and
   Wikipedia samples. Every extra token costs context, training step time, and
   inference latency, and an M1 has none of those to spare.
2. **Tiebreak: vocabulary-trim headroom.** Embedding parameters removable at
   99.9% token coverage, as a fraction of the whole model. This is the Phase 4
   compression budget, and it is decided by the tokenizer before any training happens.
3. **Hard constraints.** At most ~2B parameters, weights openly accessible, and
   convertible by both `mlx-lm` and `llama.cpp` for the Phase 6 export.
"""


def render_decision(
    results: Sequence[FertilityResult],
    skipped: dict[str, str],
    bytes_per_param: int = 2,
) -> tuple[str, FertilityResult]:
    """Pick the base model and write up why. Returns the Markdown and the winner."""
    ranked = sorted(results, key=lambda r: r.fertility_for("de"))
    winner = ranked[0]

    sections = [
        "# Base model decision",
        "",
        "## The rule, stated before the numbers",
        "",
        DECISION_RULE,
        "## Ranking",
        "",
        _ranking_table(ranked),
        "",
        f"## Decision: `{winner.spec.name}`",
        "",
        _rationale(winner, ranked),
        "",
        "## Size budget",
        "",
        _size_budget(winner, bytes_per_param),
        "",
    ]
    if skipped:
        sections += [
            "## Not evaluated",
            "",
            "\n".join(f"* **{name}** -- {reason}" for name, reason in skipped.items()),
            "",
            "The decision above stands on the candidates that could be measured. Re-running "
            "`make fertility` after resolving the access issue will regenerate this file.",
            "",
        ]
    return "\n".join(sections), winner


def _headline(result: FertilityResult):
    """The trim projection at the headline coverage threshold."""
    return min(result.projections, key=lambda p: abs(p.threshold - HEADLINE_THRESHOLD))


def _ranking_table(ranked: Sequence[FertilityResult]) -> str:
    rows = [
        "| Rank | Model | Params | German fertility | Embed share | "
        f"Trim @{HEADLINE_THRESHOLD} | Trimmed params |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for position, result in enumerate(ranked, start=1):
        spec = result.spec
        trim = _headline(result)
        rows.append(
            f"| {position} | `{spec.name}` | {spec.params / 1e9:.2f}B | "
            f"{result.fertility_for('de'):.3f} | {spec.embedding_share:.1%} | "
            f"{trim.model_fraction:.1%} ({trim.removed_mb:.0f} MB) | "
            f"{(spec.params - trim.removed_params) / 1e9:.2f}B |"
        )
    return "\n".join(rows)


def _rationale(winner: FertilityResult, ranked: Sequence[FertilityResult]) -> str:
    spec = winner.spec
    trim = _headline(winner)
    german = winner.fertility_for("de")
    worst = ranked[-1]
    gap = worst.fertility_for("de") / german - 1

    lines = [
        f"`{spec.name}` wins on the primary metric and the tiebreak at once, which is why "
        f"this is a short section.",
        "",
        f"* **Fertility {german:.3f} tokens/word** on German, the best of the "
        f"{len(ranked)} candidates measured. The worst, `{worst.spec.name}`, needs "
        f"{gap:.0%} more tokens for the same German text -- on a fixed context budget "
        f"that is {gap:.0%} less material per training example and per correction request.",
    ]

    if not spec.tie_word_embeddings:
        lines.append(
            f"* **Untied embeddings.** The embedding table and the LM head are separate "
            f"matrices, so {spec.embedding_params / 1e6:.0f}M parameters "
            f"({spec.embedding_share:.1%} of the model) sit in vocabulary rows and every "
            f"row removed in Phase 4 pays twice. Every tied-embedding candidate saves half "
            f"as much per row."
        )
    lines += [
        f"* **{trim.model_fraction:.1%} of the model is trimmable** at "
        f"{trim.threshold:.4g} coverage: {trim.removed_vocab:,} of {spec.vocab_size:,} "
        f"vocabulary rows are never touched by German or English text, worth "
        f"{trim.removed_params / 1e6:.0f}M parameters ({trim.removed_mb:.0f} MB at bf16). "
        f"That is the Phase 4 compression budget, available before any training.",
        f"* **{spec.params / 1e9:.2f}B parameters**, inside the size constraint, in standard "
        f"Hugging Face format.",
        "",
        spec.note,
    ]
    return "\n".join(lines)


def _size_budget(winner: FertilityResult, bytes_per_param: int) -> str:
    spec = winner.spec
    trim = _headline(winner)
    trimmed_params = spec.params - trim.removed_params
    base_mb = spec.params * bytes_per_param / 1e6
    trimmed_mb = trimmed_params * bytes_per_param / 1e6
    quantised_mb = trimmed_params * QUANT_BITS_PER_WEIGHT / 8 / 1e6

    return (
        "Where the roadmap's ~600MB target stands after Phase 0, using only measurements "
        "made here:\n\n"
        "| Stage | Params | Size |\n"
        "|---|---:|---:|\n"
        f"| Base, bf16 | {spec.params / 1e9:.2f}B | {base_mb:.0f} MB |\n"
        f"| Vocabulary-trimmed, bf16 | {trimmed_params / 1e9:.2f}B | {trimmed_mb:.0f} MB |\n"
        f"| Vocabulary-trimmed, 4-bit | {trimmed_params / 1e9:.2f}B | "
        f"~{quantised_mb:.0f} MB |\n\n"
        f"The 4-bit row assumes ~{QUANT_BITS_PER_WEIGHT} bits per weight, which is where "
        "llama.cpp Q4_K_M and MLX 4-bit actually land once scales and zero-points are "
        f"counted. Vocabulary trimming plus 4-bit quantisation therefore gets to roughly "
        f"{quantised_mb:.0f} MB on its own"
        + (
            ", short of the 600MB target; the Phase 4 depth-prune has to close the "
            "remainder, or the target moves."
            if quantised_mb > 600
            else ", inside the 600MB target before depth pruning is even attempted."
        )
    )
