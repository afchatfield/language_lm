"""Render the Phase 0 tokenizer analysis as Markdown and CSV."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from langlm.tokenizers.fertility import FertilityResult


def write_csv(results: Sequence[FertilityResult], path: Path) -> None:
    """One row per (model, corpus), for anyone who wants to re-plot this."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "model",
                "repo_id",
                "corpus",
                "lang",
                "sentences",
                "words",
                "tokens",
                "fertility",
                "chars_per_token",
                "bytes_per_token",
                "split_word_rate",
                "unique_tokens",
                "vocab_size",
            ]
        )
        for result in results:
            for stats in result.corpora.values():
                writer.writerow(
                    [
                        result.spec.name,
                        result.spec.repo_id,
                        stats.corpus,
                        stats.lang,
                        stats.sentences,
                        stats.words,
                        stats.tokens,
                        f"{stats.fertility:.4f}",
                        f"{stats.chars_per_token:.4f}",
                        f"{stats.bytes_per_token:.4f}",
                        f"{stats.split_word_rate:.4f}",
                        stats.unique_tokens,
                        result.spec.vocab_size,
                    ]
                )


def render_markdown(
    results: Sequence[FertilityResult],
    corpus_order: Sequence[str],
    trim_languages: Sequence[str],
    skipped: dict[str, str],
    max_sentences: int,
) -> str:
    """Build the full `tokenizer_fertility.md` report."""
    sections = [
        "# Tokenizer fertility and vocabulary-trim headroom",
        "",
        _preamble(corpus_order, trim_languages, max_sentences),
        "",
        "## Model geometry",
        "",
        _geometry_table(results),
        "",
        "## Fertility (tokens per word -- lower is better)",
        "",
        _fertility_table(results, corpus_order),
        "",
        "## Bytes per token (cross-check -- higher is better)",
        "",
        "Fertility divides by whitespace words, which flatters languages that write "
        "compounds as separate words. Bytes per token does not depend on how a word is "
        "defined, so it is the sanity check on the table above.",
        "",
        _bytes_table(results, corpus_order),
        "",
        "## Vocabulary-trim projection",
        "",
        _trim_preamble(trim_languages),
        "",
        _trim_table(results),
        "",
    ]
    if skipped:
        sections += ["## Skipped candidates", "", _skipped_list(skipped), ""]
    return "\n".join(sections)


def _preamble(corpora: Sequence[str], trim_languages: Sequence[str], max_sentences: int) -> str:
    return (
        f"Measured over the Leipzig corpora `{'`, `'.join(corpora)}`, up to "
        f"{max_sentences:,} sentences each. Only tokenizers and configs were downloaded; "
        f"no model weights were touched.\n\n"
        f"Two quantities decide the base model, and they pull against each other. "
        f"**Fertility** is how many tokens a tokenizer spends per word: a tokenizer that "
        f"shatters German compounds makes every sequence longer, costing context, training "
        f"time, and inference latency. **Trim headroom** is the share of the vocabulary that "
        f"German and English text never touch; those embedding rows can be sliced out in "
        f"Phase 4 for free. A large multilingual vocabulary tends to score well on the "
        f"second and badly on the first.\n\n"
        f"Retained languages for the trim projection: "
        f"{', '.join(f'`{lang}`' for lang in trim_languages)}. English stays because the "
        f"prompt and instruction format are English."
    )


def _geometry_table(results: Sequence[FertilityResult]) -> str:
    rows = [
        "| Model | Params | Vocab | Hidden | Layers | Tied? | Embed+head params | Share of model |",
        "|---|---:|---:|---:|---:|:--:|---:|---:|",
    ]
    for result in results:
        spec = result.spec
        rows.append(
            f"| `{spec.name}` | {spec.params / 1e9:.2f}B | {spec.vocab_size:,} | "
            f"{spec.hidden_size:,} | {spec.num_layers} | "
            f"{'yes' if spec.tie_word_embeddings else '**no**'} | "
            f"{spec.embedding_params / 1e6:.0f}M | **{spec.embedding_share:.1%}** |"
        )
    return "\n".join(rows)


def _fertility_table(results: Sequence[FertilityResult], corpus_order: Sequence[str]) -> str:
    header = "| Model | " + " | ".join(corpus_order) + " | **de mean** |"
    divider = "|---|" + "---:|" * (len(corpus_order) + 1)
    rows = [header, divider]
    best = min((r.fertility_for("de") for r in results), default=0.0)
    for result in results:
        cells = [
            f"{result.corpora[c].fertility:.3f}" if c in result.corpora else "-"
            for c in corpus_order
        ]
        german = result.fertility_for("de")
        mark = f"**{german:.3f}**" if german == best else f"{german:.3f}"
        rows.append(f"| `{result.spec.name}` | " + " | ".join(cells) + f" | {mark} |")
    return "\n".join(rows)


def _bytes_table(results: Sequence[FertilityResult], corpus_order: Sequence[str]) -> str:
    header = "| Model | " + " | ".join(corpus_order) + " |"
    divider = "|---|" + "---:|" * len(corpus_order)
    rows = [header, divider]
    for result in results:
        cells = [
            f"{result.corpora[c].bytes_per_token:.2f}" if c in result.corpora else "-"
            for c in corpus_order
        ]
        rows.append(f"| `{result.spec.name}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _trim_preamble(trim_languages: Sequence[str]) -> str:
    return (
        "`kept` is the smallest vocabulary covering that share of all token *occurrences* in "
        f"the {'+'.join(trim_languages)} corpora, plus every special and single-character "
        "token, which must survive so the tokenizer can still encode arbitrary text. "
        'Reporting a curve rather than a single "every id ever seen" count keeps the '
        "estimate honest: a long tail of ids seen once each inflates the vocabulary while "
        "carrying almost no probability mass.\n\n"
        "`saved` is the resulting reduction in the embedding table plus, where untied, the "
        "LM head -- at bf16, and as a fraction of the whole model."
    )


def _trim_table(results: Sequence[FertilityResult]) -> str:
    rows = [
        "| Model | Coverage | Vocab kept | Vocab removed | Params saved | "
        "Saved (bf16) | Saved % of model |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for index, projection in enumerate(result.projections):
            name = f"`{result.spec.name}`" if index == 0 else ""
            rows.append(
                f"| {name} | {projection.threshold:.4g} | {projection.kept_vocab:,} "
                f"({projection.kept_fraction:.0%}) | {projection.removed_vocab:,} | "
                f"{projection.removed_params / 1e6:.0f}M | {projection.removed_mb:.0f} MB | "
                f"{projection.model_fraction:.1%} |"
            )
    return "\n".join(rows)


def _skipped_list(skipped: dict[str, str]) -> str:
    return "\n".join(f"* **{name}** -- {reason}" for name, reason in skipped.items())
