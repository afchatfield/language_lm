#!/usr/bin/env python
"""Run the Phase 1 (or Phase 5) baselines and write the results table.

Three systems, plus the identity system that prices the harness itself:

    identity        changes nothing; measures what re-tokenisation costs
    languagetool    a mature rule-based checker, applying its own suggestions
    zero-shot       EuroLLM-1.7B prompted to correct the target language
    few-shot        the same model, with worked examples from the train split

Each system is run over the development split and over the overcorrection set,
and its output is written to `data/interim/phase1{,_es}/` before anything is
scored. Generating is slow and scoring is fast, so they are separate: rerunning
the script rebuilds the report from whatever outputs already exist and only
generates the ones that are missing.

    python scripts/run_baselines.py                    # everything, German
    python scripts/run_baselines.py --language es       # Spanish
    python scripts/run_baselines.py --only languagetool
    python scripts/run_baselines.py --report-only
    python scripts/run_baselines.py --limit 100        # smoke test
    python scripts/run_baselines.py --only few-shot --batch-size 16

The development split is the default and is what every tuning decision was
made against. `--split test` unseals the held-out split, which Phase 0 froze on
day one; it is guarded the same way `eval_finetuned.py` guards it, and writes to
its own report so a final number can never overwrite the development record.

    python scripts/run_baselines.py --split test        # once, at the very end
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from langlm.baselines.identity import IdentityBaseline
from langlm.baselines.languagetool import LanguageToolBaseline
from langlm.baselines.prompted import PromptedBaseline, sample_shots
from langlm.config import INTERIM_DIR, REPORTS_DIR, load_config
from langlm.data.m2 import M2Sentence
from langlm.data.splits import HELD_OUT_SPLITS, load_split, split_scoped
from langlm.eval import m2_scorer as m2
from langlm.eval import overcorrection, span_scorer

#: The order the table reads in: cheapest and least interesting first.
BASELINE_ORDER = ("identity", "languagetool", "zero-shot", "few-shot")


@dataclass(frozen=True)
class LanguageSettings:
    """Everything that differs between a German and a Spanish baseline run.

    Centralised here rather than scattered as `if language == "de"` checks
    through every function below -- the same "one dispatch point" shape
    `explanations._annotator` and `langlm.data.quality` use elsewhere in this
    project, for the same reason: it is the one place a third language would
    need to add an entry.
    """

    config_name: str
    #: Split-manifest corpus key for the real learner corpus.
    corpus: str
    #: `overcorrection` module for this language's already-correct set.
    overcorrection_module: str
    #: Passed through to `overcorrection.measure` and `errant(...)` below.
    annotator: str
    #: Measured share of already-correct sentences in the real training
    #: corpus -- German's is Falko-MERLIN's 22%, Spanish's is COWS-L2H's
    #: measured 33.0% (`load_split("cowsl2h", "train")`), not a value either
    #: language should borrow from the other.
    correct_share: float
    #: For report prose ("published German prose", "published Spanish prose").
    prose_name: str
    report_dir: Path
    hypothesis_dir: Path


LANGUAGES = {
    "de": LanguageSettings(
        config_name="phase1",
        corpus="falko_merlin",
        overcorrection_module="langlm.data.overcorrection",
        annotator="de",
        correct_share=0.22,
        prose_name="German",
        report_dir=REPORTS_DIR / "phase1",
        hypothesis_dir=INTERIM_DIR / "phase1",
    ),
    "es": LanguageSettings(
        config_name="phase1_es",
        corpus="cowsl2h",
        overcorrection_module="langlm.data.overcorrection_es",
        annotator="es",
        correct_share=0.330,
        prose_name="Spanish",
        report_dir=REPORTS_DIR / "phase5",
        hypothesis_dir=INTERIM_DIR / "phase1_es",
    ),
}


def errant(lang: LanguageSettings):
    """The ERRANT annotator module for this language, imported lazily.

    `errant_de` and `errant_es` both expose `tokenize` and `annotate_corpus`
    in the same shape (`langlm.eval.errant_core`), so this dispatch is the
    only place that needs to know there is more than one.
    """
    import importlib

    return importlib.import_module(f"langlm.eval.errant_{lang.annotator}")


def overcorrection_module(lang: LanguageSettings):
    import importlib

    return importlib.import_module(lang.overcorrection_module)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=BASELINE_ORDER, default=list(BASELINE_ORDER))
    parser.add_argument("--report-only", action="store_true", help="Score what already exists")
    parser.add_argument("--regenerate", action="store_true", help="Ignore cached outputs")
    parser.add_argument("--limit", type=int, help="Use only the first N sentences (smoke test)")
    parser.add_argument(
        "--language",
        default="de",
        choices=sorted(LANGUAGES),
        help="German kept as the default so every existing invocation is unchanged.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Which split of the learner corpus to score. Defaults to the configured "
        "dev split. Pass `test` only for a final number that nothing will be tuned on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the model batch size. Few-shot prompts are several hundred "
        "tokens longer than zero-shot ones, so the batch that fits one may not fit "
        "the other on a 16GB machine.",
    )
    args = parser.parse_args()

    lang = LANGUAGES[args.language]
    config = load_config(lang.config_name)
    split = args.split or config["evaluation"]["split"]

    held_out = split in HELD_OUT_SPLITS
    if held_out:
        print(
            f"** {split} is the held-out split. These are final numbers: nothing should be "
            f"tuned on what comes back. **"
        )
    dev = load_split(lang.corpus, split, allow_test=held_out)
    clean = load_split(overcorrection_module(lang).CORPUS, "all")
    if args.limit:
        dev, clean = dev[: args.limit], clean[: args.limit]

    print(f"{len(dev)} learner sentences ({split}), {len(clean)} correct sentences")

    if not args.report_only:
        for name in args.only:
            generate(
                name,
                dev,
                clean,
                config,
                lang,
                regenerate=args.regenerate,
                batch_size=args.batch_size,
            )

    write_report(dev, clean, config, lang, split)


# --- generation -------------------------------------------------------------


def build(name: str, config: dict, lang: LanguageSettings, batch_size: int | None = None):
    """Construct a baseline by name."""
    if name == "identity":
        return IdentityBaseline()
    if name == "languagetool":
        return LanguageToolBaseline.from_config(config_name=lang.config_name)
    settings = config["baselines"]["model"]
    shots = []
    if name == "few-shot":
        shots = sample_shots(
            load_split(lang.corpus, "train"),
            settings["few_shot_examples"],
            settings["few_shot_seed"],
            correct_share=lang.correct_share,
        )
    overrides = {"batch_size": batch_size} if batch_size else {}
    return PromptedBaseline.from_config(
        shots=shots, config_name=lang.config_name, language=lang.annotator, **overrides
    )


def generate(
    name: str,
    dev: Sequence[M2Sentence],
    clean: Sequence[M2Sentence],
    config: dict,
    lang: LanguageSettings,
    regenerate: bool = False,
    batch_size: int | None = None,
) -> None:
    """Run one baseline over both evaluation sets, unless it has been run already."""
    targets = {"dev": dev, "overcorrection": clean}
    missing = {
        key: sentences
        for key, sentences in targets.items()
        if regenerate or not path_for(name, key, len(sentences), lang).exists()
    }
    if not missing:
        print(f"{name}: already generated")
        return

    system = build(name, config, lang, batch_size)
    try:
        for key, sentences in missing.items():
            print(f"{name}: correcting {len(sentences)} sentences ({key})")
            sources = [sentence.source for sentence in sentences]
            raw = (
                system.correct_many(sources)
                if hasattr(system, "correct_many")
                else [system.correct(source) for source in sources]
            )
            # One tokeniser for every system, so that a model writing ordinary
            # prose is not charged for where it puts the full stop.
            hypotheses = [errant(lang).tokenize(text) for text in raw]
            write_lines(path_for(name, key, len(sentences), lang), hypotheses)
    finally:
        release(system)


def release(system: object) -> None:
    """Drop a baseline's model before the next one loads its own.

    Torch's MPS allocator keeps freed blocks in its own pool, so letting the
    reference fall out of scope returns the memory to torch and not to the
    machine. Two 1.7B models resident at once is what the 16GB M1 does not
    have, and the second one dies while the first is still nominally cached.
    """
    if getattr(system, "_model", None) is None:
        return
    system._model = None
    system._tokenizer = None
    gc.collect()
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    # The same argument for CUDA: on a shared card the freed blocks sit in
    # torch's pool rather than going back to the machine, and the next
    # baseline -- or a training run on the same GPU -- asks the driver, not
    # the pool, for its own.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def path_for(name: str, key: str, count: int, lang: LanguageSettings) -> Path:
    return lang.hypothesis_dir / f"{name}.{key}.{count}.txt"


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


# --- scoring ----------------------------------------------------------------


def score_baseline(
    name: str,
    dev: Sequence[M2Sentence],
    clean: Sequence[M2Sentence],
    beta: float,
    lang: LanguageSettings,
) -> dict | None:
    """Score one baseline, or return None if it has not been generated."""
    dev_path = path_for(name, "dev", len(dev), lang)
    clean_path = path_for(name, "overcorrection", len(clean), lang)
    if not dev_path.exists() or not clean_path.exists():
        return None

    hypotheses = read_lines(dev_path)
    maxmatch = m2.score(dev, hypotheses, beta=beta)
    annotated = errant(lang).annotate_corpus([s.source for s in dev], hypotheses)
    correction = span_scorer.score(dev, annotated, beta=beta, mode="correction")
    detection = span_scorer.score(dev, annotated, beta=beta, mode="detection")
    over = overcorrection.measure(
        [s.source for s in clean], read_lines(clean_path), language=lang.annotator
    )

    return {
        "name": name,
        "m2": maxmatch,
        "correction": correction,
        "detection": detection,
        "overcorrection": over,
    }


def write_report(
    dev: Sequence[M2Sentence],
    clean: Sequence[M2Sentence],
    config: dict,
    lang: LanguageSettings,
    split: str,
) -> None:
    """Score every generated baseline and write the results table."""
    beta = config["evaluation"]["beta"]
    results = [r for r in (score_baseline(n, dev, clean, beta, lang) for n in BASELINE_ORDER) if r]
    if not results:
        print("Nothing generated yet; no report written.")
        return

    lines = _report_lines(results, dev, clean, config, lang, split)
    lang.report_dir.mkdir(parents=True, exist_ok=True)
    name = "baselines.md" if lang.annotator == "de" else "baselines_es.md"
    output = split_scoped(lang.report_dir / name, split)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")

    meta = {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": split,
        "sentences": len(dev),
        "overcorrection_sentences": len(clean),
        "baselines": {
            r["name"]: {
                "f0.5": round(r["m2"].f_score, 4),
                "precision": round(r["m2"].precision, 4),
                "recall": round(r["m2"].recall, 4),
                "overcorrection_rate": round(r["overcorrection"].rate, 4),
            }
            for r in results
        },
    }
    meta_name = "baselines.json" if lang.annotator == "de" else "baselines_es.json"
    meta_path = split_scoped(lang.report_dir / meta_name, split)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _report_lines(
    results: list[dict], dev, clean, config: dict, lang: LanguageSettings, split: str
) -> list[str]:
    beta = config["evaluation"]["beta"]
    model = config["baselines"]["model"]["repo_id"]
    gold_edits = sum(len(s.edits_for()) for s in dev)
    corpus_name = "Falko-MERLIN" if lang.annotator == "de" else "COWS-L2H"

    lines = [
        f"# Phase {'1' if lang.annotator == 'de' else '5'} baselines"
        + ("" if lang.annotator == "de" else " (Spanish)"),
        "",
        f"{corpus_name} **{split}**: {len(dev):,} sentences, {gold_edits:,} gold edits. "
        f"Overcorrection set: {len(clean):,} sentences of published {lang.prose_name} prose "
        f"that need no correction. "
        + (
            "This is the held-out split, read once so that the fine-tuned model has a bar "
            "measured on the same sentences it is scored on. Every tuning decision in this "
            "project was made against dev, whose table is in the report beside this one."
            if split in HELD_OUT_SPLITS
            else "The test split has not been read."
        ),
        "",
        "## The table",
        "",
        f"| System | P | R | F{beta:g} | Overcorrection | Spurious edits |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        score, over = result["m2"], result["overcorrection"]
        lines.append(
            f"| {result['name']} | {score.precision:.4f} | {score.recall:.4f} | "
            f"**{score.f_score:.4f}** | {over.rate:.1%} | {over.edits} |"
        )

    ceiling = m2.score(dev, [sentence.target() for sentence in dev], beta=beta)
    lines += [
        "",
        f"F{beta:g} is MaxMatch (Dahlmeier and Ng, 2012) over the corrected sentences. "
        "*Overcorrection* is the share of already-correct sentences the system changed at "
        "all; *spurious edits* counts the individual changes it made to them.",
        "",
        f"The metric's own ceiling on this split is **F{beta:g} = {ceiling.f_score:.4f}**, not "
        f"1.0: that is what a system scores when it reproduces the annotator's corrections "
        f"exactly. MaxMatch recovers a system's edits from an optimal alignment between the "
        f"original and the correction, and {ceiling.counts.fn} of the corpus's "
        f"{ceiling.counts.tp + ceiling.counts.fn:,} edits are drawn in a way that no optimal "
        f"alignment produces. Read every score below against that number rather than against 1.",
        "",
        "## Detection versus correction",
        "",
        "Span-level ERRANT scoring. *Detection* asks only whether the system flagged the "
        "right span, *correction* whether it also proposed the annotator's fix. The gap "
        "between them is the system seeing the error and getting the repair wrong.",
        "",
        f"| System | Detection F{beta:g} | Correction F{beta:g} | Gap |",
        "|---|---:|---:|---:|",
    ]
    for result in results:
        detection, correction = result["detection"], result["correction"]
        lines.append(
            f"| {result['name']} | {detection.f_score:.4f} | {correction.f_score:.4f} | "
            f"{detection.f_score - correction.f_score:+.4f} |"
        )

    lines += ["", "## Where each system helps", ""]
    lines += _type_table(results)

    lines += [
        "",
        "## What the systems change in correct prose",
        "",
    ]
    for result in results:
        over = result["overcorrection"]
        if not over.changes:
            lines += [f"**{result['name']}** changed nothing.", ""]
            continue
        lines += [
            f"**{result['name']}** -- {over}. Most common: "
            + ", ".join(f"`{t}` ({c})" for t, c in over.by_type.most_common(5))
            + ".",
            "",
        ]
        for change in over.changes[:3]:
            lines += [f"> {change.source}", f">> {change.hypothesis}", ""]

    lines += [
        "## Notes",
        "",
        f"* Model: `{model}`, greedy decoding.",
        f"* Every system's output goes through the same {lang.prose_name} tokeniser before "
        "scoring. The identity row prices that step: whatever false positives it shows are "
        "the harness disagreeing with the corpus about hyphens and apostrophes, not a system "
        "making mistakes.",
        "* LanguageTool applies the first suggestion of every non-overlapping match that "
        "offers one. Matches with no suggestion are left alone.",
        "* The overcorrection set was **not** screened with LanguageTool, which would have "
        "handed LanguageTool a zero by construction.",
    ]
    return lines


def _type_table(results: list[dict], top: int = 15) -> list[str]:
    """A per-error-type recall table across systems, ordered by how common the type is."""
    reference = results[0]["correction"]
    types = [t for t, _ in reference.ranked_types(minimum=20)][:top]
    if not types:
        return ["_Not enough gold edits per type to break down._"]

    header = "| Error type | Gold | " + " | ".join(f"{r['name']} R" for r in results) + " |"
    divider = "|---|---:|" + "---:|" * len(results)
    lines = [header, divider]
    for error_type in types:
        gold = reference.by_type[error_type]
        cells = []
        for result in results:
            counts = result["correction"].by_type.get(error_type)
            cells.append(f"{counts.recall:.2f}" if counts else "0.00")
        lines.append(f"| `{error_type}` | {gold.tp + gold.fn} | " + " | ".join(cells) + " |")
    return lines


if __name__ == "__main__":
    main()
