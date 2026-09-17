"""Turning an edit into a sentence a learner can read.

The templates live in `rules/<language>.yaml` and are keyed on the corruptor
that made the error, because that is the thing that knows what happened. The
Phase 2 survey found that 78% of LanguageTool's matches on injected data are
`GERMAN_SPELLER_RULE`, whose message is the same "possible typo" for a stripped
umlaut, an ss written for ß, and a slipped finger -- three errors a learner
needs told apart.

LanguageTool is still consulted where it knows something the corruptor does not:
which case a preposition governs, which article agrees. Those refine the
explanation rather than replacing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from langlm.config import RULES_DIR


class TemplateError(KeyError):
    """Raised when an error has no explanation, which is a data bug, not a miss."""


@cache
def _annotator(language: str):
    """The ERRANT annotator module for one language, imported lazily.

    `errant_de` and `errant_es` both expose `annotate(source, hypothesis)` in
    the same shape -- see `langlm.eval.errant_core`, which is what makes them
    the same shape to begin with -- so this dispatch is the only place that
    needs to know there is more than one language. Imported lazily, and only
    the language actually asked for, because each import loads a spacy model.
    """
    if language == "de":
        from langlm.eval import errant_de as module
    elif language == "es":
        from langlm.eval import errant_es as module
    else:
        raise ValueError(f"No ERRANT annotator for language {language!r}.")
    return module


@cache
def load(language: str = "de", root: Path = RULES_DIR) -> dict:
    """The template file for a language."""
    path = Path(root) / f"{language}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No explanation templates at {path}.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def explain(
    rule: str,
    wrong: str,
    right: str,
    lt_rule_id: str | None = None,
    language: str = "de",
) -> str:
    """Explain one correction.

    Args:
        rule: The corruptor that made the error, e.g. ``sharp_s``.
        wrong: What the damaged sentence says.
        right: What it should say. Empty for an omission.
        lt_rule_id: A LanguageTool rule id, if one fired on this span. Used only
            when it adds something the corruptor could not know.
        language: Which template file to read.

    Returns:
        One or two sentences of English.

    Raises:
        TemplateError: if the corruptor has no template. Every rule must have
            one -- an example whose explanation is missing is worse in training
            data than an example that was never generated.
    """
    templates = load(language)
    entry = templates["corruptors"].get(rule)
    if entry is None:
        raise TemplateError(f"No explanation template for corruptor {rule!r}.")

    text = entry["explanation"].format(wrong=wrong, right=right).strip()

    refinement = templates.get("languagetool", {}).get(lt_rule_id or "")
    if refinement:
        text = f"{text} {refinement['explanation'].format(wrong=wrong, right=right).strip()}"

    reference = entry.get("reference")
    if reference:
        # The edition is part of the citation, not decoration. Part E was
        # renumbered in 2024, so a bare "§ 73" is ambiguous between the
        # subordinate-clause comma and whatever the 2018 text put there.
        edition = templates.get("edition", "")
        cite = f"{edition}, {reference}" if edition else reference
        text = f"{text} (Rat für deutsche Rechtschreibung, {cite})"
    return text


@cache
def load_types(language: str = "de", root: Path = RULES_DIR) -> dict:
    """The error-type template file for a language."""
    path = Path(root) / f"{language}_types.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No error-type templates at {path}.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def explain_type(
    error_type: str,
    wrong: str,
    right: str,
    language: str = "de",
) -> str:
    """Explain an edit when all we know is its ERRANT type.

    This is the real-data path. A Falko-MERLIN edit carries a type and nothing
    else -- no record of which rule the learner broke -- so the explanation can
    only be as specific as the type is. `explain` stays the better one, and is
    used wherever the corruptor is known.

    Types are compositional: an operation (`M` missing, `R` replaced, `U`
    unnecessary), a word class, and optionally `:FORM` for an inflection rather
    than a choice. The types carrying real volume are written out by hand; the
    tail is composed from those parts, which keeps 54 types consistent instead
    of unevenly hand-written.
    """
    templates = load_types(language)

    explicit = templates["explicit"].get(error_type)
    if explicit:
        return explicit.format(wrong=wrong, right=right).strip()

    parts = error_type.split(":")
    operation, category_key = (parts[0], parts[1]) if len(parts) > 1 else ("R", "OTHER")
    category = templates["categories"].get(category_key)
    if category is None:
        return templates["fallback"].format(wrong=wrong, right=right).strip()

    key = "R:FORM" if len(parts) > 2 and parts[2] == "FORM" else operation
    template = templates["operations"].get(key) or templates["operations"]["R"]
    return template.format(wrong=wrong, right=right, category=category).strip()


@dataclass(frozen=True)
class Explained:
    """One rewrite, typed and put into words for the learner."""

    start: int
    end: int
    wrong: str
    right: str
    error_type: str
    explanation: str


def explain_correction(source: str, correction: str, language: str = "de") -> list[Explained]:
    """Explain a corrected sentence by re-deriving its edits, not by being told them.

    The model is asked for a `changes` list alongside its correction, and that
    list carries a type. Rendering the learner's explanation from it makes the
    explanation only as good as the model's ability to name what it did, which
    on Falko-MERLIN dev is 0.71 against the annotator. Deriving the edits from
    the pair (source, correction) instead is deterministic and lands in the
    mid-80s -- ERRANT re-derives the annotator's own type on 89% of the edits it
    segments the same way -- so the explanation stops depending on a skill the
    model is separately bad at.

    The model's `changes` list keeps its job: it is the coverage and precision
    diagnostic in `langlm.eval.explanation_scorer`, and rewrites the model does
    not claim are measurably worse than the ones it does. It is just no longer
    the thing the learner reads.
    """
    annotator = _annotator(language)
    edits = annotator.annotate(source, correction)
    tokens = source.split()
    return [
        Explained(
            start=edit.start,
            end=edit.end,
            wrong=" ".join(tokens[edit.start : edit.end]),
            right=edit.correction,
            error_type=edit.error_type,
            explanation=explain_type(
                edit.error_type,
                " ".join(tokens[edit.start : edit.end]),
                edit.correction,
                language,
            ),
        )
        for edit in edits
    ]


def coverage(rules: list[str], language: str = "de") -> list[str]:
    """Which of the given corruptors have no template. Empty is the healthy answer."""
    templates = load(language)["corruptors"]
    return sorted(rule for rule in rules if rule not in templates)
