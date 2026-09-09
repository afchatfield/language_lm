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

from functools import cache
from pathlib import Path

import yaml

from langlm.config import RULES_DIR


class TemplateError(KeyError):
    """Raised when an error has no explanation, which is a data bug, not a miss."""


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
        text = f"{text} (Rat für deutsche Rechtschreibung, {reference})"
    return text


def coverage(rules: list[str], language: str = "de") -> list[str]:
    """Which of the given corruptors have no template. Empty is the healthy answer."""
    templates = load(language)["corruptors"]
    return sorted(rule for rule in rules if rule not in templates)
