#!/usr/bin/env python
"""Verify that LanguageTool is running *and* that its n-gram data actually loaded.

This check exists because the failure it detects is silent. With
`langtool_languageModel` unset or pointing at the wrong place, LanguageTool
starts perfectly happily and answers every request -- it just quietly stops
firing the confusion-set rules that need a language model. The symptom is a
server that only ever reports spelling mistakes, and it is easy to mistake that
for "German grammar checking is just hard".

Since das/dass and its relatives are precisely the errors this project cares
about, a green tick here is a precondition for Phase 1 and Phase 2.

    python scripts/lt_sanity_check.py
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass

from langlm.config import PHASE0_REPORT_DIR
from langlm.lt.client import SPELLING_ONLY_RULES, LanguageToolClient, LanguageToolError, Match

COMPOSE_FILE = "docker/docker-compose.yml"


#: Rules driven by the n-gram language model all carry this prefix. Verified
#: empirically against a control server started without `langtool_languageModel`:
#: every probe below fires only when the n-grams are loaded.
CONFUSION_RULE_PREFIX = "CONFUSION_RULE"


@dataclass(frozen=True)
class Probe:
    """A sentence with a known error and the correction we expect back."""

    text: str
    expect: str
    why: str
    #: True if catching this error requires the n-gram language model. These
    #: were chosen by diffing a server with n-grams against one without, so a
    #: pass here really does mean the language model loaded.
    needs_ngrams: bool


PROBES = (
    Probe(
        text="Das hat mir fiel geholfen.",
        expect="viel",
        why="fiel/viel homophone confusion",
        needs_ngrams=True,
    ),
    Probe(
        text="Ich habe ihm Garten gearbeitet.",
        expect="im",
        why="ihm/im confusion, resolvable only from context",
        needs_ngrams=True,
    ),
    Probe(
        text="Er hat mit das Buch gegeben.",
        expect="mir",
        why="mit/mir confusion",
        needs_ngrams=True,
    ),
    Probe(
        text="Die Leere an der Universität ist gut.",
        expect="Lehre",
        why="Leere/Lehre confusion; both are ordinary words, so only the language model can tell",
        needs_ngrams=True,
    ),
    Probe(
        text="Ich hoffe dass du kommst.",
        expect=",",
        why="missing comma before a subordinate clause (rule-based, no n-grams needed)",
        needs_ngrams=False,
    ),
    Probe(
        text="Ich weiß, das du morgen kommst.",
        expect="dass",
        why="das/dass (rule-based in German LanguageTool, despite the folklore)",
        needs_ngrams=False,
    ),
    Probe(
        text="Das ist ein Fehlar.",
        expect="Fehler",
        why="plain spelling (works even with no language model at all)",
        needs_ngrams=False,
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="de-DE")
    parser.add_argument("--no-report", action="store_true", help="Skip writing the report file")
    args = parser.parse_args()

    client = LanguageToolClient(language=args.language)

    print(f"LanguageTool at {client.base_url}")
    if not client.is_up():
        print("  NOT REACHABLE. Start it with `make lt-up`, then wait for it to finish booting")
        print("  (the first start with n-grams loaded takes a minute or two).")
        return 1
    print(f"  up, {len(client.languages())} languages available\n")

    results = [(probe, _run(client, probe)) for probe in PROBES]
    _print_results(results)

    # A confusion rule firing is the actual evidence: those rules exist only
    # when a language model is loaded. Checking the correction alone would be
    # weaker, because German LanguageTool catches several classic confusions
    # with hand-written rules that need no n-grams at all.
    ngram_ok = all(
        found and any(m.rule_id.startswith(CONFUSION_RULE_PREFIX) for m in matches)
        for probe, (found, matches) in results
        if probe.needs_ngrams
    )
    log_evidence = _language_model_log_line()

    print()
    if ngram_ok:
        print(f"PASS: {CONFUSION_RULE_PREFIX}_* rules are firing. The n-gram model is loaded.")
    else:
        _print_diagnosis(results, log_evidence)

    if not args.no_report:
        path = _write_report(results, ngram_ok, log_evidence)
        print(f"\nWrote {path}")

    return 0 if ngram_ok else 1


def _run(client: LanguageToolClient, probe: Probe) -> tuple[bool, list[Match]]:
    """Check one probe; return whether the expected correction was offered."""
    try:
        matches = client.check(probe.text)
    except LanguageToolError as exc:
        print(f"  request failed: {exc}")
        return False, []
    found = any(
        any(probe.expect in replacement for replacement in match.replacements) for match in matches
    )
    return found, matches


def _print_results(results: list[tuple[Probe, tuple[bool, list[Match]]]]) -> None:
    for probe, (found, matches) in results:
        tick = "ok  " if found else "MISS"
        tag = "[n-gram]" if probe.needs_ngrams else "[rule]  "
        print(f"  {tick} {tag} {probe.text}")
        print(f"           expected {probe.expect!r} -- {probe.why}")
        for match in matches:
            print(f"           got {match.rule_id}: {match.replacements[:4]}")


def _print_diagnosis(
    results: list[tuple[Probe, tuple[bool, list[Match]]]], log_evidence: str | None
) -> None:
    print("FAIL: the n-gram-dependent rules did not fire.")
    rule_ids = {m.rule_id for _, (_, matches) in results for m in matches}
    if not any(rid.startswith(CONFUSION_RULE_PREFIX) for rid in rule_ids):
        print(f"  No {CONFUSION_RULE_PREFIX}_* rule fired on any probe, which means the")
        print("  language model is not loaded, whatever else the server reports.")
    if rule_ids and rule_ids <= SPELLING_ONLY_RULES:
        print("  Every match came from the spelling dictionary, which is the classic")
        print("  signature of n-gram data that failed to load.")
    print("\n  Check, in order:")
    print("   1. `ls data/raw/ngrams/de` shows 1grams/ 2grams/ 3grams/")
    print("   2. docker/docker-compose.yml mounts that path at /ngrams")
    print("      and sets langtool_languageModel=/ngrams")
    print("   3. `make lt-logs` mentions loading a language model")
    print("   4. Java_Xmx is at least 4g -- the indexes are memory-hungry")
    if log_evidence:
        print(f"\n  Server log says: {log_evidence}")
    else:
        print("\n  No language-model line found in the server log.")


def _language_model_log_line() -> str | None:
    """Look for the server's own confirmation that it loaded a language model."""
    try:
        output = subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "logs", "--no-color", "--tail", "300"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        lowered = line.lower()
        if "language model" in lowered or "ngram" in lowered or "n-gram" in lowered:
            return line.strip()
    return None


def _write_report(
    results: list[tuple[Probe, tuple[bool, list[Match]]]],
    ngram_ok: bool,
    log_evidence: str | None,
) -> object:
    lines = [
        "# LanguageTool sanity check",
        "",
        f"**Status: {'PASS' if ngram_ok else 'FAIL'}** -- "
        + (
            f"`{CONFUSION_RULE_PREFIX}_*` rules are firing, so the n-gram language model "
            f"loaded correctly."
            if ngram_ok
            else f"no `{CONFUSION_RULE_PREFIX}_*` rule fired. The language model is not loaded."
        ),
        "",
        "Without n-gram data LanguageTool starts normally and answers every request, but "
        "silently stops firing the confusion-set rules that resolve homophones from context. "
        "Those are central to this project, so this check gates Phase 1.",
        "",
        "The probes marked *needs n-grams* were chosen by diffing this server against a "
        "control container started without `langtool_languageModel`: each one is caught only "
        "when the language model is present. That matters, because several classic German "
        "confusions -- das/dass among them -- are caught by hand-written rules that need no "
        "n-grams, and would give a false pass.",
        "",
        "| Probe | Needs n-grams | Expected | Result | Rules fired |",
        "|---|:--:|---|:--:|---|",
    ]
    for probe, (found, matches) in results:
        rules = ", ".join(f"`{m.rule_id}`" for m in matches) or "_none_"
        lines.append(
            f"| `{probe.text}` | {'yes' if probe.needs_ngrams else 'no'} | "
            f"`{probe.expect}` | {'ok' if found else '**miss**'} | {rules} |"
        )

    lines += ["", "## Server log evidence", "", f"```\n{log_evidence or '(no matching line)'}\n```"]

    PHASE0_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = PHASE0_REPORT_DIR / "lt_sanity.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
