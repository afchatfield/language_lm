#!/usr/bin/env python
"""Check every `reference:` in rules/de.yaml against the official ruleset text.

Phase 2 left six paragraph numbers unverified on the grounds that "a wrong
citation in 30k examples would teach the model to cite confidently and wrongly".
That was the right instinct aimed at the wrong half of the file: the six nulls
were harmless, and three of the numbers that *were* filled in are wrong.

They are wrong because the Rat renumbered part E in the 2024 edition. The
subordinate-clause comma is § 74 in the 2018 text and § 73 in the 2024 one; the
und/oder rule moved from § 72 to § 71. A citation written against one edition
and read against the other does not fail loudly -- it lands on the semicolon,
which is a plausible-looking answer to a question nobody asked.

So the citation is checked rather than remembered. Each one names a paragraph
and a phrase that must appear in that paragraph's rule statement; the script
downloads the pinned edition, extracts the text, and fails if a paragraph is
missing or says something else. Requires `pdftotext` (poppler-utils).

It needs the network, so it is a `make` target rather than a test. The half of
the invariant that works offline -- every uncited rule saying why it is uncited
-- is in `tests/test_explanations.py` and runs with the rest of the suite.

    python scripts/check_citations.py
    make check-citations
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

import yaml

from langlm.config import RAW_DIR, RULES_DIR
from langlm.data.download import download

URL = "https://www.rechtschreibrat.com/DOX/RfdR_Amtliches-Regelwerk_2024.pdf"
EDITION = "Amtliches Regelwerk 2024"
DESTINATION = RAW_DIR / "regelwerk"

#: What each cited paragraph has to say, as a phrase from its rule statement.
#: Keyed on the corruptor so a failure names the explanation that would ship
#: the bad citation, not just a number.
EXPECTED = {
    "noun_case": ("55", "Substantive werden großgeschrieben"),
    "sharp_s": ("25", "nach langem Vokal oder Diphthong"),
    "das_dass": ("4", "das (Artikel, Pronomen)"),
    "drop_comma_subordinate": ("73", "Das Komma zeigt Nebensätze an"),
    "drop_comma_coordinating": ("71", "Vor diesen Konjunktionen steht das Komma"),
    "drop_comma": ("72", "Das Komma zeigt einen Zusatz an"),
    "comma_before_und": ("71", "steht kein Komma"),
    "separable_prefix": ("34", "trennbare Zusammensetzungen"),
}


def ruleset_text(force: bool = False) -> str:
    """The pinned edition as plain text, downloading and converting on demand."""
    pdf = download(URL, DESTINATION / "RfdR_Amtliches-Regelwerk_2024.pdf", force=force)
    text = pdf.with_suffix(".txt")
    if not text.exists() or force:
        try:
            subprocess.run(
                ["pdftotext", "-layout", str(pdf), str(text)],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            raise SystemExit("pdftotext not found. Install poppler-utils.") from None
    return text.read_text(encoding="utf-8")


def rules_section(text: str) -> list[str]:
    """The lines of the rule part, with the Wörterverzeichnis cut off.

    The word list cites paragraphs on nearly every line, so leaving it in makes
    every number appear to exist everywhere. Its running header is the boundary.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"\s*Wörterverzeichnis\s+\w/\w\s*$", line):
            return lines[:index]
    raise SystemExit("Could not find the Wörterverzeichnis boundary; the PDF layout has changed.")


#: A paragraph heading. The layout sets the number either at the start of the
#: rule statement or flushed right at the end of its first line.
HEADING = re.compile(r"^\s*§\s?(\d+)\b|§\s?(\d+)\s*$")


def paragraph(lines: list[str], number: str) -> str:
    """One paragraph's rule statement and its E-notes.

    Runs from the paragraph's own heading to the next heading, so the E-notes
    are included -- `das (Artikel, Pronomen)` sits in § 4(6) and the comma
    before `sondern` in § 71 E2, both well past the rule statement itself.
    """
    starts = []
    for index, line in enumerate(lines):
        match = HEADING.search(line)
        if match:
            starts.append((index, match.group(1) or match.group(2)))

    body = []
    for position, (index, found) in enumerate(starts):
        if found != number:
            continue
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        body.extend(lines[index:end])
    return re.sub(r"\s+", " ", " ".join(body))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download and re-convert")
    args = parser.parse_args()

    rules = yaml.safe_load((RULES_DIR / "de.yaml").read_text(encoding="utf-8"))
    if rules.get("edition") != EDITION:
        raise SystemExit(
            f"rules/de.yaml pins edition {rules.get('edition')!r}, this script checks {EDITION!r}."
        )

    lines = rules_section(ruleset_text(force=args.force))
    failures = []

    for name, entry in rules["corruptors"].items():
        reference = entry.get("reference")
        if not reference:
            if not entry.get("reference_note"):
                failures.append(f"{name}: reference is null with no reference_note saying why")
            continue
        if name not in EXPECTED:
            failures.append(
                f"{name}: cites {reference}, which this script has nothing to check it against"
            )
            continue
        number, phrase = EXPECTED[name]
        if f"§ {number}" not in reference:
            failures.append(f"{name}: cites {reference}, expected § {number}")
            continue
        body = paragraph(lines, number)
        if not body:
            failures.append(f"{name}: § {number} does not appear in {EDITION}")
        elif phrase.lower() not in body.lower():
            failures.append(f"{name}: § {number} does not say {phrase!r}")
        else:
            print(f"  ok  {name:24} {reference}")

    cited = sum(1 for e in rules["corruptors"].values() if e.get("reference"))
    print(f"\n{cited} cited, {len(rules['corruptors']) - cited} uncited, {len(failures)} failures")
    for failure in failures:
        print(f"  FAIL {failure}", file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
