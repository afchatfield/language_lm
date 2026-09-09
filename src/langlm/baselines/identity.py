"""The system that changes nothing.

Not a strawman: it measures the floor. Every system's output is re-tokenised
before scoring so that free German prose can be compared with a tokenised
corpus, and that re-tokenisation is not quite lossless -- it disagrees with the
corpus about a handful of hyphenated numbers and apostrophes. Running the
identity system through the same pipeline prices that disagreement, so the
Phase 1 table can say how much of each system's false-positive count is the
harness rather than the system.
"""

from __future__ import annotations


class IdentityBaseline:
    """Returns its input unchanged."""

    name = "identity"

    def correct(self, source: str) -> str:
        return source
