"""The systems Phase 1 measures, and the interface they share.

Every baseline is a callable that takes a whitespace-tokenised German sentence
and returns a corrected one. That is the whole contract: it is what
LanguageTool, a prompted base model and, later, a fine-tuned one can all honour,
and it keeps the scorers ignorant of how any of them work.

Output is normalised through the same tokeniser for every system before it is
scored, so that a model which writes ordinary German is not charged for putting
the full stop next to the last word. The cost of that normalisation is itself
measured: see the identity baseline in `reports/phase1/baselines.md`.
"""

from __future__ import annotations

from typing import Protocol


class Corrector(Protocol):
    """A system that corrects one sentence at a time."""

    name: str

    def correct(self, source: str) -> str:
        """Return `source` with its errors fixed, or `source` unchanged."""
        ...
