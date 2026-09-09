"""Reader and writer for the M2 annotation format.

M2 is the standard interchange format for grammatical error correction corpora
(CoNLL-2013/14, BEA-2019, and the German Falko-MERLIN corpus used here). A file
is a sequence of blank-line-separated blocks::

    S Das ist ein falsches autos .
    A 4 6|||R:NOUN:INFL|||falsche Autos|||REQUIRED|||-NONE-|||0
    A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||1

The ``S`` line holds the whitespace-tokenised original sentence. Each ``A`` line
is one annotator's edit, with pipe-separated fields::

    A <start> <end>|||<type>|||<correction>|||<required>|||<comment>|||<annotator>

Token offsets are half-open over the ``S`` tokens, so ``start == end`` is an
insertion and an empty correction is a deletion. ``A -1 -1|||noop|||...`` marks
an annotator who found nothing to change, which is a meaningful signal rather
than an absence of one: it is exactly the "this sentence is already correct"
case that Phase 2 relies on to fight overcorrection.

This module is the foundation for the Phase 1 M2 scorer, the Phase 3 evaluation
loop, and the Phase 5 Spanish port, so it aims to round-trip byte-for-byte
rather than merely to extract what today's caller happens to need.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

#: Sentinel used by M2 for "no value here", in both the correction field of a
#: noop edit and the (almost always empty) comment field.
NONE = "-NONE-"

#: Error type written on a noop edit.
NOOP_TYPE = "noop"


class M2ParseError(ValueError):
    """Raised when a line cannot be interpreted as M2."""


@dataclass(frozen=True)
class Edit:
    """A single annotator edit over the source tokens.

    Attributes:
        start: Index of the first replaced source token (half-open range start).
        end: Index one past the last replaced source token. ``start == end``
            denotes an insertion before ``start``.
        error_type: ERRANT-style tag, e.g. ``R:NOUN:INFL``, or ``noop``.
        correction: Replacement text, whitespace-tokenised. Empty for a
            deletion; ``-NONE-`` for a noop.
        annotator: Annotator id; corpora with a single annotator use ``0``.
        required: The M2 "required" field, preserved verbatim for round-tripping.
        comment: The M2 comment field, preserved verbatim for round-tripping.
    """

    start: int
    end: int
    error_type: str
    correction: str
    annotator: int = 0
    required: str = "REQUIRED"
    comment: str = NONE

    @property
    def is_noop(self) -> bool:
        """True if this edit records "the annotator changed nothing"."""
        return self.start == -1 and self.end == -1

    @property
    def is_insertion(self) -> bool:
        return not self.is_noop and self.start == self.end

    @property
    def is_deletion(self) -> bool:
        return not self.is_noop and self.start < self.end and self.correction == ""

    @property
    def correction_tokens(self) -> list[str]:
        """The correction split into tokens (empty for deletions and noops)."""
        if self.is_noop or self.correction == "":
            return []
        return self.correction.split()

    def to_line(self) -> str:
        """Serialise back to a single M2 ``A`` line."""
        fields = "|||".join(
            [
                self.error_type,
                self.correction,
                self.required,
                self.comment,
                str(self.annotator),
            ]
        )
        return f"A {self.start} {self.end}|||{fields}"

    @classmethod
    def from_line(cls, line: str) -> Edit:
        """Parse a single M2 ``A`` line."""
        if not line.startswith("A "):
            raise M2ParseError(f"Not an annotation line: {line!r}")
        body = line[2:]
        parts = body.split("|||")
        if len(parts) < 6:
            raise M2ParseError(f"Expected 6 pipe-separated fields, got {len(parts)}: {line!r}")
        offsets, error_type, correction, required, comment, annotator = parts[:6]
        try:
            start_str, end_str = offsets.split()
            start, end = int(start_str), int(end_str)
        except ValueError as exc:  # pragma: no cover - defensive
            raise M2ParseError(f"Bad offsets in {line!r}") from exc
        return cls(
            start=start,
            end=end,
            error_type=error_type,
            correction=correction,
            annotator=int(annotator),
            required=required,
            comment=comment,
        )


@dataclass
class M2Sentence:
    """One M2 block: a source sentence plus every annotator's edits."""

    source_tokens: list[str]
    edits: list[Edit] = field(default_factory=list)

    @property
    def source(self) -> str:
        return " ".join(self.source_tokens)

    @property
    def annotators(self) -> list[int]:
        """Annotator ids present on this sentence, in ascending order."""
        return sorted({edit.annotator for edit in self.edits})

    def edits_for(self, annotator: int = 0, include_noop: bool = False) -> list[Edit]:
        """Return one annotator's edits, sorted by position.

        Noop markers are dropped by default: they carry no span to apply, and a
        caller asking for "the edits" almost always means the real ones.
        """
        selected = [e for e in self.edits if e.annotator == annotator]
        if not include_noop:
            selected = [e for e in selected if not e.is_noop]
        return sorted(selected, key=lambda e: (e.start, e.end))

    def is_correct(self, annotator: int = 0) -> bool:
        """True if this annotator recorded no changes to the sentence."""
        return not self.edits_for(annotator)

    def apply(self, annotator: int = 0) -> list[str]:
        """Materialise the corrected sentence for one annotator.

        Edits are applied right-to-left so that earlier offsets stay valid as
        later spans change length.

        Raises:
            M2ParseError: if the annotator's edits overlap, which would make the
                result depend on application order.
        """
        edits = self.edits_for(annotator)
        self._check_no_overlap(edits)

        tokens = list(self.source_tokens)
        for edit in reversed(edits):
            tokens[edit.start : edit.end] = edit.correction_tokens
        return tokens

    def target(self, annotator: int = 0) -> str:
        """The corrected sentence as a string."""
        return " ".join(self.apply(annotator))

    def to_block(self) -> str:
        """Serialise back to an M2 block (no trailing blank line)."""
        lines = [f"S {self.source}"]
        lines.extend(edit.to_line() for edit in self.edits)
        return "\n".join(lines)

    @staticmethod
    def _check_no_overlap(edits: list[Edit]) -> None:
        previous_end = -1
        for edit in edits:
            # Insertions (start == end) may legitimately sit at a previous
            # edit's boundary, so only a strict overlap is an error.
            if edit.start < previous_end:
                raise M2ParseError(
                    f"Overlapping edits at token {edit.start}: cannot apply deterministically"
                )
            previous_end = max(previous_end, edit.end)


def parse_m2(path: str | Path) -> Iterator[M2Sentence]:
    """Stream sentences from an M2 file.

    Args:
        path: Path to a ``.m2`` file.

    Yields:
        One :class:`M2Sentence` per block, in file order.
    """
    path = Path(path)
    current: M2Sentence | None = None

    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                if current is not None:
                    yield current
                    current = None
                continue
            try:
                if line.startswith("S "):
                    if current is not None:
                        # Tolerate files that omit the blank line between blocks.
                        yield current
                    current = M2Sentence(source_tokens=line[2:].split())
                elif line.startswith("A "):
                    if current is None:
                        raise M2ParseError("Annotation line before any source line")
                    current.edits.append(Edit.from_line(line))
                else:
                    raise M2ParseError(f"Unrecognised line prefix: {line!r}")
            except M2ParseError as exc:
                raise M2ParseError(f"{path}:{lineno}: {exc}") from exc

    if current is not None:
        yield current


def write_m2(sentences: Iterable[M2Sentence], path: str | Path) -> int:
    """Write sentences to an M2 file. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for sentence in sentences:
            fh.write(sentence.to_block())
            fh.write("\n\n")
            count += 1
    return count


def error_type_histogram(sentences: Iterable[M2Sentence], annotator: int = 0) -> Counter[str]:
    """Count edits by error type.

    Phase 2 weights its synthetic error injection to match this distribution, so
    that the training data looks like the errors real learners actually make.
    """
    histogram: Counter[str] = Counter()
    for sentence in sentences:
        for edit in sentence.edits_for(annotator):
            histogram[edit.error_type] += 1
    return histogram
