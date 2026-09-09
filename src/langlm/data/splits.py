"""Freezing, verifying, and guarding the evaluation splits.

The roadmap's first instruction is to freeze the test split on day one and not
look at it again until Phase 3. Discipline is a poor mechanism for that, so this
module makes it structural:

* :func:`freeze_splits` records a SHA-256 of every split file in a manifest that
  is committed to git. Any later change to the evaluation data shows up as a
  diff rather than as an unexplained jump in the numbers.
* :func:`verify_manifest` re-checks those hashes and fails on drift.
* :func:`load_split` refuses to return the test split unless the caller passes
  ``allow_test=True``, turning an accidental peek into a deliberate act that is
  visible in code review.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from langlm.config import PROJECT_ROOT, SPLIT_MANIFEST
from langlm.data.m2 import M2Sentence, parse_m2

#: Splits whose contents must not influence development decisions.
HELD_OUT_SPLITS = frozenset({"test"})


class HeldOutSplitError(RuntimeError):
    """Raised when the held-out test split is loaded without explicit opt-in."""


class ManifestError(RuntimeError):
    """Raised when the split manifest is missing or no longer matches the data."""


@dataclass(frozen=True)
class SplitRecord:
    """The frozen fingerprint of one split file."""

    path: str  # relative to the project root, so the manifest is portable
    sha256: str
    bytes: int
    sentences: int
    edits: int


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def describe_split(path: str | Path) -> SplitRecord:
    """Build the fingerprint record for one M2 file."""
    path = Path(path)
    sentences = list(parse_m2(path))
    edits = sum(len(s.edits_for(a)) for s in sentences for a in (s.annotators or [0]))
    return SplitRecord(
        path=str(path.resolve().relative_to(PROJECT_ROOT)),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        sentences=len(sentences),
        edits=edits,
    )


def freeze_splits(
    corpus: str,
    split_paths: dict[str, Path],
    source_url: str = "",
    manifest_path: Path = SPLIT_MANIFEST,
) -> dict:
    """Record the fingerprints of a corpus's splits in the manifest.

    Re-freezing a corpus that is already recorded raises rather than silently
    overwriting: if the data changed, that should be a conscious decision.

    Args:
        corpus: Corpus key, e.g. ``falko_merlin``.
        split_paths: Mapping of split name to the ``.m2`` file for that split.
        source_url: Where the data came from, for provenance.
        manifest_path: Manifest location; overridable for tests.

    Returns:
        The corpus entry that was written.
    """
    manifest = _read_manifest(manifest_path) if manifest_path.exists() else {"corpora": {}}

    if corpus in manifest["corpora"]:
        raise ManifestError(
            f"{corpus!r} is already frozen in {manifest_path}. Delete the entry deliberately "
            f"if the evaluation data really should change."
        )

    entry = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_url": source_url,
        "splits": {name: asdict(describe_split(p)) for name, p in sorted(split_paths.items())},
    }
    manifest["corpora"][corpus] = entry

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return entry


def verify_manifest(manifest_path: Path = SPLIT_MANIFEST) -> None:
    """Re-hash every frozen split and raise if anything has drifted.

    Raises:
        ManifestError: if the manifest is missing, a file is gone, or a hash
            no longer matches.
    """
    manifest = _read_manifest(manifest_path)
    problems: list[str] = []

    for corpus, entry in sorted(manifest["corpora"].items()):
        for split, record in sorted(entry["splits"].items()):
            path = PROJECT_ROOT / record["path"]
            if not path.exists():
                problems.append(f"{corpus}/{split}: missing file {record['path']}")
                continue
            actual = sha256_file(path)
            if actual != record["sha256"]:
                problems.append(
                    f"{corpus}/{split}: sha256 changed\n"
                    f"    expected {record['sha256']}\n"
                    f"    actual   {actual}"
                )

    if problems:
        raise ManifestError("Frozen evaluation data has changed:\n  " + "\n  ".join(problems))


def split_path(corpus: str, split: str, manifest_path: Path = SPLIT_MANIFEST) -> Path:
    """Look up the recorded path of a frozen split."""
    manifest = _read_manifest(manifest_path)
    try:
        record = manifest["corpora"][corpus]["splits"][split]
    except KeyError as exc:
        known = _known_splits(manifest)
        raise ManifestError(f"No frozen split {corpus!r}/{split!r}. Known: {known}") from exc
    return PROJECT_ROOT / record["path"]


def load_split(
    corpus: str,
    split: str,
    allow_test: bool = False,
    manifest_path: Path = SPLIT_MANIFEST,
) -> list[M2Sentence]:
    """Load a frozen split, refusing held-out data unless explicitly permitted.

    Args:
        corpus: Corpus key, e.g. ``falko_merlin``.
        split: Split name, e.g. ``train``, ``dev``, ``test``.
        allow_test: Must be ``True`` to load a held-out split. Pass it only from
            final evaluation code, never from anything that shapes the model.
        manifest_path: Manifest location; overridable for tests.

    Raises:
        HeldOutSplitError: when loading a held-out split without ``allow_test``.
    """
    if split in HELD_OUT_SPLITS and not allow_test:
        raise HeldOutSplitError(
            f"{corpus}/{split} is held out until final evaluation. If this really is final "
            f"evaluation, pass allow_test=True so the access is explicit and greppable."
        )
    return list(parse_m2(split_path(corpus, split, manifest_path)))


def _read_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise ManifestError(
            f"No split manifest at {manifest_path}. Run `make data` to download the corpus "
            f"and freeze the splits."
        )
    with manifest_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _known_splits(manifest: dict) -> str:
    return (
        ", ".join(
            f"{corpus}/{split}"
            for corpus, entry in sorted(manifest["corpora"].items())
            for split in sorted(entry["splits"])
        )
        or "(none)"
    )
