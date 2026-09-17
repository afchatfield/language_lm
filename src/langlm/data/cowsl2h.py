"""The COWS-L2H Spanish learner corpus.

Yamada, Davidson & Perez-Cortes, *COWS-L2H: A corpus of Spanish learner
writing* (LREC 2020), is the UC Davis Corpus of Written Spanish, L2 and
Heritage Speakers: college-level essays on a fixed set of prompts, a subset of
them corrected by one or two instructors.

Unlike Falko-MERLIN, this corpus does not ship as M2. It ships as one CSV per
(prompt, quarter) pair, one row per essay, with the raw text in ``essay`` and
up to two independent corrections in ``corrected1`` / ``corrected2`` -- most
essays have neither. There is no train/dev/test split, and corrections are
document-level: turning them into sentence-aligned pairs is this project's own
work, not upstream's.

This module downloads the CSVs (pinned to a commit so the corpus is
reproducible), keeps the essays with a `corrected1`, sentence-aligns each essay
against its correction with spacy, and writes the aligned pairs to
`data/interim/cowsl2h/`. That is as far as this module goes: the pairs are not
yet M2, because M2 needs typed edits and there is no Spanish ERRANT to type
them with. `errant_es` produces the M2 files this feeds; `freeze_splits_es.py`
freezes them.

Splitting is by learner (the `id` column, stable across a student's essays
across quarters) rather than by essay, because COWS-L2H is longitudinal and the
same student writes on several prompts -- an essay-level split would leak a
student's writing style across train and test. The assignment is a salted hash
of the id, not a stored random draw, so it reproduces without committing a
seed; it is a policy this project sets, not one upstream provides, and it is
final only once `freeze_splits_es.py` records its hash.

Licensing: COWS-L2H is released under Apache 2.0. The derived, sentence-aligned
corpus is not committed to this repository.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from langlm.config import COWSL2H_DIR, INTERIM_DIR, PROCESSED_DIR
from langlm.data.download import download

CORPUS = "cowsl2h"

# Pinned to a commit, not `master`, so the corpus a report was written against
# can be re-downloaded byte-for-byte later. Chosen as the newest commit that
# touched `csv/` as of the Phase 5 survey.
COMMIT = "6b18b773aef6d6da3b7796e9c26007981856e8e1"
RAW_BASE_URL = f"https://raw.githubusercontent.com/ucdaviscl/cowsl2h/{COMMIT}"
LICENSE_URL = f"{RAW_BASE_URL}/LICENSE"
CITATION_URL = "https://github.com/ucdaviscl/cowsl2h"

#: Every per-prompt CSV in the release, as of `COMMIT`. Hardcoded rather than
#: listed via the GitHub API at run time: the file set is part of what a
#: pinned commit promises, and an API call would let a future rename or
#: addition upstream silently change what this project trains on.
CSV_FILES = [
    "beautiful.F20.csv",
    "beautiful.S20.csv",
    "beautiful.W20.csv",
    "beautiful.W21.csv",
    "chaplin.S21.csv",
    "famous.F17.csv",
    "famous.S17.csv",
    "famous.S18.csv",
    "famous.SU17.csv",
    "famous.W18.csv",
    "place_you_dislike.S21.csv",
    "special.F18.csv",
    "special.F19.csv",
    "special.S19.csv",
    "special.W19.csv",
    "terrible.F18.csv",
    "terrible.F19.csv",
    "terrible.S19.csv",
    "terrible.W19.csv",
    "vacation.F17.csv",
    "vacation.S17.csv",
    "vacation.S18.csv",
    "vacation.SU17.csv",
    "vacation.W18.csv",
    "yourself.F20.csv",
    "yourself.S20.csv",
    "yourself.W20.csv",
    "yourself.W21.csv",
]

SPACY_MODEL = "es_core_news_sm"

INTERIM_DIR_FOR_CORPUS = INTERIM_DIR / CORPUS
PROCESSED_DIR_FOR_CORPUS = PROCESSED_DIR / CORPUS

#: Learner-id hash buckets. Chosen to land close to Falko-MERLIN's own
#: train/dev/test proportions; revisit before `freeze_splits_es.py` runs if the
#: aligned pair counts land somewhere this ratio starves dev or test.
SPLIT_RATIOS = (("train", 0.8), ("dev", 0.1), ("test", 0.1))

#: CSV fields long enough to need it (an essay can run past the 128KB default).
csv.field_size_limit(10 * 1024 * 1024)


@cache
def _nlp():
    """The Spanish spacy pipeline, loaded once. NER is the only thing dropped:
    sentence boundaries come from the parser, and nothing here reads entities.
    """
    import spacy

    try:
        return spacy.load(SPACY_MODEL, disable=["ner"])
    except OSError as exc:
        raise OSError(
            f"The spacy model {SPACY_MODEL!r} is not installed. "
            f"Run `python -m spacy download {SPACY_MODEL}`."
        ) from exc


def sentences(text: str) -> list[str]:
    """Split free text into corpus-tokenised sentences.

    Each sentence is its tokens rejoined with single spaces, the same
    convention `errant_de.tokenize` uses, so a sentence written here and one
    read back later always tokenise identically regardless of the essay's
    original spacing.
    """
    doc = _nlp()(text.replace("\r", " "))
    return [
        " ".join(tok.text for tok in sent if not tok.is_space)
        for sent in doc.sents
        if sent.text.strip()
    ]


@dataclass
class Essay:
    """One row of one CSV: an essay and up to two independent corrections."""

    learner_id: str
    prompt: str
    quarter: str
    source_file: str
    essay: str
    corrected1: str
    corrected2: str


def read_essays(csv_dir: Path = COWSL2H_DIR) -> Iterator[Essay]:
    """Yield every essay across the downloaded CSVs, corrected or not."""
    for name in CSV_FILES:
        path = csv_dir / name
        prompt, quarter = name.removesuffix(".csv").split(".", 1)
        with path.open(encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                learner_id = (row.get("id") or "").strip()
                essay = " ".join((row.get("essay") or "").split())
                if not learner_id or not essay:
                    continue
                yield Essay(
                    learner_id=learner_id,
                    prompt=prompt,
                    quarter=quarter,
                    source_file=name,
                    essay=essay,
                    corrected1=" ".join((row.get("corrected1") or "").split()),
                    corrected2=" ".join((row.get("corrected2") or "").split()),
                )


@dataclass
class AlignedPair:
    """One sentence and its correction, plus enough provenance to trace it
    back to the essay it came from."""

    learner_id: str
    prompt: str
    quarter: str
    source_file: str
    sentence_index: int
    source: str
    target: str
    target2: str | None  # from corrected2, only when that essay has one


def align_essay(essay: Essay) -> list[AlignedPair] | None:
    """Sentence-align one essay against its correction(s).

    Returns `None` when the essay and its correction do not split into the
    same number of sentences -- a corrector who merged or split a sentence,
    most often. That is not rare (the Phase 5 survey found it in 21.3% of
    corrected essays) and there is no reliable automatic fix: a corrector who
    joins two run-on sentences into one has changed the very thing being
    scored, so guessing at alignment would be scoring against a guess. Those
    essays are dropped and counted, not force-aligned.
    """
    src = sentences(essay.essay)
    tgt = sentences(essay.corrected1)
    if len(src) != len(tgt):
        return None

    tgt2 = sentences(essay.corrected2) if essay.corrected2 else None
    if tgt2 is not None and len(tgt2) != len(src):
        tgt2 = None  # keep corrected1's alignment; drop only the agreement signal

    return [
        AlignedPair(
            learner_id=essay.learner_id,
            prompt=essay.prompt,
            quarter=essay.quarter,
            source_file=essay.source_file,
            sentence_index=i,
            source=s,
            target=t,
            target2=tgt2[i] if tgt2 is not None else None,
        )
        for i, (s, t) in enumerate(zip(src, tgt, strict=True))
    ]


def assign_split(learner_id: str, ratios: tuple[tuple[str, float], ...] = SPLIT_RATIOS) -> str:
    """Deterministically bucket a learner id into train/dev/test.

    A hash rather than `random.shuffle` so re-running this on a fresh checkout
    reproduces the same split without a seed to keep in sync, and so the split
    a learner lands in does not depend on how many other learners exist -- it
    would if buckets were filled by sorted order, which would make every split
    unstable to which CSVs happen to be present.
    """
    digest = hashlib.sha256(f"cowsl2h:{learner_id}".encode()).hexdigest()
    point = int(digest, 16) / 16**64
    cursor = 0.0
    for name, share in ratios:
        cursor += share
        if point < cursor:
            return name
    return ratios[-1][0]  # pragma: no cover - float rounding at the boundary


def download_csvs(force: bool = False) -> Path:
    """Download every CSV and the licence file, skipping ones already present."""
    for name in CSV_FILES:
        download(f"{RAW_BASE_URL}/csv/{name}", COWSL2H_DIR / name, force=force)
    download(LICENSE_URL, COWSL2H_DIR / "LICENSE", force=force)
    return COWSL2H_DIR


def build_pairs(csv_dir: Path = COWSL2H_DIR) -> tuple[dict[str, list[AlignedPair]], dict[str, int]]:
    """Read every CSV, align what aligns, and bucket the result by split.

    Returns:
        A `(splits, stats)` pair. `splits` maps split name to its aligned
        pairs; `stats` counts essays seen, corrected, aligned and dropped, for
        the caller to report honestly rather than silently.
    """
    splits: dict[str, list[AlignedPair]] = {name: [] for name, _ in SPLIT_RATIOS}
    stats = {"essays": 0, "corrected": 0, "aligned": 0, "dropped_mismatch": 0}

    for essay in read_essays(csv_dir):
        stats["essays"] += 1
        if not essay.corrected1:
            continue
        stats["corrected"] += 1

        pairs = align_essay(essay)
        if pairs is None:
            stats["dropped_mismatch"] += 1
            continue

        stats["aligned"] += 1
        split = assign_split(essay.learner_id)
        splits[split].extend(pairs)

    return splits, stats


def write_pairs(splits: dict[str, list[AlignedPair]]) -> dict[str, Path]:
    """Write each split's aligned pairs to JSONL under `data/interim/cowsl2h/`."""
    INTERIM_DIR_FOR_CORPUS.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, pairs in splits.items():
        path = INTERIM_DIR_FOR_CORPUS / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for pair in pairs:
                fh.write(json.dumps(_asdict(pair), ensure_ascii=False))
                fh.write("\n")
        written[name] = path
    return written


def _asdict(pair: AlignedPair) -> dict[str, Any]:
    return {
        "learner_id": pair.learner_id,
        "prompt": pair.prompt,
        "quarter": pair.quarter,
        "source_file": pair.source_file,
        "sentence_index": pair.sentence_index,
        "source": pair.source,
        "target": pair.target,
        "target2": pair.target2,
    }


def read_pairs(split: str) -> Iterator[AlignedPair]:
    """Read one split's aligned pairs back from `data/interim/cowsl2h/`."""
    path = INTERIM_DIR_FOR_CORPUS / f"{split}.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            yield AlignedPair(**row)


def split_files() -> dict[str, Path]:
    """Paths of the typed M2 splits, without building anything.

    Unlike `falko_merlin.split_files`, these are not a copy of an upstream
    release: `scripts/build_cowsl2h_m2.py` writes them by running `errant_es`
    over the aligned pairs `write_pairs` produced.

    Raises:
        FileNotFoundError: if the corpus has not been built yet.
    """
    paths = {name: PROCESSED_DIR_FOR_CORPUS / f"{name}.m2" for name, _ in SPLIT_RATIOS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Run `python scripts/download_cowsl2h.py` then "
            f"`python scripts/build_cowsl2h_m2.py` first; missing: {missing}"
        )
    return paths
