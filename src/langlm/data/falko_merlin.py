"""The Falko-MERLIN German learner corpus.

Boyd (2018), *Using Wikipedia Edits in Low Resource Grammatical Error
Correction* (WNUT), released a merged and automatically annotated version of two
German learner corpora:

* **Falko** -- minimally corrected essays by advanced learners (C1-C2)
* **MERLIN** -- standardised exam scripts spanning A1-C1

Edits were extracted and typed with a German adaptation of ERRANT, giving the
familiar ``M:``/``R:``/``U:`` (missing / replace / unnecessary) tags. Sentences
that needed no correction carry an explicit ``noop`` marker, which is exactly
the already-correct signal Phase 2 needs to keep the model from overcorrecting.

The corpus ships as ``fm-{train,dev,test}.m2`` inside the release archive. This
module copies those into a stable ``data/processed/falko_merlin/`` layout so
that nothing downstream depends on the upstream archive's internal paths.

Licensing: Falko is CC BY 3.0, MERLIN is CC BY-SA 4.0, and the release bundles
the CC BY-SA 4.0 text. The derived data is redistributable under CC BY-SA 4.0
with attribution; it is not committed to this repository.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langlm.config import FALKO_MERLIN_DIR, PROCESSED_DIR
from langlm.data.download import download, extract

CORPUS = "falko_merlin"

RELEASE_URL = "https://github.com/adrianeboyd/boyd-wnut2018/releases/download/wnut2018/data.tar.gz"
CITATION_URL = "https://aclanthology.org/W18-6111/"

#: Split name -> filename inside the release archive.
ARCHIVE_MEMBERS = {
    "train": "fm-train.m2",
    "dev": "fm-dev.m2",
    "test": "fm-test.m2",
}

PROCESSED_DIR_FOR_CORPUS = PROCESSED_DIR / CORPUS


def download_and_prepare(force: bool = False) -> dict[str, Path]:
    """Download the release, extract the M2 files, and normalise their layout.

    Args:
        force: Re-download and re-extract even if the files are already present.

    Returns:
        Mapping of split name to the prepared ``.m2`` path.

    Raises:
        FileNotFoundError: if the archive does not contain the expected splits,
            which would mean the upstream release changed shape.
    """
    archive = download(RELEASE_URL, FALKO_MERLIN_DIR / "data.tar.gz", force=force)
    extract(archive, FALKO_MERLIN_DIR)

    extracted = FALKO_MERLIN_DIR / "data"
    if not extracted.is_dir():
        raise FileNotFoundError(f"Expected {extracted} after extracting {archive}")

    PROCESSED_DIR_FOR_CORPUS.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, Path] = {}

    for split, member in ARCHIVE_MEMBERS.items():
        source = extracted / member
        if not source.exists():
            available = sorted(p.name for p in extracted.glob("*.m2"))
            raise FileNotFoundError(
                f"{member} is missing from the release archive. Found: {available}. "
                f"The upstream release layout may have changed."
            )
        target = PROCESSED_DIR_FOR_CORPUS / f"{split}.m2"
        if force or not target.exists():
            shutil.copyfile(source, target)
        prepared[split] = target

    # Keep the licence next to the data it governs.
    licence = extracted / "LICENSE"
    if licence.exists():
        shutil.copyfile(licence, PROCESSED_DIR_FOR_CORPUS / "LICENSE")

    return prepared


def split_files() -> dict[str, Path]:
    """Paths of the prepared splits, without downloading anything.

    Raises:
        FileNotFoundError: if the corpus has not been prepared yet.
    """
    paths = {s: PROCESSED_DIR_FOR_CORPUS / f"{s}.m2" for s in ARCHIVE_MEMBERS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run `make data` first; missing: {missing}")
    return paths
