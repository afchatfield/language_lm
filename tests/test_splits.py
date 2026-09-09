"""Tests for split freezing and the held-out-data guard.

The roadmap's central methodological rule is that the test split must not
influence any decision before final evaluation. These tests assert that the rule
is enforced by the code rather than by memory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from langlm.data.splits import (
    HeldOutSplitError,
    ManifestError,
    freeze_splits,
    load_split,
    sha256_file,
    split_path,
    verify_manifest,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.m2"


@pytest.fixture
def frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A manifest over train/dev/test copies of the fixture, rooted in tmp_path."""
    # Records store paths relative to the project root, so the tests must
    # temporarily treat tmp_path as that root.
    monkeypatch.setattr("langlm.data.splits.PROJECT_ROOT", tmp_path)

    data = tmp_path / "data"
    data.mkdir()
    paths = {}
    for split in ("train", "dev", "test"):
        target = data / f"{split}.m2"
        shutil.copyfile(FIXTURE, target)
        paths[split] = target

    manifest = tmp_path / "manifest.json"
    freeze_splits("demo", paths, source_url="https://example.invalid", manifest_path=manifest)
    return manifest


def test_freeze_records_hashes_and_counts(frozen: Path):
    manifest = json.loads(frozen.read_text())
    entry = manifest["corpora"]["demo"]
    assert entry["source_url"] == "https://example.invalid"
    assert set(entry["splits"]) == {"train", "dev", "test"}
    record = entry["splits"]["train"]
    assert record["sentences"] == 4
    assert record["sha256"] == sha256_file(FIXTURE)
    # Paths are stored relative, so the manifest survives being moved.
    assert not Path(record["path"]).is_absolute()


def test_verify_passes_on_untouched_data(frozen: Path):
    verify_manifest(frozen)


def test_verify_detects_tampering(frozen: Path, tmp_path: Path):
    target = tmp_path / "data" / "dev.m2"
    target.write_text(target.read_text() + "S sneaky edit\n\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="sha256 changed"):
        verify_manifest(frozen)


def test_verify_detects_a_missing_file(frozen: Path, tmp_path: Path):
    (tmp_path / "data" / "test.m2").unlink()
    with pytest.raises(ManifestError, match="missing file"):
        verify_manifest(frozen)


def test_test_split_refuses_to_load_without_opt_in(frozen: Path):
    with pytest.raises(HeldOutSplitError, match="held out"):
        load_split("demo", "test", manifest_path=frozen)


def test_test_split_loads_with_explicit_opt_in(frozen: Path):
    assert len(load_split("demo", "test", allow_test=True, manifest_path=frozen)) == 4


def test_development_splits_load_freely(frozen: Path):
    assert len(load_split("demo", "train", manifest_path=frozen)) == 4
    assert len(load_split("demo", "dev", manifest_path=frozen)) == 4


def test_refreezing_an_existing_corpus_is_refused(frozen: Path, tmp_path: Path):
    paths = {"train": tmp_path / "data" / "train.m2"}
    with pytest.raises(ManifestError, match="already frozen"):
        freeze_splits("demo", paths, manifest_path=frozen)


def test_unknown_split_lists_what_exists(frozen: Path):
    with pytest.raises(ManifestError, match="demo/dev"):
        split_path("demo", "validation", manifest_path=frozen)


def test_missing_manifest_points_at_the_fix(tmp_path: Path):
    with pytest.raises(ManifestError, match="make data"):
        verify_manifest(tmp_path / "nope.json")
