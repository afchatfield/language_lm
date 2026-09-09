"""Tests for the M2 reader and writer.

The M2 parser underpins every evaluation number this project will report, so it
is tested against real corpus shapes: multiple annotators, insertions,
deletions, and explicit noop markers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from langlm.data.m2 import (
    Edit,
    M2ParseError,
    M2Sentence,
    error_type_histogram,
    parse_m2,
    write_m2,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.m2"


@pytest.fixture
def sentences() -> list[M2Sentence]:
    return list(parse_m2(FIXTURE))


def test_parses_every_block(sentences):
    assert len(sentences) == 4
    assert sentences[0].source == "Das ist ein falsches autos ."


def test_applies_edits_in_the_right_order(sentences):
    # Two edits on the same sentence, listed out of positional order in the
    # file: applying them must not depend on the order they appear.
    assert sentences[0].target() == "Das ist ein falsche Autos ."


def test_handles_insertion_and_deletion(sentences):
    # `U:ADV` deletes "sehr"; `M:PUNCT` inserts "heute" before the full stop.
    assert sentences[3].target() == "Ich habe viel gearbeitet heute ."


def test_noop_means_already_correct(sentences):
    assert sentences[2].is_correct()
    assert sentences[2].target() == sentences[2].source
    assert sentences[2].edits_for() == []
    # The marker itself is still available when explicitly asked for.
    assert len(sentences[2].edits_for(include_noop=True)) == 1


def test_multiple_annotators_are_kept_apart(sentences):
    sentence = sentences[1]
    assert sentence.annotators == [0, 1]
    assert sentence.target(annotator=0) == "Ich gehe nach Hause ."
    assert sentence.target(annotator=1) == sentence.source
    assert sentence.is_correct(annotator=1)


def test_round_trips_byte_for_byte(sentences, tmp_path: Path):
    output = tmp_path / "out.m2"
    write_m2(sentences, output)
    assert output.read_text(encoding="utf-8") == FIXTURE.read_text(encoding="utf-8")


def test_edit_classification():
    assert Edit(3, 3, "M:DET", "die").is_insertion
    assert Edit(3, 4, "U:ADV", "").is_deletion
    assert Edit(-1, -1, "noop", "-NONE-").is_noop
    assert not Edit(3, 4, "R:ORTH", "Autos").is_insertion


def test_overlapping_edits_are_rejected():
    sentence = M2Sentence(
        source_tokens=["a", "b", "c"],
        edits=[Edit(0, 2, "R:OTHER", "x"), Edit(1, 3, "R:OTHER", "y")],
    )
    with pytest.raises(M2ParseError, match="Overlapping"):
        sentence.apply()


def test_malformed_line_names_the_file_and_line(tmp_path: Path):
    bad = tmp_path / "bad.m2"
    bad.write_text("S ok\nA 1 2|||too|||few\n", encoding="utf-8")
    with pytest.raises(M2ParseError, match=r"bad\.m2:2"):
        list(parse_m2(bad))


def test_histogram_excludes_noops(sentences):
    histogram = error_type_histogram(sentences)
    assert histogram["R:ORTH"] == 1
    assert histogram["noop"] == 0
    assert sum(histogram.values()) == 5
