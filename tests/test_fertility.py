"""Tests for the fertility metrics and the vocabulary-trim projection.

Everything here runs against a stub tokenizer. The arithmetic is what decides
the base model, so it is worth testing on inputs whose answers can be worked out
by hand -- and worth being able to run in CI without network access.
"""

from __future__ import annotations

from collections import Counter

import pytest

from langlm.tokenizers.fertility import (
    ModelSpec,
    analyse_corpus,
    coverage_vocab_size,
    project_trim,
    protected_ids,
)


class StubTokenizer:
    """Splits on whitespace, then emits one id per three characters.

    So "abcdef" is two tokens and "ab" is one, which makes the expected
    fertility of any given sentence easy to compute by hand.
    """

    def __init__(self) -> None:
        self._vocab = {"<pad>": 0, "<eos>": 1, "a": 2, "€": 3, "<0x41>": 4, "hallo": 5}
        self.all_special_ids = [0, 1]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split():
            pieces = max(1, -(-len(word) // 3))  # ceiling division
            ids.extend(hash((word, i)) % 1000 for i in range(pieces))
        return ids

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


SPEC_TIED = ModelSpec(
    name="tied",
    repo_id="x/tied",
    params=1_000_000_000,
    vocab_size=100_000,
    hidden_size=1_000,
    num_layers=10,
    tie_word_embeddings=True,
)
SPEC_UNTIED = ModelSpec(
    name="untied",
    repo_id="x/untied",
    params=1_000_000_000,
    vocab_size=100_000,
    hidden_size=1_000,
    num_layers=10,
    tie_word_embeddings=False,
)


def test_untied_embeddings_cost_twice_as_much():
    assert SPEC_TIED.embedding_params == 100_000_000
    assert SPEC_UNTIED.embedding_params == 200_000_000
    assert SPEC_TIED.embedding_share == pytest.approx(0.10)
    assert SPEC_UNTIED.embedding_share == pytest.approx(0.20)


def test_fertility_counts_tokens_per_word():
    # "abcdef" -> 2 tokens, "gh" -> 1 token, "ijklmnopq" -> 3 tokens.
    stats = analyse_corpus(StubTokenizer(), ["abcdef gh ijklmnopq"], "demo", "de")
    assert stats.sentences == 1
    assert stats.words == 3
    assert stats.tokens == 6
    assert stats.fertility == pytest.approx(2.0)


def test_split_word_rate_counts_occurrences_not_types():
    # "gh" is one token and appears twice; "abcdef" is two tokens.
    stats = analyse_corpus(StubTokenizer(), ["gh gh abcdef"], "demo", "de")
    assert stats.words == 3
    assert stats.multi_token_words == 1
    assert stats.split_word_rate == pytest.approx(1 / 3)


def test_bytes_per_token_uses_utf8_length():
    # "üü" is 2 characters but 4 UTF-8 bytes, and one token.
    stats = analyse_corpus(StubTokenizer(), ["üü"], "demo", "de")
    assert stats.tokens == 1
    assert stats.chars_per_token == pytest.approx(2.0)
    assert stats.bytes_per_token == pytest.approx(4.0)


def test_blank_sentences_are_skipped():
    stats = analyse_corpus(StubTokenizer(), ["abc", "   ", "", "def"], "demo", "de")
    assert stats.sentences == 2


def test_max_sentences_caps_the_sample():
    stats = analyse_corpus(StubTokenizer(), ["abc"] * 100, "demo", "de", max_sentences=7)
    assert stats.sentences == 7


def test_coverage_takes_the_frequent_tail_first():
    counts = Counter({1: 90, 2: 9, 3: 1})
    # 90/100 already clears 0.9, so one token suffices.
    assert coverage_vocab_size(counts, 0.9) == 1
    # 0.95 needs the second most frequent as well.
    assert coverage_vocab_size(counts, 0.95) == 2
    assert coverage_vocab_size(counts, 1.0) == 3


def test_coverage_always_keeps_protected_ids():
    counts = Counter({1: 100})
    assert coverage_vocab_size(counts, 0.5, protected={7, 8, 9}) == 4


def test_trim_projection_scales_with_untied_embeddings():
    counts = Counter({i: 1 for i in range(1_000)})
    tied = project_trim(SPEC_TIED, counts, 1.0)
    untied = project_trim(SPEC_UNTIED, counts, 1.0)

    assert tied.kept_vocab == 1_000
    assert tied.removed_vocab == 99_000
    assert tied.removed_params == 99_000 * 1_000
    assert untied.removed_params == 2 * tied.removed_params
    assert untied.removed_bytes == 2 * untied.removed_params  # bf16
    assert untied.model_fraction == pytest.approx(0.198)


def test_trim_never_keeps_more_than_the_configured_vocab():
    # More distinct ids observed than the config declares: clamp rather than
    # report a negative saving.
    counts = Counter({i: 1 for i in range(200_000)})
    projection = project_trim(SPEC_TIED, counts, 1.0)
    assert projection.kept_vocab == SPEC_TIED.vocab_size
    assert projection.removed_vocab == 0


def test_protected_ids_shield_specials_and_byte_fallbacks_only():
    keep = protected_ids(StubTokenizer())
    assert 0 in keep and 1 in keep  # specials
    assert 4 in keep  # <0x41> byte fallback
    assert 2 in keep  # "a" is in the byte-level alphabet
    assert 3 not in keep  # "€" is a single char but not a byte stand-in
    assert 5 not in keep  # ordinary multi-character token
