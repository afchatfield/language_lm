"""Vocabulary trimming: the four artefacts have to agree or the break is silent."""

from __future__ import annotations

from collections import Counter

from langlm.compress.vocab import TrimPlan, plan, trim_tokenizer_json


class FakeTokenizer:
    """Enough of a tokenizer for the planner: a vocabulary and some specials."""

    def __init__(self, vocab: dict[str, int], special: list[int]):
        self._vocab = vocab
        self.all_special_ids = special

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def tokenizer_with(*tokens: str, special: int = 1) -> FakeTokenizer:
    vocab = {token: index for index, token in enumerate(tokens)}
    return FakeTokenizer(vocab, list(range(special)))


def test_protected_ids_survive_whatever_their_frequency() -> None:
    """A trimmed tokenizer that cannot encode arbitrary text is not a tokenizer."""
    tok = tokenizer_with("<unk>", "<0x41>", "<0x42>", "hallo", "welt")
    result = plan(tok, Counter({3: 1000}), threshold=0.5)

    kept = set(result.keep_ids)
    assert 0 in kept, "special token dropped"
    assert {1, 2} <= kept, "byte-fallback tokens dropped"
    assert 3 in kept


def test_ids_are_remapped_contiguously_and_in_order() -> None:
    """Ascending order is what keeps bos/eos at 1 and 2 without a fixup."""
    tok = tokenizer_with("<unk>", "<0x41>", "a", "b", "c")
    result = plan(tok, Counter({4: 100, 2: 50}), threshold=1.0)

    assert result.keep_ids == sorted(result.keep_ids)
    assert list(result.id_map.values()) == list(range(result.new_size))
    assert result.id_map[result.keep_ids[0]] == 0


def test_share_removed_counts_rows_not_tokens() -> None:
    result = TrimPlan(keep_ids=[0, 1], id_map={0: 0, 1: 1}, original_size=10, protected=2)
    assert result.new_size == 2
    assert result.removed == 8
    assert result.share_removed == 0.8


def tokenizer_json(vocab: dict[str, int], merges: list[str], added=()) -> dict:
    return {"model": {"vocab": vocab, "merges": merges}, "added_tokens": list(added)}


def test_a_merge_dies_with_the_token_it_produces() -> None:
    """The consistency the roadmap means by "rebuild the merge table".

    `a b -> ab` has both parts still present, but `ab` itself was trimmed. Keeping
    the merge would leave the tokenizer able to produce a token that has no row.
    """
    data = tokenizer_json({"a": 0, "b": 1, "ab": 2}, ["a b"])
    result = plan(tokenizer_with("a", "b", "ab", special=0), Counter({0: 10, 1: 10}), threshold=1.0)
    assert 2 not in set(result.keep_ids)

    trimmed = trim_tokenizer_json(data, result)
    assert trimmed["model"]["merges"] == []
    assert set(trimmed["model"]["vocab"]) == {"a", "b"}


def test_a_merge_survives_when_all_three_parts_do() -> None:
    keep = TrimPlan(keep_ids=[0, 1, 2], id_map={0: 0, 1: 1, 2: 2}, original_size=3, protected=0)
    data = tokenizer_json({"a": 0, "b": 1, "ab": 2}, ["a b"])
    assert trim_tokenizer_json(data, keep)["model"]["merges"] == ["a b"]


def test_a_merge_dies_with_either_of_its_parts() -> None:
    dropped_right = TrimPlan(keep_ids=[0, 2], id_map={0: 0, 2: 1}, original_size=3, protected=0)
    data = tokenizer_json({"a": 0, "b": 1, "ab": 2}, ["a b"])
    assert trim_tokenizer_json(data, dropped_right)["model"]["merges"] == []


def test_merge_order_is_preserved() -> None:
    """In BPE the order of the merge list is the priority, not a detail."""
    keep = TrimPlan(
        keep_ids=[0, 1, 2, 3, 4],
        id_map={i: i for i in range(5)},
        original_size=5,
        protected=0,
    )
    merges = ["a b", "ab c", "b c"]
    data = tokenizer_json({"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}, merges)
    assert trim_tokenizer_json(data, keep)["model"]["merges"] == ["a b", "ab c"]


def test_added_tokens_are_remapped_to_their_new_ids() -> None:
    """A stale added-token id points the fast tokenizer at a row that is gone."""
    keep = TrimPlan(keep_ids=[0, 2], id_map={0: 0, 2: 1}, original_size=3, protected=0)
    data = tokenizer_json(
        {"<unk>": 0, "b": 1, "</s>": 2},
        [],
        added=[{"id": 0, "content": "<unk>"}, {"id": 2, "content": "</s>"}],
    )
    added = trim_tokenizer_json(data, keep)["added_tokens"]
    assert [(entry["id"], entry["content"]) for entry in added] == [(0, "<unk>"), (1, "</s>")]


def test_vocab_and_merges_stay_consistent_after_trimming() -> None:
    """The invariant the whole module exists to hold."""
    keep = TrimPlan(keep_ids=[0, 1, 3], id_map={0: 0, 1: 1, 3: 2}, original_size=4, protected=0)
    data = tokenizer_json({"a": 0, "b": 1, "ab": 2, "ba": 3}, ["a b", "b a"])
    trimmed = trim_tokenizer_json(data, keep)

    survivors = set(trimmed["model"]["vocab"])
    for merge in trimmed["model"]["merges"]:
        left, right = merge.split(" ", 1)
        assert left in survivors and right in survivors
        assert left + right in survivors
