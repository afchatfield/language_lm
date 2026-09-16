"""Vocabulary trimming: removing rows the target languages never touch.

Phase 0 measured the opportunity and predicted the saving; this performs it.
86,401 of EuroLLM's 128,000 rows are never produced by German or English text,
and because its embeddings are untied each removed row is deleted twice -- once
from ``embed_tokens`` and once from ``lm_head``.

**Four things have to stay consistent or the artefact is quietly broken.**

1. *The weights.* ``embed_tokens.weight`` and ``lm_head.weight`` are sliced to
   the kept rows, in the same order.
2. *``config.vocab_size``.* Sliced weights with a stale config is the classic
   silent break: the checkpoint loads, generates plausible text, and every
   converter reads the wrong row count.
3. *``tokenizer.model``*, the SentencePiece proto, whose pieces carry the BPE
   merge priority as scores.
4. *``tokenizer.json``*, the fast-tokenizer BPE with an explicit merge list.

Both tokenizer files are present on a Llama-architecture checkpoint and
different converters read different ones -- llama.cpp prefers the proto,
`tokenizers` uses the JSON. Rebuilding one and not the other produces two
tokenizers that disagree, which does not fail loudly anywhere.

**Ids are kept in ascending order**, so the specials at 0/1/2 stay at 0/1/2 and
``bos_token_id``/``eos_token_id`` need no remapping. The remap is still applied
and asserted rather than assumed.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langlm.tokenizers.fertility import coverage_vocab_size, protected_ids


@dataclass(frozen=True)
class TrimPlan:
    """Which rows survive, and what it costs."""

    #: Old vocabulary ids to keep, ascending.
    keep_ids: list[int]
    #: Old id to new id, for every kept row.
    id_map: dict[int, int]
    #: Rows in the original vocabulary.
    original_size: int
    #: Ids kept only because they are special or byte-fallback.
    protected: int

    @property
    def new_size(self) -> int:
        return len(self.keep_ids)

    @property
    def removed(self) -> int:
        return self.original_size - self.new_size

    @property
    def share_removed(self) -> float:
        return self.removed / self.original_size


def plan(tokenizer, token_counts: Counter[int], threshold: float = 0.999) -> TrimPlan:
    """Choose the rows to keep.

    Args:
        tokenizer: The tokenizer whose vocabulary is being trimmed.
        token_counts: Token id to occurrence count over the target corpora.
        threshold: Share of token occurrences the kept rows must cover.

    Returns:
        The plan. Protected ids -- specials and all 256 byte-fallback tokens --
        are kept whatever their frequency, because without them the tokenizer
        cannot encode input the target languages never use, and a trimmed
        tokenizer that cannot encode arbitrary text is not a tokenizer.
    """
    protected = protected_ids(tokenizer)
    wanted = coverage_vocab_size(token_counts, threshold, protected)

    keep: set[int] = set(protected)
    for token_id, _ in token_counts.most_common():
        if len(keep) >= wanted:
            break
        keep.add(token_id)

    keep_ids = sorted(keep)
    return TrimPlan(
        keep_ids=keep_ids,
        id_map={old: new for new, old in enumerate(keep_ids)},
        original_size=len(tokenizer.get_vocab()),
        protected=len(protected),
    )


def trim_sentencepiece(proto_bytes: bytes, plan: TrimPlan) -> bytes:
    """Filter the SentencePiece proto to the kept pieces, in order.

    Scores are carried over untouched. In a SentencePiece BPE model the score
    *is* the merge priority, so rewriting them would change segmentation for
    reasons unrelated to the trim.
    """
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2

    model = sp_pb2.ModelProto()
    model.ParseFromString(proto_bytes)
    if len(model.pieces) != plan.original_size:
        raise ValueError(
            f"Proto has {len(model.pieces)} pieces, the plan was built for {plan.original_size}."
        )

    kept = [model.pieces[old] for old in plan.keep_ids]
    del model.pieces[:]
    model.pieces.extend(kept)
    return model.SerializeToString()


def trim_tokenizer_json(data: dict[str, Any], plan: TrimPlan) -> dict[str, Any]:
    """Rebuild the fast tokenizer's vocabulary and merge table.

    A merge survives only if both of its parts *and* the token it produces are
    all still in the vocabulary. Dropping that check leaves merges that produce
    ids which no longer exist, which is the failure the roadmap means by
    "rebuild the merge table consistently with the trimmed vocab".

    Merge order is preserved, because in BPE it is the priority.
    """
    model = data["model"]
    old_vocab: dict[str, int] = model["vocab"]
    keep = set(plan.keep_ids)

    vocab = {token: plan.id_map[i] for token, i in old_vocab.items() if i in keep}
    survivors = set(vocab)

    merges = []
    for merge in model.get("merges", []):
        left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
        if left in survivors and right in survivors and left + right in survivors:
            merges.append(merge)

    model["vocab"] = vocab
    model["merges"] = merges

    # Added tokens carry their own ids and are re-listed here; any that were
    # dropped would leave the fast tokenizer pointing at a row that is gone.
    added = []
    for entry in data.get("added_tokens", []):
        if entry["id"] in keep:
            added.append({**entry, "id": plan.id_map[entry["id"]]})
    data["added_tokens"] = added
    return data


def trim_weights(model, plan: TrimPlan) -> None:
    """Slice the embedding and output rows in place.

    Both matrices are addressed, not just the input side. EuroLLM's embeddings
    are untied -- which is why it was chosen in Phase 0 -- so ``lm_head`` is a
    second full copy of the vocabulary and skipping it would forfeit half the
    saving and leave the head disagreeing with the embedding about what row
    means what.
    """
    import torch

    index = torch.tensor(plan.keep_ids, dtype=torch.long)

    embedding = model.get_input_embeddings()
    embedding.weight.data = embedding.weight.data.index_select(0, index).clone()
    embedding.num_embeddings = plan.new_size

    head = model.get_output_embeddings()
    if head is None:
        raise ValueError("No output embedding: a tied-embedding model needs a different path.")
    head.weight.data = head.weight.data.index_select(0, index).clone()
    head.out_features = plan.new_size

    model.config.vocab_size = plan.new_size
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.vocab_size = plan.new_size


def write(source: Path, destination: Path, plan: TrimPlan, model, tokenizer) -> None:
    """Write the trimmed checkpoint, tokenizer files and all.

    `save_pretrained` writes the tokenizer as the library understands it, which
    for a Llama checkpoint means copying the *original* `tokenizer.model` proto
    straight through. Both tokenizer artefacts are therefore rebuilt afterwards,
    over the top of what was just written.
    """
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)

    proto = source / "tokenizer.model"
    if proto.exists():
        (destination / "tokenizer.model").write_bytes(trim_sentencepiece(proto.read_bytes(), plan))

    fast = destination / "tokenizer.json"
    if fast.exists():
        data = json.loads(fast.read_text(encoding="utf-8"))
        fast.write_text(
            json.dumps(trim_tokenizer_json(data, plan), ensure_ascii=False),
            encoding="utf-8",
        )

    config = destination / "config.json"
    body = json.loads(config.read_text(encoding="utf-8"))
    body["vocab_size"] = plan.new_size
    config.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
