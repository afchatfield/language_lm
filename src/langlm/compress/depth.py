"""Depth pruning: dropping transformer layers that barely change their input.

The vocabulary trim is free -- 21.9% of parameters for 0.0011 of F0.5 -- but it
stops at 755 MB against a 600 MB target. Everything left has to come from depth,
and depth is not free: a vocabulary row the target language never emits is dead
weight, whereas every layer is on the path of every token.

**The signal.** In a residual network each layer computes ``x + f(x)``. A layer
whose output points almost exactly where its input already pointed has added
little, and the cosine similarity between a block's input and output measures
exactly that. Layers with the highest similarity are the cheapest to remove.

This is the Gromov et al. / "ShortGPT" observation and it is a *heuristic*, not a
theorem: similarity is measured one layer at a time on a frozen model, while
removing a layer changes what every later layer sees. So the ranking proposes
candidates and the quality-versus-size curve decides, which is why the roadmap
asks for a sweep rather than a single point.

**Contiguity.** Removing a contiguous block is the variant that reports well and
the one this measures, because the residual stream's representation drifts
gradually with depth: two individually-redundant layers taken from opposite ends
of the network remove two *different* transformations, while a contiguous run
removes one slightly longer version of the same one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerScore:
    """How little one layer changed the residual stream."""

    index: int
    #: Mean cosine similarity between the block's input and its output.
    similarity: float

    @property
    def redundancy(self) -> float:
        """Higher means safer to drop."""
        return self.similarity


@dataclass(frozen=True)
class Window:
    """A contiguous run of layers considered for removal."""

    start: int
    count: int
    #: Mean similarity across the window, which is what ranks it.
    similarity: float

    @property
    def end(self) -> int:
        return self.start + self.count

    @property
    def layers(self) -> list[int]:
        return list(range(self.start, self.end))


def measure(model, tokenizer, texts, device: str = "cuda", batch_size: int = 8) -> list[LayerScore]:
    """Mean input-output cosine similarity for every decoder layer.

    Hidden states are taken with ``output_hidden_states``, which returns the
    embedding output followed by one tensor per layer -- so ``hidden[i]`` is
    layer ``i``'s input and ``hidden[i + 1]`` its output.

    Padding is masked out. Without that, a batch of short sentences is mostly
    padding and every layer looks redundant, because a layer really does leave
    the representation of a pad token almost untouched.
    """
    import torch

    model.eval()
    layers = model.config.num_hidden_layers
    totals = [0.0] * layers
    counts = [0] * layers

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)

        mask = encoded["attention_mask"].bool()
        for index in range(layers):
            before = out.hidden_states[index]
            after = out.hidden_states[index + 1]
            similarity = torch.nn.functional.cosine_similarity(before, after, dim=-1)
            selected = similarity[mask]
            totals[index] += selected.sum().item()
            counts[index] += selected.numel()

    return [
        LayerScore(index=i, similarity=totals[i] / counts[i] if counts[i] else 0.0)
        for i in range(layers)
    ]


def best_window(scores: list[LayerScore], count: int, protect_edges: int = 1) -> Window:
    """The most redundant contiguous run of `count` layers.

    Args:
        scores: Per-layer similarity, in layer order.
        count: How many layers to drop.
        protect_edges: Layers at each end that are never dropped. The first
            block turns embeddings into something the rest of the stack expects
            and the last one prepares the representation the LM head reads;
            both routinely score as redundant on this metric and are not.

    Returns:
        The window with the highest mean similarity.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    low, high = protect_edges, len(scores) - protect_edges
    if high - low < count:
        raise ValueError(f"Cannot drop {count} layers from {len(scores)} with edges protected.")

    best = None
    for start in range(low, high - count + 1):
        window = scores[start : start + count]
        mean = sum(s.similarity for s in window) / count
        if best is None or mean > best.similarity:
            best = Window(start=start, count=count, similarity=mean)
    assert best is not None
    return best


def prune(model, window: Window):
    """Remove a window of decoder layers, in place.

    The surviving layers are renumbered, because `transformers` addresses them
    by position and a gap in `layer_idx` breaks the KV cache rather than merely
    looking untidy.
    """
    import torch

    layers = model.model.layers
    kept = [layer for i, layer in enumerate(layers) if i not in set(window.layers)]
    for position, layer in enumerate(kept):
        layer.self_attn.layer_idx = position

    model.model.layers = torch.nn.ModuleList(kept)
    model.config.num_hidden_layers = len(kept)
    return model
