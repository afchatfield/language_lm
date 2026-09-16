"""Depth pruning: picking the window, and renumbering what survives."""

from __future__ import annotations

import pytest

from langlm.compress.depth import LayerScore, best_window


def scores(*values: float) -> list[LayerScore]:
    return [LayerScore(index=i, similarity=v) for i, v in enumerate(values)]


def test_picks_the_most_redundant_contiguous_run() -> None:
    # Layers 3 and 4 barely change their input; everything else does.
    result = best_window(scores(0.1, 0.2, 0.2, 0.9, 0.95, 0.2, 0.1), count=2)
    assert result.layers == [3, 4]
    assert result.similarity == pytest.approx(0.925)


def test_the_window_is_contiguous_even_when_scattered_layers_score_higher() -> None:
    """Two redundant layers at opposite ends remove two different transformations.

    The residual stream drifts gradually with depth, so a contiguous run removes
    one slightly longer version of the same transformation. Layers 1 and 6 are
    the two highest-scoring layers here and must still not be chosen together;
    the adjacent pair 3-4 wins despite neither being in the top two.
    """
    values = scores(0.1, 0.99, 0.1, 0.80, 0.85, 0.1, 0.99, 0.1)
    assert sorted(values, key=lambda s: -s.similarity)[:2] == [values[1], values[6]]

    result = best_window(values, count=2)
    assert result.layers == [3, 4]


def test_edges_are_protected() -> None:
    """The first and last blocks score as redundant on this metric and are not."""
    result = best_window(scores(0.99, 0.99, 0.2, 0.3, 0.4, 0.99, 0.99), count=2)
    assert 0 not in result.layers
    assert 6 not in result.layers


def test_protect_edges_can_be_widened() -> None:
    """Two layers at each end are then off limits however redundant they look."""
    result = best_window(scores(0.9, 0.9, 0.1, 0.2, 0.3, 0.9, 0.9), count=2, protect_edges=2)
    assert set(result.layers) <= {2, 3, 4}


def test_window_end_and_layers_agree() -> None:
    result = best_window(scores(0.1, 0.5, 0.6, 0.7, 0.1), count=2)
    assert result.layers == list(range(result.start, result.end))
    assert len(result.layers) == result.count


def test_asking_for_more_layers_than_fit_is_an_error() -> None:
    with pytest.raises(ValueError, match="Cannot drop"):
        best_window(scores(0.1, 0.2, 0.3, 0.4), count=3)


def test_a_non_positive_count_is_an_error() -> None:
    with pytest.raises(ValueError, match="positive"):
        best_window(scores(0.1, 0.2, 0.3), count=0)


def test_surviving_layers_are_renumbered() -> None:
    """A gap in `layer_idx` breaks the KV cache rather than merely looking untidy."""
    torch = pytest.importorskip("torch")
    from langlm.compress.depth import Window, prune

    class Attn:
        def __init__(self, idx):
            self.layer_idx = idx

    class Layer(torch.nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.self_attn = Attn(idx)
            self.tag = idx

    class Inner(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(i) for i in range(n)])

    class Model(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.model = Inner(n)
            self.config = type("C", (), {"num_hidden_layers": n})()

    model = prune(Model(6), Window(start=2, count=2, similarity=0.9))

    assert model.config.num_hidden_layers == 4
    assert [layer.tag for layer in model.model.layers] == [0, 1, 4, 5]
    assert [layer.self_attn.layer_idx for layer in model.model.layers] == [0, 1, 2, 3]
