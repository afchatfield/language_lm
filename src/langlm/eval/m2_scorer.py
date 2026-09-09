"""MaxMatch (M2) scoring: the standard F0.5 metric for grammatical error correction.

A GEC system emits a *corrected sentence*, not a list of edits, so scoring has to
recover the edits before it can compare them with the gold annotation. There are
many ways to describe the same rewrite -- ``[der -> die]`` and
``[der Frau -> die Frau]`` produce identical output -- and MaxMatch
(Dahlmeier and Ng, 2012) resolves that by choosing, among all edit sequences that
turn the source into the hypothesis, the one that overlaps the gold annotation
most. A system is not penalised for describing a correct rewrite differently from
the annotator.

The algorithm here follows the original in structure:

1. Build the lattice of *every optimal* Levenshtein alignment between the source
   and the hypothesis. Substitution costs 2, the same as a deletion plus an
   insertion, so both readings of a rewrite stay in the lattice instead of one
   being arbitrarily preferred.
2. Add phrase-level arcs by composing runs of adjacent token-level arcs, so that
   ``[a b -> c d]`` can be recovered as one edit rather than two. A composition
   may swallow at most ``max_unchanged`` matching tokens, which is what stops the
   whole sentence collapsing into a single edit.
3. Find the cheapest path through that lattice, where cost is the pair
   ``(-matched, unmatched)`` compared lexicographically: maximise the number of
   gold edits recovered first, and among the ties prefer the description that
   invents the fewest extra edits.

Step 3 is where this differs from the reference implementation, which encodes the
same preference as float weights with an epsilon term. Comparing the pair exactly
means the tie-break cannot be swamped by a long sentence, and removes the only
tunable constant in the scorer.

The precision-weighted F0.5 is the field's convention, and the right one here:
a grammar checker that invents corrections is worse than one that misses them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from langlm.data.m2 import Edit, M2Sentence

#: A lattice vertex: (tokens of the source consumed, tokens of the hypothesis consumed).
Vertex = tuple[int, int]

#: A system edit: a half-open source token span and the text that replaces it.
SystemEdit = tuple[int, int, str]

#: An arc: the vertex it leads to, and the edit it represents (``None`` for a match).
Arc = tuple[Vertex, SystemEdit | None]

#: Cost of a substitution, in units where an insertion or a deletion costs 1.
#: Two, so that a substitution ties with delete-then-insert and the lattice keeps
#: both alignments.
COST_SUB = 2

#: How many unchanged tokens a phrase-level edit may span. Two is the reference
#: implementation's value: enough for ``[in the end -> at the end]``, not enough
#: for a whole clause to become one edit.
MAX_UNCHANGED = 2

#: Cap on the token span of a phrase-level edit, per side. The longest gold edit
#: in Falko-MERLIN covers 9 tokens.
MAX_SPAN = 10


@dataclass(frozen=True)
class Counts:
    """True positives, false positives and false negatives, in edits."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    def f(self, beta: float = 0.5) -> float:
        """F-beta. ``beta=0.5`` weights precision twice as heavily as recall."""
        p, r = self.precision, self.recall
        if p == 0.0 and r == 0.0:
            return 0.0
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r)


@dataclass(frozen=True)
class M2Score:
    """A corpus-level score plus the counts it was computed from."""

    counts: Counts
    beta: float = 0.5
    sentences: int = 0

    @property
    def precision(self) -> float:
        return self.counts.precision

    @property
    def recall(self) -> float:
        return self.counts.recall

    @property
    def f_score(self) -> float:
        return self.counts.f(self.beta)

    def __str__(self) -> str:
        c = self.counts
        return (
            f"P={self.precision:.4f} R={self.recall:.4f} F{self.beta:g}={self.f_score:.4f} "
            f"(tp={c.tp} fp={c.fp} fn={c.fn}, {self.sentences} sentences)"
        )


def _distance_matrices(
    source: Sequence[str], hyp: Sequence[str], cost_sub: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Levenshtein distance from the start, and to the end, of every vertex.

    An arc lies on some optimal alignment exactly when
    ``forward[u] + cost(u, v) + backward[v]`` equals the total distance, so both
    directions are needed.
    """
    n, m = len(source), len(hyp)

    forward = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        forward[i][0] = i
    for j in range(1, m + 1):
        forward[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = forward[i - 1][j - 1] + (0 if source[i - 1] == hyp[j - 1] else cost_sub)
            forward[i][j] = min(forward[i - 1][j] + 1, forward[i][j - 1] + 1, sub)

    backward = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        backward[i][m] = n - i
    for j in range(m - 1, -1, -1):
        backward[n][j] = m - j
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            sub = backward[i + 1][j + 1] + (0 if source[i] == hyp[j] else cost_sub)
            backward[i][j] = min(backward[i + 1][j] + 1, backward[i][j + 1] + 1, sub)

    return forward, backward


def _token_arcs(
    source: Sequence[str], hyp: Sequence[str], cost_sub: int = COST_SUB
) -> dict[Vertex, list[Arc]]:
    """Every token-level arc that lies on an optimal alignment."""
    n, m = len(source), len(hyp)
    forward, backward = _distance_matrices(source, hyp, cost_sub)
    total = forward[n][m]

    arcs: dict[Vertex, list[Arc]] = {}
    for i in range(n + 1):
        for j in range(m + 1):
            out: list[Arc] = []
            if i < n and j < m:
                same = source[i] == hyp[j]
                cost = 0 if same else cost_sub
                if forward[i][j] + cost + backward[i + 1][j + 1] == total:
                    edit = None if same else (i, i + 1, hyp[j])
                    out.append(((i + 1, j + 1), edit))
            if i < n and forward[i][j] + 1 + backward[i + 1][j] == total:
                out.append(((i + 1, j), (i, i + 1, "")))  # deletion
            if j < m and forward[i][j] + 1 + backward[i][j + 1] == total:
                out.append(((i, j + 1), (i, i, hyp[j])))  # insertion
            if out:
                arcs[(i, j)] = out
    return arcs


def _phrase_arcs(
    token_arcs: dict[Vertex, list[Arc]],
    source: Sequence[str],
    hyp: Sequence[str],
    max_unchanged: int = MAX_UNCHANGED,
    max_span: int = MAX_SPAN,
) -> dict[Vertex, list[Arc]]:
    """Add arcs formed by composing runs of adjacent token-level arcs.

    The edit an arc represents depends only on its two endpoints, so
    compositions are deduplicated by endpoint rather than by path, which bounds
    the work per starting vertex at ``max_span`` squared however tangled the
    lattice is. A composition whose replacement text equals the source span it
    covers is dropped: it changes nothing, and would otherwise let a run of
    matching tokens masquerade as an edit.
    """
    arcs: dict[Vertex, list[Arc]] = {v: list(out) for v, out in token_arcs.items()}

    for start, base in token_arcs.items():
        reached: set[Vertex] = {target for target, _ in base}
        # States are (vertex, unchanged tokens swallowed so far) and are visited
        # once each: two paths reaching the same state can be extended in exactly
        # the same ways, so exploring both would only duplicate work.
        seen_states: set[tuple[Vertex, int]] = {(start, 0)}
        frontier: list[tuple[Vertex, int]] = [(start, 0)]
        for _ in range(2 * max_span):
            nxt: list[tuple[Vertex, int]] = []
            for vertex, unchanged in frontier:
                for target, edit in token_arcs.get(vertex, ()):
                    swallowed = unchanged + (1 if edit is None else 0)
                    if swallowed > max_unchanged:
                        continue
                    if target[0] - start[0] > max_span or target[1] - start[1] > max_span:
                        continue
                    if (target, swallowed) not in seen_states:
                        seen_states.add((target, swallowed))
                        nxt.append((target, swallowed))
                    if target in reached:
                        continue
                    reached.add(target)
                    replacement = " ".join(hyp[start[1] : target[1]])
                    if replacement == " ".join(source[start[0] : target[0]]):
                        continue
                    arcs.setdefault(start, []).append((target, (start[0], target[0], replacement)))
            if not nxt:
                break
            frontier = nxt

    return arcs


def extract_edits(
    source: Sequence[str],
    hyp: Sequence[str],
    gold: Iterable[Edit] = (),
    max_unchanged: int = MAX_UNCHANGED,
) -> list[SystemEdit]:
    """Recover the system's edits, described so as to match `gold` where possible.

    Args:
        source: The original sentence's tokens.
        hyp: The system's corrected sentence, tokenised the same way.
        gold: Gold edits to align the description to. With none supplied the
            result is simply the shortest edit sequence.
        max_unchanged: Unchanged tokens a single phrase-level edit may span.

    Returns:
        The chosen edits, in source order.
    """
    if list(source) == list(hyp):
        return []

    gold_keys = {(e.start, e.end, e.correction) for e in gold if not e.is_noop}
    token_arcs = _token_arcs(source, hyp)
    arcs = _phrase_arcs(token_arcs, source, hyp, max_unchanged=max_unchanged)

    end = (len(source), len(hyp))
    best: dict[Vertex, tuple[int, int]] = {(0, 0): (0, 0)}
    previous: dict[Vertex, tuple[Vertex, SystemEdit | None]] = {}

    for vertex in sorted(arcs, key=lambda v: (v[0] + v[1], v[0])):
        cost = best.get(vertex)
        if cost is None:
            continue
        for target, edit in arcs[vertex]:
            if edit is None:
                candidate = cost
            elif edit in gold_keys:
                candidate = (cost[0] - 1, cost[1])
            else:
                candidate = (cost[0], cost[1] + 1)
            if target not in best or candidate < best[target]:
                best[target] = candidate
                previous[target] = (vertex, edit)

    edits: list[SystemEdit] = []
    vertex = end
    while vertex != (0, 0):
        vertex, edit = previous[vertex]
        if edit is not None:
            edits.append(edit)
    edits.reverse()
    return edits


def sentence_counts(
    source: Sequence[str],
    hyp: Sequence[str],
    gold: Sequence[Edit],
) -> Counts:
    """Score one hypothesis against one annotator's gold edits."""
    real_gold = [e for e in gold if not e.is_noop]
    gold_keys = {(e.start, e.end, e.correction) for e in real_gold}
    system = extract_edits(source, hyp, real_gold)
    tp = sum(1 for edit in system if edit in gold_keys)
    return Counts(tp=tp, fp=len(system) - tp, fn=len(gold_keys) - tp)


def score(
    sentences: Sequence[M2Sentence],
    hypotheses: Sequence[str],
    beta: float = 0.5,
) -> M2Score:
    """Score a corpus of hypotheses against gold M2 sentences.

    Args:
        sentences: Gold sentences, in the order they were corrected.
        hypotheses: One corrected sentence per gold sentence, whitespace-tokenised
            the same way the source is.
        beta: F-score beta; 0.5 by convention in GEC.

    Raises:
        ValueError: if the two sequences are of different lengths, which almost
            always means a system dropped or merged a sentence.
    """
    if len(sentences) != len(hypotheses):
        raise ValueError(
            f"{len(hypotheses)} hypotheses for {len(sentences)} sentences: "
            f"the scorer aligns them by position, so they must correspond one to one."
        )

    total = Counts()
    for sentence, hypothesis in zip(sentences, hypotheses, strict=True):
        best: Counts | None = None
        best_key: tuple[float, int, int, int] | None = None
        for annotator in sentence.annotators or [0]:
            counts = sentence_counts(
                sentence.source_tokens,
                hypothesis.split(),
                sentence.edits_for(annotator),
            )
            # Pick the annotator that leaves the corpus best off so far, the way
            # the reference scorers do: a local choice can trade tp for fp in a
            # way that only the running totals can adjudicate.
            running = total + counts
            key = (running.f(beta), running.tp, -running.fp, -running.fn)
            if best_key is None or key > best_key:
                best, best_key = counts, key
        total = total + (best or Counts())

    return M2Score(counts=total, beta=beta, sentences=len(sentences))
