"""The corruption framework: damage a correct sentence, and say what you did.

A corruptor is a function from a correct sentence to the places in it that could
plausibly be got wrong. It proposes; it does not decide. The sampler decides,
because the error mix across the corpus is a corpus-level property and no
individual corruptor can see it.

The output is an :class:`~langlm.data.m2.M2Sentence` whose *source* is the
damaged text and whose edits repair it. That direction matters and is easy to
get backwards: the training example shows the model broken German and asks for
the fix, so the corrupted form is the input and the original is the target. A
corruptor that returns the edit the other way round produces a corpus that
teaches a model to introduce errors, and nothing about the file format would
complain.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langlm.data.m2 import Edit, M2Sentence


@dataclass(frozen=True)
class Corruption:
    """One way a correct sentence could be got wrong.

    Attributes:
        start: Index of the first token replaced, in the *correct* sentence.
        end: One past the last token replaced. ``start == end`` inserts.
        replacement: What the damaged sentence says instead, whitespace
            tokenised. Empty deletes the span.
        error_type: The ERRANT tag of the edit that repairs it, so the injected
            distribution is measured in the same units the evaluation uses.
        rule: Which corruptor produced this, for diagnosis when a sample looks
            wrong. Not written to the M2 file.
    """

    start: int
    end: int
    replacement: str
    error_type: str
    rule: str

    @property
    def is_deletion(self) -> bool:
        return not self.replacement

    def damaged_tokens(self, tokens: Sequence[str]) -> list[str]:
        """The correct sentence, with this corruption applied."""
        replacement = self.replacement.split() if self.replacement else []
        return [*tokens[: self.start], *replacement, *tokens[self.end :]]


#: A corruptor: given the tokens of a correct sentence and, where it needs one,
#: a parse of it, every way that sentence could plausibly be got wrong.
#: Returning nothing is normal -- most sentences offer no purchase for most
#: rules.
Corruptor = Callable[[Sequence[str], "object | None"], list[Corruption]]

#: Every registered corruptor, by name.
REGISTRY: dict[str, Corruptor] = {}

#: Those of them that cannot work without a parse. Kept separate so a caller
#: with no parser gets the orthographic rules rather than an exception, and so
#: the generator knows whether paying for a parse would buy it anything.
NEEDS_PARSE: set[str] = set()


def _register(name: str, function: Corruptor, needs_parse: bool) -> Corruptor:
    if name in REGISTRY:
        raise ValueError(f"Two corruptors are registered as {name!r}.")
    REGISTRY[name] = function
    if needs_parse:
        NEEDS_PARSE.add(name)
    return function


def corruptor(name: str) -> Callable[..., Corruptor]:
    """Register a corruptor that works on tokens alone.

    German allows most of them to: articles and prepositions are closed classes
    that can be listed, and a capitalised non-initial token is reliably a noun.
    The parse is only needed for word order and morphology.
    """

    def register(function):
        return _register(name, lambda tokens, doc=None: function(tokens), needs_parse=False)

    return register


def parse_corruptor(name: str) -> Callable[..., Corruptor]:
    """Register a corruptor that needs a parsed sentence.

    The wrapped function is handed the spacy doc and is skipped entirely when
    there is not one, rather than falling back to a guess. A word-order rule
    that guesses at where the finite verb is produces damage that is not the
    error it claims to be, and the explanation attached to it would be a lie.
    """

    def register(function):
        return _register(
            name,
            lambda tokens, doc=None: function(tokens, doc) if doc is not None else [],
            needs_parse=True,
        )

    return register


def propose(
    tokens: Sequence[str],
    rules: Sequence[str] | None = None,
    doc: object | None = None,
) -> list[Corruption]:
    """Every corruption the given rules can find in a sentence."""
    names = rules if rules is not None else list(REGISTRY)
    found: list[Corruption] = []
    for name in names:
        found.extend(REGISTRY[name](tokens, doc))
    return found


def apply(tokens: Sequence[str], corruptions: Sequence[Corruption]) -> M2Sentence:
    """Damage a sentence with several non-overlapping corruptions.

    Args:
        tokens: The correct sentence.
        corruptions: What to do to it. Overlapping spans are a caller error;
            :func:`choose` is what avoids them.

    Returns:
        An M2 sentence whose source is the damaged text and whose edits restore
        the original.

    Raises:
        ValueError: if two corruptions overlap, which would make the edit
            offsets meaningless rather than merely wrong.
    """
    ordered = sorted(corruptions, key=lambda c: c.start)
    for earlier, later in itertools.pairwise(ordered):
        if earlier.end > later.start:
            raise ValueError(f"Overlapping corruptions: {earlier} and {later}.")

    damaged: list[str] = []
    edits: list[Edit] = []
    cursor = 0
    for corruption in ordered:
        damaged.extend(tokens[cursor : corruption.start])
        start = len(damaged)
        replacement = corruption.replacement.split() if corruption.replacement else []
        damaged.extend(replacement)
        edits.append(
            Edit(
                start=start,
                end=start + len(replacement),
                error_type=corruption.error_type,
                correction=" ".join(tokens[corruption.start : corruption.end]),
                # Which rule did this, carried in the M2 comment field because
                # that is the one place the format has for it. The explanation
                # needs it and the error type cannot supply it: `R:SPELL` is
                # shared by the umlaut, ß, das/dass and keyboard rules, which
                # want four different explanations.
                comment=corruption.rule,
            )
        )
        cursor = corruption.end
    damaged.extend(tokens[cursor:])

    return M2Sentence(source_tokens=damaged, edits=edits)


def _score(
    error_type: str,
    weights: dict[str, float] | None,
    emitted: Counter | None,
) -> float:
    """How much this type is wanted right now."""
    if weights is None:
        return 1.0
    target = weights.get(error_type, 0.0)
    if emitted is None:
        return max(target, 1e-6)
    total = sum(emitted.values())
    observed = emitted[error_type] / total if total else 0.0
    # A type already over its share is not banned, only starved: banning it
    # would make the mix depend on the order the sentences happen to arrive.
    return max(target - observed, target * 0.05, 1e-6)


def choose(
    corruptions: Sequence[Corruption],
    count: int,
    rng: random.Random,
    weights: dict[str, float] | None = None,
    emitted: Counter | None = None,
) -> list[Corruption]:
    """Pick `count` non-overlapping corruptions, preferring the wanted types.

    Weights are over error types rather than over corruptors, because the target
    distribution is expressed in ERRANT tags -- the units the evaluation and the
    Falko-MERLIN histogram both use. A type with no weight is not forbidden,
    only unlikely; giving it zero would mean a rule silently never fires.

    The type is drawn first and a candidate second, which matters more than it
    looks. Rules differ enormously in how much purchase they get: the keyboard
    rule offers a candidate for nearly every word, while the word-order rule
    offers at most one and only in a sentence with a subordinate clause.
    Weighting the candidates directly would hand a type with fifty of them fifty
    times the mass of a type with one at the same weight -- and the realised mix
    would be a function of how prolific each rule is rather than of the
    distribution the weights were derived to produce.

    Drawing the type per sentence is still not enough on its own, because a type
    has to be *available* to be drawn and they are not equally available: nearly
    every sentence can take a spelling error, while word order needs a
    subordinate clause and missing punctuation needs a comma. Passing `emitted`
    -- the counts so far across the whole corpus -- scores each type by how far
    it is below its target share rather than by the target alone, so the scarce
    types are taken whenever a sentence does offer them and the corpus converges
    on the intended mix instead of on whatever the sentences happened to allow.
    """
    remaining = list(corruptions)
    chosen: list[Corruption] = []
    while remaining and len(chosen) < count:
        by_type: dict[str, list[Corruption]] = defaultdict(list)
        for candidate in remaining:
            by_type[candidate.error_type].append(candidate)
        types = list(by_type)
        scores = [_score(t, weights, emitted) for t in types]
        pick = rng.choice(by_type[rng.choices(types, weights=scores, k=1)[0]])
        chosen.append(pick)
        remaining = [c for c in remaining if c.end <= pick.start or c.start >= pick.end]
    return sorted(chosen, key=lambda c: c.start)
