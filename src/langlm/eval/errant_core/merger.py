"""Rule-based edit merging, shared across languages.

An alignment between two sentences is a sequence of token-level operations, and
merging decides which runs of them describe one error rather than several.
``[ein System -> ein Systeme]`` is one agreement error; ``[der Mann geht -> die
Frau kommt]`` is three. Getting this wrong changes the denominator of every
score.

Adapted from ERRANT's English merger (Bryant et al., 2017; MIT licensed) for
German, and moved here unchanged when Phase 5 ported Spanish, because nothing
in it actually reads German: every rule is keyed on universal POS tags and
dependency labels, not a lexicon. What changed from the English original, when
this was German-only, was:

* The possessive-suffix rules are gone. They keyed off the Penn tag ``POS``,
  which the German tagset has no equivalent of -- German marks the genitive on
  the noun itself.
* The auxiliary/particle merge covers German separable prefixes and the ``zu``
  of an infinitive group, both tagged ``PART`` -- a universal tag Spanish's
  ``ir a + infinitive`` and clitic-climbing constructions also use, which is
  the reason this rule was safe to keep rather than to fork.

If a Spanish construction ever needs a rule this shape cannot express, it
belongs here as a new branch keyed on a universal tag, not as a per-language
fork of the file: the whole point of moving it was that one rule set serves
every language ERRANT's alignment already tags consistently.
"""

from __future__ import annotations

from itertools import combinations, groupby
from re import sub
from string import punctuation
from typing import Any

import spacy.symbols as POS
from errant.edit import Edit
from rapidfuzz.distance import Indel

#: Content-word tags. A run that touches one of these is merged rather than
#: split: content words carry the error, function words around them are context.
OPEN_POS = {POS.ADJ, POS.AUX, POS.ADV, POS.NOUN, POS.VERB}

#: Character similarity above which two substituted tokens are treated as
#: variants of the same word and split apart from their neighbours.
SIMILAR = 0.75


def get_rule_edits(alignment: Any) -> list[Edit]:
    """Turn an alignment into the list of edits it describes."""
    edits: list[Edit] = []
    # Runs of matches (M) and transpositions (T) are handled wholesale; every
    # other run of adjacent operations goes through the merge rules.
    for op, group in groupby(
        alignment.align_seq, lambda x: x[0][0] if x[0][0] in {"M", "T"} else False
    ):
        group = list(group)
        if op == "M":
            continue
        if op == "T":
            edits.extend(Edit(alignment.orig, alignment.cor, seq[1:]) for seq in group)
            continue
        for seq in _process_seq(group, alignment):
            edits.append(Edit(alignment.orig, alignment.cor, seq[1:]))
    return edits


def _process_seq(seq: list, alignment: Any) -> list:
    """Recursively decide where a run of adjacent operations should be cut."""
    if len(seq) <= 1:
        return seq

    ops = [op[0] for op in seq]
    # A run of pure deletions or pure insertions is one edit: a multi-token edit
    # that involves no substitution is nearly always a single phrase.
    if set(ops) == {"D"} or set(ops) == {"I"}:
        return _merge(seq)

    content = False
    # Largest spans first, so the widest applicable rule wins.
    combos = sorted(combinations(range(len(seq)), 2), key=lambda x: x[1] - x[0], reverse=True)

    for start, end in combos:
        if "S" not in ops[start : end + 1]:
            continue
        orig = alignment.orig[seq[start][1] : seq[end][2]]
        cor = alignment.cor[seq[start][3] : seq[end][4]]

        # Case changes at a boundary belong with the token that caused them.
        if orig[-1].lower == cor[-1].lower:
            # [Auto -> Das grosse Auto]
            if start == 0 and (
                (len(orig) == 1 and cor[0].text[0].isupper())
                or (len(cor) == 1 and orig[0].text[0].isupper())
            ):
                return _merge(seq[start : end + 1]) + _process_seq(seq[end + 1 :], alignment)
            # [, wir -> . Wir]: a sentence split recased the following word.
            if (len(orig) > 1 and _is_punct(orig[-2])) or (len(cor) > 1 and _is_punct(cor[-2])):
                return (
                    _process_seq(seq[: end - 1], alignment)
                    + _merge(seq[end - 1 : end + 1])
                    + _process_seq(seq[end + 1 :], alignment)
                )

        # Word-boundary errors: [statt dessen -> stattdessen], [Haus tür -> Haustür].
        # German compounding makes these common enough to be worth a rule.
        if _squash(orig) == _squash(cor):
            return (
                _process_seq(seq[:start], alignment)
                + _merge(seq[start : end + 1])
                + _process_seq(seq[end + 1 :], alignment)
            )

        # One phrase becoming another of the same kind, or a verb group being
        # rebuilt: [zu essen -> essend], [bereiten vor -> vorbereiten].
        pos_set = {tok.pos for tok in orig} | {tok.pos for tok in cor}
        if len(orig) != len(cor) and (
            len(pos_set) == 1 or pos_set.issubset({POS.AUX, POS.PART, POS.VERB})
        ):
            return (
                _process_seq(seq[:start], alignment)
                + _merge(seq[start : end + 1])
                + _process_seq(seq[end + 1 :], alignment)
            )

        # The splitting rules only apply once the window is down to two ops.
        if end - start < 2:
            if len(orig) == len(cor) == 2:
                return _process_seq(seq[: start + 1], alignment) + _process_seq(
                    seq[start + 1 :], alignment
                )
            # A near-identical token at either end is its own (spelling) edit.
            if (ops[start] == "S" and _char_similarity(orig[0], cor[0]) > SIMILAR) or (
                ops[end] == "S" and _char_similarity(orig[-1], cor[-1]) > SIMILAR
            ):
                return _process_seq(seq[: start + 1], alignment) + _process_seq(
                    seq[start + 1 :], alignment
                )
            # A determiner at the end belongs to the following noun phrase.
            if end == len(seq) - 1 and (
                (ops[-1] in {"D", "S"} and orig[-1].pos == POS.DET)
                or (ops[-1] in {"I", "S"} and cor[-1].pos == POS.DET)
            ):
                return [*_process_seq(seq[:-1], alignment), seq[-1]]

        if not pos_set.isdisjoint(OPEN_POS):
            content = True

    return _merge(seq) if content else seq


def _squash(tokens: Any) -> str:
    """Lower-case the tokens and drop the characters that only mark boundaries."""
    return sub("['-]", "", "".join(tok.lower_ for tok in tokens))


def _is_punct(token: Any) -> bool:
    return token.pos == POS.PUNCT or token.text in punctuation


def _char_similarity(a: Any, b: Any) -> float:
    return 1 - Indel.normalized_distance(a.text, b.text)


def _merge(seq: list) -> list:
    """Collapse a run of operations into one edit spanning all of them."""
    return [("X", seq[0][1], seq[-1][2], seq[0][3], seq[-1][4])] if seq else seq
