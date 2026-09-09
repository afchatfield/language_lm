"""Error typing for German edits.

Every edit gets a label of the form ``<operation>:<category>``: ``M`` for a
missing token, ``U`` for an unnecessary one, ``R`` for a replacement, and a
category that is either a part of speech (``R:DET``), an inflection of one
(``R:DET:FORM``), or one of the cross-cutting classes ``SPELL``, ``ORTH``,
``MORPH``, ``WO`` and ``OTHER``.

This is the scheme Boyd (2018) defined for German and used to annotate the
Falko-MERLIN M2 files this project scores against, so the labels here are
comparable with the corpus's own. The scheme is deliberately flatter than
ERRANT's English one: English splits inflection into ``:NUM``, ``:TENSE`` and
``:SVA``, whereas German marks case, number, gender and person on the same
ending and a single ``:FORM`` is all that can be read off reliably.

The ordering of the rules matters, and it is the same as ERRANT's: orthography
before word order, spelling before part of speech, shared lemma before string
similarity. A misspelling that happens to change part of speech is a
misspelling.
"""

from __future__ import annotations

from typing import Any

from errant.edit import Edit
from rapidfuzz.distance import Levenshtein

from langlm.eval.errant_de.spelling import is_known_word

#: Universal POS tags renamed to the corpus's category names.
POS_MAP = {"PROPN": "PNOUN", "CCONJ": "CONJ"}

#: Tags that make an uninformative category. An edit that lands on one of these
#: is reported as OTHER rather than as ``R:X``.
RARE_POS = frozenset({"INTJ", "NUM", "SYM", "X", "SPACE"})

#: Parts of speech that inflect in German, and so can take a ``:FORM`` label.
#: Determiners and pronouns are in the list -- unlike English, they decline.
INFLECTING_POS = frozenset({"ADJ", "ADV", "AUX", "DET", "NOUN", "PRON", "VERB"})

#: Character similarity above which two words are called variant spellings.
#: ERRANT's English threshold, which transfers: it is a property of how far
#: typists' fingers slip, not of the language.
SPELL_SIMILARITY = 0.55


def classify(edit: Edit) -> Edit:
    """Set ``edit.type`` and return the edit."""
    if not edit.o_toks and not edit.c_toks:
        edit.type = "UNK"
    elif not edit.o_toks:
        edit.type = "M:" + one_sided_type(edit.c_toks)
    elif not edit.c_toks:
        edit.type = "U:" + one_sided_type(edit.o_toks)
    elif edit.o_str == edit.c_str:
        edit.type = "UNK"
    elif edit.o_toks[-1].lower == edit.c_toks[-1].lower and (
        len(edit.o_toks) > 1 or len(edit.c_toks) > 1
    ):
        # A trailing token that only changed case came along for the ride; e.g.
        # [, deshalb -> . Deshalb]. Type the edit as if it were not there.
        full_o, full_c = edit.o_toks, edit.c_toks
        edit.o_toks, edit.c_toks = full_o[:-1], full_c[:-1]
        edit = classify(edit)
        edit.o_toks, edit.c_toks = full_o, full_c
    else:
        edit.type = "R:" + two_sided_type(edit.o_toks, edit.c_toks)
    return edit


def pos_of(token: Any) -> str:
    """The token's part of speech under the corpus's category names."""
    return POS_MAP.get(token.pos_, token.pos_)


def one_sided_type(tokens: Any) -> str:
    """Type an edit that inserts or deletes tokens, where only one side exists."""
    tags = [pos_of(tok) for tok in tokens]
    deps = [tok.dep_ for tok in tokens]

    if len(set(tags)) == 1 and tags[0] not in RARE_POS:
        return tags[0]
    # A missing or superfluous auxiliary shows up as a mixed POS run whose
    # dependencies agree; e.g. a dropped perfect auxiliary.
    if set(deps) <= {"oc", "aux"} and set(tags) <= {"AUX", "VERB"}:
        return "AUX"
    # Verb groups: the "zu" of an infinitive, or a separable prefix, plus its verb.
    if set(tags) == {"PART", "VERB"}:
        return "VERB"
    return "OTHER"


def two_sided_type(o_toks: Any, c_toks: Any) -> str:
    """Type an edit that replaces one non-empty span with another."""
    o_pos = [pos_of(tok) for tok in o_toks]
    c_pos = [pos_of(tok) for tok in c_toks]

    # Same letters, different case or spacing: [autos -> Autos], [so gar -> sogar].
    # German capitalises every noun, so this is a large and distinct class.
    if _squash(o_toks) == _squash(c_toks):
        return "ORTH"
    # The same words in a different order.
    if sorted(tok.lower_ for tok in o_toks) == sorted(tok.lower_ for tok in c_toks):
        return "WO"

    if len(o_toks) == len(c_toks) == 1:
        return _single_token_type(o_toks[0], c_toks[0], o_pos[0], c_pos[0])

    # Multi-token replacements are rare and hard to be specific about.
    if len(set(o_pos + c_pos)) == 1 and o_pos[0] not in RARE_POS:
        return o_pos[0]
    return "OTHER"


def _single_token_type(o_tok: Any, c_tok: Any, o_pos: str, c_pos: str) -> str:
    """Type a one-for-one token replacement, the commonest and most informative case."""
    same_lemma = o_tok.lemma_ == c_tok.lemma_

    # A word the learner did not spell correctly is a spelling error even when
    # the misspelling changes the part of speech, so this test comes first.
    if o_tok.text.isalpha() and not is_known_word(o_tok.text):
        if same_lemma:
            return _inflection_type(o_pos, c_pos)
        if _similarity(o_tok, c_tok) > SPELL_SIMILARITY:
            return "SPELL"
        # Short words are similar to everything, so the ratio is uninformative;
        # fall back to what the correction is.
        if len(o_tok.text) <= 4 and len(c_tok.text) <= 4:
            return "SPELL"
        return c_pos if c_pos not in RARE_POS else "OTHER"

    # A real word in the wrong form: [der -> die], [Männer -> Männern].
    if same_lemma:
        return _inflection_type(o_pos, c_pos)
    # A real word of the right kind but the wrong choice: [mit -> in].
    if o_pos == c_pos and o_pos not in RARE_POS:
        return o_pos
    # Confusions across a part-of-speech boundary; [das -> dass] lands here, as
    # it does in the corpus, because the two are genuinely different words.
    return "OTHER"


def _inflection_type(o_pos: str, c_pos: str) -> str:
    """Type an edit whose two sides share a lemma."""
    if o_pos == c_pos:
        # Only some parts of speech inflect. The rest reach here because the
        # lemmatiser collapses a whole class to one lemma -- every punctuation
        # mark lemmatises to the same thing -- and for those the shared lemma
        # says nothing, so the edit is typed by its part of speech alone.
        return f"{o_pos}:FORM" if o_pos in INFLECTING_POS else o_pos
    # Same word, different part of speech: a derivational or agreement change
    # too tangled to name.
    return "MORPH"


def _squash(tokens: Any) -> str:
    return "".join(tok.lower_ for tok in tokens)


def _similarity(a: Any, b: Any) -> float:
    return Levenshtein.normalized_similarity(a.lower_, b.lower_)
