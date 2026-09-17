"""The error-typing logic shared by every language's classifier.

Every edit gets a label of the form ``<operation>:<category>``: ``M`` for a
missing token, ``U`` for an unnecessary one, ``R`` for a replacement, and a
category that is either a part of speech (``R:DET``), an inflection of one
(``R:DET:FORM``), or one of the cross-cutting classes ``SPELL``, ``ORTH``,
``MORPH``, ``WO`` and ``OTHER``. This is the scheme Boyd (2018) defined for
German ERRANT, itself an adaptation of Bryant et al.'s English one, and it
reads as universal POS tags and dependency labels almost everywhere -- which
is what let it move here unchanged when Phase 5 ported Spanish.

The ordering of the rules matters, and it is ERRANT's own: orthography before
word order, spelling before part of speech, shared lemma before string
similarity. A misspelling that happens to change part of speech is a
misspelling.

**The one thing that is not universal** is deciding when two spellings of a
replacement are "the same word, cosmetically different" -- ``R:ORTH`` rather
than a real edit. German's answer is case and spacing: ``[autos -> Autos]``,
because German capitalises every noun. Spanish does not capitalise nouns, but
it carries a diacritic on almost the same axis -- ``[esta -> está]`` is the
same word, stressed -- so the check that decides this is the one parameter
`make_classifier` takes rather than hard-codes. Everything else -- what counts
as a spelling error, an inflection, a word-order swap -- reads only universal
POS tags, lemmas and string similarity, which mean the same thing regardless
of which language's spacy model produced them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from errant.edit import Edit
from rapidfuzz.distance import Levenshtein

#: Universal POS tags renamed to the corpus's category names. Shared because
#: the renaming is about the label being read, not the language: "PNOUN" and
#: "CONJ" are just names this project's reports use everywhere.
POS_MAP = {"PROPN": "PNOUN", "CCONJ": "CONJ"}

#: Tags that make an uninformative category. An edit that lands on one of these
#: is reported as OTHER rather than as ``R:X``.
RARE_POS = frozenset({"INTJ", "NUM", "SYM", "X", "SPACE"})

#: Parts of speech that inflect for gender, number, case, tense or person, and
#: so can take a ``:FORM`` label rather than being read as a different word.
#: The German and Spanish sets happen to coincide -- both decline determiners
#: and pronouns, unlike English -- so this stays a shared default rather than
#: a per-language override; a language where it does not hold can still pass
#: its own set to `make_classifier`.
INFLECTING_POS = frozenset({"ADJ", "ADV", "AUX", "DET", "NOUN", "PRON", "VERB"})

#: Character similarity above which two words are called variant spellings.
#: ERRANT's English threshold, which transfers: it is a property of how far
#: typists' fingers slip, not of the language.
SPELL_SIMILARITY = 0.55

#: `(orig_tokens, corrected_tokens) -> bool`: true when the two spans are the
#: same text up to the one cosmetic axis a language cares about (case for
#: German, diacritics for Spanish).
SameSpelling = Callable[[Any, Any], bool]


def squash(tokens: Any) -> str:
    """Lower-case the tokens and join them with no separator.

    The default notion of "same spelling, different casing/spacing":
    ``[so gar -> sogar]`` squashes equal on both sides. A language whose
    cosmetic axis is not case (Spanish's is diacritics) composes this with its
    own normalisation rather than replacing it -- see `errant_es.classifier`.
    """
    return "".join(tok.lower_ for tok in tokens)


def make_classifier(
    is_known_word: Callable[[str], bool],
    same_spelling: SameSpelling | None = None,
    pos_map: dict[str, str] = POS_MAP,
    rare_pos: frozenset[str] = RARE_POS,
    inflecting_pos: frozenset[str] = INFLECTING_POS,
    spell_similarity: float = SPELL_SIMILARITY,
):
    """Build a `classify(edit) -> edit` function for one language.

    Args:
        is_known_word: True if a string is a word of the target language, cased
            exactly as written. The line between a misspelling and a wrong
            word choice.
        same_spelling: See `SameSpelling`. Defaults to `squash` equality, which
            is right for a language whose only cosmetic axis is case.
        pos_map, rare_pos, inflecting_pos, spell_similarity: Override only if a
            language's facts genuinely differ from the shared defaults above --
            not as a place to encode a language's grammar.
    """
    same_spelling = same_spelling or (lambda o, c: squash(o) == squash(c))

    def pos_of(token: Any) -> str:
        return pos_map.get(token.pos_, token.pos_)

    def one_sided_type(tokens: Any) -> str:
        """Type an edit that inserts or deletes tokens, where only one side exists."""
        tags = [pos_of(tok) for tok in tokens]
        deps = [tok.dep_ for tok in tokens]

        if len(set(tags)) == 1 and tags[0] not in rare_pos:
            return tags[0]
        # A missing or superfluous auxiliary shows up as a mixed POS run whose
        # dependencies agree; e.g. a dropped perfect auxiliary.
        if set(deps) <= {"oc", "aux"} and set(tags) <= {"AUX", "VERB"}:
            return "AUX"
        # Verb groups: an infinitive marker or separable prefix, plus its verb.
        if set(tags) == {"PART", "VERB"}:
            return "VERB"
        return "OTHER"

    def _inflection_type(o_pos: str, c_pos: str) -> str:
        """Type an edit whose two sides share a lemma."""
        if o_pos == c_pos:
            # Only some parts of speech inflect. The rest reach here because
            # the lemmatiser collapses a whole class to one lemma -- every
            # punctuation mark lemmatises to the same thing -- and for those
            # the shared lemma says nothing, so the edit is typed by its part
            # of speech alone.
            return f"{o_pos}:FORM" if o_pos in inflecting_pos else o_pos
        # Same word, different part of speech: a derivational or agreement
        # change too tangled to name.
        return "MORPH"

    def _similarity(a: Any, b: Any) -> float:
        return Levenshtein.normalized_similarity(a.lower_, b.lower_)

    def _single_token_type(o_tok: Any, c_tok: Any, o_pos: str, c_pos: str) -> str:
        """Type a one-for-one token replacement, the commonest, most informative case."""
        same_lemma = o_tok.lemma_ == c_tok.lemma_

        # A word the learner did not spell correctly is a spelling error even
        # when the misspelling changes the part of speech, so this test comes
        # first.
        if o_tok.text.isalpha() and not is_known_word(o_tok.text):
            if same_lemma:
                return _inflection_type(o_pos, c_pos)
            if _similarity(o_tok, c_tok) > spell_similarity:
                return "SPELL"
            # Short words are similar to everything, so the ratio is
            # uninformative; fall back to what the correction is.
            if len(o_tok.text) <= 4 and len(c_tok.text) <= 4:
                return "SPELL"
            return c_pos if c_pos not in rare_pos else "OTHER"

        # A real word in the wrong form: [der -> die], [Männer -> Männern].
        if same_lemma:
            return _inflection_type(o_pos, c_pos)
        # A real word of the right kind but the wrong choice: [mit -> in].
        if o_pos == c_pos and o_pos not in rare_pos:
            return o_pos
        # Confusions across a part-of-speech boundary; [das -> dass] lands
        # here, as it does in the corpus, because the two are genuinely
        # different words.
        return "OTHER"

    def two_sided_type(o_toks: Any, c_toks: Any) -> str:
        """Type an edit that replaces one non-empty span with another."""
        o_pos = [pos_of(tok) for tok in o_toks]
        c_pos = [pos_of(tok) for tok in c_toks]

        # Same word, cosmetically different: [autos -> Autos], [esta -> está].
        if same_spelling(o_toks, c_toks):
            return "ORTH"
        # The same words in a different order.
        if sorted(tok.lower_ for tok in o_toks) == sorted(tok.lower_ for tok in c_toks):
            return "WO"

        if len(o_toks) == len(c_toks) == 1:
            return _single_token_type(o_toks[0], c_toks[0], o_pos[0], c_pos[0])

        # Multi-token replacements are rare and hard to be specific about.
        if len(set(o_pos + c_pos)) == 1 and o_pos[0] not in rare_pos:
            return o_pos[0]
        return "OTHER"

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
            # A trailing token that only changed case came along for the ride;
            # e.g. [, deshalb -> . Deshalb]. Type the edit as if it were not
            # there.
            full_o, full_c = edit.o_toks, edit.c_toks
            edit.o_toks, edit.c_toks = full_o[:-1], full_c[:-1]
            edit = classify(edit)
            edit.o_toks, edit.c_toks = full_o, full_c
        else:
            edit.type = "R:" + two_sided_type(edit.o_toks, edit.c_toks)
        return edit

    return classify
