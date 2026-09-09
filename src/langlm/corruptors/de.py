"""German error injection.

Each rule finds places a correct sentence could plausibly be got wrong in the
way a learner gets it wrong -- not in the way a random-noise process would. The
error types are ERRANT tags, so the injected mix can be compared directly with
the Falko-MERLIN histogram and with the per-type recall the Phase 1 baselines
reported.

Almost everything here works on tokens alone. German makes that possible in a
way English would not: articles and prepositions are closed classes that can be
sentence-initial is a noun with few enough exceptions to be useful. Word
order and the separable prefix are the exceptions: they need a parse, and are
registered with `parse_corruptor` so a caller without one gets the other rules
rather than an exception.
gets the other rules rather than an exception.
"""

from __future__ import annotations

from collections.abc import Sequence

from langlm.corruptors.base import Corruption, corruptor, parse_corruptor

# --- closed classes ---------------------------------------------------------

#: Definite and indefinite articles, grouped so a swap stays within the class
#: and produces a wrong article rather than a non-word.
DEFINITE = ("der", "die", "das", "den", "dem", "des")
INDEFINITE = ("ein", "eine", "einen", "einem", "einer", "eines")

#: Prepositions a learner confuses with each other. Grouped by what actually
#: gets mixed up -- the two-way locatives, then the rest -- rather than by an
#: alphabetical list, which would produce swaps no learner would ever make.
PREPOSITION_SETS = (
    ("in", "an", "auf", "über", "unter", "neben", "zwischen", "vor", "hinter"),
    ("mit", "bei", "von", "zu", "nach", "aus", "seit"),
    ("für", "um", "durch", "gegen", "ohne"),
)

#: Keyboard neighbours on a German QWERTZ layout, for typos that look like
#: typing rather than like a random letter substitution.
NEIGHBOURS = {
    "a": "qsy",
    "b": "vgn",
    "c": "xdv",
    "d": "sfec",
    "e": "wrd",
    "f": "dgrv",
    "g": "fhtb",
    "h": "gjzn",
    "i": "uok",
    "j": "hkum",
    "k": "jlim",
    "l": "kop",
    "m": "njk",
    "n": "bhm",
    "o": "ipl",
    "p": "ol",
    "q": "wa",
    "r": "etf",
    "s": "adwx",
    "t": "rzg",
    "u": "zij",
    "v": "cfb",
    "w": "qes",
    "x": "ycs",
    "y": "xas",
    "z": "tuh",
}


def _is_noun(token: str, index: int) -> bool:
    """A capitalised token that is not sentence-initial is almost always a noun."""
    return index > 0 and token[:1].isupper() and token.isalpha() and len(token) > 2


# --- orthography ------------------------------------------------------------


@parse_corruptor("noun_case")
def noun_case(tokens: Sequence[str], doc) -> list[Corruption]:
    """Lower-case a common noun. The most common error in German learner text.

    Proper nouns are skipped, and that is a pedagogical choice rather than a
    technical one. English capitalises names too, so an English speaker does not
    write `josef` -- the error they actually make is on common nouns, where
    German capitalises and English does not. Injecting `Tel` -> `tel` would
    teach the model to fix something nobody gets wrong, and the explanation
    attached to it ("this is a noun") reads oddly about a place name.

    Measured on the first build: without this, one noun-case example in five was
    a proper noun.
    """
    if len(doc) != len(tokens):
        return []
    return [
        Corruption(i, i + 1, token.lower(), "R:ORTH", "noun_case")
        for i, token in enumerate(tokens)
        if _is_noun(token, i) and doc[i].tag_ == "NN"
    ]


@corruptor("sharp_s")
def sharp_s(tokens: Sequence[str]) -> list[Corruption]:
    """Write ß as ss. Wrong after a long vowel; the reform did not abolish it."""
    return [
        Corruption(i, i + 1, token.replace("ß", "ss"), "R:SPELL", "sharp_s")
        for i, token in enumerate(tokens)
        if "ß" in token
    ]


@corruptor("umlaut")
def umlaut(tokens: Sequence[str]) -> list[Corruption]:
    """Strip the umlaut diacritic -- what a learner without the keyboard does."""
    table = str.maketrans("äöüÄÖÜ", "aouAOU")
    return [
        Corruption(i, i + 1, token.translate(table), "R:SPELL", "umlaut")
        for i, token in enumerate(tokens)
        if any(character in token for character in "äöüÄÖÜ")
    ]


@corruptor("das_dass")
def das_dass(tokens: Sequence[str]) -> list[Corruption]:
    """Swap das and dass, which sound identical and are not interchangeable."""
    swap = {"das": "dass", "dass": "das", "Das": "Dass", "Dass": "Das"}
    return [
        Corruption(i, i + 1, swap[token], "R:SPELL", "das_dass")
        for i, token in enumerate(tokens)
        if token in swap
    ]


# --- morphology and agreement -----------------------------------------------


@corruptor("article_form")
def article_form(tokens: Sequence[str]) -> list[Corruption]:
    """Use the wrong article. Gender and case, the two things German marks here."""
    found = []
    for i, token in enumerate(tokens):
        lower = token.lower()
        for group in (DEFINITE, INDEFINITE):
            if lower in group:
                for other in group:
                    if other != lower:
                        replacement = other.capitalize() if token[0].isupper() else other
                        found.append(
                            Corruption(i, i + 1, replacement, "R:DET:FORM", "article_form")
                        )
    return found


@corruptor("preposition")
def preposition(tokens: Sequence[str]) -> list[Corruption]:
    """Substitute a preposition a learner would plausibly reach for instead."""
    found = []
    for i, token in enumerate(tokens):
        lower = token.lower()
        for group in PREPOSITION_SETS:
            if lower in group:
                for other in group:
                    if other != lower:
                        replacement = other.capitalize() if token[0].isupper() else other
                        found.append(Corruption(i, i + 1, replacement, "R:ADP", "preposition"))
    return found


# --- omission ---------------------------------------------------------------


@corruptor("drop_article")
def drop_article(tokens: Sequence[str]) -> list[Corruption]:
    """Leave the article out, as speakers of article-less languages do."""
    return [
        Corruption(i, i + 1, "", "M:DET", "drop_article")
        for i, token in enumerate(tokens)
        if token.lower() in DEFINITE + INDEFINITE
    ]


#: Conjunctions and pronouns that open a subordinate clause. A comma before one
#: of these is there for a different reason than a comma before `sondern`, and
#: the learner needs telling which.
SUBORDINATORS = frozenset(
    {
        "dass",
        "weil",
        "wenn",
        "ob",
        "obwohl",
        "da",
        "damit",
        "als",
        "während",
        "bevor",
        "nachdem",
        "sodass",
        "falls",
        "indem",
        "seit",
        "sobald",
        "bis",
        "der",
        "die",
        "das",
        "dem",
        "den",
        "welche",
        "welcher",
        "welches",
        "wo",
    }
)

#: Conjunctions that take a comma before them even though they join two main
#: clauses -- the case English speakers most often miss, because English does
#: not require one.
COORDINATORS = frozenset({"sondern", "aber", "doch", "jedoch", "denn", "allein"})


@corruptor("drop_comma")
def drop_comma(tokens: Sequence[str]) -> list[Corruption]:
    """Leave out a comma. German requires them where English does not.

    The rule name records *which* comma rule was broken, because German has
    several and the explanation has to name the right one. Reading the first
    build, a comma before `sondern` and a comma around an apposition were both
    being explained as separating a subordinate clause, which is true of neither.
    """
    found = []
    for i, token in enumerate(tokens):
        if token != ",":
            continue
        following = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
        if following in SUBORDINATORS:
            rule = "drop_comma_subordinate"
        elif following in COORDINATORS:
            rule = "drop_comma_coordinating"
        else:
            rule = "drop_comma"
        found.append(Corruption(i, i + 1, "", "M:PUNCT", rule))
    return found


# --- typing -----------------------------------------------------------------


@corruptor("keyboard")
def keyboard(tokens: Sequence[str]) -> list[Corruption]:
    """Hit the neighbouring key.

    Only the first substitutable letter of each word is offered. A rule that
    offered every position would swamp the sampler with one error type and make
    the injected distribution a function of word length.
    """
    found = []
    for i, token in enumerate(tokens):
        if not token.isalpha() or len(token) < 4:
            continue
        for position, character in enumerate(token.lower()):
            if character in NEIGHBOURS:
                damaged = token[:position] + NEIGHBOURS[character][0] + token[position + 1 :]
                found.append(Corruption(i, i + 1, damaged, "R:SPELL", "keyboard"))
                break
    return found


#: Rules whose output is a plain misspelling rather than a real word. Kept
#: separate because a corpus of nothing but these teaches spell-checking, and
#: the project already has a spell checker.
NONWORD_RULES = frozenset({"umlaut", "sharp_s", "keyboard"})


# --- agreement and case (still token-only) ----------------------------------

#: German adjective endings. Swapping one for another is the single most common
#: agreement error in learner German, because the correct ending depends on
#: gender, case, number *and* what kind of article precedes it.
ADJECTIVE_ENDINGS = ("em", "en", "er", "es", "e")

#: Prepositions that take the accusative for movement and the dative for
#: position. Choosing between them is a decision English never asks a speaker
#: to make, which is why learners get it wrong long after they know the case
#: endings themselves.
TWO_WAY = frozenset({"in", "an", "auf", "über", "unter", "neben", "zwischen", "vor", "hinter"})

#: Dative and accusative forms that a two-way preposition chooses between.
#: Masculine and neuter only: the feminine `der`/`die` pair is ambiguous with
#: the genitive and the plural, so a swap there is often not the error it
#: claims to be.
CASE_SWAP = {"dem": "den", "den": "dem", "einem": "einen", "einen": "einem"}


@parse_corruptor("adjective_form")
def adjective_form(tokens: Sequence[str], doc) -> list[Corruption]:
    """Give an attributive adjective the wrong ending.

    This was a token-only rule that recognised adjectives by their ending, on
    the theory that a lower-case word ending in `-en` before a capitalised noun
    is attributive. It is not: the first build had it firing on articles,
    prepositions, quantifiers and once on the modal verb `koennen`, 22.6% of the
    time. The endings are shared by too much of the closed-class vocabulary for
    a list of exceptions to close the gap, so it asks the tagger instead.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for i, token in enumerate(tokens[:-1]):
        if doc[i].tag_ != "ADJA" or not token.islower():
            continue
        if not _is_noun(tokens[i + 1], i + 1):
            continue
        for ending in ADJECTIVE_ENDINGS:
            if token.endswith(ending) and len(token) > len(ending) + 2:
                stem = token[: -len(ending)]
                found.extend(
                    Corruption(i, i + 1, stem + other, "R:ADJ:FORM", "adjective_form")
                    for other in ADJECTIVE_ENDINGS
                    if other != ending
                )
                break
    return found


@corruptor("two_way_case")
def two_way_case(tokens: Sequence[str]) -> list[Corruption]:
    """Put the wrong case after a two-way preposition."""
    found = []
    for i, token in enumerate(tokens[:-1]):
        if token.lower() not in TWO_WAY:
            continue
        following = tokens[i + 1]
        swapped = CASE_SWAP.get(following.lower())
        if swapped:
            replacement = swapped.capitalize() if following[0].isupper() else swapped
            found.append(Corruption(i + 1, i + 2, replacement, "R:DET:FORM", "two_way_case"))
    return found


# --- word order (needs a parse) ---------------------------------------------

FINITE_TAGS = frozenset({"VVFIN", "VAFIN", "VMFIN"})


@parse_corruptor("verb_final")
def verb_final(tokens: Sequence[str], doc) -> list[Corruption]:
    """Put the finite verb second in a subordinate clause instead of last.

    German sends the finite verb to the end of a subordinate clause, and English
    does not, so learners write `dass er faehrt morgen nach Berlin` for `dass er
    morgen nach Berlin faehrt`. The clause is rewritten as a single span rather
    than as a move, because one span is one edit and a move would be two.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.tag_ != "KOUS":
            continue
        verb = token.head
        if verb.tag_ not in FINITE_TAGS or verb.i <= token.i + 2:
            continue
        clause = list(range(token.i + 1, verb.i + 1))
        if len(clause) < 3:
            continue
        # Subject first, then the verb, then everything that was between them.
        reordered = [tokens[verb.i], *[tokens[i] for i in clause[1:-1]]]
        found.append(
            Corruption(
                token.i + 1,
                verb.i + 1,
                " ".join([tokens[clause[0]], *reordered]),
                "R:WO",
                "verb_final",
            )
        )
    return found


@parse_corruptor("separable_prefix")
def separable_prefix(tokens: Sequence[str], doc) -> list[Corruption]:
    """Glue a separable prefix back onto its verb.

    `Der Zug faehrt um acht Uhr ab` becomes `Der Zug abfaehrt um acht Uhr`,
    which is what a learner writes who has met the infinitive `abfahren` and not
    yet met the rule that splits it.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.tag_ != "PTKVZ":
            continue
        verb = token.head
        if verb.tag_ not in FINITE_TAGS or verb.i >= token.i:
            continue
        merged = tokens[token.i] + tokens[verb.i]
        between = [tokens[i] for i in range(verb.i + 1, token.i)]
        found.append(
            Corruption(
                verb.i,
                token.i + 1,
                " ".join([merged, *between]),
                "R:WO",
                "separable_prefix",
            )
        )
    return found
