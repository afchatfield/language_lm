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
from langlm.eval.errant_de.spelling import is_known_word

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
    """Swap das and dass, which sound identical and are not interchangeable.

    Typed `R:OTHER`, not `R:SPELL`, because that is what the corpus calls it:
    220 of the 221 das/dass edits in Falko-MERLIN train and dev are `R:OTHER`.
    The two words are a determiner and a conjunction, so no part-of-speech
    label covers the pair and ERRANT falls through to `OTHER` -- which makes
    this rule the only injected coverage of dev's second-largest type.
    """
    swap = {"das": "dass", "dass": "das", "Das": "Dass", "Dass": "Das"}
    return [
        Corruption(i, i + 1, swap[token], "R:OTHER", "das_dass")
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


# --- pronouns ---------------------------------------------------------------

#: Personal pronouns whose dative and accusative forms learners exchange. Only
#: the unambiguous pairs: `ihr` is dative singular feminine, a possessive and a
#: second-person plural subject all at once, so a swap there is as likely to
#: produce a different error as the intended one.
PRONOUN_CASE = {
    "mir": "mich",
    "mich": "mir",
    "dir": "dich",
    "dich": "dir",
    "ihm": "ihn",
    "ihn": "ihm",
}


@corruptor("pronoun_form")
def pronoun_form(tokens: Sequence[str]) -> list[Corruption]:
    """Exchange a dative pronoun for its accusative, or the other way round.

    German marks on the pronoun what English marks with word order -- `mir` and
    `mich` are both `me` -- so the choice has nothing in the learner's first
    language to hang on.
    """
    found = []
    for i, token in enumerate(tokens):
        swapped = PRONOUN_CASE.get(token.lower())
        if swapped:
            replacement = swapped.capitalize() if token[0].isupper() else swapped
            found.append(Corruption(i, i + 1, replacement, "R:PRON:FORM", "pronoun_form"))
    return found


@parse_corruptor("drop_pronoun")
def drop_pronoun(tokens: Sequence[str], doc) -> list[Corruption]:
    """Leave out a subject or reflexive pronoun.

    Two learners make this error for two reasons. A speaker of a pro-drop
    language -- Italian, Spanish, Polish -- omits the subject because their own
    grammar lets them; German never does, because its verb endings no longer
    identify the subject on their own. An English speaker omits `sich`, because
    the English verb it translates is not reflexive: *he interests for music*.
    """
    if len(doc) != len(tokens):
        return []
    return [
        Corruption(token.i, token.i + 1, "", "M:PRON", "drop_pronoun")
        for token in doc
        # Not sentence-initial: dropping the first word leaves the next one
        # lower-case, which is a second error the edit does not claim.
        if token.i > 0 and (token.tag_ == "PRF" or (token.tag_ == "PPER" and token.dep_ == "sb"))
    ]


# --- prepositions and auxiliaries -------------------------------------------


@parse_corruptor("drop_preposition")
def drop_preposition(tokens: Sequence[str], doc) -> list[Corruption]:
    """Leave out a preposition the verb or the phrase requires.

    `warten auf`, `denken an`, `sich interessieren für`: which preposition a
    German verb governs is not predictable from the English one, and the learner
    who does not know it often writes none at all. Contracted forms (`im`,
    `zum`) are left alone -- removing one would take an article with it, and the
    edit would be two errors wearing the label of one.
    """
    if len(doc) != len(tokens):
        return []
    return [
        Corruption(token.i, token.i + 1, "", "M:ADP", "drop_preposition")
        for token in doc
        if token.tag_ == "APPR" and token.i > 0
    ]


#: The finite forms of the three auxiliaries, grouped by tense. A swap stays
#: inside its group so that the error is the agreement one a learner makes,
#: rather than a change of tense wearing an agreement label.
#: The second person is left out of every group, and not for tidiness. It barely
#: occurs in edited prose, so the sentences available to damage do not offer it
#: a plausible context -- `die Saison warst die Ausgabe` is not an error any
#: learner makes. What learners do make is the third-person number error, which
#: is also what the corpus shows: `ist` for `sind` and `wird` for `werden` are
#: the two commonest auxiliary edits in Falko-MERLIN.
AUXILIARY_FORMS = (
    ("bin", "ist", "sind"),
    ("war", "waren"),
    ("habe", "hat", "haben"),
    ("hatte", "hatten"),
    ("werde", "wird", "werden"),
    ("wurde", "wurden"),
)


@parse_corruptor("auxiliary_form")
def auxiliary_form(tokens: Sequence[str], doc) -> list[Corruption]:
    """Make the auxiliary disagree with its subject.

    Asks the tagger rather than matching the surface form, because `haben`,
    `werden` and `sein` are also full verbs and their forms are some of the most
    frequent words in the language: `sie werden` as a passive auxiliary and `sie
    werden` as *they become* want different explanations, and `wird` inside a
    quoted title wants neither.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.tag_ != "VAFIN":
            continue
        lower = token.text.lower()
        for group in AUXILIARY_FORMS:
            if lower not in group:
                continue
            found.extend(
                Corruption(
                    token.i,
                    token.i + 1,
                    other.capitalize() if token.text[0].isupper() else other,
                    "R:AUX:FORM",
                    "auxiliary_form",
                )
                for other in group
                if other != lower
            )
            break
    return found


@parse_corruptor("drop_auxiliary")
def drop_auxiliary(tokens: Sequence[str], doc) -> list[Corruption]:
    """Leave out the auxiliary of a perfect or a passive.

    `Ich habe gegessen` becomes `Ich gegessen`. The participle is where the
    meaning is, so it is the auxiliary that a learner under pressure drops --
    and German, unlike English, cannot recover it from anywhere else.
    """
    if len(doc) != len(tokens):
        return []
    return [
        Corruption(token.i, token.i + 1, "", "M:AUX", "drop_auxiliary")
        for token in doc
        if token.tag_ == "VAFIN"
        and token.i > 0
        and any(child.tag_ in {"VVPP", "VAPP", "VMPP"} for child in token.children)
    ]


# --- verb and noun morphology -----------------------------------------------


@parse_corruptor("verb_infinitive")
def verb_infinitive(tokens: Sequence[str], doc) -> list[Corruption]:
    """Use the infinitive where the sentence needs a finite verb.

    The commonest shape of learner German at A2: the verb is known in the form
    the dictionary lists it in, and the endings that agree it with a subject
    come later. Forms that are already identical to the infinitive -- the
    first and third person plural of nearly every German verb -- offer nothing
    to damage and are skipped.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.tag_ not in {"VVFIN", "VMFIN"}:
            continue
        infinitive = token.lemma_
        if not infinitive or infinitive.lower() == token.text.lower():
            continue
        if not is_known_word(infinitive):
            continue
        if token.text[0].isupper():
            infinitive = infinitive.capitalize()
        # A modal is an auxiliary to the tagger and to the corpus, so `kann` ->
        # `können` is an R:AUX:FORM edit however much it feels like a verb one.
        error_type = "R:AUX:FORM" if token.tag_ == "VMFIN" else "R:VERB:FORM"
        found.append(Corruption(token.i, token.i + 1, infinitive, error_type, "verb_infinitive"))
    return found


@parse_corruptor("genitive_s")
def genitive_s(tokens: Sequence[str], doc) -> list[Corruption]:
    """Leave the genitive ending off a masculine or neuter noun.

    `der Preis des Hauses` becomes `der Preis des Haus`. The case is already
    marked on the article, so to a learner the ending on the noun looks like
    something the sentence has said once already.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.tag_ != "NN" or token.morph.get("Case") != ["Gen"]:
            continue
        if token.morph.get("Number") != ["Sing"]:
            continue
        for ending in ("es", "s"):
            stem = token.text[: -len(ending)]
            # The stem must itself be a word. Without that test the rule strips
            # the `s` off `Praxis` and calls the non-word it leaves a case error.
            if token.text.endswith(ending) and len(stem) > 2 and is_known_word(stem):
                found.append(Corruption(token.i, token.i + 1, stem, "R:NOUN:FORM", "genitive_s"))
                break
    return found


@parse_corruptor("dative_plural_n")
def dative_plural_n(tokens: Sequence[str], doc) -> list[Corruption]:
    """Leave the -n off a dative plural.

    `mit den Kindern` becomes `mit den Kinder`. German adds an -n to the plural
    in the dative and nowhere else, so the form appears only in a case the
    learner is already unsure of. The parse decides, not the surface: `den` is a
    dative plural and an accusative singular masculine both, and stripping the
    -n from the second produces a non-word rather than a case error.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.tag_ != "NN" or token.morph.get("Case") != ["Dat"]:
            continue
        if token.morph.get("Number") != ["Plur"] or not token.text.endswith("n"):
            continue
        stem = token.text[:-1]
        if len(stem) > 3 and is_known_word(stem):
            found.append(Corruption(token.i, token.i + 1, stem, "R:NOUN:FORM", "dative_plural_n"))
    return found


# --- punctuation a German sentence does not take ----------------------------


@parse_corruptor("comma_after_fronted")
def comma_after_fronted(tokens: Sequence[str], doc) -> list[Corruption]:
    """Put an English comma after a fronted adverbial.

    *Yesterday, I went home* is English punctuation; `Gestern , ging ich nach
    Hause` is not German. The comma is wrong precisely because German has
    already marked the fronting by moving the verb, so nothing is left for the
    comma to do.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.tag_ not in FINITE_TAGS or token.dep_ != "ROOT" or token.i < 1:
            continue
        if tokens[token.i - 1] == ",":
            continue
        # The verb is second and something other than the subject is first, so
        # what precedes it is a fronted element rather than a subject.
        if any(child.dep_ == "sb" and child.i < token.i for child in token.children):
            continue
        found.append(Corruption(token.i, token.i, ",", "U:PUNCT", "comma_after_fronted"))
    return found


@parse_corruptor("comma_before_und")
def comma_before_und(tokens: Sequence[str], doc) -> list[Corruption]:
    """Put a comma before `und` where it joins two words rather than two clauses.

    English writes *bread, and butter*; German writes `Brot und Butter` and
    nothing else. Restricted to coordinations with no verb after the
    conjunction, because a comma before an `und` that joins two main clauses is
    permitted and injecting one would be labelling correct German as an error.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.tag_ != "KON" or token.text.lower() not in {"und", "oder"}:
            continue
        if token.i == 0 or tokens[token.i - 1] == ",":
            continue
        rest = doc[token.i + 1 :]
        if any(later.tag_ in FINITE_TAGS for later in rest):
            continue
        found.append(Corruption(token.i, token.i, ",", "U:PUNCT", "comma_before_und"))
    return found
