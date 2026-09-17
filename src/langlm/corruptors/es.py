"""Spanish error injection.

Each rule finds places a correct sentence could plausibly be got wrong in the
way a learner gets it wrong -- not in the way a random-noise process would.
Mirrors `corruptors/de.py` in shape and discipline; what differs is which
things are closed classes (articles and the `por`/`para`-style preposition
pairs, same as German) and which need a parse (gender/number agreement is
productive here, the way it is not for German's fixed article table).

**Two roadmap-named phenomena are not here, and that is a finding, not an
oversight.** `es_core_news_sm` -- the small model, kept small for the same
reason `de_core_news_sm` is -- lemmatises and tenses irregular preterites
unreliably: `estuvimos` came back `Tense=Pres`, `hube` came back `Tense=Pres`
with lemma `hubir`, `tuviste` with lemma `ener`, and `estuviste` was not tagged
as a verb at all. A rule keyed on those tags would silently mistype its own
training data, which is exactly the failure Phase 2's own inspection caught
for German ("correct correction, wrong explanation") -- so **preterite vs
imperfect is declined**, the same way cLang-8 and CEDEL2 were declined
elsewhere in this project, on the same grounds: not free, not this pass.
**Leísmo/laísmo is declined for a different reason** -- it is standard usage
in large parts of the Spanish-speaking world, and COWS-L2H's own annotators
plainly agree (`R:PRON:FORM` is 1.17% of all edits, `R:PRON` 0.58%): "correcting"
it would teach the model to fight a dialect rather than an error. A rule that
generated data no honest annotator would have marked is worse than no rule.

`ser`/`estar` survives, but scoped to the present tense only: `es_core_news_sm`
mistags the noun "era" (the digital *era*, a geological *era*) as this verb's
copula form in every construction tested, and the collision is specific to the
imperfect -- present-tense forms (`es`, `son`, `estoy`, `están`, ...) have no
such homograph. `subjunctive_for_indicative` survives on the same terms,
restricted to the two verbs whose subjunctive the tagger gets right often
enough to trust (`sea`/`sean`/`estemos` tagged correctly; `seas`/`estés` did
not, so the rule simply never fires there -- a missed corruption, not a wrong
one, and the safe side of that trade).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from langlm.corruptors.base import Corruption, Corruptor
from langlm.corruptors.base import corruptor as _corruptor
from langlm.corruptors.base import parse_corruptor as _parse_corruptor
from langlm.eval.errant_es.spelling import is_known_word

#: Spanish's own tables, never the German ones `base.py` defaults to. `de.py`
#: and this file reuse rule names -- `article_form`, `preposition`,
#: `adjective_form`, `verb_infinitive`, `keyboard`, `drop_article`,
#: `drop_preposition` -- for rules that do different things in each language,
#: so importing both into one namespace would raise on the first collision.
REGISTRY: dict[str, Corruptor] = {}
NEEDS_PARSE: set[str] = set()

corruptor = partial(_corruptor, registry=REGISTRY, needs_parse_set=NEEDS_PARSE)
parse_corruptor = partial(_parse_corruptor, registry=REGISTRY, needs_parse_set=NEEDS_PARSE)

# --- closed classes ---------------------------------------------------------

#: Spanish has no case, so the article table is gender x number only -- half
#: the size of German's, which also varies by case.
DEFINITE = ("el", "la", "los", "las")
INDEFINITE = ("un", "una", "unos", "unas")

#: Confusable preposition sets. `por`/`para` is the headline pair the roadmap
#: names -- both translate as English "for", and nothing in a learner's first
#: language forces the distinction. The second and third groups are the other
#: two English speakers reach for by habit: "en" doing the work three English
#: prepositions split between them, and "con"/"sin" for the times a learner
#: negates the wrong way.
PREPOSITION_SETS = (
    ("por", "para"),
    ("a", "en", "de"),
    ("con", "sin"),
)

#: Keyboard neighbours on a Spanish QWERTY layout. The one letter German's
#: table does not need: `ñ` sits to the right of `l`, its own key, not a
#: diacritic a learner can drop the way an umlaut can -- so it appears here as
#: a neighbour to type instead of, never as something `drop_accent` strips.
NEIGHBOURS = {
    "a": "qsz",
    "b": "vgn",
    "c": "xdv",
    "d": "sfxc",
    "e": "wrd",
    "f": "dgrvc",
    "g": "fhtb",
    "h": "gjyn",
    "i": "ujko",
    "j": "hkun",
    "k": "jlmi",
    "l": "kopñ",
    "m": "njk",
    "n": "bhjm",
    "ñ": "lp",
    "o": "iklp",
    "p": "olñ",
    "q": "wa",
    "r": "edft",
    "s": "adwxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qes",
    "x": "zsdc",
    "y": "tugh",
    "z": "asx",
}

# --- orthography -------------------------------------------------------------

#: Stressed vowel -> plain vowel. The same table `errant_es.classifier` uses to
#: recognise a stress mark as the same word restressed, not `ñ` or `ü` -- see
#: that module's docstring for why folding those would be wrong rather than
#: merely incomplete.
_UNSTRESS = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")


@corruptor("drop_accent")
def drop_accent(tokens: Sequence[str]) -> list[Corruption]:
    """Drop a stress mark. The single largest orthography class in COWS-L2H.

    Unlike German's noun-capitalisation rule, this needs no parse and no
    closed list: any word carrying a. e. i. o or u with a stress mark is a
    candidate, because the mark is what makes this an orthography error rather
    than a different one -- `errant_es.classifier` folds a stress-only
    difference into `R:ORTH` by the same rule that recognises it here.
    """
    return [
        Corruption(i, i + 1, token.translate(_UNSTRESS), "R:ORTH", "drop_accent")
        for i, token in enumerate(tokens)
        if token != token.translate(_UNSTRESS)
    ]


@corruptor("keyboard")
def keyboard(tokens: Sequence[str]) -> list[Corruption]:
    """Hit the neighbouring key. Same rule as German's, Spanish layout.

    Only the first substitutable letter of each word is offered, for the same
    reason German's version limits itself: offering every position would swamp
    the sampler with one error type and make the injected distribution a
    function of word length rather than of what learners actually do.

    Skips a slip that happens to land on a real word -- `tías` (aunts) one key
    over is `rías` (estuaries), a genuine Spanish noun. Spanish's lexicon is
    dense enough that this is not rare, and unlike German's version of this
    rule, which never hit it in the corpus this project measured against, a
    stress-test over 400 real COWS-L2H sentences found it on the first pass.
    Left unguarded, the explanation "«{wrong}» is not a Spanish word" would be
    false on exactly the sentences it fires on.
    """
    found = []
    for i, token in enumerate(tokens):
        if not token.isalpha() or len(token) < 4:
            continue
        for position, character in enumerate(token.lower()):
            if character in NEIGHBOURS:
                damaged = token[:position] + NEIGHBOURS[character][0] + token[position + 1 :]
                if not is_known_word(damaged):
                    found.append(Corruption(i, i + 1, damaged, "R:SPELL", "keyboard"))
                break
    return found


# --- punctuation --------------------------------------------------------------


@corruptor("drop_inverted_punctuation")
def drop_inverted_punctuation(tokens: Sequence[str]) -> list[Corruption]:
    """Leave out the opening ¿ or ¡. English has no equivalent to forget.

    Spanish is one of the only languages that marks a question or exclamation
    at both ends, and a learner whose first language marks only the end -- which
    is every other language this project touches -- has nothing in their own
    grammar to remind them the opening mark exists.
    """
    return [
        Corruption(i, i + 1, "", "M:PUNCT", "drop_inverted_punctuation")
        for i, token in enumerate(tokens)
        if token in ("¿", "¡")
    ]


# --- determiners and agreement ------------------------------------------------


@parse_corruptor("article_form")
def article_form(tokens: Sequence[str], doc) -> list[Corruption]:
    """Use the wrong article. Gender and number, the two things Spanish marks
    here -- there is no case to also get wrong, unlike German's four-way table.

    This has to be a parse-based rule and German's equivalent does not: `el`,
    `la`, `los`, `las` are also Spanish's third-person object clitics ("no la
    veo" -- I don't see *her*), homographic with the articles rather than a
    separate paradigm the way German's determiners and pronouns are. Matching
    the surface form alone -- what the first version of this rule did -- swaps
    a clitic pronoun and calls the result a wrong article, which a
    stress-test over real COWS-L2H sentences caught within the first sample.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.pos_ != "DET":
            continue
        lower = token.text.lower()
        for group in (DEFINITE, INDEFINITE):
            if lower in group:
                found.extend(
                    Corruption(
                        token.i,
                        token.i + 1,
                        other.capitalize() if token.text[0].isupper() else other,
                        "R:DET:FORM",
                        "article_form",
                    )
                    for other in group
                    if other != lower
                )
    return found


@parse_corruptor("drop_article")
def drop_article(tokens: Sequence[str], doc) -> list[Corruption]:
    """Leave the article out, as speakers of article-less first languages do.

    Parse-based for the same reason `article_form` is: `el`/`la`/`los`/`las`
    are also object clitics, and dropping one of those is `drop_object_clitic`'s
    job, not this rule's -- they are different errors with different
    explanations even though the two rules would otherwise pick the same span.
    """
    if len(doc) != len(tokens):
        return []
    return [
        Corruption(token.i, token.i + 1, "", "M:DET", "drop_article")
        for token in doc
        if token.pos_ == "DET" and token.text.lower() in DEFINITE + INDEFINITE
    ]


#: Endings a regular Spanish adjective takes for gender and number. Checked
#: longest first so `-os`/`-as` is not read as a hit on the bare `-o`/`-a`
#: ending one character short.
_ADJECTIVE_ENDINGS = ("os", "as", "o", "a")

#: Which swaps this rule offers: one axis at a time -- gender OR number, never
#: both. `-a` -> `-o` is gender alone; `-a` -> `-as` is number alone; `-a` ->
#: `-os` flips both and is not offered. Restricting to one axis is the more
#: realistic learner error on its own grounds and was worth doing regardless,
#: but it is not a fix for the mistype rate below: a second stress-test after
#: adding it found single-axis swaps failing too (`exitosa` -> `exitosas`,
#: gender held constant, still came back `R:OTHER`), so the real cause is the
#: same one `verb_infinitive` has -- inconsistent re-lemmatisation of an
#: inflected form under `es_core_news_sm`, not how many features moved.
_ADJECTIVE_SWAPS = {"os": ("as", "o"), "as": ("os", "a"), "o": ("a", "os"), "a": ("o", "as")}


@parse_corruptor("adjective_form")
def adjective_form(tokens: Sequence[str], doc) -> list[Corruption]:
    """Give an adjective the wrong gender or number ending.

    Unlike German's version of this rule, which has to ask the tagger to tell
    an adjective from a preposition sharing its ending, Spanish adjective
    agreement is productive rather than closed -- so the only thing worth a
    parse here is confirming the word actually is an adjective, not guessing
    which suffix means what.

    The `is_known_word` check exists because some adjectives ending in `-a` do
    not inflect for gender at all -- `indígena` has no `indígeno` -- and
    spacy's morphology tags `indígena` `Gender=Fem` regardless, the same as it
    tags a real feminine like `roja`, so the tag cannot tell the two classes
    apart. The dictionary can: `indígenos` was the actual, wrong output of an
    earlier version of this rule, found the same way German's original
    `adjective_form` bug was, by running it over real sentences and reading
    what came out.

    **This rule's measured mistype rate is higher still: 33.3% of 48 fires**
    over the same sample, for the same reason `verb_infinitive` has one:
    `es_core_news_sm` does not always re-lemmatise an inflected adjective back
    to the same lemma once the sentence around it has been damaged, and when
    it does not, `errant_es` reasonably calls the edit `R:ADJ` or `R:OTHER`
    rather than `R:ADJ:FORM`. As with `verb_infinitive`, the corrected
    *sentence* is never wrong -- only the type label attached to how the edit
    is explained. This is the rule most worth revisiting if the project ever
    moves past the small spacy model.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.pos_ != "ADJ":
            continue
        lower = token.text.lower()
        for ending in _ADJECTIVE_ENDINGS:
            if lower.endswith(ending) and len(lower) > len(ending) + 2:
                stem = token.text[: -len(ending)]
                found.extend(
                    Corruption(token.i, token.i + 1, stem + other, "R:ADJ:FORM", "adjective_form")
                    for other in _ADJECTIVE_SWAPS[ending]
                    if is_known_word(stem + other)
                )
                break
    return found


# --- prepositions --------------------------------------------------------------


@corruptor("preposition")
def preposition(tokens: Sequence[str]) -> list[Corruption]:
    """Substitute a preposition a learner would plausibly reach for instead."""
    found = []
    for i, token in enumerate(tokens):
        lower = token.lower()
        for group in PREPOSITION_SETS:
            if lower in group:
                found.extend(
                    Corruption(
                        i,
                        i + 1,
                        other.capitalize() if token[0].isupper() else other,
                        "R:ADP",
                        "preposition",
                    )
                    for other in group
                    if other != lower
                )
    return found


#: The only two obligatory contractions in Spanish. Left alone for the same
#: reason German's `drop_preposition` leaves `im`/`zum` alone: removing one
#: takes an article with it, and the edit would be two errors wearing the
#: label of one.
_CONTRACTIONS = frozenset({"al", "del"})


@parse_corruptor("drop_preposition")
def drop_preposition(tokens: Sequence[str], doc) -> list[Corruption]:
    """Leave out a preposition. Spanish governs prepositions by verb the same
    way German does -- `pensar en`, `soñar con` -- and the learner who does not
    know which one a verb takes often writes none at all.
    """
    if len(doc) != len(tokens):
        return []
    return [
        Corruption(token.i, token.i + 1, "", "M:ADP", "drop_preposition")
        for token in doc
        if token.pos_ == "ADP" and token.i > 0 and token.text.lower() not in _CONTRACTIONS
    ]


# --- pronouns: the pro-drop headline ------------------------------------------

#: (person, number) -> the pronoun a finite verb's own morphology names
#: unambiguously. Third person is not here on purpose: Spanish subject
#: pronouns split by gender in the third person (él/ella) and by dialect in
#: the second plural (vosotros/ustedes), and a verb's own agreement carries
#: neither -- inserting one would be a guess wearing the shape of a correction.
_SUBJECT_PRONOUN = {("1", "Sing"): "yo", ("2", "Sing"): "tú", ("1", "Plur"): "nosotros"}


@parse_corruptor("redundant_subject_pronoun")
def redundant_subject_pronoun(tokens: Sequence[str], doc) -> list[Corruption]:
    """Insert a subject pronoun Spanish leaves to the verb ending.

    The single largest correctable category in COWS-L2H (`U:PRON`, 8.15% of
    all edits) has no German counterpart at all: German never drops a subject,
    so `errant_de` never had to type this direction. It is the clearest sign
    the two languages needed different corruptors rather than the same one
    with different words plugged in -- an over-explicit "yo" is the transfer
    error an English or heritage-Spanish speaker makes, not the omission a
    German learner makes.

    Sentence-initial verbs are skipped: inserting a pronoun there would also
    have to move the capital letter, which is a second edit the correction
    does not claim to make -- the same reason German's `drop_pronoun` skips
    position zero.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.i == 0 or token.pos_ not in {"VERB", "AUX"}:
            continue
        if token.morph.get("VerbForm") != ["Fin"]:
            continue
        person = token.morph.get("Person")
        number = token.morph.get("Number")
        if not person or not number:
            continue
        pronoun = _SUBJECT_PRONOUN.get((person[0], number[0]))
        if pronoun is None:
            continue
        if any(child.dep_ in {"nsubj", "nsubj:pass", "expl"} for child in token.children):
            continue  # already has an explicit subject; not this rule's job
        found.append(Corruption(token.i, token.i, pronoun, "U:PRON", "redundant_subject_pronoun"))
    return found


@parse_corruptor("drop_object_clitic")
def drop_object_clitic(tokens: Sequence[str], doc) -> list[Corruption]:
    """Leave out an object or reflexive clitic.

    `Le dije la verdad` becomes `Dije la verdad`, and `Me llamo Juan` becomes
    `Llamo Juan` -- the reflexive that names a person is not optional the way
    German's `sich` sometimes reads as optional to an English speaker, but a
    learner still drops it under the same pressure that drops any clitic that
    carries no stress of its own.
    """
    if len(doc) != len(tokens):
        return []
    return [
        Corruption(token.i, token.i + 1, "", "M:PRON", "drop_object_clitic")
        for token in doc
        if token.pos_ == "PRON" and token.i > 0 and token.dep_ in {"obj", "iobj", "expl"}
    ]


# --- ser / estar and the subjunctive -------------------------------------------

#: Present tense only. `es_core_news_sm` mistags the noun "era" (a geological
#: or historical *era*) as this verb's copula in every construction tested --
#: `Estamos en la era digital` parses `era` as `AUX`/`cop` just as readily as
#: the real copula does -- and the collision is specific to the imperfect.
#: Present-tense forms have no such homograph.
_SER_ESTAR = {
    "soy": "estoy",
    "estoy": "soy",
    "eres": "estás",
    "estás": "eres",
    "es": "está",
    "está": "es",
    "somos": "estamos",
    "estamos": "somos",
    "sois": "estáis",
    "estáis": "sois",
    "son": "están",
    "están": "son",
}


@parse_corruptor("ser_estar")
def ser_estar(tokens: Sequence[str], doc) -> list[Corruption]:
    """Swap `ser` for `estar`, or the reverse -- the confusion the roadmap
    names first, because English collapses both into one verb: "to be".

    Skips identity statements -- `Su esposa es Michelle Obama` -- where the
    complement is a proper name. Spanish grammar requires `ser` there and
    forbids `estar` outright, which is not a subtle preference: the swap
    produces something so ungrammatical that `es_core_news_sm`'s own re-parse
    stops recognising the result as a copula at all, typing it `R:OTHER`
    instead of the `R:AUX` this rule claims. Found the same way every other
    guard in this file was, by running it over real sentences first.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.pos_ != "AUX" or token.dep_ != "cop":
            continue
        if token.head.pos_ == "PROPN":
            continue
        swap = _SER_ESTAR.get(token.text.lower())
        if swap is None:
            continue
        replacement = swap.capitalize() if token.text[0].isupper() else swap
        # errant_es's classifier tags this copula AUX, not VERB, so a swap
        # between two AUX forms of different lemmas lands on R:AUX -- verified
        # against a real annotation rather than assumed from the German rule
        # this is modelled on, where the equivalent swap is a VERB.
        found.append(Corruption(token.i, token.i + 1, replacement, "R:AUX", "ser_estar"))
    return found


#: Present subjunctive of `ser` and `estar`, keyed by surface form so the swap
#: does not depend on the tagger's lemma -- which is exactly what fails for
#: this pair in the forms `SUBJUNCTIVE_INDICATIVE` deliberately excludes: see
#: the module docstring for `seas`/`estés`.
_SUBJUNCTIVE_INDICATIVE = {
    "sea": "es",
    "seamos": "somos",
    "sean": "son",
    "esté": "está",
    "estemos": "estamos",
    "estén": "están",
}


@parse_corruptor("subjunctive_for_indicative")
def subjunctive_for_indicative(tokens: Sequence[str], doc) -> list[Corruption]:
    """Use the indicative where `que` requires the subjunctive.

    Restricted to `ser`/`estar`, and only the person/number combinations
    `es_core_news_sm` tags `Mood=Sub` correctly and consistently -- `sea`,
    `seamos`, `sean`, `esté`, `estemos`, `estén` all did across every sentence
    tested; `seas` and `estés` did not, tagging as a noun and an unrelated verb
    respectively. A combination this rule cannot verify is one it does not
    touch, which trades recall for never mistyping its own training data.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.pos_ != "AUX" or token.morph.get("Mood") != ["Sub"]:
            continue
        replacement = _SUBJUNCTIVE_INDICATIVE.get(token.text.lower())
        if replacement is None:
            continue
        # "que" within the last few tokens is the textbook trigger this rule
        # targets, not a claim about the clause boundary the parse would give
        # more precisely -- a cheap, robust proxy for the one thing that
        # matters: a reader can see why the subjunctive was called for here.
        window = tokens[max(0, token.i - 4) : token.i]
        if not any(word.lower() == "que" for word in window):
            continue
        replacement = replacement.capitalize() if token.text[0].isupper() else replacement
        # Same lemma, same AUX tag as `ser_estar` -- an inflection, not a word
        # choice, so `errant_es` calls it R:AUX:FORM. Verified, not assumed.
        found.append(
            Corruption(
                token.i, token.i + 1, replacement, "R:AUX:FORM", "subjunctive_for_indicative"
            )
        )
    return found


# --- verb morphology -----------------------------------------------------------


@parse_corruptor("verb_infinitive")
def verb_infinitive(tokens: Sequence[str], doc) -> list[Corruption]:
    """Use the infinitive where the sentence needs a finite verb.

    The commonest shape of early learner Spanish, the same way it is for
    German: the verb is known in its dictionary form, and the ending that
    agrees it with a subject has not been learned yet. `ser`, `estar` and
    `haber` are excluded -- they are what `ser_estar` and the auxiliary rules
    already cover, and a bare `ser`/`estar` in running text reads as a
    different, stranger error than the one this rule is naming.

    **This rule mistypes its own output roughly one time in five (22.3% of
    121 fires, one corruption sampled per sentence over 800 real COWS-L2H
    sentences), and that number is measured, not guessed.** Spanish
    infinitives nominalise -- `el fumar es malo`, "smoking is bad" -- so a
    bare infinitive standing where a finite verb should be is genuinely
    ambiguous to a parser, not only to a reader, and `es_core_news_sm`'s
    re-parse of the damaged sentence often lands somewhere other than the
    clean `VERB` reading it started from (`R:OTHER`, `R:MORPH`, `R:VERB`,
    `R:AUX:FORM` instead of `R:VERB:FORM`). Whether the verb sat in the
    clause's root position was tested as a predictor over the full candidate
    set, not just the one sample per sentence above, and does not separate the
    failures from the successes -- 12.9% bad at root, 16.6% off it -- so this
    ships with the measured rate attached rather than a scope-down the
    evidence does not support. The **correction itself is never wrong** --
    `apply()` always restores the real original word, so F0.5 and the
    corrected sentence are unaffected -- only the `type` this rule claims for
    its own edit sometimes is, which lands on the per-type histogram and the
    `changes` list the model is trained to emit.
    """
    if len(doc) != len(tokens):
        return []
    found = []
    for token in doc:
        if token.pos_ not in {"VERB", "AUX"} or token.morph.get("VerbForm") != ["Fin"]:
            continue
        if token.lemma_ in {"ser", "estar", "haber"}:
            continue
        infinitive = token.lemma_
        if not infinitive or infinitive.lower() == token.text.lower():
            continue
        if not is_known_word(infinitive):
            continue
        if token.text[0].isupper():
            infinitive = infinitive.capitalize()
        found.append(Corruption(token.i, token.i + 1, infinitive, "R:VERB:FORM", "verb_infinitive"))
    return found
