"""A softened equality for replacement text, and the tiers it is built from.

The strict scorers ask whether the system's replacement is the annotator's
replacement, character for character. That is the right default, but on a
single-annotator corpus it charges the system for disagreements no reader would
call errors: `Kartoffel-Suppe` for `Kartoffelsuppe`, `Wirtschaft` for `Ökonomie`.
This module supplies an equivalence that forgives exactly those, so a lenient
score can be reported beside the strict one.

The design constraint is that the relation must be *tight*. Dropping the
replacement entirely and scoring on spans alone -- the obvious way to be
generous -- is not a softer metric but a broken one: the MaxMatch lattice is
free to describe an edit however it likes, so with the text unconstrained it
re-describes its rewrites to land on whatever spans the gold annotation touches.
Measured on Falko-MERLIN dev, that route reports F0.5 = 0.869 against a strict
0.694, with a recall of 0.847 that the same system's ERRANT detection recall
(0.661) says it cannot have. Every tier below therefore constrains the text.

Three tiers, in order of how much evidence they need:

* **typographic** -- the same characters under Unicode normalisation, with
  curly quotes, dashes and non-breaking spaces folded. No linguistics involved.
* **compound** -- closed against hyphenated writing of one compound
  (`Chemiestudentin` / `Chemie-Studentin`). Both are correct German; the
  Bindestrich is the writer's call. A *spaced* form is not included, because in
  German that is the error itself.
* **synonym** -- a lexical substitution licensed by OpenThesaurus, restricted to
  a same-length, same-part-of-speech replacement of content words. This is the
  only tier that can be wrong, so it is the only one that is optional and the
  only one with a measured error rate: on 60 hand-judged disputes from dev it
  accepted 8, all 8 of which a reader marked acceptable, and missed 10 that a
  reader accepted but no thesaurus links (`Händler` for a gold `Handeler` that
  is not a word, `kostspielig` for a gold `köstlich` that misreads the sentence).

What the whole thing is worth is small and worth knowing: it moves dev F0.5 from
0.694 to about 0.702. The gap between detection and correction is not the
annotator being fussy; it is the model choosing the wrong word.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from langlm.config import RAW_DIR

#: Where `scripts/download_thesaurus.py` puts OpenThesaurus.
THESAURUS_PATH = RAW_DIR / "openthesaurus" / "openthesaurus.txt"

#: Parts of speech whose choice is a matter of wording rather than of grammar.
#: A determiner or a preposition swapped for another is a grammatical decision
#: with one right answer, so those never qualify however close the two words are.
CONTENT_POS = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV"})

#: Characters that differ between keyboards and typesetters rather than between
#: words.
_TYPOGRAPHIC = {
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
}


def typographic(text: str) -> str:
    """Normalise the characters a writer does not choose deliberately."""
    text = unicodedata.normalize("NFC", text)
    for source, target in _TYPOGRAPHIC.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def same_compound(gold: str, hypothesis: str) -> bool:
    """True for one compound written closed on one side and hyphenated on the other.

    Case is ignored on the parts because hyphenation moves it: `Chemiestudentin`
    against `Chemie-Studentin` is the same word written two admissible ways. A
    form containing a space is refused -- `Kartoffel Suppe` is the learner's
    error, not a variant of it.
    """
    gold, hypothesis = typographic(gold), typographic(hypothesis)
    if " " in gold or " " in hypothesis:
        return False
    if ("-" in gold) == ("-" in hypothesis):
        return False
    return gold.replace("-", "").lower() == hypothesis.replace("-", "").lower()


@cache
def load_thesaurus(path: Path = THESAURUS_PATH) -> Mapping[str, frozenset[str]]:
    """Read OpenThesaurus into a lowercase word -> synonyms mapping.

    The text export is one synonym set per line, semicolon separated, with
    parenthesised usage notes that are dropped. Multi-word entries are dropped
    too: this is used to license a single-token substitution.

    Raises:
        FileNotFoundError: with the command that fetches the file.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No thesaurus at {path}. Run `python scripts/download_thesaurus.py` "
            f"to fetch OpenThesaurus, or score without the synonym tier."
        )
    groups: dict[str, set[str]] = defaultdict(set)
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or ";" not in line:
                continue
            words = [word.split("(")[0].strip() for word in line.rstrip("\n").split(";")]
            words = [word for word in words if word and " " not in word]
            for word in words:
                lowered = word.lower()
                groups[lowered].update(other.lower() for other in words if other != word)
    return {word: frozenset(others) for word, others in groups.items()}


#: Morphological features the sentence imposes on whichever word fills the slot.
#: Gender is deliberately absent: it belongs to the word, not to the position, so
#: two synonyms may differ in it and still be interchangeable. A feature missing
#: from one side is not compared, because a bare fragment out of context is often
#: tagged with nothing at all.
GOVERNED_FEATURES = ("Number", "Case", "Tense", "Mood", "Person", "Degree")


def _same_inflection(left, right) -> bool:
    """Whether two tokens are inflected alike on the features that are governed."""
    for feature in GOVERNED_FEATURES:
        ours, theirs = left.morph.get(feature), right.morph.get(feature)
        if ours and theirs and ours != theirs:
            return False
    return True


@dataclass(frozen=True)
class Leniency:
    """An equivalence over replacement strings, assembled from the tiers above.

    Attributes:
        synonyms: Word to synonyms, or ``None`` to leave the synonym tier off.
        compounds: Whether to forgive closed against hyphenated compounds.
    """

    synonyms: Mapping[str, frozenset[str]] | None = None
    compounds: bool = True

    @classmethod
    def strict(cls) -> Leniency:
        """No leniency at all: the identity of the strict scorers."""
        return cls(synonyms=None, compounds=False)

    @classmethod
    def full(cls, path: Path = THESAURUS_PATH) -> Leniency:
        """Every tier, thesaurus included."""
        return cls(synonyms=load_thesaurus(path), compounds=True)

    def __call__(self, gold: str, hypothesis: str) -> bool:
        return self.equivalent(gold, hypothesis)

    def equivalent(self, gold: str, hypothesis: str) -> bool:
        """Whether the hypothesis's replacement text may stand for the gold's."""
        if gold == hypothesis:
            return True
        if typographic(gold) == typographic(hypothesis):
            return True
        if self.compounds and same_compound(gold, hypothesis):
            return True
        return bool(self.synonyms) and self._synonymous(gold, hypothesis)

    def _synonymous(self, gold: str, hypothesis: str) -> bool:
        """A same-shape content-word substitution the thesaurus licenses.

        Both sides are parsed so that the comparison can be made on lemmas: the
        annotator's `Ökonomie` and the model's `Wirtschaft` are listed together,
        but `Gesellschaft` and `Gemeinschaften` are not until the second is
        reduced to its lemma. Requiring the same part-of-speech sequence keeps
        the tier from licensing a change of construction, which is grammar
        rather than word choice.

        The lemma finds the link; it must not also excuse the inflection. Going
        through lemmas means `Zwänge` reaches `Druck`, and accepting that would
        forgive a change of number -- exactly the kind of error the system is
        being scored on. So the surface forms have to agree on the features the
        sentence governs as well.
        """
        from langlm.eval import errant_de

        gold_tokens = errant_de.parse(gold)
        hyp_tokens = errant_de.parse(hypothesis)
        if not gold_tokens or len(gold_tokens) != len(hyp_tokens):
            return False
        for left, right in zip(gold_tokens, hyp_tokens, strict=True):
            if left.pos_ != right.pos_ or left.pos_ not in CONTENT_POS:
                return False
            if left.text == right.text:
                continue
            if not _same_inflection(left, right):
                return False
            if not self._linked(left.text, left.lemma_, right.text, right.lemma_):
                return False
        return True

    def _linked(self, gold: str, gold_lemma: str, hyp: str, hyp_lemma: str) -> bool:
        assert self.synonyms is not None
        left = {gold.lower(), gold_lemma.lower()}
        right = {hyp.lower(), hyp_lemma.lower()}
        return any(other in self.synonyms.get(word, ()) for word in left for other in right)


def matcher(
    gold: Iterable[tuple[int, int, str]],
    equivalent: Leniency | Callable[[str, str], bool] | None,
) -> Callable[[tuple[int, int, str]], bool]:
    """Build the predicate the scorers use to ask "does this edit match gold?".

    With no leniency this is set membership, which is what the strict scorers
    have always done. With leniency it falls back to asking whether some gold
    edit on the same span proposes an equivalent replacement -- the span still
    has to be right, which is what stops the softening from turning into the
    span-only metric this module's docstring warns about.
    """
    exact = set(gold)
    if equivalent is None:
        return exact.__contains__

    by_span: dict[tuple[int, int], list[str]] = defaultdict(list)
    for start, end, correction in exact:
        by_span[(start, end)].append(correction)

    def match(edit: tuple[int, int, str]) -> bool:
        if edit in exact:
            return True
        start, end, replacement = edit
        return any(
            equivalent(candidate, replacement) for candidate in by_span.get((start, end), ())
        )

    return match
