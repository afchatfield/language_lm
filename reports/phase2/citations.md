# Phase 2: the Regelwerk citations

Phase 2 closed with one debt: six of fifteen `reference:` fields in `rules/de.yaml`
were `null` because the Rat für deutsche Rechtschreibung paragraph had not been
checked, and the note said to verify them before Phase 3. Nobody did, the file grew
to 25 rules, and the count reached 20 null of 25.

The debt was real and it was pointed at the wrong half of the file. The nulls were
harmless. **Three of the five numbers that were filled in are wrong.**

## What was wrong

Checked against the *Amtliches Regelwerk 2024*, the current edition, free from
rechtschreibrat.com:

| Rule | Cited | § says, in 2024 | Correct |
|---|---|---|---|
| `drop_comma_subordinate` | § 74 | the semicolon | **§ 73** |
| `comma_before_und` | § 72 | the comma marks a Zusatz | **§ 71** |
| `comma_after_fronted` | § 71 | no comma in an und/oder Reihung | **none exists** |
| `noun_case` | § 55 | Substantive werden großgeschrieben | § 55 ✓ |
| `sharp_s` | § 25 | ß nach langem Vokal oder Diphthong | § 25 ✓ |

Two of the three are not carelessness, they are edition drift. The 2024 edition
renumbered part E. In the 2018 text § 74 *is* "Nebensätze grenzt man mit Komma ab"
and § 72 *is* the und/oder rule, so both citations were right when they were
written and both rotted in place when the Rat republished. The file never recorded
which edition it was citing, so nothing could notice.

That is the failure mode worth naming. A stale citation does not look stale. It
points at a real paragraph that says something plausible about punctuation, and the
only way to catch it is to look.

The third is a plain error. `comma_after_fronted` teaches that German puts no comma
after a fronted phrase, which is true, and cited § 71, which is about something
else. There is no paragraph to cite: part E is written as a list of cases where a
comma *is* set, and never forbids one. The absence after a fronted phrase follows
from no rule applying, which is not a citation.

## What the nulls actually were

Not unfinished work. The Regelwerk covers sound-letter assignment, compound
spelling, hyphens, capitalisation, punctuation and line breaking — and nothing
else. Seventeen of the 25 rules are grammar: agreement, case government, declension,
word order, verb government. The Rat has no jurisdiction over any of them and never
will, so no amount of checking would have filled those fields in.

Three of the twenty nulls were citable and had simply not been looked up:

| Rule | Now cites | Why it is orthography |
|---|---|---|
| `das_dass` | § 4(6), § 4 E2 | § 4(6) lists `das (Artikel, Pronomen)` among the single-consonant exceptions; E2 sends `dass (Konjunktion)` back to § 2. The Regelwerk contrasts the pair on exactly the article-versus-conjunction axis the explanation uses. |
| `separable_prefix` | § 34 | trennbare Zusammensetzungen are written as one word only in the infinitive, the participles, and in a subordinate clause at final position — so the prefix detaching in a main clause is Getrennt- und Zusammenschreibung, not syntax. |
| `drop_comma_coordinating` | § 71 E2 | "Vor diesen Konjunktionen steht das Komma", of `aber, doch, jedoch, sondern`. |

Final state: **8 cited, 17 uncited**, every uncited one carrying a `reference_note`
saying why no citation exists, and a test asserting that it does. `null` now means
"no citation is possible", which is a finished state, and it can no longer be
confused with "nobody got to it".

## What this cost the model

Nothing. `langlm.train.format` keeps explanation prose out of the SFT target — the
model emits an ERRANT type and `langlm.explanations` renders the sentence at serving
time — so no citation has ever been in a supervised token. The wrong paragraph
numbers lived only as metadata in `data/processed/training/de.jsonl`, where 1,502
edits carried one of the three (687 × § 74, 621 × § 71, and 185 of the § 72 uses).

They were re-rendered in place rather than rebuilt: `make training-set` draws fresh
random corruptions and would produce a different corpus, breaking the correspondence
with the Phase 3 checkpoints. 6,832 explanations across 9 rules were re-cited, the
621 uncitable ones stripped, and the pre-fix file kept beside it.

So the serving path was fixed by fixing the YAML, and no retraining is implied. The
debt note's reasoning — "a wrong citation in 30k examples would teach the model to
cite confidently and wrongly" — turned out not to apply, because the schema had
already moved the explanation out of the model's mouth. It would have applied to the
corpus as it stood when the note was written.

## Why it will not rot again

`scripts/check_citations.py` (`make check-citations`, and in `make phase2`)
downloads the pinned edition, cuts off the Wörterverzeichnis — which cites
paragraphs on nearly every line and makes any number appear to exist everywhere —
and checks that each cited paragraph both exists and contains a phrase from the rule
it is supposed to state. The paragraph's E-notes are included, which is what makes
§ 4(6) and § 71 E2 checkable at all.

The numbers are verified against the PDF rather than remembered, and the edition is
pinned in `rules/de.yaml` and rendered into every citation, so the next renumbering
fails the check instead of silently repointing 700 explanations at the semicolon.
Restoring the two old numbers fails it today; that is the control that says the check
works.
