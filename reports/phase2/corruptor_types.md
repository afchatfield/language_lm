# Phase 2: do the corruptors produce the errors they claim?

Each rule was run over 600 clean sentences; one of the corruptions it offered was applied, and the damaged sentence re-annotated with the same German ERRANT the evaluation uses. The column is how often the tag the rule claims is among the tags ERRANT derives.

| Rule | Claims | Fired | ERRANT agrees |
|---|---|---:|---:|
| `pronoun_form` | `R:PRON:FORM` | 10 | 0.0% * |
| `separable_prefix` | `R:WO` | 67 | 0.0% * |
| `verb_final` | `R:WO` | 66 | 19.7% * |
| `dative_plural_n` | `R:NOUN:FORM` | 126 | 69.8% * |
| `umlaut` | `R:SPELL` | 433 | 82.4% * |
| `adjective_form` | `R:ADJ:FORM` | 306 | 83.3% * |
| `genitive_s` | `R:NOUN:FORM` | 62 | 91.9% |
| `article_form` | `R:DET:FORM` | 523 | 92.0% |
| `drop_article` | `M:DET` | 523 | 92.9% |
| `sharp_s` | `R:SPELL` | 76 | 93.4% |
| `preposition` | `R:ADP` | 495 | 94.3% |
| `das_dass` | `R:OTHER` | 132 | 95.5% |
| `keyboard` | `R:SPELL` | 600 | 97.5% |
| `verb_infinitive` | `R:VERB:FORM` | 343 | 98.0% |
| `two_way_case` | `R:DET:FORM` | 62 | 98.4% |
| `auxiliary_form` | `R:AUX:FORM` | 273 | 99.3% |
| `drop_preposition` | `M:ADP` | 462 | 99.8% |
| `comma_after_fronted` | `U:PUNCT` | 273 | 100.0% |
| `comma_before_und` | `U:PUNCT` | 106 | 100.0% |
| `drop_auxiliary` | `M:AUX` | 157 | 100.0% |
| `drop_comma` | `M:PUNCT` | 288 | 100.0% |
| `drop_pronoun` | `M:PRON` | 162 | 100.0% |
| `noun_case` | `R:ORTH` | 586 | 100.0% |

\* a known disagreement, where ERRANT and the corpus differ and the corpus wins:

* `adjective_form` -- an ending swap sometimes produces another real word
* `dative_plural_n` -- spacy lemmatises long compounds inconsistently, so ERRANT reads the number change as a different word
* `pronoun_form` -- spacy gives `mir` and `mich` different lemmas; the corpus calls the pair `R:PRON:FORM` on 154 of 156 edits
* `separable_prefix` -- same: the prefix moving reads as two edits, not one
* `sharp_s` -- `ss` for `ß` is a real spelling in Switzerland and ERRANT types some of them as orthography
* `umlaut` -- a stripped umlaut sometimes lands on a real word, which ERRANT types by part of speech rather than as a misspelling
* `verb_final` -- ERRANT has no move operation and describes a reordering as a deletion plus an insertion

## What ERRANT says instead

* `pronoun_form` claims `R:PRON:FORM`, gets `R:PRON` (10)
* `separable_prefix` claims `R:WO`, gets `R:VERB+M:ADP` (44), `R:VERB+M:ADV` (19), `R:OTHER+M:ADP` (2)
* `verb_final` claims `R:WO`, gets `U:AUX+M:AUX` (31), `U:VERB+M:VERB` (17), `U:ADJ+M:VERB` (4)
* `dative_plural_n` claims `R:NOUN:FORM`, gets `R:NOUN` (28), `R:OTHER` (7), `R:MORPH` (3)
* `umlaut` claims `R:SPELL`, gets `R:NOUN` (37), `R:AUX:FORM` (13), `R:VERB` (8)
* `adjective_form` claims `R:ADJ:FORM`, gets `R:ADJ` (40), `R:MORPH` (7), `R:OTHER` (3)

Every rule below 90% is a known disagreement.

