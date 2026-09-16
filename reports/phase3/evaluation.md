# Phase 3: the fine-tuned model

Falko-MERLIN **dev**: 2,503 sentences. Overcorrection set: 400 sentences of published German prose. Scored with the same MaxMatch and ERRANT machinery as Phase 1, so these numbers sit beside those ones honestly. The test split has not been read.

## The table

| System | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| identity | 0.0417 | 0.0002 | 0.0008 | 0.0% |
| languagetool | 0.5653 | 0.1992 | 0.4134 | 28.7% |
| zero-shot | 0.4761 | 0.0905 | 0.2571 | 10.0% |
| few-shot | 0.6600 | 0.2675 | 0.5102 | 15.0% |
| **fine-tuned** | 0.7273 | 0.5881 | **0.6944** | 13.0% |

The bar is the best Phase 1 baseline, **F0.5 = 0.5102** (few-shot), not LanguageTool's 0.4134. This model **beats** it.

## Detection versus correction

| | Detection F0.5 | Correction F0.5 | Gap |
|---|---:|---:|---:|
| fine-tuned | 0.7417 | 0.6174 | +0.1243 |

## How much of that gap is disagreement rather than error

The same output, rescored with `langlm.eval.leniency`, which lets the system's replacement stand for the annotator's when the two are typographic variants, the same compound written closed or hyphenated, or thesaurus synonyms in the same inflection. The span still has to be right.

| | M2 F0.5 | Correction F0.5 |
|---|---:|---:|
| strict | 0.6944 | 0.6174 |
| lenient | 0.6990 | 0.6216 |

The headline stays the strict number. The lenient one is worth +0.0046, which is the answer to whether the detection-correction gap above is the annotator being fussy: it is not. Almost all of that gap is the model finding the right word to change and then changing it wrongly -- most often a misspelling whose intended word is recoverable, answered with a different word (`rabben` corrected to `finden`, not `rauben`).

## Where it helps

Per-error-type recall. This is the table Phase 2 is meant to be fixed from -- but read it as coverage, not as volume: across the 36 types with dev volume, the correlation between a type's train/dev volume ratio and its F0.5 is only r = 0.27, while types the corruptors produce at all average F0.5 0.679 against 0.488 for types they never produce. A weak type is one to start injecting, not one to inject more of.

| Error type | Gold | Recall |
|---|---:|---:|
| `R:SPELL` | 816 | 0.64 |
| `R:DET:FORM` | 693 | 0.79 |
| `M:PUNCT` | 582 | 0.70 |
| `R:OTHER` | 555 | 0.30 |
| `R:ORTH` | 529 | 0.70 |
| `R:ADJ:FORM` | 277 | 0.79 |
| `R:NOUN:FORM` | 260 | 0.72 |
| `U:PUNCT` | 220 | 0.58 |
| `R:ADP` | 218 | 0.38 |
| `M:DET` | 188 | 0.39 |
| `R:MORPH` | 186 | 0.59 |
| `R:WO` | 162 | 0.57 |
| `M:PRON` | 157 | 0.29 |
| `R:PUNCT` | 140 | 0.29 |
| `R:VERB:FORM` | 117 | 0.58 |

## Did it explain what it did?

Scored against the edits ERRANT derives from the model's *own* correction, not against the gold annotation. Whether an edit should have been made is the correction score's question; this one asks whether the model can say what it just did, because a rewrite it cannot describe is one it cannot explain.

| | Value | Reads as |
|---|---:|---|
| Coverage | 0.8984 | of 5,472 rewrites it made, 4,916 were listed |
| Precision | 0.9649 | of 5,468 changes it listed, 5,276 match a real rewrite |
| Type accuracy | 0.7106 | of the 5,276 that match, 3,749 carry the type ERRANT assigns |

Type accuracy is what the learner actually sees, because the type is the key `rules/de_types.yaml` renders the explanation from. Read all three against the oracle rather than against 1.00. Scoring the gold correction and the gold changes through this same scorer gives coverage 0.9795, precision 1.0000 and type accuracy 0.8387: ERRANT segments a rewrite slightly differently from the annotator, and re-derives the annotator's type on 86% of edits, the rest being coarsenings such as `M:PUNCT` to `M:OTHER`.

## Did it answer in the schema?

0 with no readable correction, 5 cut off but recovered, 995 left unchanged, out of 2,903 answers.

An answer with no readable correction is scored as an unchanged sentence, which is why it is counted here rather than left to look like restraint. Cut-off answers are counted separately because the schema puts the sentence first: the correction survives and only the `changes` list is lost, so they cost the explanation metrics above rather than the score.
