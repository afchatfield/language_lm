# Phase 3: the fine-tuned model

Falko-MERLIN **test**: 2,337 sentences. Overcorrection set: 400 sentences of published German prose. Scored with the same MaxMatch and ERRANT machinery as the baselines, so these numbers sit beside those ones honestly. This is the held-out split.

## The table

| System | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| identity | 0.1026 | 0.0007 | 0.0033 | 0.0% |
| languagetool | 0.5890 | 0.2091 | 0.4321 | 28.7% |
| zero-shot | 0.4634 | 0.0880 | 0.2501 | 10.0% |
| few-shot | 0.6431 | 0.2574 | 0.4948 | 15.2% |
| cw0125-test | 0.7330 | 0.6242 | 0.7083 | 14.0% |
| iter4-cw025-dpo-b5-test | 0.7038 | 0.6180 | 0.6848 | 12.5% |
| iter5-dpo-test | 0.7065 | 0.6200 | 0.6873 | 15.5% |
| iter5-test | 0.7325 | 0.6210 | 0.7071 | 13.0% |
| **iter4-cw025-test** | 0.7322 | 0.6176 | **0.7060** | 9.5% |

The bar is the strongest system it is measured against, **F0.5 = 0.7083** (cw0125-test), ahead of iter5-test's 0.7071. This model **does not beat** it.

## Detection versus correction

| | Detection F0.5 | Correction F0.5 | Gap |
|---|---:|---:|---:|
| iter4-cw025-test | 0.7387 | 0.6224 | +0.1163 |

## How much of that gap is disagreement rather than error

The same output, rescored with `langlm.eval.leniency`, which lets the system's replacement stand for the annotator's when the two are typographic variants, the same compound written closed or hyphenated, or thesaurus synonyms in the same inflection. The span still has to be right.

| | M2 F0.5 | Correction F0.5 |
|---|---:|---:|
| strict | 0.7060 | 0.6224 |
| lenient | 0.7094 | 0.6258 |

The headline stays the strict number. The lenient one is worth +0.0034, which is the answer to whether the detection-correction gap above is the annotator being fussy: it is not. Almost all of that gap is the model finding the right word to change and then changing it wrongly -- most often a misspelling whose intended word is recoverable, answered with a different word (`rabben` corrected to `finden`, not `rauben`).

## Where it helps

Per-error-type recall. This is the table the data pipeline is meant to be fixed from -- but read it as coverage, not as volume: across the 36 types with dev volume, the correlation between a type's train/dev volume ratio and its F0.5 is only r = 0.27, while types the corruptors produce at all average F0.5 0.679 against 0.488 for types they never produce. A weak type is one to start injecting, not one to inject more of.

| Error type | Gold | Recall |
|---|---:|---:|
| `R:SPELL` | 834 | 0.66 |
| `R:DET:FORM` | 580 | 0.80 |
| `M:PUNCT` | 565 | 0.70 |
| `R:ORTH` | 495 | 0.68 |
| `R:OTHER` | 485 | 0.31 |
| `R:ADJ:FORM` | 239 | 0.78 |
| `R:NOUN:FORM` | 219 | 0.78 |
| `M:DET` | 217 | 0.47 |
| `U:PUNCT` | 206 | 0.55 |
| `R:ADP` | 197 | 0.42 |
| `R:MORPH` | 169 | 0.63 |
| `M:PRON` | 136 | 0.32 |
| `R:WO` | 135 | 0.56 |
| `R:VERB:FORM` | 121 | 0.62 |
| `R:PUNCT` | 115 | 0.36 |

## Did it explain what it did?

Scored against the edits ERRANT derives from the model's *own* correction, not against the gold annotation. Whether an edit should have been made is the correction score's question; this one asks whether the model can say what it just did, because a rewrite it cannot describe is one it cannot explain.

| | Value | Reads as |
|---|---:|---|
| Coverage | 0.8568 | of 5,251 rewrites it made, 4,499 were listed |
| Precision | 0.9161 | of 5,317 changes it listed, 4,871 match a real rewrite |
| Type accuracy | 0.6921 | of the 4,871 that match, 3,371 carry the type ERRANT assigns |

Type accuracy is what the learner actually sees, because the type is the key `rules/de_types.yaml` renders the explanation from. Read all three against the oracle rather than against 1.00. Scoring the gold correction and the gold changes through this same scorer gives coverage 0.9795, precision 1.0000 and type accuracy 0.8387: ERRANT segments a rewrite slightly differently from the annotator, and re-derives the annotator's type on 86% of edits, the rest being coarsenings such as `M:PUNCT` to `M:OTHER`.

## Did it answer in the schema?

0 with no readable correction, 6 cut off but recovered, 968 left unchanged, out of 2,737 answers.

An answer with no readable correction is scored as an unchanged sentence, which is why it is counted here rather than left to look like restraint. Cut-off answers are counted separately because the schema puts the sentence first: the correction survives and only the `changes` list is lost, so they cost the explanation metrics above rather than the score.
