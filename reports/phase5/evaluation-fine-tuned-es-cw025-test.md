# Phase 5: the fine-tuned model

COWS-L2H **test**: 4,284 sentences. Overcorrection set: 400 sentences of published Spanish prose. Scored with the same MaxMatch and ERRANT machinery as the baselines, so these numbers sit beside those ones honestly. This is the held-out split.

## The table

| System | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| identity | 0.5000 | 0.0001 | 0.0007 | 0.0% |
| languagetool | 0.4547 | 0.1630 | 0.3348 | 27.3% |
| zero-shot | 0.4632 | 0.0610 | 0.1997 | 11.2% |
| few-shot | 0.6092 | 0.1210 | 0.3371 | 8.0% |
| **fine-tuned-es-cw025-test** | 0.6878 | 0.4867 | **0.6353** | 6.0% |

The bar is the strongest system it is measured against, **F0.5 = 0.3371** (few-shot), ahead of languagetool's 0.3348. This model **beats** it.

## Detection versus correction

| | Detection F0.5 | Correction F0.5 | Gap |
|---|---:|---:|---:|
| fine-tuned-es-cw025-test | 0.6864 | 0.5932 | +0.0932 |

## Where it helps

Per-error-type recall. This is the table the data pipeline is meant to be fixed from. Read it as coverage rather than volume: a type the corruptors never produce is one to start injecting, not one to inject more of. The German equivalent of this table carries a measured correlation between volume and F0.5; no such measurement exists for Spanish yet, so none is quoted.

| Error type | Gold | Recall |
|---|---:|---:|
| `R:OTHER` | 996 | 0.23 |
| `R:ORTH` | 811 | 0.79 |
| `U:PRON` | 785 | 0.71 |
| `R:VERB:FORM` | 359 | 0.44 |
| `R:SPELL` | 356 | 0.76 |
| `R:DET:FORM` | 342 | 0.69 |
| `R:ADP` | 310 | 0.37 |
| `M:ADP` | 284 | 0.58 |
| `U:DET` | 268 | 0.50 |
| `R:VERB` | 244 | 0.30 |
| `M:PUNCT` | 219 | 0.32 |
| `M:DET` | 210 | 0.48 |
| `R:ADJ:FORM` | 205 | 0.61 |
| `R:NOUN` | 197 | 0.26 |
| `U:ADP` | 140 | 0.47 |

## Did it explain what it did?

Scored against the edits ERRANT derives from the model's *own* correction, not against the gold annotation. Whether an edit should have been made is the correction score's question; this one asks whether the model can say what it just did, because a rewrite it cannot describe is one it cannot explain.

| | Value | Reads as |
|---|---:|---|
| Coverage | 0.9309 | of 5,397 rewrites it made, 5,024 were listed |
| Precision | 0.9689 | of 5,374 changes it listed, 5,207 match a real rewrite |
| Type accuracy | 0.8270 | of the 5,207 that match, 4,306 carry the type ERRANT assigns |

Type accuracy is what the learner actually sees, because the type is the key `rules/de_types.yaml` renders the explanation from. Read all three against the oracle rather than against 1.00. Scoring the gold correction and the gold changes through this same scorer gives coverage 0.9795, precision 1.0000 and type accuracy 0.8387: ERRANT segments a rewrite slightly differently from the annotator, and re-derives the annotator's type on 86% of edits, the rest being coarsenings such as `M:PUNCT` to `M:OTHER`.

## Did it answer in the schema?

0 with no readable correction, 0 cut off but recovered, 2,147 left unchanged, out of 4,684 answers.

An answer with no readable correction is scored as an unchanged sentence, which is why it is counted here rather than left to look like restraint. Cut-off answers are counted separately because the schema puts the sentence first: the correction survives and only the `changes` list is lost, so they cost the explanation metrics above rather than the score.
