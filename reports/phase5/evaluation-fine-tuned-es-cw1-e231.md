# Phase 5: the fine-tuned model

COWS-L2H **dev**: 4,716 sentences. Overcorrection set: 400 sentences of published Spanish prose. Scored with the same MaxMatch and ERRANT machinery as the baselines, so these numbers sit beside those ones honestly. The test split has not been read.

## The table

| System | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| identity | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| languagetool | 0.4241 | 0.1588 | 0.3179 | 27.3% |
| zero-shot | 0.3872 | 0.0510 | 0.1671 | 11.2% |
| few-shot | 0.5664 | 0.1088 | 0.3077 | 8.0% |
| fine-tuned-es-cw025 | 0.6510 | 0.4531 | 0.5987 | 6.0% |
| fine-tuned-es | 0.6373 | 0.4559 | 0.5904 | 7.0% |
| **fine-tuned-es-cw1-e231** | 0.6331 | 0.4597 | **0.5887** | 5.8% |

The bar is the strongest system it is measured against, **F0.5 = 0.5987** (fine-tuned-es-cw025), ahead of fine-tuned-es's 0.5904. This model **does not beat** it.

## Detection versus correction

| | Detection F0.5 | Correction F0.5 | Gap |
|---|---:|---:|---:|
| fine-tuned-es-cw1-e231 | 0.6317 | 0.5339 | +0.0977 |

## Where it helps

Per-error-type recall. This is the table the data pipeline is meant to be fixed from. Read it as coverage rather than volume: a type the corruptors never produce is one to start injecting, not one to inject more of. The German equivalent of this table carries a measured correlation between volume and F0.5; no such measurement exists for Spanish yet, so none is quoted.

| Error type | Gold | Recall |
|---|---:|---:|
| `R:OTHER` | 1251 | 0.20 |
| `R:ORTH` | 838 | 0.74 |
| `U:PRON` | 594 | 0.76 |
| `R:VERB:FORM` | 357 | 0.39 |
| `R:DET:FORM` | 327 | 0.66 |
| `U:DET` | 303 | 0.58 |
| `M:ADP` | 279 | 0.57 |
| `R:ADP` | 268 | 0.39 |
| `R:SPELL` | 257 | 0.65 |
| `R:VERB` | 250 | 0.24 |
| `M:PUNCT` | 249 | 0.38 |
| `M:DET` | 247 | 0.49 |
| `R:ADJ:FORM` | 221 | 0.67 |
| `R:NOUN` | 199 | 0.32 |
| `R:AUX:FORM` | 158 | 0.31 |

## Did it explain what it did?

Scored against the edits ERRANT derives from the model's *own* correction, not against the gold annotation. Whether an edit should have been made is the correction score's question; this one asks whether the model can say what it just did, because a rewrite it cannot describe is one it cannot explain.

| | Value | Reads as |
|---|---:|---|
| Coverage | 0.9462 | of 5,912 rewrites it made, 5,594 were listed |
| Precision | 0.9819 | of 5,927 changes it listed, 5,820 match a real rewrite |
| Type accuracy | 0.8316 | of the 5,820 that match, 4,840 carry the type ERRANT assigns |

Type accuracy is what the learner actually sees, because the type is the key `rules/de_types.yaml` renders the explanation from. Read all three against the oracle rather than against 1.00. Scoring the gold correction and the gold changes through this same scorer gives coverage 0.9795, precision 1.0000 and type accuracy 0.8387: ERRANT segments a rewrite slightly differently from the annotator, and re-derives the annotator's type on 86% of edits, the rest being coarsenings such as `M:PUNCT` to `M:OTHER`.

## Did it answer in the schema?

0 with no readable correction, 0 cut off but recovered, 2,356 left unchanged, out of 5,116 answers.

An answer with no readable correction is scored as an unchanged sentence, which is why it is counted here rather than left to look like restraint. Cut-off answers are counted separately because the schema puts the sentence first: the correction survives and only the `changes` list is lost, so they cost the explanation metrics above rather than the score.
