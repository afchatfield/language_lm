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
| fine-tuned-es-cw1-e231 | 0.6331 | 0.4597 | 0.5887 | 5.8% |
| fine-tuned-es | 0.6373 | 0.4559 | 0.5904 | 7.0% |
| **trimmed-nohealing-es** | 0.6453 | 0.4534 | **0.5949** | 7.5% |

The bar is the strongest system it is measured against, **F0.5 = 0.5987** (fine-tuned-es-cw025), ahead of fine-tuned-es's 0.5904. This model **does not beat** it.

## Detection versus correction

| | Detection F0.5 | Correction F0.5 | Gap |
|---|---:|---:|---:|
| trimmed-nohealing-es | 0.6370 | 0.5456 | +0.0913 |

## Where it helps

Per-error-type recall. This is the table the data pipeline is meant to be fixed from. Read it as coverage rather than volume: a type the corruptors never produce is one to start injecting, not one to inject more of. The German equivalent of this table carries a measured correlation between volume and F0.5; no such measurement exists for Spanish yet, so none is quoted.

| Error type | Gold | Recall |
|---|---:|---:|
| `R:OTHER` | 1251 | 0.20 |
| `R:ORTH` | 838 | 0.75 |
| `U:PRON` | 594 | 0.74 |
| `R:VERB:FORM` | 357 | 0.36 |
| `R:DET:FORM` | 327 | 0.63 |
| `U:DET` | 303 | 0.58 |
| `M:ADP` | 279 | 0.56 |
| `R:ADP` | 268 | 0.41 |
| `R:SPELL` | 257 | 0.67 |
| `R:VERB` | 250 | 0.26 |
| `M:PUNCT` | 249 | 0.34 |
| `M:DET` | 247 | 0.49 |
| `R:ADJ:FORM` | 221 | 0.67 |
| `R:NOUN` | 199 | 0.35 |
| `R:AUX:FORM` | 158 | 0.25 |

## Did it explain what it did?

Scored against the edits ERRANT derives from the model's *own* correction, not against the gold annotation. Whether an edit should have been made is the correction score's question; this one asks whether the model can say what it just did, because a rewrite it cannot describe is one it cannot explain.

| | Value | Reads as |
|---|---:|---|
| Coverage | 0.9179 | of 5,667 rewrites it made, 5,202 were listed |
| Precision | 0.8989 | of 6,035 changes it listed, 5,425 match a real rewrite |
| Type accuracy | 0.8100 | of the 5,425 that match, 4,394 carry the type ERRANT assigns |

Type accuracy is what the learner actually sees, because the type is the key `rules/de_types.yaml` renders the explanation from. Read all three against the oracle rather than against 1.00. Scoring the gold correction and the gold changes through this same scorer gives coverage 0.9795, precision 1.0000 and type accuracy 0.8387: ERRANT segments a rewrite slightly differently from the annotator, and re-derives the annotator's type on 86% of edits, the rest being coarsenings such as `M:PUNCT` to `M:OTHER`.

## Did it answer in the schema?

2 with no readable correction, 1 cut off but recovered, 2,421 left unchanged, out of 5,116 answers.

An answer with no readable correction is scored as an unchanged sentence, which is why it is counted here rather than left to look like restraint. Cut-off answers are counted separately because the schema puts the sentence first: the correction survives and only the `changes` list is lost, so they cost the explanation metrics above rather than the score.

## Sample answers

```json
{"correction": "Trump es bajo , gordo y viejo .", "changes": [{"was": "corto", "now": "bajo", "type": "R:ADJ"}]}
```

```json
{"correction": "Por la mañana después , * FIRST_NAME * se despertó y se sintió mal por la inconveniencia .", "changes": [{"was": "después", "now": "después después", "type": "M:ADV"}, {"was": "sentía", "now": "se sintió", "type": "R:OTHER"}]}
```

```json
{"correction": "La razón principal por la que yo aprecio tanto a Leonardo DiCaprio , es por todo el trabajo que hace para luchar contra el cambio climático .", "changes": []}
```

```json
{"correction": "¡ Me gusta escuchar música mucho ! Cuando era niño me gustaba correr rápidamente .", "changes": [{"was": "Me gusta", "now": "¡ Me gusta", "type": "M:PUNCT"}]}
```

```json
{"correction": "Una persona famosa tiene muchas cosas de lujo y no se preocupa mucho por el dinero .", "changes": []}
```

