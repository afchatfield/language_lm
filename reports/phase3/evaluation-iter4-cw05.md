# Phase 3: the fine-tuned model

Falko-MERLIN **dev**: 2,503 sentences. Overcorrection set: 400 sentences of published German prose. Scored with the same MaxMatch and ERRANT machinery as Phase 1, so these numbers sit beside those ones honestly. The test split has not been read.

## The table

| System | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| identity | 0.0417 | 0.0002 | 0.0008 | 0.0% |
| languagetool | 0.5653 | 0.1992 | 0.4134 | 28.7% |
| zero-shot | 0.4761 | 0.0905 | 0.2571 | 10.0% |
| few-shot | 0.6600 | 0.2675 | 0.5102 | 15.0% |
| fine-tuned-iter3 | 0.7268 | 0.5953 | 0.6960 | 12.5% |
| fine-tuned | 0.7273 | 0.5881 | 0.6944 | 13.0% |
| **iter4-cw05** | 0.7304 | 0.6072 | **0.7019** | 9.5% |

The bar is the best Phase 1 baseline, **F0.5 = 0.6960** (few-shot), not LanguageTool's 0.4134. This model **beats** it.

## Detection versus correction

| | Detection F0.5 | Correction F0.5 | Gap |
|---|---:|---:|---:|
| iter4-cw05 | 0.7476 | 0.6256 | +0.1220 |

## How much of that gap is disagreement rather than error

The same output, rescored with `langlm.eval.leniency`, which lets the system's replacement stand for the annotator's when the two are typographic variants, the same compound written closed or hyphenated, or thesaurus synonyms in the same inflection. The span still has to be right.

| | M2 F0.5 | Correction F0.5 |
|---|---:|---:|
| strict | 0.7019 | 0.6256 |
| lenient | 0.7060 | 0.6291 |

The headline stays the strict number. The lenient one is worth +0.0041, which is the answer to whether the detection-correction gap above is the annotator being fussy: it is not. Almost all of that gap is the model finding the right word to change and then changing it wrongly -- most often a misspelling whose intended word is recoverable, answered with a different word (`rabben` corrected to `finden`, not `rauben`).

## Where it helps

Per-error-type recall. This is the table Phase 2 is meant to be fixed from -- but read it as coverage, not as volume: across the 36 types with dev volume, the correlation between a type's train/dev volume ratio and its F0.5 is only r = 0.27, while types the corruptors produce at all average F0.5 0.679 against 0.488 for types they never produce. A weak type is one to start injecting, not one to inject more of.

| Error type | Gold | Recall |
|---|---:|---:|
| `R:SPELL` | 816 | 0.65 |
| `R:DET:FORM` | 693 | 0.80 |
| `M:PUNCT` | 582 | 0.71 |
| `R:OTHER` | 555 | 0.32 |
| `R:ORTH` | 529 | 0.68 |
| `R:ADJ:FORM` | 277 | 0.79 |
| `R:NOUN:FORM` | 260 | 0.74 |
| `U:PUNCT` | 220 | 0.61 |
| `R:ADP` | 218 | 0.42 |
| `M:DET` | 188 | 0.41 |
| `R:MORPH` | 186 | 0.61 |
| `R:WO` | 162 | 0.57 |
| `M:PRON` | 157 | 0.33 |
| `R:PUNCT` | 140 | 0.36 |
| `R:VERB:FORM` | 117 | 0.61 |

## Did it explain what it did?

Scored against the edits ERRANT derives from the model's *own* correction, not against the gold annotation. Whether an edit should have been made is the correction score's question; this one asks whether the model can say what it just did, because a rewrite it cannot describe is one it cannot explain.

| | Value | Reads as |
|---|---:|---|
| Coverage | 0.8974 | of 5,565 rewrites it made, 4,994 were listed |
| Precision | 0.9489 | of 5,614 changes it listed, 5,327 match a real rewrite |
| Type accuracy | 0.7197 | of the 5,327 that match, 3,834 carry the type ERRANT assigns |

Type accuracy is what the learner actually sees, because the type is the key `rules/de_types.yaml` renders the explanation from. Read all three against the oracle rather than against 1.00. Scoring the gold correction and the gold changes through this same scorer gives coverage 0.9795, precision 1.0000 and type accuracy 0.8387: ERRANT segments a rewrite slightly differently from the annotator, and re-derives the annotator's type on 86% of edits, the rest being coarsenings such as `M:PUNCT` to `M:OTHER`.

## Did it answer in the schema?

1 with no readable correction, 4 cut off but recovered, 993 left unchanged, out of 2,903 answers.

An answer with no readable correction is scored as an unchanged sentence, which is why it is counted here rather than left to look like restraint. Cut-off answers are counted separately because the schema puts the sentence first: the correction survives and only the `changes` list is lost, so they cost the explanation metrics above rather than the score.

## Sample answers

```json
{"correction": "Heute braucht die Welt praxisorientierte Spezialisten , die nicht nur Theorie wissen .", "changes": []}
```

```json
{"correction": "Diese Ereignisse haben doch einen Einfluss auf die Täter sowie die Opfer , die sich dadurch verbunden lassen .", "changes": [{"was": "ein", "now": "einen", "type": "R:DET:FORM"}, {"was": ", der", "now": ", die der", "type": "M:PRON"}]}
```

```json
{"correction": "Sie argumentierten , dass das Studium als Folge von geringem Wert war .", "changes": [{"was": "der", "now": "das", "type": "R:DET:FORM"}]}
```

```json
{"correction": "Bitte schreiben Sie mir so bald wie möglich , damit ich meine Entscheidung treffen kann .", "changes": []}
```

```json
{"correction": "Sie hatten keine Möglichkeit , sich auszubilden .", "changes": [{"was": "Möglichkeit sich", "now": "Möglichkeit , sich", "type": "M:PUNCT"}]}
```

