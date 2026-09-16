# Phase 3: the fine-tuned model

Falko-MERLIN **dev**: 2,503 sentences. Overcorrection set: 400 sentences of published German prose. Scored with the same MaxMatch and ERRANT machinery as Phase 1, so these numbers sit beside those ones honestly. The test split has not been read.

## The table

| System | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| identity | 0.0417 | 0.0002 | 0.0008 | 0.0% |
| languagetool | 0.5653 | 0.1992 | 0.4134 | 28.7% |
| zero-shot | 0.4761 | 0.0905 | 0.2571 | 10.0% |
| few-shot | 0.6600 | 0.2675 | 0.5102 | 15.0% |
| cw0125-test | 0.7330 | 0.6242 | 0.7083 | 14.0% |
| cw0125 | 0.7289 | 0.6304 | 0.7068 | 13.2% |
| fine-tuned-iter3 | 0.7268 | 0.5953 | 0.6960 | 12.5% |
| iter4-cw025-dpo-b5-test | 0.7038 | 0.6180 | 0.6848 | 12.5% |
| iter4-cw025-dpo-b5 | 0.7096 | 0.6193 | 0.6895 | 12.5% |
| iter4-cw025-dpo | 0.6867 | 0.6274 | 0.6739 | 15.0% |
| iter4-cw025-test | 0.7322 | 0.6176 | 0.7060 | 9.5% |
| iter4-cw025 | 0.7339 | 0.6174 | 0.7072 | 9.8% |
| iter4-cw05 | 0.7304 | 0.6072 | 0.7019 | 9.5% |
| iter5-dpo-test | 0.7065 | 0.6200 | 0.6873 | 15.5% |
| iter5-dpo | 0.7066 | 0.6326 | 0.6905 | 15.5% |
| iter5-test | 0.7325 | 0.6210 | 0.7071 | 13.0% |
| iter5 | 0.7278 | 0.6287 | 0.7056 | 13.0% |
| trimmed-healed | 0.7284 | 0.6022 | 0.6991 | 7.2% |
| trimmed-nohealing | 0.7327 | 0.6164 | 0.7061 | 9.5% |
| fine-tuned | 0.7273 | 0.5881 | 0.6944 | 13.0% |
| **resft-pruned6** | 0.6912 | 0.5557 | **0.6591** | 16.8% |

The bar is the best Phase 1 baseline, **F0.5 = 0.7083** (few-shot), not LanguageTool's 0.4134. This model **does not beat** it.

## Detection versus correction

| | Detection F0.5 | Correction F0.5 | Gap |
|---|---:|---:|---:|
| resft-pruned6 | 0.7069 | 0.5806 | +0.1264 |

## How much of that gap is disagreement rather than error

The same output, rescored with `langlm.eval.leniency`, which lets the system's replacement stand for the annotator's when the two are typographic variants, the same compound written closed or hyphenated, or thesaurus synonyms in the same inflection. The span still has to be right.

| | M2 F0.5 | Correction F0.5 |
|---|---:|---:|
| strict | 0.6591 | 0.5806 |
| lenient | 0.6626 | 0.5848 |

The headline stays the strict number. The lenient one is worth +0.0036, which is the answer to whether the detection-correction gap above is the annotator being fussy: it is not. Almost all of that gap is the model finding the right word to change and then changing it wrongly -- most often a misspelling whose intended word is recoverable, answered with a different word (`rabben` corrected to `finden`, not `rauben`).

## Where it helps

Per-error-type recall. This is the table Phase 2 is meant to be fixed from -- but read it as coverage, not as volume: across the 36 types with dev volume, the correlation between a type's train/dev volume ratio and its F0.5 is only r = 0.27, while types the corruptors produce at all average F0.5 0.679 against 0.488 for types they never produce. A weak type is one to start injecting, not one to inject more of.

| Error type | Gold | Recall |
|---|---:|---:|
| `R:SPELL` | 816 | 0.64 |
| `R:DET:FORM` | 693 | 0.77 |
| `M:PUNCT` | 582 | 0.66 |
| `R:OTHER` | 555 | 0.26 |
| `R:ORTH` | 529 | 0.67 |
| `R:ADJ:FORM` | 277 | 0.75 |
| `R:NOUN:FORM` | 260 | 0.72 |
| `U:PUNCT` | 220 | 0.56 |
| `R:ADP` | 218 | 0.34 |
| `M:DET` | 188 | 0.34 |
| `R:MORPH` | 186 | 0.52 |
| `R:WO` | 162 | 0.51 |
| `M:PRON` | 157 | 0.27 |
| `R:PUNCT` | 140 | 0.21 |
| `R:VERB:FORM` | 117 | 0.53 |

## Did it explain what it did?

Scored against the edits ERRANT derives from the model's *own* correction, not against the gold annotation. Whether an edit should have been made is the correction score's question; this one asks whether the model can say what it just did, because a rewrite it cannot describe is one it cannot explain.

| | Value | Reads as |
|---|---:|---|
| Coverage | 0.8013 | of 5,481 rewrites it made, 4,392 were listed |
| Precision | 0.8619 | of 5,599 changes it listed, 4,826 match a real rewrite |
| Type accuracy | 0.6890 | of the 4,826 that match, 3,325 carry the type ERRANT assigns |

Type accuracy is what the learner actually sees, because the type is the key `rules/de_types.yaml` renders the explanation from. Read all three against the oracle rather than against 1.00. Scoring the gold correction and the gold changes through this same scorer gives coverage 0.9795, precision 1.0000 and type accuracy 0.8387: ERRANT segments a rewrite slightly differently from the annotator, and re-derives the annotator's type on 86% of edits, the rest being coarsenings such as `M:PUNCT` to `M:OTHER`.

## Did it answer in the schema?

0 with no readable correction, 3 cut off but recovered, 957 left unchanged, out of 2,903 answers.

An answer with no readable correction is scored as an unchanged sentence, which is why it is counted here rather than left to look like restraint. Cut-off answers are counted separately because the schema puts the sentence first: the correction survives and only the `changes` list is lost, so they cost the explanation metrics above rather than the score.

## Sample answers

```json
{"correction": "Heute braucht die Welt praxisorientierte Spezialisten , die nicht nur Theorie wissen .", "changes": []}
```

```json
{"correction": "Diese Ereignisse haben doch einen Einfluss auf die Täter sowie die Opfer , die sich dadurch verbunden lassen .", "changes": [{"was": "ein", "now": "einen", "type": "R:DET:FORM"}, {"was": "der", "now": "die", "type": "R:DET:FORM"}]}
```

```json
{"correction": "Sie argumentierten , dass das Studium in der Folge von geringem Wert war .", "changes": [{"was": "der", "now": "das", "type": "R:DET:FORM"}, {"was": "als", "now": "in", "type": "R:ADP"}, {"was": "als Folge", "now": "als der Folge", "type": "M:DET"}]}
```

```json
{"correction": "Bitte schreiben Sie mir so bald wie möglich , damit ich meine Entscheidung treffen kann .", "changes": []}
```

```json
{"correction": "Sie hatten keine Möglichkeit sich auszubilden .", "changes": []}
```

