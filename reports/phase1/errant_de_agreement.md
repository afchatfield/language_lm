# German ERRANT: agreement with the corpus annotation

Falko-MERLIN **dev**, 2,503 sentences. Each gold correction was applied and then re-annotated from scratch by `langlm.eval.errant_de`; the result is compared with the corpus's own M2 annotation, which was produced by Boyd's (2018) German ERRANT.

| | |
|---|---:|
| Edits both annotators drew | 5,950 |
| Drawn only by this one | 317 |
| Drawn only by the corpus | 435 |
| Span precision | 0.9494 |
| Span recall | 0.9319 |
| **Type agreement on shared edits** | **0.8909** |

Span agreement is the number that matters. It bounds every score computed with this annotator: a system judged against differently-drawn edits is judged against the wrong thing. Type agreement only affects how the per-type breakdown reads, and some of the gap is irreducible -- the two annotators run different spacy versions, and the labels are downstream of the lemmas and tags those produce.

## Where the labels differ

The widest gap is `R:MORPH`, where the two annotators agree on 9% of the labels. The disagreement is about naming rather than finding: 181 of the corpus's 186 edits of this type are drawn identically here, but called `R:OTHER` (45), `R:NOUN:FORM` (30), `R:NOUN` (25) instead. The corpus reserves that label for any pair of words built on the same stem, which takes a stemmer; this classifier asks the narrower question of whether the two words share a lemma, and sends the rest to the part-of-speech buckets.

| Corpus says | This says | Count |
|---|---|---:|
| `R:MORPH` | `R:OTHER` | 45 |
| `R:MORPH` | `R:NOUN:FORM` | 30 |
| `R:ADJ:FORM` | `R:MORPH` | 26 |
| `R:MORPH` | `R:NOUN` | 25 |
| `R:NOUN:FORM` | `R:NOUN` | 24 |
| `R:SPELL` | `R:NOUN` | 20 |
| `R:MORPH` | `R:SPELL` | 20 |
| `R:SPELL` | `R:NOUN:FORM` | 19 |
| `R:DET:FORM` | `R:DET` | 19 |
| `R:OTHER` | `R:SPELL` | 18 |
| `R:VERB:FORM` | `R:SPELL` | 16 |
| `R:PRON:FORM` | `R:PRON` | 16 |
| `R:ADJ:FORM` | `R:SPELL` | 15 |
| `R:MORPH` | `R:DET:FORM` | 15 |
| `R:NOUN:FORM` | `R:SPELL` | 14 |
| `R:VERB:FORM` | `R:OTHER` | 13 |
| `R:VERB:FORM` | `R:VERB` | 12 |
| `R:NOUN` | `R:OTHER` | 12 |
| `R:OTHER` | `R:ADV` | 11 |
| `R:OTHER` | `R:SCONJ` | 10 |

## By error type

*Recall* is the share of the corpus's edits of this type that were drawn at all; *agreement* is the share of those that were also given the same label.

| Error type | Gold edits | Span recall | Type agreement |
|---|---:|---:|---:|
| `R:SPELL` | 816 | 0.96 | 0.91 |
| `R:DET:FORM` | 693 | 0.99 | 0.96 |
| `M:PUNCT` | 582 | 0.93 | 1.00 |
| `R:OTHER` | 555 | 0.87 | 0.83 |
| `R:ORTH` | 529 | 0.92 | 1.00 |
| `R:ADJ:FORM` | 277 | 1.00 | 0.80 |
| `R:NOUN:FORM` | 260 | 1.00 | 0.84 |
| `U:PUNCT` | 220 | 0.94 | 1.00 |
| `R:ADP` | 218 | 0.99 | 0.98 |
| `M:DET` | 188 | 0.91 | 1.00 |
| `R:MORPH` | 186 | 0.97 | 0.09 |
| `R:WO` | 162 | 0.98 | 1.00 |
| `M:PRON` | 157 | 0.80 | 0.96 |
| `R:PUNCT` | 140 | 0.98 | 1.00 |
| `R:VERB:FORM` | 117 | 0.93 | 0.58 |
| `R:AUX:FORM` | 101 | 0.94 | 0.86 |
| `M:ADP` | 90 | 0.87 | 0.99 |
| `R:NOUN` | 80 | 0.96 | 0.75 |
| `M:AUX` | 76 | 0.75 | 0.95 |
| `U:VERB` | 67 | 1.00 | 0.96 |
| `M:OTHER` | 65 | 0.86 | 1.00 |
| `U:PRON` | 65 | 0.71 | 0.96 |
| `R:PRON:FORM` | 65 | 0.98 | 0.62 |
| `U:AUX` | 65 | 0.77 | 0.94 |
| `M:VERB` | 61 | 0.98 | 0.97 |
| `R:VERB` | 58 | 0.97 | 0.75 |
| `R:PRON` | 51 | 0.92 | 0.89 |
| `U:OTHER` | 50 | 0.86 | 0.91 |
| `M:PART` | 44 | 0.82 | 1.00 |
| `U:ADP` | 40 | 0.88 | 1.00 |
| `U:DET` | 36 | 0.81 | 0.97 |
| `R:AUX` | 30 | 0.87 | 0.96 |
| `U:ADV` | 29 | 0.90 | 0.73 |
| `M:ADV` | 26 | 0.96 | 0.96 |
| `R:ADV` | 22 | 1.00 | 0.86 |
| `U:PART` | 21 | 0.67 | 1.00 |
