# Phase 5 baselines (Spanish)

COWS-L2H **dev**: 4,716 sentences, 7,563 gold edits. Overcorrection set: 400 sentences of published Spanish prose that need no correction. The test split has not been read.

## The table

| System | P | R | F0.5 | Overcorrection | Spurious edits |
|---|---:|---:|---:|---:|---:|
| identity | 0.0000 | 0.0000 | **0.0000** | 0.0% | 0 |
| languagetool | 0.4232 | 0.1588 | **0.3175** | 27.3% | 151 |

F0.5 is MaxMatch (Dahlmeier and Ng, 2012) over the corrected sentences. *Overcorrection* is the share of already-correct sentences the system changed at all; *spurious edits* counts the individual changes it made to them.

The metric's own ceiling on this split is **F0.5 = 0.9949**, not 1.0: that is what a system scores when it reproduces the annotator's corrections exactly. MaxMatch recovers a system's edits from an optimal alignment between the original and the correction, and 47 of the corpus's 7,563 edits are drawn in a way that no optimal alignment produces. Read every score below against that number rather than against 1.

## Detection versus correction

Span-level ERRANT scoring. *Detection* asks only whether the system flagged the right span, *correction* whether it also proposed the annotator's fix. The gap between them is the system seeing the error and getting the repair wrong.

| System | Detection F0.5 | Correction F0.5 | Gap |
|---|---:|---:|---:|
| identity | 0.0000 | 0.0000 | +0.0000 |
| languagetool | 0.3951 | 0.2849 | +0.1102 |

## Where each system helps

| Error type | Gold | identity R | languagetool R |
|---|---:|---:|---:|
| `R:OTHER` | 1251 | 0.00 | 0.02 |
| `R:ORTH` | 838 | 0.00 | 0.69 |
| `U:PRON` | 594 | 0.00 | 0.00 |
| `R:VERB:FORM` | 357 | 0.00 | 0.07 |
| `R:DET:FORM` | 327 | 0.00 | 0.28 |
| `U:DET` | 303 | 0.00 | 0.05 |
| `M:ADP` | 279 | 0.00 | 0.06 |
| `R:ADP` | 268 | 0.00 | 0.07 |
| `R:SPELL` | 257 | 0.00 | 0.51 |
| `R:VERB` | 250 | 0.00 | 0.02 |
| `M:PUNCT` | 249 | 0.00 | 0.33 |
| `M:DET` | 247 | 0.00 | 0.01 |
| `R:ADJ:FORM` | 221 | 0.00 | 0.21 |
| `R:NOUN` | 199 | 0.00 | 0.14 |
| `R:AUX:FORM` | 158 | 0.00 | 0.02 |

## What the systems change in correct prose

**identity** changed nothing.

**languagetool** -- 109/400 sentences modified (27.3%), 151 spurious edits. Most common: `R:SPELL` (101), `R:ORTH` (12), `R:PNOUN` (11), `M:PUNCT` (8), `R:OTHER` (5).

> Fue fundada en el año 2006 por George Alexander Nader .
>> Fue fundada en el año 2006 por George Alexander Nacer .

> A diferencia de otros modelos , el Shinkirō no tiene armas de corto alcance , careciendo de Slash Harkens o espadas .
>> A diferencia de otros modelos , el Sintió no tiene armas de corto alcance , careciendo de Flash Harqueña o espadas .

> Los usuarios pueden crear un monedero a través de la aplicación móvil de Binance , que también servirá para actividades DeFi como apuestas , préstamos y empréstitos .
>> Los usuarios pueden crear un monedero a través de la aplicación móvil de Binance , que también servirá para actividades De Fi como apuestas , préstamos y empréstitos .

## Notes

* Model: `utter-project/EuroLLM-1.7B`, greedy decoding.
* Every system's output goes through the same Spanish tokeniser before scoring. The identity row prices that step: whatever false positives it shows are the harness disagreeing with the corpus about hyphens and apostrophes, not a system making mistakes.
* LanguageTool applies the first suggestion of every non-overlapping match that offers one. Matches with no suggestion are left alone.
* The overcorrection set was **not** screened with LanguageTool, which would have handed LanguageTool a zero by construction.
