# Phase 5 baselines (Spanish)

COWS-L2H **test**: 4,284 sentences, 7,333 gold edits. Overcorrection set: 400 sentences of published Spanish prose that need no correction. This is the held-out split, read once so that the fine-tuned model has a bar measured on the same sentences it is scored on. Every tuning decision in this project was made against dev, whose table is in the report beside this one.

## The table

| System | P | R | F0.5 | Overcorrection | Spurious edits |
|---|---:|---:|---:|---:|---:|
| identity | 0.5000 | 0.0001 | **0.0007** | 0.0% | 0 |
| languagetool | 0.4547 | 0.1630 | **0.3348** | 27.3% | 151 |
| zero-shot | 0.4632 | 0.0610 | **0.1997** | 11.2% | 172 |
| few-shot | 0.6092 | 0.1210 | **0.3371** | 8.0% | 37 |

F0.5 is MaxMatch (Dahlmeier and Ng, 2012) over the corrected sentences. *Overcorrection* is the share of already-correct sentences the system changed at all; *spurious edits* counts the individual changes it made to them.

The metric's own ceiling on this split is **F0.5 = 0.9986**, not 1.0: that is what a system scores when it reproduces the annotator's corrections exactly. MaxMatch recovers a system's edits from an optimal alignment between the original and the correction, and 20 of the corpus's 7,333 edits are drawn in a way that no optimal alignment produces. Read every score below against that number rather than against 1.

## Detection versus correction

Span-level ERRANT scoring. *Detection* asks only whether the system flagged the right span, *correction* whether it also proposed the annotator's fix. The gap between them is the system seeing the error and getting the repair wrong.

| System | Detection F0.5 | Correction F0.5 | Gap |
|---|---:|---:|---:|
| identity | 0.0007 | 0.0000 | +0.0007 |
| languagetool | 0.4270 | 0.3078 | +0.1192 |
| zero-shot | 0.2289 | 0.1679 | +0.0610 |
| few-shot | 0.3883 | 0.3149 | +0.0735 |

## Where each system helps

| Error type | Gold | identity R | languagetool R | zero-shot R | few-shot R |
|---|---:|---:|---:|---:|---:|
| `R:OTHER` | 996 | 0.00 | 0.02 | 0.03 | 0.06 |
| `R:ORTH` | 811 | 0.00 | 0.73 | 0.15 | 0.33 |
| `U:PRON` | 785 | 0.00 | 0.00 | 0.00 | 0.01 |
| `R:VERB:FORM` | 359 | 0.00 | 0.11 | 0.06 | 0.11 |
| `R:SPELL` | 356 | 0.00 | 0.54 | 0.30 | 0.44 |
| `R:DET:FORM` | 342 | 0.00 | 0.28 | 0.05 | 0.20 |
| `R:ADP` | 310 | 0.00 | 0.04 | 0.01 | 0.03 |
| `M:ADP` | 284 | 0.00 | 0.02 | 0.04 | 0.09 |
| `U:DET` | 268 | 0.00 | 0.04 | 0.02 | 0.05 |
| `R:VERB` | 244 | 0.00 | 0.02 | 0.05 | 0.09 |
| `M:PUNCT` | 219 | 0.00 | 0.30 | 0.00 | 0.08 |
| `M:DET` | 210 | 0.00 | 0.01 | 0.01 | 0.15 |
| `R:ADJ:FORM` | 205 | 0.00 | 0.20 | 0.09 | 0.11 |
| `R:NOUN` | 197 | 0.00 | 0.08 | 0.08 | 0.16 |
| `U:ADP` | 140 | 0.00 | 0.01 | 0.01 | 0.04 |

## What the systems change in correct prose

**identity** changed nothing.

**languagetool** -- 109/400 sentences modified (27.3%), 151 spurious edits. Most common: `R:SPELL` (101), `R:ORTH` (12), `R:PNOUN` (11), `M:PUNCT` (8), `R:OTHER` (5).

> Fue fundada en el año 2006 por George Alexander Nader .
>> Fue fundada en el año 2006 por George Alexander Nacer .

> A diferencia de otros modelos , el Shinkirō no tiene armas de corto alcance , careciendo de Slash Harkens o espadas .
>> A diferencia de otros modelos , el Sintió no tiene armas de corto alcance , careciendo de Flash Harqueña o espadas .

> Los usuarios pueden crear un monedero a través de la aplicación móvil de Binance , que también servirá para actividades DeFi como apuestas , préstamos y empréstitos .
>> Los usuarios pueden crear un monedero a través de la aplicación móvil de Binance , que también servirá para actividades De Fi como apuestas , préstamos y empréstitos .

**zero-shot** -- 45/400 sentences modified (11.2%), 172 spurious edits. Most common: `R:OTHER` (89), `U:OTHER` (22), `M:PNOUN` (6), `R:ADP` (6), `R:NOUN` (6).

> A diferencia de otros modelos , el Shinkirō no tiene armas de corto alcance , careciendo de Slash Harkens o espadas .
>> El Shinkirō no tiene armas de corto alcance , careciendo de Slash Harkens o espadas .

> En la Casa Blanca tomaron nota sobre la perspectiva del presidente electo y se comprometieron en colaborar con el futuro gobierno de La Libertad Avanza .
>> La Casa Blanca tomó nota sobre la perspectiva del presidente electo y se comprometió en colaborar con el futuro gobierno de La Libertad Avanza .

> Comentarios desactivados en Los puestos afean la ciudad : ¡ Hay , que desalojar !
>> Los puestos afean la ciudad : ¡ Hay , que desalojar !

**few-shot** -- 32/400 sentences modified (8.0%), 37 spurious edits. Most common: `M:DET` (5), `R:OTHER` (4), `M:PUNCT` (4), `R:NOUN` (3), `R:PUNCT` (3).

> A pesar de todo , los britanos le habían elegido para liderar la resistencia .
>> A pesar de todo , los británicos le habían elegido para liderar la resistencia .

> Entrevista con el directivo del Agoncillo , Daniel Pascual , que defiende la participación de su equipo en la fase de ascenso a 2ªRFEF .
>> Entrevista con el director del Agoncillo , Daniel Pascual , que defiende la participación de su equipo en la fase de ascenso a 2ªRFEF .

> Como quedase plasmado en su comunicado del lunes , el presidente de la CEOE ha reiterado que “ no estamos de acuerdo con la amnistía ” .
>> Como queda plasmado en su comunicado del lunes , el presidente de la CEOE ha reiterado que “ no estamos de acuerdo con la amnistía ” .

## Notes

* Model: `utter-project/EuroLLM-1.7B`, greedy decoding.
* Every system's output goes through the same Spanish tokeniser before scoring. The identity row prices that step: whatever false positives it shows are the harness disagreeing with the corpus about hyphens and apostrophes, not a system making mistakes.
* LanguageTool applies the first suggestion of every non-overlapping match that offers one. Matches with no suggestion are left alone.
* The overcorrection set was **not** screened with LanguageTool, which would have handed LanguageTool a zero by construction.
