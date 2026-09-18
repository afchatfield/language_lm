# Phase 5 baselines (Spanish)

COWS-L2H **dev**: 4,716 sentences, 7,563 gold edits. Overcorrection set: 400 sentences of published Spanish prose that need no correction. The test split has not been read.

## The table

| System | P | R | F0.5 | Overcorrection | Spurious edits |
|---|---:|---:|---:|---:|---:|
| identity | 0.0000 | 0.0000 | **0.0000** | 0.0% | 0 |
| languagetool | 0.4241 | 0.1588 | **0.3179** | 27.3% | 151 |
| zero-shot | 0.3872 | 0.0510 | **0.1671** | 11.2% | 172 |
| few-shot | 0.5664 | 0.1088 | **0.3077** | 8.0% | 37 |

F0.5 is MaxMatch (Dahlmeier and Ng, 2012) over the corrected sentences. *Overcorrection* is the share of already-correct sentences the system changed at all; *spurious edits* counts the individual changes it made to them.

The metric's own ceiling on this split is **F0.5 = 0.9948**, not 1.0: that is what a system scores when it reproduces the annotator's corrections exactly. MaxMatch recovers a system's edits from an optimal alignment between the original and the correction, and 50 of the corpus's 7,563 edits are drawn in a way that no optimal alignment produces. Read every score below against that number rather than against 1.

## Detection versus correction

Span-level ERRANT scoring. *Detection* asks only whether the system flagged the right span, *correction* whether it also proposed the annotator's fix. The gap between them is the system seeing the error and getting the repair wrong.

| System | Detection F0.5 | Correction F0.5 | Gap |
|---|---:|---:|---:|
| identity | 0.0000 | 0.0000 | +0.0000 |
| languagetool | 0.3951 | 0.2849 | +0.1102 |
| zero-shot | 0.1916 | 0.1374 | +0.0542 |
| few-shot | 0.3480 | 0.2831 | +0.0649 |

## Where each system helps

| Error type | Gold | identity R | languagetool R | zero-shot R | few-shot R |
|---|---:|---:|---:|---:|---:|
| `R:OTHER` | 1251 | 0.00 | 0.02 | 0.02 | 0.05 |
| `R:ORTH` | 838 | 0.00 | 0.69 | 0.11 | 0.25 |
| `U:PRON` | 594 | 0.00 | 0.00 | 0.00 | 0.02 |
| `R:VERB:FORM` | 357 | 0.00 | 0.07 | 0.07 | 0.12 |
| `R:DET:FORM` | 327 | 0.00 | 0.28 | 0.05 | 0.17 |
| `U:DET` | 303 | 0.00 | 0.05 | 0.02 | 0.04 |
| `M:ADP` | 279 | 0.00 | 0.06 | 0.02 | 0.10 |
| `R:ADP` | 268 | 0.00 | 0.07 | 0.03 | 0.06 |
| `R:SPELL` | 257 | 0.00 | 0.51 | 0.30 | 0.43 |
| `R:VERB` | 250 | 0.00 | 0.02 | 0.04 | 0.06 |
| `M:PUNCT` | 249 | 0.00 | 0.33 | 0.00 | 0.12 |
| `M:DET` | 247 | 0.00 | 0.01 | 0.02 | 0.16 |
| `R:ADJ:FORM` | 221 | 0.00 | 0.21 | 0.07 | 0.15 |
| `R:NOUN` | 199 | 0.00 | 0.14 | 0.13 | 0.19 |
| `R:AUX:FORM` | 158 | 0.00 | 0.02 | 0.03 | 0.03 |

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
