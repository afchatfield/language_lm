# Phase 1 baselines

Falko-MERLIN **test**: 2,337 sentences, 5,963 gold edits. Overcorrection set: 400 sentences of published German prose that need no correction. This is the held-out split, read once so that the fine-tuned model has a bar measured on the same sentences it is scored on. Every tuning decision in this project was made against dev, whose table is in the report beside this one.

## The table

| System | P | R | F0.5 | Overcorrection | Spurious edits |
|---|---:|---:|---:|---:|---:|
| identity | 0.1026 | 0.0007 | **0.0033** | 0.0% | 0 |
| languagetool | 0.5890 | 0.2091 | **0.4321** | 28.7% | 142 |
| zero-shot | 0.4634 | 0.0880 | **0.2501** | 10.0% | 95 |
| few-shot | 0.6431 | 0.2574 | **0.4948** | 15.2% | 76 |

F0.5 is MaxMatch (Dahlmeier and Ng, 2012) over the corrected sentences. *Overcorrection* is the share of already-correct sentences the system changed at all; *spurious edits* counts the individual changes it made to them.

The metric's own ceiling on this split is **F0.5 = 0.9974**, not 1.0: that is what a system scores when it reproduces the annotator's corrections exactly. MaxMatch recovers a system's edits from an optimal alignment between the original and the correction, and 37 of the corpus's 5,963 edits are drawn in a way that no optimal alignment produces. Read every score below against that number rather than against 1.

## Detection versus correction

Span-level ERRANT scoring. *Detection* asks only whether the system flagged the right span, *correction* whether it also proposed the annotator's fix. The gap between them is the system seeing the error and getting the repair wrong.

| System | Detection F0.5 | Correction F0.5 | Gap |
|---|---:|---:|---:|
| identity | 0.0099 | 0.0000 | +0.0099 |
| languagetool | 0.5964 | 0.4081 | +0.1883 |
| zero-shot | 0.2827 | 0.2101 | +0.0726 |
| few-shot | 0.5734 | 0.4566 | +0.1168 |

## Where each system helps

| Error type | Gold | identity R | languagetool R | zero-shot R | few-shot R |
|---|---:|---:|---:|---:|---:|
| `R:SPELL` | 834 | 0.00 | 0.58 | 0.18 | 0.50 |
| `R:DET:FORM` | 580 | 0.00 | 0.23 | 0.07 | 0.23 |
| `M:PUNCT` | 565 | 0.00 | 0.20 | 0.02 | 0.31 |
| `R:ORTH` | 495 | 0.00 | 0.54 | 0.15 | 0.39 |
| `R:OTHER` | 485 | 0.00 | 0.04 | 0.04 | 0.09 |
| `R:ADJ:FORM` | 239 | 0.00 | 0.26 | 0.12 | 0.34 |
| `R:NOUN:FORM` | 219 | 0.00 | 0.23 | 0.19 | 0.54 |
| `M:DET` | 217 | 0.00 | 0.00 | 0.00 | 0.11 |
| `U:PUNCT` | 206 | 0.00 | 0.02 | 0.05 | 0.06 |
| `R:ADP` | 197 | 0.00 | 0.04 | 0.00 | 0.04 |
| `R:MORPH` | 169 | 0.00 | 0.09 | 0.18 | 0.37 |
| `M:PRON` | 136 | 0.00 | 0.00 | 0.01 | 0.03 |
| `R:WO` | 135 | 0.00 | 0.00 | 0.05 | 0.11 |
| `R:VERB:FORM` | 121 | 0.00 | 0.15 | 0.10 | 0.21 |
| `R:PUNCT` | 115 | 0.00 | 0.04 | 0.00 | 0.05 |

## What the systems change in correct prose

**identity** changed nothing.

**languagetool** -- 115/400 sentences modified (28.7%), 142 spurious edits. Most common: `R:SPELL` (94), `R:ORTH` (22), `R:NOUN` (9), `R:PNOUN` (4), `R:DET:FORM` (3).

> Im Wittener Sportausschuss herrscht Einigkeit : Ringer der SU Annen müssen höher gefördert werden als üblich .
>> Im Wittener Sportausschuss herrscht Einigkeit : Ringer der USA Innen müssen höher gefördert werden als üblich .

> Freilebende Kühe gebären ihr erstes Kalb in der Regel im vierten Lebensjahr .
>> Frei lebende Kühe gebären ihr erstes Kalb in der Regel im vierten Lebensjahr .

> Der gesamte Gebäudekomplex , bestehend aus Alter Kirche , Pfarrscheuer und altem Pfarrhaus steht unter Denkmalschutz .
>> Der gesamte Gebäudekomplex , bestehend aus Alter Kirche , Pfarrscheune und altem Pfarrhaus steht unter Denkmalschutz .

**zero-shot** -- 40/400 sentences modified (10.0%), 95 spurious edits. Most common: `R:OTHER` (34), `U:OTHER` (17), `R:WO` (4), `R:SPELL` (4), `U:ADP` (3).

> Im Wittener Sportausschuss herrscht Einigkeit : Ringer der SU Annen müssen höher gefördert werden als üblich .
>> Annen müssen höher gefördert werden als üblich .

> Dass eine falsche Eingabe eine gültige IBAN ergibt , ist daher unwahrscheinlich .
>> Die IBAN ist falsch .

> An einem Cabrio wurde ein Seitenfenster heruntergedrückt , um in das Fahrzeuginnere zu gelangen .
>> Ein Cabrio wurde ein Seitenfenster heruntergedrückt , um in das Fahrzeuginnere zu gelangen .

**few-shot** -- 61/400 sentences modified (15.2%), 76 spurious edits. Most common: `R:NOUN` (10), `R:OTHER` (9), `M:PUNCT` (8), `U:OTHER` (6), `R:SPELL` (5).

> Der gesamte Gebäudekomplex , bestehend aus Alter Kirche , Pfarrscheuer und altem Pfarrhaus steht unter Denkmalschutz .
>> Der gesamte Gebäudekomplex , bestehend aus der alten Kirche , der Pfarrscheuer und dem alten Pfarrhaus , steht unter Denkmalschutz .

> An den gefundenen Waffen würden im Labor Spuren gesucht - Fingerabdrücke oder DNA-Spuren , hieß es .
>> An den gefundenen Waffen würden Spuren im Labor gesucht - Fingerabdrücke oder DNA-Spuren , hieß es .

> Im anschließenden , beinahe sieben Jahre dauernden Gefecht zwischen Katharina und Heinrich wurden u. a. Details und Zeugen über ihre erste Ehe zu Arthur ausgegraben .
>> Im anschließenden , beinahe sieben Jahre dauernden Gefecht zwischen Katharina und Heinrich wurden u. a. Details und Zeugen über ihre erste Ehe , zu Arthur ausgegraben .

## Notes

* Model: `utter-project/EuroLLM-1.7B`, greedy decoding.
* Every system's output goes through the same German tokeniser before scoring. The identity row prices that step: whatever false positives it shows are the harness disagreeing with the corpus about hyphens and apostrophes, not a system making mistakes.
* LanguageTool applies the first suggestion of every non-overlapping match that offers one. Matches with no suggestion are left alone.
* The overcorrection set was **not** screened with LanguageTool, which would have handed LanguageTool a zero by construction.
