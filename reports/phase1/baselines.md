# Phase 1 baselines

Falko-MERLIN **dev**: 2,503 sentences, 6,385 gold edits. Overcorrection set: 400 sentences of published German prose that need no correction. The test split has not been read.

## The table

| System | P | R | F0.5 | Overcorrection | Spurious edits |
|---|---:|---:|---:|---:|---:|
| identity | 0.0417 | 0.0002 | **0.0008** | 0.0% | 0 |
| languagetool | 0.5653 | 0.1992 | **0.4134** | 28.7% | 142 |
| zero-shot | 0.4761 | 0.0905 | **0.2571** | 10.0% | 104 |
| few-shot | 0.6600 | 0.2675 | **0.5102** | 15.0% | 78 |

F0.5 is MaxMatch (Dahlmeier and Ng, 2012) over the corrected sentences. *Overcorrection* is the share of already-correct sentences the system changed at all; *spurious edits* counts the individual changes it made to them.

The metric's own ceiling on this split is **F0.5 = 0.9976**, not 1.0: that is what a system scores when it reproduces the annotator's corrections exactly. MaxMatch recovers a system's edits from an optimal alignment between the original and the correction, and 35 of the corpus's 6,385 edits are drawn in a way that no optimal alignment produces. Read every score below against that number rather than against 1.

## Detection versus correction

Span-level ERRANT scoring. *Detection* asks only whether the system flagged the right span, *correction* whether it also proposed the annotator's fix. The gap between them is the system seeing the error and getting the repair wrong.

| System | Detection F0.5 | Correction F0.5 | Gap |
|---|---:|---:|---:|
| identity | 0.0047 | 0.0000 | +0.0047 |
| languagetool | 0.5869 | 0.3849 | +0.2020 |
| zero-shot | 0.2892 | 0.2197 | +0.0695 |
| few-shot | 0.5835 | 0.4715 | +0.1120 |

## Where each system helps

| Error type | Gold | identity R | languagetool R | zero-shot R | few-shot R |
|---|---:|---:|---:|---:|---:|
| `R:SPELL` | 816 | 0.00 | 0.52 | 0.19 | 0.49 |
| `R:DET:FORM` | 693 | 0.00 | 0.22 | 0.06 | 0.24 |
| `M:PUNCT` | 582 | 0.00 | 0.21 | 0.02 | 0.37 |
| `R:OTHER` | 555 | 0.00 | 0.03 | 0.04 | 0.13 |
| `R:ORTH` | 529 | 0.00 | 0.54 | 0.16 | 0.42 |
| `R:ADJ:FORM` | 277 | 0.00 | 0.25 | 0.17 | 0.44 |
| `R:NOUN:FORM` | 260 | 0.00 | 0.21 | 0.21 | 0.52 |
| `U:PUNCT` | 220 | 0.00 | 0.05 | 0.06 | 0.04 |
| `R:ADP` | 218 | 0.00 | 0.05 | 0.00 | 0.04 |
| `M:DET` | 188 | 0.00 | 0.01 | 0.01 | 0.05 |
| `R:MORPH` | 186 | 0.00 | 0.12 | 0.15 | 0.34 |
| `R:WO` | 162 | 0.00 | 0.00 | 0.04 | 0.09 |
| `M:PRON` | 157 | 0.00 | 0.01 | 0.01 | 0.03 |
| `R:PUNCT` | 140 | 0.00 | 0.04 | 0.00 | 0.02 |
| `R:VERB:FORM` | 117 | 0.00 | 0.16 | 0.10 | 0.23 |

## What the systems change in correct prose

**identity** changed nothing.

**languagetool** -- 115/400 sentences modified (28.7%), 142 spurious edits. Most common: `R:SPELL` (94), `R:ORTH` (22), `R:NOUN` (9), `R:PNOUN` (4), `R:DET:FORM` (3).

> Im Wittener Sportausschuss herrscht Einigkeit : Ringer der SU Annen müssen höher gefördert werden als üblich .
>> Im Wittener Sportausschuss herrscht Einigkeit : Ringer der USA Innen müssen höher gefördert werden als üblich .

> Freilebende Kühe gebären ihr erstes Kalb in der Regel im vierten Lebensjahr .
>> Frei lebende Kühe gebären ihr erstes Kalb in der Regel im vierten Lebensjahr .

> Der gesamte Gebäudekomplex , bestehend aus Alter Kirche , Pfarrscheuer und altem Pfarrhaus steht unter Denkmalschutz .
>> Der gesamte Gebäudekomplex , bestehend aus Alter Kirche , Pfarrscheune und altem Pfarrhaus steht unter Denkmalschutz .

**zero-shot** -- 40/400 sentences modified (10.0%), 104 spurious edits. Most common: `R:OTHER` (41), `U:OTHER` (15), `R:WO` (4), `R:ADV` (3), `U:ADP` (3).

> Im Wittener Sportausschuss herrscht Einigkeit : Ringer der SU Annen müssen höher gefördert werden als üblich .
>> Annen müssen höher gefördert werden als üblich .

> Dass eine falsche Eingabe eine gültige IBAN ergibt , ist daher unwahrscheinlich .
>> Die IBAN ist falsch .

> An einem Cabrio wurde ein Seitenfenster heruntergedrückt , um in das Fahrzeuginnere zu gelangen .
>> Ein Cabrio wurde ein Seitenfenster heruntergedrückt , um in das Fahrzeuginnere zu gelangen .

**few-shot** -- 60/400 sentences modified (15.0%), 78 spurious edits. Most common: `R:OTHER` (10), `M:PUNCT` (9), `R:NOUN` (9), `R:SPELL` (5), `U:OTHER` (5).

> Der gesamte Gebäudekomplex , bestehend aus Alter Kirche , Pfarrscheuer und altem Pfarrhaus steht unter Denkmalschutz .
>> Der gesamte Gebäudekomplex , bestehend aus der alten Kirche , der Pfarrscheuer und dem alten Pfarrhaus , steht unter Denkmalschutz .

> An den gefundenen Waffen würden im Labor Spuren gesucht - Fingerabdrücke oder DNA-Spuren , hieß es .
>> An den gefundenen Waffen würden Spuren im Labor gesucht - Fingerabdrücke oder DNA-Spuren , hieß es .

> Inzwischen sei das Tempo auf der Warngauer Straße außerhalb des Ortes durchgängig auf 70 km / h beschränkt worden , führte er an .
>> Inzwischen sei das Tempo auf der Warngau Straße außerhalb des Ortes durchgängig auf 70 km / h beschränkt worden , führte er an .

## Notes

* Model: `utter-project/EuroLLM-1.7B`, greedy decoding.
* Every system's output goes through the same German tokeniser before scoring. The identity row prices that step: whatever false positives it shows are the harness disagreeing with the corpus about hyphens and apostrophes, not a system making mistakes.
* LanguageTool applies the first suggestion of every non-overlapping match that offers one. Matches with no suggestion are left alone.
* The overcorrection set was **not** screened with LanguageTool, which would have handed LanguageTool a zero by construction.
