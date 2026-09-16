# Phase 3: the held-out test split

Frozen on day one of Phase 0, enforced in code (`load_split` refuses it without
`allow_test=True`), and read here for the first time. 2,337 sentences, 5,963 gold
edits — the same annotation density as dev, 2.55 edits per corrected sentence.

Read once, for five models, with no decision taken afterwards that feeds back
into any of them. What the numbers are for is choosing which model Phase 4
compresses, and establishing the "uncompressed" row of that phase's ablation
table.

## The table

| System | P | R | F0.5 | Lenient F0.5 | Overcorrection | Expl. type acc. |
|---|---:|---:|---:|---:|---:|---:|
| cw0125 | 0.7330 | 0.6242 | **0.7083** | 0.7109 | 14.0% | 0.712 |
| iter5 | 0.7325 | 0.6210 | 0.7071 | 0.7109 | 13.0% | 0.734 |
| **iter4-cw025** | 0.7322 | 0.6176 | 0.7060 | 0.7094 | **9.5%** | 0.692 |
| iter5 + DPO | 0.7065 | 0.6200 | 0.6873 | 0.6917 | 15.5% | 0.723 |
| iter4-cw025 + DPO (beta 0.5) | 0.7038 | 0.6180 | 0.6848 | 0.6882 | 12.5% | 0.679 |

## Six iterations of tuning did not overfit dev

The first thing to check, because it is the thing that would invalidate
everything else.

| System | dev | test | delta |
|---|---:|---:|---:|
| iter4-cw025 | 0.7072 | 0.7060 | −0.0012 |
| cw0125 | 0.7068 | 0.7083 | +0.0015 |
| iter5 | 0.7056 | 0.7071 | +0.0015 |
| iter5 + DPO | 0.6905 | 0.6873 | −0.0032 |
| iter4-cw025 + DPO | 0.6895 | 0.6848 | −0.0047 |

Every model lands within 0.005 of its dev score and the mean absolute gap is
0.0024. Six iterations, four ablation studies and a hyperparameter sweep were all
conducted against dev, and none of it bought a dev-specific advantage. The dev
numbers quoted throughout Phase 3 can be read as honest estimates of held-out
performance.

The DPO models are the two largest gaps, both negative. A model that has been
walked away from its reference generalises very slightly worse, which is
consistent with `dpo.md`'s account of what went wrong and is a second, independent
confirmation of it: DPO loses 0.021 and 0.020 on test, against 0.018 and 0.015 on
dev. It did not fail on dev for a dev-shaped reason.

## The three SFT models are one model

They span 0.0023 of F0.5. More tellingly, **the ranking inverts between splits**:

```
dev:   iter4-cw025  >  cw0125  >  iter5
test:  cw0125       >  iter5   >  iter4-cw025
```

Precision is identical to three decimal places across all three (0.7322, 0.7325,
0.7330). This is what `nbest.md` predicted when it reported that iteration 3's
+0.0016 had a 95% interval straddling zero, and it settles the question: the last
three data iterations produced one model with three names, and F0.5 cannot choose
between them. Anyone quoting "cw0125 is the best model" from this table is
quoting noise.

## So the choice is made on overcorrection

One axis does separate them, and it is the axis this project has said matters
since Phase 1:

| System | Overcorrection | dev | replicates? |
|---|---:|---:|---|
| iter4-cw025 | **9.5%** | 9.8% | yes |
| iter5 | 13.0% | 13.0% | yes |
| cw0125 | 14.0% | 13.3% | yes |

A 4.5-point gap that holds across both splits, against an F0.5 difference that
changes sign between them. Phase 1 called LanguageTool's 28.7% "the strongest
argument in the project so far for a model that knows when to keep quiet", and
the whole 22%-negative-examples design exists to serve that. Choosing a model that
edits correct German 47% more often, to buy 0.0023 of an F0.5 that cannot be
measured to that precision, would be choosing against the project's own stated
priority.

> **Phase 4 compresses `iter4-cw025`.** F0.5 0.7060, overcorrection 9.5%, on
> held-out test. That is the "uncompressed" row.

Its explanation type accuracy is the lowest of the three (0.692 against 0.734),
which is the one real cost of this choice and is worth carrying into Phase 6 —
though it is measured against an oracle of 0.8387, not against 1.00, since ERRANT
re-derives the annotator's own type on only 86% of gold edits.

## What this table does not say

**It does not say the model beats LanguageTool by 0.29.** The Phase 1 baselines —
LanguageTool 0.4134, few-shot 0.5102 — were computed on *dev*, and
`scripts/run_baselines.py` is hard-wired to the configured dev split. Comparing a
test row against a dev baseline is not a comparison. Before any of this reaches a
README, `run_baselines.py` needs the same `--split` treatment
`scripts/eval_finetuned.py` just got, and LanguageTool and few-shot need one run
each on test. LanguageTool is cheap; few-shot is a single generation pass.

Until that is done, the only defensible claims from this page are the ones
internal to it: which of our models to compress, and what the uncompressed row is.

## Provenance

`scripts/eval_finetuned.py --split test`, which prints a warning naming the split
and passes `allow_test=True` only when the flag is given. Scored with the same
MaxMatch and German ERRANT machinery as every other number in Phases 1 and 3.
Per-model reports are `reports/phase3/evaluation-*-test.md`.
