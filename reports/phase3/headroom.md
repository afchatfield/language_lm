# Phase 3: how much is left, and where it is

Iteration 4 scores **F0.5 0.7072** on Falko-MERLIN dev (`iter4-cw025`). The
question this answers is whether that is near the ceiling or near the ceiling of
*this recipe*, because those have different consequences: the first says move to
Phase 4, the second says one more data iteration is owed.

It is the second. The measurements below are over the cached dev generations of
`iter4-cw025`, no retraining involved.

## The error budget

6,385 gold edits over 2,503 sentences: tp 3,942, fp 1,429, fn 2,443.

| Bucket | Count | F0.5 if that bucket were perfect |
|---|---:|---:|
| FP overlapping a gold span the model missed | 843 | 0.8675 |
| FP touching no gold span at all | 586 | 0.7722 |
| FN the model attempted and got wrong | 1,133 | -- |
| FN the model never touched | 1,310 | -- |
| both FP buckets fixed | | 0.9481 |

Two facts reframe the problem.

**Detection is finished.** Only **215 of 6,385** gold edits (3.4%) sit in a
sentence the model never entered, and **1,118 of the 1,970** sentences needing
correction come back byte-identical to the gold target. The model gets into the
right sentence and, most of the time, the right region. Nothing in the remaining
loss is about willingness to edit -- which is also why `rounds > 1` screened at
-0.001 and why the overcorrection rate is already down to 9.8%.

**The 843 are not a word-choice problem.** The expectation from ablation 1 was
"right span, wrong word", and the leniency experiment was built to test whether
those disputes were defensible. Segmented by how the model's span relates to the
annotator's, that is not what they are:

| | Count |
|---|---:|
| model edit **wider** than the gold edit | 558 |
| model **merged two or more** gold edits into one rewrite | 221 |
| staggered spans | 32 |
| same span, different replacement | 29 |
| model edit narrower than gold | 3 |

Exact-span disagreement is 29 cases out of 843. The failure is that the model
reaches into a longer stretch of the sentence than the annotator did and gets
part of it wrong:

| source | gold | model |
|---|---|---|
| `für ihren Verständnis` | `für Ihr Verständnis` | `für ihr Verständnis` |
| `die meisten Firma` | `den meisten Firmen` | `der meisten Firmen` |
| `soll sich an dieses Problem zu überlegen` | `soll sich dieses Problem überlegen` | `soll sich diesem Problem zu überlegen` |

Case, gender, and the capitalised formal `Sie`. That is German competence, not
retrieval, and no scorer setting reaches it. It is consistent with ablation 1's
conclusion arrived at from the other direction.

## The 586 spurious edits

522 of them are on sentences that *do* need correction, so this is the model
over-editing inside a sentence it is already working on rather than damaging
correct prose. The distribution is a true long tail -- the most frequent
spurious edit occurs **twice** in the whole dev set -- which independently
confirms ablation 3: there is no pattern here for a filter to key on.

Dev is single-annotator (2,503 sentences, one annotator each), so the metric
does under-credit defensible alternatives. Ablation 1 measured that at +0.0046
and it is not the explanation for anything above.

## Against the published numbers

[Luhtaru et al. 2024](https://aclanthology.org/2024.findings-emnlp.727/), the
same benchmark, Falko-MERLIN **test**:

| System | Params | Training data | F0.5 |
|---|---:|---|---:|
| Llama, gold only | 7B | Falko-MERLIN train | 74.31 |
| Llama + 1M probabilistic errors + gold | 7B | + rule-generated synthetic | 75.85 |
| mT5-xxl ([Rothe et al. 2021](https://arxiv.org/abs/2106.03830)) | 13B | multilingual synthetic + cLang-8 | 75.96 |
| Llama + 100k back-translated errors + gold | 7B | + learned synthetic | 76.25 |
| **Llama + 1M back-translated errors + gold** | 7B | + learned synthetic | **76.75** |
| **this project** (dev) | **1.7B** | gold + 10k rule-generated | **70.72** |

Dev and test are not the same split and the last row is not strictly comparable
with the rest. Taken as approximate, though, the table says two things worth
acting on. A 7B model trained on **exactly our gold data and nothing else**
reaches 74.31, so 1.7B at 70.72 is about three and a half points behind a model
four times its size -- and the remaining gap is real headroom rather than metric
artefact. And the documented route past gold-only is not more rules: rule-style
probabilistic generation at 1M sentences is worth +1.5, learned back-translation
at the same volume +2.4, and **100k back-translated sentences buy nearly all of
what 1M buys**.

## What is spent, and what is not

Spent, with numbers already in `ablations.md` and `screen.json`: corruptor
coverage, dictionary post-editing, type filtering, metric leniency, multi-round
inference, checkpoint selection. None of it should be re-spent.

Not spent:

**1. Back-translated error generation.** Ablation 5 ended on "the corruptors
were written for the types that are easy to generate, and those are largely the
frequent, mechanical ones". The corollary is that the hard types need a
generator that was not written by hand. `R:OTHER` -- lexical choice -- is the
single largest miss bucket at 350 fn plus 185 wrong-fix attempts against a
recall of 0.32, and no rule will ever produce it. Everything needed is already
in the repo: 19,237 Falko pairs read backwards are the generator's training set,
and the clean corpus is what it runs over.

**2. `changes_weight` below 0.25.** 1.0 -> 0.5 -> 0.25 gives 0.6944 -> 0.7019 ->
0.7072 and the trend has not turned over. Since ablation 4 moved the
learner-facing explanation onto ERRANT-derived types, the `changes` list is a
diagnostic rather than an output, so lowering the weight further costs nothing a
reader sees. Two screen runs.

**3. cLang-8 German**, ~114k sentences, which is what the mT5-xxl row was built
on. Targets are machine-generated, so it is distillation rather than gold.

**4. N-best rerank with a minimum-edit prior.** Decoding is greedy. 558 of the
843 bad false positives are *wider than gold*, and reranking beam candidates by
edit distance to the source penalises exactly that. Expectations should be low
-- four serving-side changes have bought nothing here -- but this one differs in
kind, in that it changes which outputs are reachable rather than which are kept.

## What this predicts

Iteration 5 is (1) plus (2), and the prediction it should be held to is **0.72
to 0.74 on dev**. Recorded before the run, in the same spirit as iteration 3's
prediction of 0.72, which was wrong and is written up as wrong.

Past that the base model is the binding constraint, and trading it for a larger
one is the trade Phase 4 exists not to make.
