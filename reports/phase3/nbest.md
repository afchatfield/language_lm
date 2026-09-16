# Phase 3: the n-best rerank, and why the oracle gap is not a reserve

`headroom.md` §4 proposed reranking beam candidates by edit distance to the
source: decoding is greedy, 558 of the 843 harmful false positives are *wider*
than the gold edit against 3 that are narrower, and a one-directional bias is
the kind a prior can correct where a better search cannot. Expectations were set
low -- four serving-side changes had already returned nothing -- but this one
"differs in kind, in that it changes which outputs are reachable rather than
which are kept."

It does change what is reachable. That turns out not to be the same as changing
what is reached.

Everything below is `checkpoints/phase3-de-iter4-cw025`, the published 0.7072
model, with `scripts/nbest_rerank.py` at beam 5. Generations are cached under
`data/interim/phase3/nbest/`. The test split was not touched.

## The screen said yes and it was wrong

Swept on the 600-sentence screen, a light minimum-edit penalty looked like the
best serving-side result the project had seen: F0.5 0.7257 against greedy's
0.7092, +0.0165, with precision 0.7356 -> 0.7711 exactly as predicted.

The penalty was chosen on the same 600 sentences it was scored on. Held out on
the other 1,902 dev sentences with the penalty fixed at 0.02 beforehand:

| selection | F0.5 | P | R |
|---|---:|---:|---:|
| greedy (the published path) | 0.7064 | 0.7330 | 0.6170 |
| beam top-1 | 0.7148 | 0.7413 | 0.6255 |
| min-edit, penalty 0.02 | 0.7155 | 0.7565 | 0.5881 |
| random pick from the beam | 0.5889 | 0.5934 | 0.5717 |
| MBR (consensus of the beam) | 0.7039 | 0.7294 | 0.6176 |
| learned reranker, 9 features | 0.7148 | 0.7413 | 0.6255 |
| **oracle (per-sentence best of 5)** | **0.8380** | 0.8654 | 0.7439 |

**The prior is worth +0.0007 over beam top-1.** The screen's +0.0085 was the
tuning leak and nothing else. What survives is beam search itself: +0.0084 over
greedy here and +0.0080 on the screen, reproducible, for five times the
generation. `headroom.md` §4 is answered, and the answer is no.

## The oracle is real, and that is the interesting part

0.8380 against 0.7148 is 0.12 of F0.5 sitting inside a five-wide beam, and the
model reaches the annotator's correction on 40.7% of sentences where it does not
rank it first. Two controls say that is not an artefact of maximising over five
noisy candidates:

* **Random selection collapses to 0.5889**, 0.126 *below* top-1. The beam's own
  ordering carries most of what the model knows; the oracle is not picking up
  metric noise.
* **It replicates.** Screen 0.8396 and 39.9%, held-out 0.8380 and 40.7%.

So the model does generate something much better than what it says, four times
in ten.

## Three selectors, none of which reach it

**Minimum edit distance.** +0.0007. It buys precision (0.7413 -> 0.7565) and
sells more recall than it buys (0.6255 -> 0.5881), which is the trade F0.5
should like and does not, because the edits it suppresses are about as often
right as wrong.

**MBR.** -0.0109 against top-1, below even greedy. Choosing the candidate most
like its siblings selects the consensus of a beam whose members share a prefix,
which is closer to "the most cautious paraphrase" than to "the most likely
correction".

**A learned reranker.** Logistic regression over nine gold-free features --
sequence score, margin to top-1, rank, edit distance, edits per token, length
delta, diff-op count, MBR risk, identical siblings -- fitted on n-best over 2,000
Falko-MERLIN *train* sentences where the gold says which candidate won, then
applied to held-out dev. It **never overrides top-1**, on any of the 1,902
sentences. The weights say why: rank -2.08, score margin +1.07, score +1.04, and
every shape feature under 0.21 in absolute value. Given the features it had, the
model's own ranking was the best predictor available, so the classifier learned
to repeat it.

## Why no selector works

The best candidate's position in the beam:

| position | share |
|---|---:|
| rank 0 (top-1 already best or tied) | 59.3% |
| rank 1 | 19.2% |
| rank 2 | 9.7% |
| rank 3 | 6.4% |
| rank 4 | 5.4% |

And the score of committing to one position:

| always pick | F0.5 |
|---|---:|
| rank 0 | 0.7148 |
| rank 1 | 0.6032 |
| rank 2 | 0.5642 |
| rank 3 | 0.5428 |
| rank 4 | 0.5282 |

Quality decays monotonically with rank and the winners are smeared thinly across
the tail -- no pocket, no threshold, no cluster. The likelihood ordering is
already close to the best ordering available *from surface statistics*. What
separates `der meisten Firmen` from `den meisten Firmen` is German case
agreement, not sequence score, edit count, or beam consensus, and a reranker
that cannot read German cannot tell them apart.

This is the same conclusion `ablations.md` reached about the dictionary gate and
`headroom.md` reached from the false-positive segmentation, arrived at from a
third direction. The dictionary was predicted there to "belong in an n-best
rerank against the source"; that home does not help it, because the n-best
rerank has the same blindness the post-editor had.

## What this retires, and what it leaves

**Retired.** Minimum-edit reranking, MBR, and feature-based reranking over this
candidate set. The oracle gap should not be quoted as headroom again: it is a
measurement of the model's uncertainty, not a reserve that better serving can
draw on.

**Standing.** Beam search is a real +0.0084 for 5x generation. Whether that is
worth paying is a serving decision, not an open research question. Note it costs
nothing in explanations -- the correction comes first in the schema so generation
stops at the `changes` marker, and since `ablations.md` §4 the learner-facing
explanation is derived by ERRANT from the correction rather than read from the
model's `changes` list.

**Untouched by this result.** Preference training. A reranker must identify the
better candidate from features computed *outside* the model; DPO on the same
pairs changes the model's parameters, so it is not bounded by what those features
separate -- it can learn the case agreement rather than a proxy for it. That is a
different information channel and this experiment does not close it. It does
lower the prior: the pairs would teach the same German the SFT corpus already
teaches, and iteration 5 showed that more data of that kind is not what is
missing. Roughly 40% of sentences yield an on-policy preference pair, so the
data is free; the cost is a generation pass over train plus a DPO run.

The labels would come from the gold annotation rather than from a larger model,
so this stays distillation-free, which is why cLang-8 was declined and this was
not.
