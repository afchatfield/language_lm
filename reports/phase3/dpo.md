# Phase 3: preference training, and the last thing the beam had to offer

`nbest.md` retired reranking and left exactly one thing standing:

> **Untouched by this result.** Preference training. A reranker must identify the
> better candidate from features computed *outside* the model; DPO on the same
> pairs changes the model's parameters, so it is not bounded by what those
> features separate — it can learn the case agreement rather than a proxy for it.

That is now spent too. It does not work, it fails in a way that says something,
and the way it fails closes the oracle gap as a research direction rather than
leaving it open for a cleverer method.

Everything below is Falko-MERLIN **dev**. The test split was not touched by any
of it.

## The result

| System | F0.5 | Overcorrection | Rewrites |
|---|---:|---:|---:|
| **iter4-cw025** (the bar) | **0.7072** | 9.8% | 5,658 |
| + DPO, beta 0.1 | 0.6739 | 15.0% | 6,285 |
| + DPO, beta 0.5 | 0.6895 | 12.5% | — |
| **iter5** | **0.7056** | 13.0% | 5,872 |
| + DPO, beta 0.1 | 0.6905 | 15.5% | — |

Two checkpoints, three runs, nothing above its own starting point.

## Why the beam was worth using even though beam search is dead

Worth stating plainly, because it looks like the retired idea. `nbest.md` killed
beam search as a *serving* strategy: min-edit, MBR and a nine-feature learned
reranker all landed within 0.001 of beam top-1, and the published path stayed
greedy. None of that is reopened here. **After DPO the model is still served
greedy — the beam is not in the inference path at all.**

Here the beam is an offline data-generation device, run once over Falko train. It
is not asked to *pick* the better candidate, only to *exhibit* two candidates so
the gold annotation can label which is better. A reranker cannot tell `der
meisten Firmen` from `den meisten Firmen` because the features it reads cannot
read German. Gradient descent has no such limit; it is not scoring features, it
is moving weights. That was the argument, and it was a reasonable one.

## The pairs

A pair is emitted where the model is wrong about its own beam: the candidate
scoring best against the annotator is not the one the model ranked first, and it
is strictly better in true positives minus false ones rather than merely
narrower.

| | iter4-cw025 | iter5 |
|---|---:|---:|
| pairs | 4,289 of 19,237 (22.3%) | 4,324 of 19,237 (22.5%) |
| mean gain (tp − fp) | 1.64 | 1.63 |
| mean chosen rank | 1.80 | 1.78 |

The 22% is lower than the 40% `nbest.md` quotes because that figure counted
sentences where the beam *reaches* the annotator's correction; a pair also
requires the chosen side to be strictly better, so ties and equal-but-narrower
candidates drop out.

The two checkpoints agree to within a few tenths of a percent on every column,
which is worth more than it looks: the oracle gap is a stable property of this
model class trained this way, not an accident of one checkpoint.

## What went wrong, and why more tuning will not fix it

The first run separated the pairs and damaged the model at the same time.
Preference accuracy on held-out pairs reached 0.681 against a chance level of
0.500, so the pairs are learnable. But `rewards/chosen` was **−0.88** and
`rewards/rejected` **−0.99**: both negative, meaning the policy assigned *lower*
likelihood than the reference to both sides of every pair. It was not pulling the
chosen answer up. It was pushing the rejected answer down harder than the chosen
one, and drifting away from both.

That drift is visible in the output. Rewrites rose from 5,658 to 6,285 and
overcorrection from 9.8% to 15.0% — the model started editing more, and more
widely, which is precisely the failure the whole project is built against.

The fix for drift is a tighter KL anchor, so beta went from 0.1 to 0.5. It worked
as a diagnosis: preference accuracy rose to 0.753, and the underlying log-ratio
drift — reward divided by beta, since the reward scales with it — shrank from
−9.1 to −5.4. Less drift, better separation, and F0.5 recovered from 0.6739 to
0.6895.

Still short of 0.7072, and the shape of that recovery is the finding:

| beta | F0.5 | Overcorrection |
|---:|---:|---:|
| 0.1 | 0.6739 | 15.0% |
| 0.5 | 0.6895 | 12.5% |
| ∞ (the reference, untrained) | 0.7072 | 9.8% |

Tightening the anchor moves F0.5 monotonically back toward the SFT model and
never past it. The limit of the sweep is the model we already have. There is no
beta worth finding, because the best available outcome of this procedure is doing
nothing, and every departure from the reference is paid for in precision.

iter5 makes the same point from the other end. Its run barely moved: eval loss
0.6791 against ln 2 = 0.6931, margin 0.032, drift only −1.2. It neither drifted
nor learned — and still lost 0.0151. A procedure that hurts when it does almost
nothing is not one that a better setting rescues.

## What this retires

**The oracle gap.** `nbest.md` said it should not be quoted as headroom again,
having shown no *selector* could reach it. This shows no *optimiser* reaches it
either, on pairs drawn from that same gap, with the gold annotation supplying the
labels. The 0.8380 oracle is a measurement of the model's uncertainty. It is not
a reserve, and nothing left in the toolbox draws on it.

**Preference training on gold-labelled on-policy pairs**, at least in this shape.
What remains untried is a different pair *source* — pairs that disagree about
German rather than about how far to edit — but nothing in the repo produces those,
and `headroom.md`'s conclusion stands: past this point the base model is the
binding constraint, and trading it for a larger one is the trade Phase 4 exists
not to make.

## Cost

Two beam passes over Falko-MERLIN train at 34 minutes each, three DPO runs at
about four minutes each, five dev evaluations. Roughly two hours of H200 time for
a negative result, which is the cheapest of the three serving-side or
training-side ideas the oracle gap has now absorbed.

The code stays: `src/langlm/train/preference.py`, `scripts/build_preference_pairs.py`,
`scripts/train_dpo.py`, `configs/phase3_dpo.yaml`, and the pair caches. Phase 5
ports this repo to Spanish, and a negative result that has to be re-derived from
scratch is a negative result that gets re-run.

## One implementation note worth keeping

The first version of the pair builder truncated both completions at the `changes`
marker, reasoning from iteration 4 that the `changes` list is 70% of the tokens
and F0.5 never reads it, so it did not deserve preference gradient either.

That was wrong, and not by a little. TRL appends an EOS token to both sides of
every pair and offers no way to switch it off, so a truncated completion teaches
the model that an answer *ends* at `{"correction": ..., "changes":` — which would
have broken the schema on every sentence rather than the 1 in 2,903 it breaks on
now. Iteration 4's argument is about where gradient is spent on a target being
imitated; a DPO completion is not a target, it is one half of a comparison, and
the comparison has to be between two things the model could actually emit.

Both sides now carry a `changes` list rebuilt by the same German ERRANT that
derives the learner-facing explanation, and `tests/test_preference.py` pins the
invariant that a completion parses as a whole answer.
