# Phase 4: what compression costs

The gate (`gate.md`) established that a vocabulary-trimmed checkpoint converts
and runs. This measures what the compression is worth, on Falko-MERLIN **dev**,
scored with the same MaxMatch and German ERRANT machinery as every number in
Phases 1 and 3.

**Two results.** The vocabulary trim is free: 21.9% of the parameters for 0.0011
of F0.5, with no healing and no retraining. Depth pruning is not: the 600 MB
target is reachable, at 593 MB, and it costs 0.046 of F0.5.

## The table

| Variant | Params | Q4_K_M | F0.5 | P | R | Overcorr. | Expl. type |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uncompressed (corpus A) | 1.657B | 997 MB | **0.7072** | 0.7339 | 0.6174 | 9.8% | 0.696 |
| Vocab-trimmed, adapter reused | 1.294B | 755 MB | **0.7061** | 0.7327 | 0.6164 | **9.5%** | 0.694 |
| Vocab-trimmed + LM healing | 1.294B | 755 MB | 0.6991 | 0.7284 | 0.6022 | 7.2% | 0.693 |
| Vocab-trimmed + re-SFT (corpus B) | 1.294B | 755 MB | 0.7048 | 0.7274 | 0.6269 | 14.5% | 0.750 |
| + drop 2 layers (22) | 1.200B | 698 MB | 0.6811 | 0.7035 | 0.6042 | 16.5% | 0.716 |
| + drop 4 layers (20) | 1.105B | 647 MB | 0.6716 | 0.6997 | 0.5787 | 15.0% | 0.690 |
| + drop 6 layers (18) | 1.011B | **593 MB** | 0.6591 | 0.6912 | 0.5557 | 16.8% | 0.689 |

Sizes are measured `llama-quantize` output at Q4_K_M, not projections.

## The vocabulary trim is free

**21.9% of the parameters and 242 MB at Q4 for 0.0011 of F0.5**, and the Phase 3
adapter was reused unchanged -- no healing, no retraining, not a single GPU-hour
of recovery. Every metric moves by about 0.001, five times smaller than the
±0.005 spread between dev and test measured across five models in
`reports/phase3/test.md`. Overcorrection is marginally *better*.

That works because the LoRA targets `q/k/v/o/gate/up/down_proj` and none of them
touch the vocabulary dimension, so an adapter trained on the full vocabulary is
dimension-valid on a trimmed base. The trim removes rows the model had no use
for; the rows it keeps are bit-identical.

## Healing is not needed, and costs F0.5

A short LM-objective pass over 60,000 clean German sentences -- the roadmap's
healing step -- moved F0.5 from 0.7061 to **0.6991**, a loss of 0.0070. It bought
2.3pp of overcorrection (9.5% → 7.2%) by making the model more conservative:
recall fell 0.6164 → 0.6022 while precision barely moved.

That is a real trade and a defensible one if restraint is the priority, but it is
not what healing was for. Healing exists to repair damage, and the ablation above
shows there was no damage to repair. **The step is dropped from the pipeline.**

## The control, and what it caught

The depth-pruned variants had to be re-trained from scratch, and the corpus on
disk is no longer the one the uncompressed baseline used: iteration 5's
back-translation recipe (75,938 examples, "corpus B") replaced iteration 4's
(28,733, "corpus A"), which cannot be regenerated because the corruptions are
drawn at random.

So a control was trained: the same corpus B recipe on the **unpruned** trimmed
base. Without it the depth rows would have been compared against a corpus-A
number and pruning would have been charged for someone else's regression.

It was charged for one. Raw against the 0.7061 reference, the first prune point
looks like it nearly doubles overcorrection, 9.5% → 16.5%. **The control is
already at 14.5% with zero layers dropped.** Five of those six points are the
corpus. Pruning adds 1--2pp on top, not 7.

| Compared against | drop 2 | drop 4 | drop 6 |
|---|---:|---:|---:|
| 0.7061 (corpus A, no control) | −0.025 | −0.034 | −0.047 |
| **0.7048 (corpus B control)** | **−0.024** | **−0.033** | **−0.046** |

On F0.5 the two readings nearly coincide, because corpus B is F0.5-neutral. On
overcorrection they differ by a factor of three, and overcorrection is the
metric this project has prioritised since Phase 1.

### A second verdict on iteration 5

Phase 3 retired the back-translation corpus because it scored 0.7056 against the
0.7072 bar -- a loss inside the noise. Here the same data, on a different base,
with a different recipe, is again F0.5-neutral (0.7048 vs 0.7061, −0.0013) and
**costs 5.0pp of overcorrection** (9.5% → 14.5%).

It is not uniformly worse. Recall improves (0.6164 → 0.6269) and explanation type
accuracy improves substantially (0.694 → 0.750), which is consistent with a
corpus that covers more error types. But it buys those by editing more, and the
edits it adds are wrong often enough to leave F0.5 flat while overcorrection
climbs. `reports/phase3/dpo.md` called iteration 5 a null result; on this evidence
it was a negative one, and the Phase 3 write-up understated it.

## The depth curve

Measured by input-output cosine similarity over 512 German sentences
(`layer_similarity.json`). 24 layers: layer 0 scores 0.309 and layer 23 scores
0.500, while the 22 interior layers sit in a narrow band, mean 0.924, range
0.873--0.943. The edges do most of the work and the interior is close to
uniform -- which is why the heuristic cannot rank the interior finely, and why
the roadmap's insistence on a curve rather than a single point is right.

The chosen windows are contiguous and central: 13--14, then 12--15, then 11--16.

**The curve is sublinear, in the useful direction.** The first two layers cost
0.024 and the next four cost only 0.022 more. Most of the price is paid for
disturbing the residual stream at all rather than for the number of layers
removed, which argues for taking more once any are taken.

**Depth pruning does not survive without re-training, and the vocabulary trim
did.** Applying the Phase 3 adapter to a depth-pruned base -- the trick that made
the vocabulary trim free -- produces a model that emits no schema whatsoever.
Dropping 2 of 24 layers is enough: the output becomes free prose running to the
full 1024-token cap, and at 6 layers it degenerates into repetition (`Die Leute
mögen die Leute, die Leute`). The adapter survives losing 69% of the vocabulary
and not 8% of the depth, because removing layers changes the residual stream that
the layers around them were tuned against. Every depth row above is a full re-SFT.

## Where the size target lands

| | |
|---|---:|
| Target | 600 MB |
| Vocabulary trim alone | 755 MB |
| Vocabulary trim + 6 dropped layers | **593 MB** |

**The target is met.** Phase 0 predicted vocabulary trimming would land near 782
MB and that the depth prune would have to close the gap; it landed at 755 MB and
the depth prune closed it.

What it costs is 0.046 of F0.5, and the result is still far above every baseline
this project has: 0.6591 against few-shot's 0.5102 and LanguageTool's 0.4134.

## Two defensible claims

The exit criterion asks for one. There are two, and they answer different
questions.

> **39.0% fewer parameters and 40.5% smaller on disk (997 → 593 MB) for 0.046 of
> F0.5**, hitting the 600 MB offline target.

> **21.9% fewer parameters and 24.2% smaller (997 → 755 MB) for 0.0011 of F0.5**,
> which is free within measurement error and needs no retraining at all.

## What should ship

**The vocabulary-trimmed model with the Phase 3 corpus-A adapter: 755 MB, F0.5
0.7061, overcorrection 9.5%** -- unless 600 MB is a hard constraint.

It is the best row in the table on both metrics that matter, it costs nothing to
produce beyond the trim itself, and its 9.5% overcorrection is the lowest of any
configuration except the healed one. The depth-pruned variants reach 600 MB but
give up 0.046 of F0.5 *and* inherit corpus B's overcorrection regression, because
corpus A no longer exists to re-train them on.

That last point is a limitation rather than a conclusion: **the depth rows may be
pessimistic by roughly 5pp of overcorrection**, since they were necessarily
trained on the corpus that costs 5pp. Rebuilding a corpus-A-equivalent and
re-running the 6-layer point is the one experiment that would settle whether 593
MB can be had at single-digit overcorrection.

## Not measured here

**Tokens/sec on the M1.** The ablation table asks for it and this is a
Linux/CUDA box. It belongs with the MLX conversion, which is also Apple-Silicon
only and also outstanding (`gate.md`).

**The test split.** Every number here is dev. The test split was read once at the
end of Phase 3 for five models; the compression variants have not touched it, and
should not until the shipping configuration is chosen.
