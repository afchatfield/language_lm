# Phase 3 ablations: what does not work, and why

Four things tried against the cached dev generations of `checkpoints/phase3-de`,
no retraining involved. Three of them fail. They are written down because each
one is an idea worth having, and knowing the number is what stops it being had
again.

Baseline throughout: **M2 F0.5 0.6944** (P 0.7273, R 0.5881), ERRANT detection
0.7417 against correction 0.6174 on Falko-MERLIN dev, 2,503 sentences.

## 1. Easing the metric for like-for-like word choice

The detection-correction gap is 700 edits where the model finds the right span
and writes the wrong replacement. The question was how much of that is the model
being wrong and how much is one annotator's wording being treated as the only
answer.

**Not by scoring spans alone.** Dropping the replacement from the comparison
gives F0.5 = 0.8692, which is not a softer metric but a broken one. MaxMatch
chooses how to *describe* a system's edits so as to match the gold annotation;
with the replacement text unconstrained it re-describes rewrites onto whatever
spans the annotator touched. The tell is a recall of 0.8469 against the same
system's ERRANT detection recall of 0.6607: it cannot have found 85% of the
gold spans when a direct span comparison says it found 66%.

**Tightly, then.** `langlm.eval.leniency` accepts a replacement for the
annotator's when they are typographic variants, the same compound written closed
or hyphenated (`Chemiestudentin` / `Chemie-Studentin`), or OpenThesaurus
synonyms in the same inflection. The span still has to be right.

| | M2 F0.5 | ERRANT correction F0.5 |
|---|---:|---:|
| strict | 0.6944 | 0.6174 |
| typographic + compound | 0.6955 | 0.6183 |
| + synonyms | 0.6990 | 0.6216 |

**+0.0046 in total.** Sixty of the disputes were judged by hand from the full
sentence: 18 of 60 were defensible, and the shipped relation accepts 11 of those
18 with no false accepts. The seven it misses are not thesaurus cases -- they
are places where the model is *better* than the annotation (`Händler` for a gold
`Handeler`, which is not a word; `kostspielig` where the gold `köstlich`
misreads the sentence) and no lexical resource will reach them.

So the answer to the original question is that the gap is real error. The wrong
replacements are dominated not by defensible paraphrase but by misspellings
whose intended word is recoverable, answered with a different word:

| gold | model |
|---|---|
| `rabben` → `rauben` | `finden` |
| `Verbrenchen` → `Verbrechen` | `Verbrennen` |
| `gelont` → `gelungen` | `gelogen` |
| `leiben` → `lieben` | `leben` |

This corrects the earlier reading of that bucket. "Different word entirely" is
41.6% of the wrong fixes, but it is not mostly defensible German with a single
annotator refusing credit -- most of it is the model failing at recovery.

## 2. Repairing spellings from a dictionary at serving time

R:SPELL is the largest gold type (816 edits, 12.9% of dev) with the largest
detection-correction gap (+0.218, about 178 edits). A dictionary should be able
to help: of those 178, the nearest in-vocabulary word to what the learner wrote
*is* the gold correction in 71.

It does not survive contact with a gate. Building a 236k-type lexicon from
Leipzig frequencies plus the Hunspell stems, and overriding the model whenever
it rewrote an out-of-vocabulary token into another out-of-vocabulary token:

| rule | F0.5 |
|---|---:|
| baseline | 0.6944 |
| override when the model's word is not in the lexicon | 0.6884 |
| ...and only for distance-1 repairs | 0.6924 |
| override every out-of-vocabulary source token | 0.6697 |

At the edit level the tightest gate fires 67 times: 7 fixes, 43 breaks. The
breaks are all the same failure -- a type list cannot judge German inflection and
compounding, so `Studienprogramme`, `Referate`, `Grüsse`, `festgekettet` and
`geehrte` all read as non-words and get "repaired" into something worse. A real
morphological check (Hunspell with affix expansion) would stop the false
triggers, but it would not reach the prize: the 71 recoverable cases are ones
where the model's answer *is* a valid German word, just the wrong one, and no
lexicon gate can see that. The dictionary belongs in the corruptors or in an
n-best rerank against the source, not in a post-editor.

## 3. Filtering by error type

Already ruled out for the type the model claims. It is also worth nothing for
the type ERRANT *derives* from the model's own correction, which is the better
feature -- it is deterministic, and available whether or not the model's
`changes` list survived generation. Greedily dropping the least precise types
with real volume:

| dropped | F0.5 | P | R |
|---|---:|---:|---:|
| nothing | 0.6944 | 0.7273 | 0.5881 |
| `M:OTHER` | 0.6945 | 0.7292 | 0.5836 |
| `+ U:OTHER` | 0.6943 | 0.7306 | 0.5793 |
| `+ R:VERB` | 0.6954 | 0.7333 | 0.5765 |
| `+ R:OTHER` | 0.6895 | 0.7452 | 0.5308 |

The best is +0.0010. Precision does climb, but never fast enough to pay for the
recall, even under a metric that weights it double. There is no serving-side
filter on this output worth having.

## 4. Explanations from derived types

The one that works, and it is not a score change. The model names its own edit
correctly 0.7106 of the time. ERRANT, given only the pair (source, the model's
own correction), re-derives the annotator's label on **0.8909** of the edits it
segments the same way -- 0.8367 measured over every gold edit including the ones
it segments differently.

`langlm.explanations.explain_correction` renders the learner-facing explanation
that way, so the explanation no longer depends on a skill the model is
separately bad at. The `changes` list keeps its diagnostic job in
`langlm.eval.explanation_scorer`; it is just no longer what the learner reads.

## The coverage lever, as it looked before it was tried

Nothing on the serving path moves the score. The two levers left were the ones
that need a training run: coverage of the 45 error types no corruptor produces,
and R:SPELL recovery.

The coverage number as measured then: the synthetic half produced 19,480 edits
across 8 types. Dev recall on those 8 was **0.667**; on the 45 types it never
produced, over 2,920 gold edits -- 45.7% of the annotation -- it was **0.404**.
Section 5 below is what happened when that was acted on, and why the inference
from those two numbers did not hold.

## 5. Widening the corruptors, and why the coverage argument was wrong

The corruptors now produce 17 error types rather than 8, taking the share of
dev's gold edits in uncovered types from 45.7% to 20.0%
(`reports/phase2/injection_weights.md`, and `corruptor_types.md` for the check
that each new rule produces the error it claims). Retrained on the rebuilt
corpus, same recipe, same effective batch:

| | P | R | F0.5 | Overcorrection | Type accuracy |
|---|---:|---:|---:|---:|---:|
| iteration 2 | 0.7273 | 0.5881 | 0.6944 | 13.0% | 0.7106 |
| iteration 3 | 0.7268 | 0.5953 | 0.6960 | 12.5% | 0.7229 |

**+0.0016, which is nothing.** A paired bootstrap over the 2,503 dev sentences
puts the 95% interval on that delta at [-0.0051, +0.0084]; iteration 3 is ahead
in 66% of resamples. Recall on the newly covered types moved 0.453 to 0.463, and
on everything else 0.579 to 0.583 -- the same drift, in a corpus where the
per-type gold counts are 65 to 260 and a swing of 0.04 is three or four edits.

The prediction was recall 0.547 -> 0.667 and F0.5 around 0.72. It was wrong, and
the reason is a mistake in the statistic this whole section rests on. "45.7% of
gold edits are in types no corruptor produces" is true, and it was read as "the
model has never seen these errors". It has. The corruptors are one half of the
training corpus; the other half is 19,237 real Falko-MERLIN sentences carrying
49,058 edits across all 54 types. Every type called uncovered was already there:

| type | real half | synthetic added |
|---|---:|---:|
| `R:OTHER` | 3,869 | 1,198 |
| `R:NOUN:FORM` | 1,930 | 785 |
| `U:PUNCT` | 1,756 | 807 |
| `M:PRON` | 1,177 | 494 |
| `R:VERB:FORM` | 963 | 439 |
| `R:PRON:FORM` | 517 | 74 |

So this was never zero-to-something. It was a 13-36% top-up on a base the model
already had, and it bought what a 13-36% top-up buys. The correlation the
argument was built on -- types with synthetic coverage average F0.5 0.679 against
0.488 for types without -- runs the other way round: the corruptors were written
for the types that are easy to generate, and those are largely the frequent,
mechanical ones that any model does well on. Coverage was a marker of easiness,
not a cause of it.

What did improve is worth keeping: overcorrection 13.0% -> 12.5% on published
prose, and the model's own type accuracy 0.7106 -> 0.7229. The wider rule set is
also what a Phase 5 Spanish port would be built from, and the `das_dass`
mislabel it turned up was a real bug. But as a lever on F0.5 it is spent.

## Where this leaves the score

Four serving-side changes and one corpus change, and none of them moved it. The
score has been 0.694-0.696 throughout. That is now the interesting fact: the
remaining loss is not in the metric's strictness, not in the type mix of the
training data, and not in anything a filter or a dictionary can reach from
outside the model. What is left is the model -- 1.7B parameters, and the honest
next lever is a larger base, which collides with Phase 4's compression story
exactly as the earlier reading said it would.
