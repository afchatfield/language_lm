# Spanish port — Phase 5

> **Status: trained and measured.** Two adapters, both scored against the full
> Phase 5 baseline table on COWS-L2H dev, and one read of the held-out test
> split. Every number below came from a run on real data; none is projected.

Phase 5's purpose is to test whether the German architecture generalises, or
whether it only ever worked because it was built once, for one language, with
every design choice unconsciously shaped around it. The honest answer so far
is: **mostly, with two real exceptions worth naming up front.**

## What is not the same as German, and why

**The abstraction was one item short.** The original plan read "add a
corruptor file"; it should have read "add a corruptor file and an ERRANT
annotator." No public Spanish ERRANT existed to install — the closest
candidate (a Stanza-based multilingual reimplementation) covers English,
German, Czech, Korean and Chinese, not Spanish, and runs on a different NLP
stack than everything else here. `errant_de` was split into a shared core
(`errant_core`) plus a language profile, verified to change nothing about
German (`tests/test_errant_de.py`, still 14/14) before `errant_es` was written
against the same core.

**Two learner-error phenomena named in the original plan are declined, not
silently dropped:**

- *Preterite vs imperfect.* `es_core_news_sm` — the small spacy model, kept
  small for the same reason the German model is — tags irregular preterites
  unreliably enough that a corruptor built on those tags would mistype its own
  training data: `estuvimos` came back tagged present tense, `hube` came back
  with the lemma `hubir`, `estuviste` was not tagged as a verb at all. This is
  the same failure class Phase 2's own inspection caught for German, just
  found by testing before shipping rather than after.
- *Leísmo/laísmo.* Standard usage across large parts of the Spanish-speaking
  world, and COWS-L2H's own annotators evidently treat it that way —
  `R:PRON:FORM` is 1.17% of all real edits, not the volume a systematically
  corrected dialect feature would produce. A rule here would teach the model
  to fight a dialect, not an error.

**One phenomenon has no German analogue at all.** Spanish is pro-drop —
`tengo un perro`, not `yo tengo un perro` — and over-explicit subject
pronouns are the single largest correctable error class in the real corpus
(`U:PRON`, 8.15%, ahead of every category except stress-mark orthography).
German never drops a subject, so `errant_de` never had to type this direction
of edit. `redundant_subject_pronoun` is the rule that targets it, restricted
to the persons a Spanish verb ending names unambiguously (first singular,
second singular, first plural — never third, which splits by gender the verb
does not carry).

## The corpus

[COWS-L2H](https://github.com/ucdaviscl/cowsl2h) (Yamada, Davidson &
Perez-Cortes, LREC 2020) in place of Falko-MERLIN: real Spanish learner
essays with instructor corrections, Apache 2.0. Unlike Falko-MERLIN it ships
as document-level CSV with no train/dev/test split and no typed edits, so
building on it meant three steps German's corpus arrived already having done:

| | German (Falko-MERLIN) | Spanish (COWS-L2H) |
|---|---:|---:|
| real sentence pairs | ~24,000 | **46,473** (train 37,473 / dev 4,716 / test 4,284) |
| already correct | — | 33.0% |
| split by | essay | **learner id** (salted hash — the corpus is longitudinal, one student writes several essays, and an essay-level split would leak a student's style across train and test) |

5,382 essays, 2,881 corrected; 634 (22.0%) were dropped rather than
force-aligned because their essay and correction split into different
numbers of sentences — a corrector merging or splitting a run-on, most often,
and there is no honest automatic fix for that.

## The pipeline, and what each stage produced

Every number below came from an actual run against real data, not a config
default.

1. **`scripts/download_cowsl2h.py`** — downloads COWS-L2H pinned to a commit,
   sentence-aligns each essay against its correction with spacy, splits by
   learner id. Output: `data/interim/cowsl2h/{train,dev,test}.jsonl`.
2. **`scripts/build_cowsl2h_m2.py`** — types the aligned pairs with
   `errant_es`. **37,473 / 4,716 / 4,284 sentences, 64,909 / 7,563 / 7,333
   edits.**
3. **`scripts/freeze_splits_es.py`** — SHA-256 into `data/splits/manifest.json`,
   the same guard that keeps Falko-MERLIN test off-limits until final
   evaluation. Also writes the real error-type histogram
   (`reports/phase5/error_type_histogram_es.md`, 51 types, 72,472 train+dev
   edits) that everything downstream is weighted against.
4. **`corruptors/es.py`** — 13 rules: stress marks, keyboard typos, the
   missing opening ¿/¡, article agreement (form and omission), adjective
   agreement, preposition confusion (por/para headline) and omission,
   redundant subject pronouns, dropped object clitics, ser/estar, subjunctive
   after `que`, and the bare infinitive for a finite verb. Every rule's
   claimed ERRANT type is verified against real `errant_es` output, not
   assumed — two of the thirteen were wrong on the first pass (`ser_estar`
   and `subjunctive_for_indicative` claimed `R:VERB(:FORM)`; `errant_es` tags
   the copula `AUX`, giving `R:AUX(:FORM)`) and caught by testing rather than
   shipped. A subsequent stress test over 800 real, already-correct COWS-L2H
   sentences found and fixed five more defects, the most serious being
   `article_form` swapping Spanish's article/object-clitic homograph (`la` is
   both "the" and "her") and calling the result a wrong article.
   `tests/test_corruptors_es.py` (32 cases) regression-tests all five.
5. **`scripts/build_clean_corpus_es.py`** — the same filters `clean_de.py`
   applies (`langlm.data.quality`, now parameterised by language; the Spanish
   character set was read off a real survey of `spa_news_2023_100K`, not
   guessed), over `spa_news_2023_100K` + `spa_wikipedia_2021_100K`. **60,000
   sentences kept of 200,000 read.**
6. **`scripts/build_overcorrection_set_es.py`** — 400 frozen sentences, held
   out of the clean corpus by construction, the Spanish counterpart of the
   400-sentence German overcorrection set Phase 1's baselines are scored
   against.
7. **`scripts/build_training_set_es.py`** — real (COWS-L2H) + synthetic
   (corruptors over the clean corpus), mixed. **97,473 examples: 37,473 real +
   60,000 synthetic, 51 distinct error types, 0 broken examples, 0 edits with
   no explanation.** Read against the same 100-sample exit criterion Phase 2
   used, which caught one real defect: an `M:PUNCT` explanation that
   overclaimed "likely the opening ¿ or ¡" on an edit that was actually a
   missing comma. Fixed before the numbers above.

## Two rules ship with a measured, non-zero mistype rate

`verb_infinitive` (22.3% of 121 fires) and `adjective_form` (33.3% of 48
fires), measured over the same 800-sentence stress test, sometimes have
`es_core_news_sm` re-lemmatise the damaged word to a different lemma than the
original once the sentence around it is ungrammatical, so `errant_es` calls
the edit `R:OTHER`/`R:MORPH`/`R:VERB` rather than the `:FORM` type the rule
claims. **The correction itself is never wrong in either case** — only the
type label the per-type histogram and the model's own `changes` list see.
`adjective_form` is the rule most worth revisiting if this project ever moves
past the small spacy model.

## One scoping decision, made explicitly

**No back-translation.** German's own iteration 5 added a back-translation
generator after its first successful training run, predicted +0.02–0.03 F0.5,
and measured −0.0016 to −0.0017 instead (`reports/phase3/headroom.md`) — it
did not earn its keep even for German. This port stops at the data state
German was in when it first passed the exit criterion (real data primary,
synthetic supporting), not at every later German iteration. There is now a
Spanish training run to set a bar a generator would have to clear — 0.5987 —
but building one to test against it is a decision for a later iteration, not
a gap in this one.

## Baselines

All four, on COWS-L2H dev (4,716 sentences, 7,563 gold edits):

| System | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| identity | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| LanguageTool (`es`) | 0.4241 | 0.1588 | **0.3179** | 27.3% |
| zero-shot EuroLLM | 0.3872 | 0.0510 | **0.1671** | 11.2% |
| few-shot EuroLLM | 0.5664 | 0.1088 | **0.3077** | 8.0% |

**There is no clear bar, and that is itself the divergence from German.** In
German the prompted base model won outright — few-shot 0.5102 against
LanguageTool's 0.4134. In Spanish the same prompt, model and shot count leave
the two indistinguishable: LanguageTool leads by 0.0102, and a paired bootstrap
over the 4,716 dev sentences puts that difference at 95% CI [−0.0297, +0.0087],
p = 0.29. LanguageTool is quoted as the bar below because it is the larger of
the two, not because it is measurably better.

**It is not the corpus's editing style.** That was the first hypothesis and it
is false: COWS-L2H corrects sentences almost exactly as minimally as
Falko-MERLIN does. Mean character-level edit distance over corrected sentences
is 9.087 against German's 9.082, medians 6 and 6; in token terms Spanish edits
are *smaller* (mean 3.27, median 2) than German's (3.63, median 3), and
identical once normalised by sentence length (0.226 against 0.220). Pairs that
share no content word at all — genuine sentence-alignment failures — are 10 of
25,110 corrected sentences, 0.04%.

**Part of it was the prompt this seed drew.** Shot 5 of 8 was

> **Input:** Y el lugar donde sucedió mi historia o cuando comenzó mi infancia
> es el mejor lugar en mi memoria.
> **Output:** Ese recuerdo fue mi recuerdo más hermoso.

with a churn of 0.86 — the top 4.4% of the corpus by how much of the sentence
it replaces, and the only shot in either language's prompt above 0.25 (German's
run 0.00 to 0.22). An eighth of the pattern was an instruction to replace the
sentence rather than repair it. `sample_shots` now rejects an example above
`MAX_SHOT_CHURN` (0.5) and draws a replacement, which swapped that one shot for
a real correction and moved few-shot from 0.2985 to **0.3077**, recall 0.1072 to
0.1088, overcorrection 9.0% to 8.0%.

The rejection happens *after* the draw rather than by filtering the pool first,
and that distinction was measured rather than assumed: filtering the pool
changes what every later draw indexes into, and it silently replaced six of
German's eight shots and pushed its worst churn from 0.22 to 0.33. As
implemented, a prompt with nothing to reject spends no randomness and comes out
bit-identical, so **no German number moves**.

**The rest is the model.** With a clean prompt few-shot still alters only 22.4%
of dev sentences when 65.9% of them need a change: a base model given eight
examples is reluctant to edit, and its recall of 0.1088 against LanguageTool's
0.1588 is the whole of the remaining difference. LanguageTool's accent and
spelling rules do not read a prompt, so they fire on every sentence matching
them.

The rest of the baseline was checked before this was believed: the instruction
really is the Spanish one, the shots really are drawn from COWS-L2H train,
`max_new_tokens` binds on 2 of 4,716 sentences, and the shot mix follows the
same rule German's does (2 of 8 unchanged there, 3 of 8 here, mirroring each
corpus's own correct share). **The few-shot row is therefore a fair reading of
this prompt, and an unlucky reading of the method** — see the note below.

## What the fine-tuning is worth

Two runs, identical but for the loss weight on the `changes` list:

| System | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| best baseline (LanguageTool) | 0.4241 | 0.1588 | 0.3179 | 27.3% |
| fine-tuned, `changes` weight 1.0 | 0.6373 | 0.4559 | 0.5904 | 7.0% |
| **fine-tuned, `changes` weight 0.25** | 0.6510 | 0.4531 | **0.5987** | 6.0% |

**+0.2808 over the bar**, at less than a quarter of LanguageTool's
overcorrection rate. German's own margin over its bar was +0.1842.

The `changes` down-weighting German measured at +0.0128 is worth **+0.0100**
here (paired bootstrap over dev, 10,000 resamples, 95% CI [+0.0027, +0.0173],
p = 0.0070), and it buys that entirely in precision (+0.0137) with recall
flat. It is not free: explanation coverage falls 0.948 → 0.925 and type
accuracy 0.834 → 0.818, which is what down-weighting the field the
explanations come from should do.

A control run settled which of the two changes mattered. The `0.25` adapter
selects its best checkpoint by evaluation loss (epoch 1.98) while the `1.0`
one shipped its final weights, so the two differed in more than the loss
weight. Scoring the `1.0` run at a comparable checkpoint gives 0.5887 against
its own final 0.5904 — a difference of +0.0016, CI [−0.0030, +0.0062],
p = 0.48. **Checkpoint choice is noise here and the loss weight is not**,
which also means the evaluation-loss minimum at epoch 2 is not an F0.5
minimum: the loss rose from epoch 2 to 3 while F0.5 did not fall.

### The held-out split

Read once, on the `checkpoints/phase3-es-cw025` adapter, after the recipe was
chosen. COWS-L2H test, 4,284 sentences, 7,333 gold edits:

| | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| **fine-tuned, `changes` weight 0.25** | 0.6878 | 0.4867 | **0.6353** | 6.0% |

**Higher than dev's 0.5987, by +0.0366** — the opposite of the direction a
model that had been tuned against its dev split would move, which is the point
of having kept the split sealed. The two splits are disjoint sets of learners,
not disjoint samples of one pool (COWS-L2H is longitudinal, and the split is
by salted learner id), so they are not interchangeable draws and a gap in
either direction is expected. Explanation quality holds: coverage 0.931,
precision 0.969, type accuracy 0.827, against dev's 0.925 / 0.969 / 0.818.

`reports/phase5/evaluation-fine-tuned-es-cw025-test.md` carries this one row
and no bar. It used to table the dev baselines beside it, which invited exactly
the comparison that reading a sealed split once is meant to avoid — 0.6353 is a
test number and the 0.3179 it sat next to was not. The report generator now
records which split each result came from and compares only within one, so a
held-out run stands alone and the comparisons stay on dev, where every other
system in this document was measured.

### Where it wins, and where it does not

| Error type | Gold | LanguageTool R | few-shot R | fine-tuned R |
|---|---:|---:|---:|---:|
| `R:ORTH` (stress marks) | 838 | 0.69 | 0.21 | **0.74** |
| `U:PRON` (pro-drop) | 594 | 0.00 | 0.02 | **0.73** |
| `R:OTHER` | 1251 | 0.02 | 0.05 | 0.20 |
| `M:PUNCT` | 249 | 0.33 | 0.15 | 0.34 |
| `R:VERB:FORM` | 357 | 0.07 | 0.12 | 0.36 |
| `R:AUX:FORM` | 158 | 0.02 | 0.04 | 0.26 |

`U:PRON` is the vindication of the one rule with no German analogue: the
redundant subject pronoun, 594 gold edits that **no** baseline corrects at
all, and the fine-tuned model reaches 0.73. `R:ORTH` takes the stress marks
off LanguageTool, which is where the rule engine earned most of its score.

The weak rows are the ones this port declined in advance and said so. Verb
morphology — `R:VERB:FORM` 0.36, `R:AUX:FORM` 0.26, `R:VERB` 0.25 — is
exactly where preterite/imperfect was dropped because `es_core_news_sm` could
not tag it reliably, and where `verb_infinitive` ships with a measured 22.3%
mistype rate. The declined corruptor is legible in the scoreboard.

## Does the architecture generalise?

Yes, and the margin says so more strongly than the absolute number does:

| | German | Spanish |
|---|---:|---:|
| best baseline | 0.5102 (few-shot) | 0.3179 (LanguageTool) |
| fine-tuned | 0.7072 | 0.5987 |
| margin over the bar | +0.1842 | **+0.2808** |
| overcorrection | 13.0% | **6.0%** |
| metric ceiling | 0.9976 | 0.9948 |

Spanish sits about 0.10 F0.5 below German in absolute terms, and the obvious
explanation is wrong. `R:OTHER` — the untypeable-rewrite bucket no corruptor
produces — is 16.5% of Spanish gold edits against 8.7% of German's, so the
two corpora are not equally tractable. Scoring both languages with that type
removed entirely lifts German 0.6174 → 0.6465 and Spanish 0.5496 → 0.5788:
**+0.0291 and +0.0292, and the gap does not move.** Corpus composition
explains none of it. What remains is concentrated in verb morphology and
punctuation, which is a statement about what this port chose not to build
rather than about whether the architecture travels.

## What is genuinely not done yet

- **No cross-lingual pruning experiment** (evaluating the German-pruned model
  on Spanish and vice versa). Phase 4 trimmed the vocabulary to the rows
  German and English text touches, which makes this a real test rather than a
  formality: Spanish's accented forms are single tokens, and whether they
  survived that trim is not something to guess at.
- **No preference training.** German's iteration 6 ran DPO over its own beam;
  this port stops at supervised fine-tuning, for the same reason it stops
  short of back-translation.
- **No leniency relation.** Every tier of German's is a fact about German —
  OpenThesaurus synonyms, closed against hyphenated compounds — and there is
  no Spanish resource standing in for it, so a Spanish run reports the strict
  number alone rather than a leniency that quietly means something else.

## Running it

```bash
make train-data-es     # corpus, ERRANT typing, frozen splits, clean text, training set
make dry-run-es        # loss mask, schema, one forward pass. Trains nothing.
make train-es          # the real run
make evaluate-es       # score it against the Phase 5 baselines
```

`make phase5` chains all four. The baselines are separate, because
LanguageTool needs a server the training box does not otherwise want:

```bash
make lt-up ngrams-es   # LanguageTool with the Spanish n-grams
make phase1-es
```

To score a second recipe without overwriting the first one's report:

```bash
make evaluate-es ADAPTER_ES=checkpoints/phase3-es-cw025 NAME_ES=fine-tuned-es-cw025
```

Running the scripts directly works too, but note the order: the overcorrection
set is held out of the clean corpus *by construction*, so it has to be built
first or `build_clean_corpus_es.py` fails looking for it.

Needs the same Docker-free local setup as German's `make train-data`, plus
`es_core_news_sm` (`python -m spacy download es_core_news_sm`, already added
to `environment.yml`) and the Spanish Hunspell dictionary, fetched
automatically on first use the same way German's is.
