# Spanish port — Phase 5

> **Status: data and training pipeline ready; not yet trained.** Everything
> below has been built and run for real — corpus, annotator, synthetic error
> injection, training set — and `train_sft.py --language es --dry-run` passes
> against real data. No adapter has been trained yet, so there is no F0.5
> number in this document. When one exists it will be measured, not guessed,
> the same way every number in the main [README](README.md) is.

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
did not earn its keep even for German. There is no Spanish training run yet
to set a bar for a generator to fail to clear, so this port stops at the data
state German was in when it first passed the exit criterion (real data
primary, synthetic supporting), not at every later German iteration.

## Baselines

`run_baselines.py --language es` now exists (`configs/phase1_es.yaml`;
LanguageTool set to generic `es`, matching the downloaded `ngrams-es`
archive). Identity and LanguageTool are measured:

| System | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| identity | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| LanguageTool (`es`) | 0.4232 | 0.1588 | **0.3175** | 27.3% |

Weaker than German's LanguageTool baseline (F0.5 0.4134), which the Spanish
grammar rule file's size predicted before any of this ran — but LanguageTool
was never the bar that mattered for German either; Phase 1 found the prompted
base model beat it. Zero-shot and few-shot EuroLLM are what decide that for
Spanish, and are either running or complete by the time this is read: see
`reports/phase5/baselines_es.md` for the current numbers.

## What is genuinely not done yet

- **No trained model, no F0.5, no per-type breakdown.** `train_sft.py
  --language es` is verified ready; it has not been run to completion.
- **No Makefile targets.** Every script above is run directly with `python`;
  nothing is wired into `make train-data` / `make training-set` / `make
  train`'s Spanish equivalents yet.
- **No cross-lingual pruning experiment** (evaluating the German-pruned model
  on Spanish and vice versa) — waits on a trained Spanish model to exist.

## Running it

```bash
python scripts/download_cowsl2h.py
python scripts/build_cowsl2h_m2.py
python scripts/freeze_splits_es.py

python scripts/build_clean_corpus_es.py
python scripts/build_overcorrection_set_es.py
python scripts/build_training_set_es.py

python scripts/train_sft.py --language es --limit 64 --dry-run   # check first
python scripts/train_sft.py --language es                        # the real run
```

Needs the same Docker-free local setup as German's `make train-data`, plus
`es_core_news_sm` (`python -m spacy download es_core_news_sm`, already added
to `environment.yml`) and the Spanish Hunspell dictionary, fetched
automatically on first use the same way German's is.
