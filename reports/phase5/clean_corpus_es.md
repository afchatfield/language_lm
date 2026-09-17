# Phase 5: the Spanish clean corpus

Source: `spa_news_2023_100K`, `spa_wikipedia_2021_100K`, shuffled with seed 20260909.
Mirrors `reports/phase2/clean_corpus.md`'s funnel for German.

## The funnel

| Stage | Sentences |
|---|---:|
| read from Leipzig | 200,000 (100.0%) |
| duplicate | 1 (0.0%) |
| failed the prose filter | 17,457 (8.7%) |
| outside the length window | 3,759 (1.9%) |
| unknown lower-case word | 22,900 (11.5%) |
| held out for overcorrection | 400 (0.2%) |
| pooled for parsing | 155,483 (77.7%) |

The pool is then shuffled and parsed until the corpus is full, so the stage below counts the sentences the parser actually saw rather than the whole pool.

| Stage | Sentences |
|---|---:|
| parsed | 62,498 (100.0%) |
| fragment: no finite verb, or a stranded subordinate clause | 2,498 (4.0%) |
| kept | 60,000 (96.0%) |

**60,000** sentences in the corpus.

## Why fragments are dropped

Leipzig is news and Wikipedia, so some of it is headlines and captions, and subordinate clauses printed as whole sentences. None of that is ungrammatical; it is poor material to damage on purpose -- a word-order or agreement corruptor has no main clause to work on, and a model asked to restore a verbless caption is learning something other than grammar.

A sentence is kept if it has a finite verb (`VerbForm=Fin` in the morphology -- `es_core_news_sm` gives Spanish's `tag_` the same universal value as `pos_`, with no TIGER-style tag that names finiteness the way German's `VVFIN`/`VAFIN`/`VMFIN` do, so this reads the feature German's version reads off the tag directly) and is not a stranded subordinate clause -- an opening subordinating conjunction that governs the root. That second check is the same one German's version makes, unchanged: `SCONJ` is a universal tag.

This filter is **not** applied to the Phase 1 overcorrection set, for the same reason German's is not: that set is frozen once built, and changing its membership after the fact would move numbers already reported.

## Held out

The 400 sentences of the Spanish overcorrection set (`overcorrection_es`) were removed here rather than remembered later. They come from these same corpora, and training on them would make the overcorrection measurement a memory test rather than a test of restraint. Matched on a whitespace- and case-insensitive key, because the frozen set stores its sentences tokenised and the Leipzig files do not.

## What the filters do not ask

No filter here asks LanguageTool whether a sentence is correct, for the same reason German's does not: doing so would define correct Spanish, in the training data, as Spanish LanguageTool has no opinion about -- the standard Phase 1's baseline is measured against. Spelling is checked against the Spanish Hunspell dictionary instead, lower-case words only, so a capitalised unknown (usually a name) does not bias the corpus towards sentences with no proper nouns in them.
