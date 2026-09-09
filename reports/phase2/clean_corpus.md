# Phase 2: the clean corpus

Source: `deu_news_2024_100K`, `deu_wikipedia_2021_100K`, shuffled with seed 20260909.

## The funnel

| Stage | Sentences |
|---|---:|
| read from Leipzig | 200,000 (100.0%) |
| duplicate | 1 (0.0%) |
| failed the prose filter | 13,287 (6.6%) |
| outside the length window | 10,567 (5.3%) |
| unknown lower-case word | 8,199 (4.1%) |
| held out for overcorrection | 400 (0.2%) |
| pooled for parsing | 167,546 (83.8%) |

The pool is then shuffled and parsed until the corpus is full, so the stage below counts the sentences the parser actually saw rather than the whole pool.

| Stage | Sentences |
|---|---:|
| parsed | 62,454 (100.0%) |
| fragment: no finite verb, or a stranded subordinate clause | 2,454 (3.9%) |
| kept | 60,000 (96.1%) |

**60,000** sentences in the corpus.

## Why fragments are dropped

Leipzig is news and Wikipedia, so some of it is headlines and captions -- `Eisdisco am Kirlteich , 14 bis 22 Uhr .` -- and subordinate clauses printed as whole sentences. None of that is ungrammatical; German headlines are legitimate. It is poor material to damage on purpose: a word-order corruptor has no main clause to reorder, and a model asked to restore a verbless caption is learning something other than grammar.

A sentence is kept if it has a finite verb and is not a stranded subordinate clause -- an opening subordinating conjunction that governs the root. `Obwohl es regnete , ging er spazieren .` is kept, because there the conjunction governs a subordinate verb and the root belongs to the main clause. The finite-verb half inherits the parser's mistakes and drops some real sentences with them, which the corpus can afford.

This filter is **not** applied to the Phase 1 overcorrection set. That set is frozen and the published baselines are scored against it; changing its membership now would move numbers that have already been reported.

## Held out

The 400 sentences of the Phase 1 overcorrection set were removed here rather than remembered later. They are drawn from these same corpora, and they are the instrument that measures whether a trained model leaves correct German alone -- training on them would make that measurement a memory test. The match is made on a whitespace- and case-insensitive key, because the frozen set stores its sentences tokenised and the Leipzig files do not.

## What the filters do not ask

No filter here asks LanguageTool whether a sentence is correct. Doing so would define correct German, in the training data, as German LanguageTool has no opinion about -- the standard Phase 1 measured against and beat. Spelling is checked against a Hunspell dictionary instead, and only for lower-case words: a capitalised unknown is usually a proper noun, and rejecting those would bias the corpus towards sentences with no names in them.
