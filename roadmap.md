# Multilingual GEC + Compression Pipeline — Roadmap

A small, compressed language model that finds and explains grammar/spelling errors,
starting with German, extending to Spanish, running locally on a 16GB M1.

**Base model candidate:** EuroLLM-1.7B (confirm in Phase 0)
**Target footprint:** ~600MB (vocab-trimmed + 4-bit)
**Compute:** Stick to M1 16gb for now, we can reevaluate if we need a H200 later (which i have for thesis, therefore don't want to use much for personal use)

---

## Phase 0 — Setup and scoping
*~3–5 days*

- [x] Repo skeleton: `corruptors/`, `rules/`, `eval/`, `compress/`, `data/`
- [x] Download Falko-MERLIN, write the M² format loader
- [x] **Freeze the test split on day one.** Do not look at it again until Phase 3
      — enforced in code: `load_split(..., "test")` raises without `allow_test=True`
- [x] Self-hosted LanguageTool in Docker
  - [x] n-grams downloaded and mounted, `langtool_languageModel=/ngrams` (3.1GB extracted)
  - [x] `Java_Xmx=4g` or higher
  - [x] Sanity check: `make lt-check`. Note the folklore is wrong — German das/dass is
        caught by a *hand-written* rule and fires with no n-grams at all. The probes were
        picked by diffing against a control server without n-grams; the real n-gram
        signal is `CONFUSION_RULE_*` (fiel/viel, ihm/im, mit/mir, Leere/Lehre).
- [x] Tokenizer fertility analysis: EuroLLM-1.7B vs Qwen3-1.7B vs Gemma 3 1B
      (added Qwen3.5-0.8B and Qwen3.5-2B; Gemma 3 1B skipped — gated on HF)
  - [x] Tokens/word on German corpus
  - [x] Fraction of vocab ever touched by German text → pre-computes vocab trim ratio
        (reported as a coverage *curve* at 99 / 99.9 / 99.99 / 100% of token occurrences)

**Exit criterion:** ✅ Falko-MERLIN loads, LT runs locally, base model chosen with a number to justify it.

> **Outcome: EuroLLM-1.7B.** Best German fertility (1.693 tokens/word) *and* the most
> trimmable vocabulary (21.4% of the model at 99.9% coverage), helped by untied embeddings
> that make every removed row pay twice. Qwen3-1.7B loses on both axes and is really 2.03B
> parameters. Full writeup: `reports/phase0/base_model_decision.md`.
>
> Two numbers to carry forward. **22% of Falko-MERLIN sentences need no correction**, which
> sets the Phase 2 negative-example ratio empirically rather than by guess. And **vocab trim
> + 4-bit lands at ~782MB**, not 600MB — Phase 4's depth prune has to close that gap, or the
> target moves. Recorded now so it stays a prediction rather than a retrofit.

---

## Phase 1 — Eval harness and baselines
*~1 week — do not skip this phase*

- [x] M²-scorer (F0.5) implemented
- [x] German ERRANT running, per-error-type breakdowns
      — agrees with the corpus annotation at 0.949 span precision / 0.932 recall
      (`reports/phase1/errant_de_agreement.md`)
- [x] Overcorrection set built: 400 already-correct sentences from Leipzig news and Wikipedia
  - [x] Metric: % of sentences the system modifies
- [x] Baseline 1: LanguageTool alone
- [x] Baseline 2: base model zero-shot
- [x] Baseline 3: base model few-shot (8 examples, 22% of them already correct)

**Exit criterion:** ✅ three numbers in a table. `reports/phase1/baselines.md`.

> If LT's F0.5 is very high, that is not bad news. It means the contribution is the
> *explanations* and the *offline 600MB footprint*. Reframe the project accordingly and
> carry that framing through to the README.

> **Outcome: the prompted base model already beats LanguageTool.**
>
> | System | P | R | F0.5 | Overcorrection |
> |---|---:|---:|---:|---:|
> | LanguageTool | 0.5653 | 0.1992 | 0.4134 | 28.7% |
> | zero-shot | 0.4761 | 0.0905 | 0.2571 | 10.0% |
> | few-shot | 0.6600 | 0.2675 | **0.5102** | 15.0% |
>
> So the contingency above does not fire, and **Phase 3 has to beat 0.5102, not 0.4134**.
> Setting the bar at LanguageTool would have been setting it too low.
>
> Three things to carry forward. **LanguageTool modifies 28.7% of already-correct
> published prose** — `SU Annen` → `USA Innen`, `Freilebende` → `Frei lebende` — which is
> the strongest argument in the project so far for a model that knows when to keep quiet.
> **Detection is not the differentiator: correction is.** LT and few-shot find nearly the
> same spans (detection F0.5 0.5869 vs 0.5835); few-shot wins because its repairs are
> right more often (gap +0.1120 vs LT's +0.2020). And **zero-shot's 10% overcorrection
> rate is not a virtue** — it touches fewer correct sentences but wrecks the ones it
> touches, 2.6 spurious edits per modified sentence against LT's 1.2, because it
> paraphrases and truncates instead of correcting (`Dass eine falsche Eingabe eine gültige
> IBAN ergibt , ist daher unwahrscheinlich .` → `Die IBAN ist falsch .`).
>
> The identity row scores 0 spurious edits, so none of the above is the tokeniser
> manufacturing false positives, and the metric's ceiling on this split is 0.9976.

---

## Phase 2 — Data pipeline
*~2 weeks — this is the real work*

- [x] Clean German source text: Leipzig news + Wikipedia, quality-filtered
      — 60,000 sentences frozen; the 400-sentence overcorrection set is excluded by
      construction (`reports/phase2/clean_corpus.md`)
- [x] Error injection module (`corruptors/de.py`) — 13 rules, 4 of them parse-based
  - [x] Noun capitalisation — common nouns only; see the outcome note
  - [x] Comma rules — split three ways by which comma rule is broken
  - [x] ß / ss, umlaut stripping
  - [x] Article gender (der/die/das)
  - [x] Adjective declension endings
  - [x] Case errors after Wechselpräpositionen
  - [x] Verb-final word order in subordinate clauses
  - [x] Separable prefix placement
  - [x] das / dass
  - [x] Keyboard-adjacency typos
- [x] **Weight injection distribution to match the Falko-MERLIN error histogram from Phase 1**
      — blended with Phase 1 recall; `reports/phase2/injection_weights.md`
- [x] Run LT over corrupted text → rule IDs + messages (`reports/phase2/lt_rule_survey.md`)
- [x] Explanation templating (`rules/de.yaml`)
  - [x] Templates written — 13 keyed on the corruptor, 9 on a LanguageTool rule id
  - [x] ~~LLM-rewrite the long tail~~ — there is no tail; see the outcome note
- [x] Negative examples: **22%** correct sentences with empty edit lists — matches the measured
      Falko-MERLIN rate (`reports/phase0/error_type_histogram.md`) ← main defence against overcorrection
- [x] Target ~30k examples — 30,000 built, 35,062 edits

**Exit criterion:** ✅ inspected, three defects found and fixed, rebuilt.
`reports/phase2/training_set.md`.

> **Outcome: 30,000 examples, and the inspection earned its place in the plan.**
>
> The exit criterion is written as a formality — read a hundred samples, fix the pipeline if
> more than five are wrong. It caught three real defects that every automated invariant
> passed straight over, because all three produced a *correct* correction with a *wrong*
> explanation, and the round-trip check cannot see the difference.
>
> **Recognising a word class by its shape does not work.** `adjective_form` identified
> adjectives by their ending and fired on articles, prepositions, quantifiers and once on
> the modal verb `können` — 22.6% of the time. The lesson is narrower than "use the parser":
> *listing* a closed class works fine, which is why the article and preposition rules are
> sound on tokens alone; *inferring* class membership from a suffix does not, because German
> shares those endings across half the closed-class vocabulary.
>
> **`noun_case` was lowercasing proper nouns** one time in five. The correction was right and
> the example was worthless: English capitalises names too, so no English speaker writes
> `josef`. It now injects common nouns only.
>
> **`drop_comma` explained every comma as separating a subordinate clause.** German commas
> also go before `sondern`, and around appositions — and the generic case turned out to be
> the *largest* of the three at 2,844 of 4,614. Now split by what follows the comma.
>
> Two findings for later phases. **LanguageTool detects only 63% of what we plant**, and
> under 10% of preposition errors — so Phase 1's blind spots are its blind spots too, which
> is the strongest evidence yet for what this project is for. And **the roadmap's "top ~100
> rule ids" does not exist**: 50 rule ids fired in total and two of them account for 93% of
> matches, with the dominant one saying only "Möglicher Tippfehler gefunden". The
> explanations are keyed on our own corruptors instead, which know what they did.
>
> ~~One debt: six of the fifteen `reference:` fields in `rules/de.yaml` are `null` because the
> Rat für deutsche Rechtschreibung paragraph has not been checked.~~ **Settled, and the debt
> was pointed the wrong way.** The nulls were harmless -- 17 of the 25 rules are grammar, which
> the Regelwerk does not legislate at all. Three of the five *filled-in* numbers were wrong:
> the 2024 edition renumbered part E, so the subordinate-clause comma moved § 74 → § 73 and the
> und/oder rule § 72 → § 71, while `comma_after_fronted` cited a paragraph about something else
> and has no citation to make. Now 8 cited and verified against the official PDF by
> `make check-citations`, 17 carrying a note saying why no citation exists, and the edition
> pinned in the file and rendered into every citation. It cost the model nothing -- explanation
> prose was never in the SFT target -- so no retraining is implied.
> `reports/phase2/citations.md`.

---

## Phase 3 — German SFT
*~1 week*

- [x] LoRA config: r=32, alpha=64, all linear layers, lr 1e-4, cosine, 2–3 epochs
- [x] **Verify loss masking on the prompt** before launching anything long
      — printed by every run, and pinned by `tests/test_train_format.py`
- [x] Structured JSON output schema (span-based edits)
- [x] ~~Constrained decoding at inference~~ — **not needed.** The model emitted
      1 invalid JSON in 2,903 answers and 0 overlapping spans. The schema was
      the easy half.
- [ ] Evaluate → read per-error-type breakdown → fix data in Phase 2 → retrain
  - [x] Iteration 1 — synthetic-only data. **F0.5 0.1891**, worse than the
        untrained baseline on 14 of 15 error types. See the outcome note.
  - [x] Iteration 2 — real learner errors as the primary signal. **F0.5 0.6944**,
        which clears the few-shot bar of 0.5102 and the exit criterion with it.
  - [x] Iteration 3 — corruptors widened from 8 error types to 17. **F0.5 0.6960**,
        a delta of +0.0016 whose 95% interval straddles zero. The coverage
        argument was wrong and `reports/phase3/ablations.md` §5 says why.
  - [x] Iteration 4 — the `changes` list down-weighted in the loss, since it is
        70% of the supervised tokens and F0.5 never reads it. **F0.5 0.7019** at
        weight 0.5 and **0.7072** at 0.25, with overcorrection down to 9.8%.
  - [x] Iteration 5 — back-translated errors: a generator trained on Falko read
        backwards, run over clean German, to reach the types no rule can write.
        Predicted 0.72–0.74 in `reports/phase3/headroom.md`; **got 0.7056**,
        below the 0.7072 bar. Recall rose (0.6174 → 0.6287) and precision paid
        for it. The second wrong prediction in a row, written up as wrong.
  - [x] `changes_weight` below 0.25 — **the trend turned over.** 1.0 → 0.5 →
        0.25 → 0.125 gives 0.6944 → 0.7019 → 0.7072 → 0.7068. 0.25 is the
        optimum and this line is closed.
  - [x] N-best rerank with a minimum-edit prior — **+0.0007 over beam top-1**,
        retired along with MBR and a learned reranker. What survives is beam
        search itself at +0.0084 for 5x generation, which is a serving cost
        decision rather than an open question. `reports/phase3/nbest.md`.
  - [x] Iteration 6 — DPO on gold-labelled pairs from the model's own beam, the
        one channel `nbest.md` left open. **Negative on both checkpoints**:
        0.7072 → 0.6895 and 0.7056 → 0.6905, and tightening the KL anchor walks
        F0.5 monotonically back toward the untrained reference without ever
        passing it. `reports/phase3/dpo.md`.
  - [ ] Still unspent: cLang-8 German (machine-generated targets, so distillation
        rather than gold — declined on the same grounds it always was).

> **Iteration 1: the synthetic corpus made the model worse.** F0.5 0.1891 against
> few-shot's 0.5102 and zero-shot's 0.2571 — precision 0.25, so three edits in four
> were wrong. Recall fell on 14 of 15 error types; `M:DET` (+0.04) was the only gain.
>
> **Injection share did not predict improvement.** `R:SPELL` took the largest share of
> the corpus at 25.1% and lost the most recall, 0.49 → 0.12. `R:WO` had 8.0% and went
> 0.09 → 0.00. The Phase 2 blend — over-sample the types the baseline is blind to —
> was measured and it does not work.
>
> **The cause was the shape of the data, not its error mix.** Our inputs were pristine
> news German carrying one or two surgical errors. Falko sentences average 2.55 edits
> and 39% carry three or more; the synthetic corpus averaged 1.17 and contained *none*
> with three, because `errors_per_sentence` was capped at 2. The model learned to find
> a mechanical corruption sitting in otherwise-perfect prose, and overwrote the German
> competence the base model already had — which is where few-shot's 0.49 on spelling
> came from.
>
> **Falko-MERLIN train had been sitting unused the whole time**: 19,237 real annotated
> sentences, 54 error types, real density, real negatives. It had only ever been read
> to draw few-shot prompt examples.

*Expect 2–4 loops. Quality comes from data iteration, not hyperparameter search.*

**Exit criterion:** beat the LT baseline on F0.5, **or** match it with substantially better
explanations. Overcorrection rate under control.

- [x] **Held-out test split read**, once, for five models. `reports/phase3/test.md`.
      Six iterations of dev tuning did not overfit it: every model lands within 0.005
      of its dev score, mean absolute gap 0.0024. **Phase 4 compresses `iter4-cw025`
      — F0.5 0.7060, overcorrection 9.5% on test.** The three SFT models are one
      model with three names (they span 0.0023 and the ranking *inverts* between
      splits), so the choice was made on overcorrection, where a 4.5-point gap
      replicates across both.
- [x] Re-run the Phase 1 baselines on test before any README claim.
      `run_baselines.py --split test` now exists, guarded the same way
      `eval_finetuned.py` guards it and writing to `baselines-test.md` so a final
      number can never land on top of the development record. **The README's
      "+0.29" was flattered at both ends.** LanguageTool scores *better* on test
      than on dev (0.4321 against 0.4134) and few-shot scores *worse* (0.4948
      against 0.5102), so the honest margin is **+0.2739**, not +0.29. The
      conclusion is unchanged; the number can now be quoted on its own.
      `reports/phase1/baselines-test.md`.

      | System | dev | test |
      |---|---:|---:|
      | identity | 0.0008 | 0.0033 |
      | LanguageTool | 0.4134 | 0.4321 |
      | zero-shot | 0.2571 | 0.2501 |
      | few-shot | 0.5102 | 0.4948 |
      | iter4-cw025 | 0.7072 | 0.7060 |

> **Met at iteration 2 and held since.** F0.5 0.7072 against LanguageTool's 0.4134 and
> few-shot's 0.5102, overcorrection 9.8% against LanguageTool's 28.7%. The contingency
> below did not fire.
>
> What the four iterations cost is worth recording next to what they bought. Iterations 3
> and 4 moved F0.5 by +0.0016 and +0.0112 respectively, and the larger of those came from
> the loss function rather than from the data — after a full data iteration aimed at
> coverage returned nothing. `reports/phase3/ablations.md` has the four serving-side
> changes that returned nothing either.
>
> The error budget is in `reports/phase3/headroom.md`. Short version: detection is
> finished — 3.4% of gold edits sit in a sentence the model never enters, and 1,118 of
> 1,970 sentences needing correction come back byte-identical to the annotator's. What is
> left is German competence inside spans the model has already found.
>
> **Four more things were tried after that note and none of them moved it.** Iteration 5's
> back-translation, `changes_weight` 0.125, the n-best rerank, and DPO on the model's own
> beam. The last one matters most: `nbest.md` had shown no *selector* reaches the 0.8380
> oracle, and `dpo.md` shows no *optimiser* reaches it either, on pairs drawn from that
> same gap with gold labels. The oracle gap is a measurement of the model's uncertainty
> and is now retired as a source of headroom. Past this point the base model is the
> binding constraint, which is the constraint Phase 4 exists not to relax.

> ⚠️ If after three data iterations you still can't beat LT, stop grinding. Pivot the framing to
> "small local model matching a mature rule-based system, with explanations, at 600MB offline."
> That is still a good project. Owning it beats pretending on a CV.

---

## Phase 4 — Compression
*~1.5 weeks*

**Check first, before spending GPU hours:**

- [x] Trim a base model's vocab *without* training, confirm `llama.cpp` still converts it
      — passes. GGUF f16 and Q4_K_M both convert and run. **`mlx-lm` is untested: it is
      Apple-Silicon only and the trim ran on the Linux/CUDA box, so this half has to be
      run on the M1** before the gate is fully passed.
- [x] Standard HF format, `vocab_size` correctly updated in config — all four artefacts
      agree at 39,399 (config, `embed_tokens`, `lm_head`, and *both* tokenizer files).

> **Gate outcome: `reports/phase4/gate.md`.** 128,000 → 39,399 rows, **21.9% of parameters
> and 726 MB at bf16**, against Phase 0's predicted 21.4% / 708 MB. At Q4_K_M the trimmed
> model is **755 MB against the 600 MB target**, so the depth prune has to find another
> **155 MB** or the target moves.
>
> **The gate caught a real defect.** Counting coverage on prose alone deleted the JSON
> structural tokens — `▁{`, `▁[`, `}]`, `:"` — because they barely occur in news and
> Wikipedia. The checkpoint converted cleanly, passed every structural check, corrected
> German correctly, and returned it as *bare text with no schema*. The trim now counts the
> task corpus too; `--no-task-data` reproduces the failure. **A vocabulary trim has to be
> counted over everything the deployed model reads or writes** — which Phase 5 must repeat
> for Spanish, where the same mistake would again look like success.

**Then:**

- [x] Vocab trim on the base model — 128,000 → 39,399 rows
  - [x] Tokenize German+English corpus, frequency threshold
  - [x] Keep all specials and byte-fallbacks
  - [x] Slice embedding + LM head rows
  - [x] Remap token IDs
  - [x] Rebuild tokenizer merge table consistently with trimmed vocab (260,662 → 89,212)
- [x] ~~Heal: short LM-objective finetune on German text~~ — **dropped.** There was no
      damage to repair: the trim is free. Healing *cost* 0.0070 of F0.5, buying 2.3pp of
      overcorrection by making the model more conservative.
- [x] Task SFT on trimmed model, same Phase 3 recipe — run as the depth-sweep control
- [x] Depth prune (second experiment)
  - [x] Layer-wise input/output cosine similarity — edges do the work (L0 0.309, L23 0.500);
        the 22 interior layers are near-uniform at 0.924 ± 0.035
  - [x] Sweep 2 / 4 / 6 dropped layers → curve, not a point
  - [x] ~~Heal each with LoRA~~ — a full re-SFT each, because the adapter does **not**
        survive layer removal at all (see the outcome note)

**Ablation table:**

| Variant | F0.5 | Overcorrection % | Size | tok/s (M1) | Peak RSS |
|---|---:|---:|---:|---:|---:|
| Uncompressed | 0.7072 | 9.8% | 997 MB | not measured | not measured |
| **Vocab-trimmed** | **0.7061** | **9.5%** | **755 MB** | **62.3 ± 0.7** | **477 MB** |
| Vocab-trimmed + depth-pruned (−6) | 0.6591 | 16.8% | **593 MB** | 79.2 ± 3.6 | 379 MB |

`tok/s (M1)` is `tg128`, `llama-bench -p 512 -n 128 -r 5`, llama.cpp b29c606e2 (10964),
Q4_K_M on Metal, M1 / 16 GB. Prompt processing is `pp512` 651.5 ± 8.2 vs 865.6 ± 20.8.
At sentence shape (`-p 64 -n 32`) the gap widens slightly: 62.3 / 573.8 vs 83.0 / 801.6,
so a 64-token prompt and a 32-token correction take **625 ms** against **465 ms**. Peak RSS
is `/usr/bin/time -l` on a single `llama-cli` generation; weights are mmapped and unified,
so it undercounts the file on disk. The uncompressed row was never converted to GGUF.

The depth prune is worth a genuine **+27% tok/s and −21% resident memory** — and it buys
160 ms on an operation already far under a second, well past typing speed. Speed is not
the axis this decision turns on; overcorrection is.

**Exit criterion:** ✅ two claims, `reports/phase4/compression.md`.

> **21.9% fewer parameters and 24.2% smaller (997 → 755 MB) for 0.0011 of F0.5** — free
> within measurement error, with **no healing and no retraining**, because the LoRA targets
> no vocabulary-dimension module and so stays valid on a trimmed base.
>
> **39.0% fewer parameters and 40.5% smaller (997 → 593 MB) for 0.046 of F0.5** — the
> 600 MB offline target, met. Phase 0 predicted the trim would land near 782 MB and that
> the depth prune would have to close the gap; it landed at 755 MB and the prune closed it.
>
> **Depth pruning does not survive without re-training, and the vocabulary trim did.**
> Reusing the Phase 3 adapter on a depth-pruned base produces a model that emits no schema
> at all — free prose at 2 layers dropped, degenerate repetition at 6. It survives losing
> 69% of the vocabulary but not 8% of the depth.
>
> **The control earned its keep.** Read naively the first prune point looks like it nearly
> doubles overcorrection (9.5% → 16.5%); the zero-layer control on the same corpus is
> already at 14.5%. Five of those six points belong to iteration 5's back-translation
> corpus, not to pruning — which is a **second, independent verdict on iteration 5**:
> F0.5-neutral again, but costing 5.0pp of overcorrection. Phase 3 called it null; it was
> negative, and `reports/phase3/dpo.md` understated it.
>
> **Recommendation: ship the vocabulary-trimmed model at 755 MB** unless 600 MB is hard.
> It is the best row on both metrics that matter and costs nothing to produce. Caveat: the
> depth rows may be pessimistic by ~5pp of overcorrection, since corpus A no longer exists
> to re-train them on.

---

## Phase 5 — Spanish port
*~1 week — the test of whether the architecture is real*

**Corpus downloaded and sentence-aligned** (`scripts/download_cowsl2h.py`, pinned to
commit `6b18b773`, spacy `es_core_news_sm` — not the regex splitter the initial survey
used, so these numbers are the real ones):

| | German (Falko-MERLIN) | Spanish (COWS-L2H) |
|---|---:|---:|
| aligned sentence pairs | ~24,000 | **46,473** (train 37,473 / dev 4,716 / test 4,284) |
| carrying real edits | — | 31,151 (67.0%) |
| already correct, the noop signal | — | 15,322 |
| learners | — | 883 (716 / 85 / 82) |

5,382 essays, 2,881 with a `corrected1` (53.5%). Split by learner id, salted-hash
bucketed 80/10/10 — essay-level splitting would leak a student's style across train
and test, since the corpus is longitudinal. Of the 2,881 corrected essays, 634 (22.0%)
were dropped: their essay and correction split into different numbers of sentences,
most often a corrector merging or splitting a run-on, and there is no honest automatic
fix for that — force-aligning would score against a guess.

**Now typed and frozen.** `errant_es` turned the aligned pairs into M2
(`scripts/build_cowsl2h_m2.py`), and `scripts/freeze_splits_es.py` recorded their
SHA-256 in the manifest already committed for German: 37,473 / 4,716 / 4,284 sentences,
64,909 / 7,563 / 7,333 edits. The build order above reads 1-2-3 but the real dependency
runs 1-3-2: freezing needs M2, and M2 needs a typing annotator, so `errant_es` had to
exist before the freeze could — noting the resequencing rather than quietly doing it.

> **The upstream corrections carry their own noise**, independent of anything this
> pipeline does: `corrected1` on one `vacation.S17` essay reads "...el año pasadosolo
> yo..." — a corrector's edit fused two words with no space between them, verified in
> the raw CSV before writing any code to explain it away. Nothing here auto-repairs it;
> a wrong fix would be worse than a visible one. `errant_es` needs a Spanish Hunspell
> dictionary for the same reason `errant_de/spelling.py` needs `de_DE_frami` — telling a
> misspelling from a wrong word choice — and that same dictionary is what will catch
> artifacts like this one instead of typing them as a real edit.

- [x] `corruptors/es.py`: 13 rules — `drop_accent`, `keyboard`, `drop_inverted_punctuation`,
      `article_form`, `drop_article`, `adjective_form`, `preposition` (por/para headline),
      `drop_preposition`, `redundant_subject_pronoun`, `drop_object_clitic`, `ser_estar`,
      `subjunctive_for_indicative`, `verb_infinitive`. Every rule's claimed type verified
      against real `errant_es` output, not assumed — two guesses were wrong on the first
      pass (`ser_estar`/`subjunctive_for_indicative` claimed `R:VERB(:FORM)`, `errant_es`
      calls the copula `AUX` and gives `R:AUX(:FORM)`) and caught immediately by testing
      rather than shipping. A stress test over 800 real, already-correct COWS-L2H
      sentences then found and fixed five more defects — `article_form`/`drop_article`
      swapping an object clitic pronoun ("la" is homographic with the article), `keyboard`
      occasionally landing on a real word by coincidence, `adjective_form` inflecting a
      gender-invariant adjective (`indígena` has no `indígeno`), and `ser_estar` firing on
      identity statements where Spanish grammar forbids `estar` outright. `tests/test_corruptors_es.py`
      (32 cases) regression-tests every one of the five.
  - **Two roadmap-named phenomena are declined, honestly, not silently dropped.**
    Preterite vs imperfect: `es_core_news_sm` tags irregular preterites unreliably enough
    to be unsafe (`estuvimos` → `Tense=Pres`, `hube` → lemma `hubir`, `estuviste` not
    tagged as a verb at all) — a rule built on those tags would mistype its own training
    data the way the German file's own inspection warned against. Leísmo/laísmo: it is
    standard usage across large parts of the Spanish-speaking world, and COWS-L2H's own
    annotators evidently agree — `R:PRON:FORM` is 1.17% of edits, not the mass a
    systematically "corrected" dialect feature would produce — so a rule here would be
    teaching the model to fight a dialect, not an error.
  - **Two shipped rules carry a measured, non-zero mistype rate**, found and quantified by
    the same stress test rather than hidden: `verb_infinitive` (22.3% of 121 fires) and
    `adjective_form` (33.3% of 48 fires) sometimes have `es_core_news_sm` re-lemmatise the
    damaged form to a different lemma than the original, so `errant_es` calls the edit
    `R:OTHER`/`R:MORPH`/`R:VERB` rather than the `:FORM` type the rule claims. The
    **correction itself is never wrong** in either case — only the type label the
    `changes` list and the per-type histogram see. `adjective_form` is the rule most
    worth revisiting if the project ever moves past the small spacy model.
  - **Found and fixed a real, separate bug on the way**: `explanations.explain_correction`
    hard-coded `errant_de` regardless of its own `language` parameter, so
    `explain_correction(..., language="es")` silently annotated Spanish text with the
    German annotator and German dictionary. Not something this session introduced — a
    latent gap with only one language to expose it — but it would have made every Spanish
    explanation wrong. Fixed with a small `_annotator(language)` dispatch; German's path
    reverified unchanged.
  - **`corruptor()`/`parse_corruptor()`/`propose()` in `corruptors/base.py` gained an
    optional `registry`/`needs_parse_set` parameter**, defaulting to German's own tables
    so every existing call site is unchanged. Needed because `de.py` and `es.py` reuse
    seven rule names (`article_form`, `drop_article`, `preposition`, `drop_preposition`,
    `adjective_form`, `keyboard`, `verb_infinitive`) for rules that do different things in
    each language — the previous single global registry would have raised at import time
    on the first shared name.
- [x] `rules/es.yaml` + `rules/es_types.yaml` — explanation templates for every corruptor
      and every real-data ERRANT type down to ~1% of edits, calibrated against the actual
      Spanish histogram rather than copied from German's. Two entries read differently on
      purpose: `U:PRON` is German's *smallest* hand-written entry (1.36%, repetition) and
      Spanish's *largest* correctable one (8.15%, the pro-drop pattern with no German
      analogue); `R:ORTH` is stress marks here, not capitalisation. No `languagetool:`
      refinement section and no citation apparatus yet — an LT Spanish rule-id survey and
      whatever the RAE equivalent of a checkable numbered ruleset would be are real,
      separate work, left undone rather than faked with an unverified citation.
- [x] `errant_es` — **the item Phase 5 was missing.** No public Spanish ERRANT existed
      to install, so `errant_de` was split into `errant_core` (alignment plumbing +
      merger, moved unchanged — verified against `tests/test_errant_de.py`, still 14/14)
      and a per-language profile (classifier + spelling). `errant_es.classifier` swaps in
      one thing: German's default "same word, cosmetically different" is case-insensitive
      equality; Spanish's is stress-mark-insensitive equality (`esta`/`está`), and it is
      deliberately *not* Unicode decomposition, which would strip the tilde off `ñ` and
      call `año`/`ano` the same word. `errant_es.spelling` uses the LibreOffice `es_ES`
      Hunspell dictionary, same as `de_DE_frami` for German. Verified against real
      COWS-L2H sentences and a 14-case test suite (`tests/test_errant_es.py`), covering
      the case ERRANT's decomposition would get wrong.
- [x] ~~Eval data: COWS-L2H / CEDEL2~~ → **COWS-L2H only.** CEDEL2 is free, large and
      current (6,559 participants, 1.5M words, v3 October 2025) but ships FreeLing POS
      tags and **no corrected parallel text**, so it can neither train nor score GEC.
      It was on this list in error.
- [x] Sentence-align essay to correction — a step German never needed, because
      Falko-MERLIN arrived as sentence-level M2 with noop markers and COWS-L2H is
      document-level CSV. `src/langlm/data/cowsl2h.py`, spacy-based, 46,473 pairs.
- [x] Freeze `es` train/dev/test splits by learner, not by essay: the corpus is
      longitudinal and the same student writes on several prompts. `scripts/freeze_splits_es.py`,
      manifest verified. Error-type histogram in `reports/phase5/error_type_histogram_es.md`
      (51 types, 72,472 train+dev edits) — the Spanish counterpart of the table Phase 2
      used to weight German synthetic error injection, now ready for `corruptors/es.py`.
- [x] Run the full data + training pipeline end to end, up to the point training can
      start — **and the claim did not hold.** Two things needed real code changes, not just
      new files: `train/format.py`'s prompt instruction was hardcoded `"Correct the German
      sentence"` (would have trained Spanish under a lying prompt), and `train/data.py`'s
      training-file path was hardcoded to `de.jsonl`. Both fixed additively — `language`
      parameters defaulting to `"de"`, every existing German call site unchanged, full suite
      still green. `train_sft.py --language es` now exists and its dry-run was actually run:
      correct prompt, sane loss mask, real forward pass.
  - Built for real, not stubbed: `quality.py` parameterized by language (Spanish `ALLOWED`
    regex read off a real character survey of `spa_news_2023_100K`, not guessed);
    `clean_es.py` (60,000-sentence clean corpus, `spa_news_2023_100K` + `spa_wikipedia_2021_100K`,
    the second downloaded this session); `overcorrection_es.py` + 400 frozen sentences;
    `learner_es.py` (COWS-L2H → training records); `injection.py` gained a `registry`
    parameter so it can draw from `corruptors.es` instead of German's; `configs/phase2_es.yaml`,
    `phase3_es.yaml`, `phase1_es.yaml`. `build_training_set_es.py` produced **97,473 examples**
    (37,473 real COWS-L2H + 60,000 synthetic, 51 error types, 0 broken, 0 unexplained) —
    verified by reading the 100-sample exit-criterion output, which caught one real defect
    (`M:PUNCT`'s explanation overclaimed "likely ¿ or ¡" when the real edit was a comma) and
    fixed it before the numbers above.
  - **One scoping call, made explicitly**: no back-translation. German's own iteration 5
    measured negative against its first successful run's bar; there is no equivalent bar for
    Spanish yet to measure against, so building the generator now would be speculative. The
    negative-example rate is COWS-L2H's own measured 33.0%, not Falko-MERLIN's 22% carried
    over.
- [x] Spanish baselines. `run_baselines.py` generalized the same way the training pipeline was —
      a `LanguageSettings` dispatch table (config, corpus, annotator, measured negative-example
      share, report paths) rather than per-language forks, German's own path re-verified
      unchanged. `overcorrection.measure` and `PromptedBaseline` gained the same `language`
      parameter, defaulting to German. `configs/phase1_es.yaml` points LanguageTool at generic
      `es` (matching the `ngrams-es` archive, not a country variant). The Spanish n-grams
      (1.7GB) were downloaded and verified live against the running server — a real confusion
      pair from LanguageTool's own Spanish rule data (`acido`/`ácido`) was caught, not assumed
      from the archive existing on disk.

  | System | P | R | F0.5 | Overcorrection |
  |---|---:|---:|---:|---:|
  | identity | 0.0000 | 0.0000 | 0.0000 | 0.0% |
  | LanguageTool (`es`) | 0.4232 | 0.1588 | **0.3175** | 27.3% |

  Confirms the prediction made from the rule-file size alone before any of this ran (`rules/es.yaml`
  had no LT survey yet, but German's `grammar.xml` was already known to be 4.9 MB against
  Spanish's 2.1 MB): the Spanish LanguageTool baseline is real but weaker than German's
  (F0.5 0.3175 vs 0.4134), on a similar overcorrection rate (27.3% vs 28.7%). Zero-shot and
  few-shot EuroLLM — the bar that actually matters, per Phase 1's own finding that the prompted
  base model, not LanguageTool, is the honest target — were running at the time of this entry;
  see `reports/phase5/baselines_es.md` for whichever numbers are current.
- [x] Spanish baselines on test as well as dev, the Phase 5 counterpart of the
      Phase 3 item above. The held-out Spanish table had one row and no bar until
      now. **Margin over the strongest baseline: +0.2982** (few-shot, 0.3371).
      The two baselines swap places between splits — LanguageTool leads few-shot
      by 0.0102 on dev and trails by 0.0023 on test — which is the same tie the
      dev entry already called a tie, seen a second time.
      `reports/phase5/baselines_es-test.md`.
- [x] Spanish SFT. Two recipes trained and scored against the full baseline table on
      dev; `cw025` carried forward, matching German's own choice of `changes_weight`.
      **F0.5 0.5987 dev, 0.6353 test, overcorrection 6.0%.** The Phase 5 list never
      had a checkbox for training the model it exists to produce — noting that the
      plan was short an item rather than back-dating one.
- [x] **Compress the Spanish model** — the Phase 4 recipe, re-run end to end.
      `reports/phase5/compression_es.md`.
  - [x] Vocabulary trim counted over Spanish + English prose *and* the Spanish task
        corpus — `trim_vocab.py --language es` exists because the task corpus is
        what carries the schema, and Phase 4's gate said in writing that Phase 5
        had to repeat that check. It was repeated and it passes: 2 unreadable
        answers in 5,116, the untrimmed model's own rate.
  - [x] `merge_adapter.py` — the merge step that produced German's `ship-*`
        checkpoints was never scripted; now it is.
  - [x] Q4_K_M, measured not projected. Both trimmed models re-quantised with one
        llama.cpp build so the sizes are comparable, and German's reproduces
        Phase 4's published 755 MB. Spanish's uncompressed 996.7 MB is also the
        first *measured* value for that row in either language — Phase 4 never
        converted its uncompressed model to GGUF — and it corroborates the 997 MB
        that has been carried since.
- [ ] Cross-lingual pruning experiment: evaluate German-pruned model on Spanish and vice versa
      → how much does language-specific pruning cost cross-lingually?
      **Half-answered.** The tokenizer half is measured in `compression_es.md`: the two
      kept sets share 29,061 rows, a bilingual model's floor is their 50,224-row union
      (27.5% above either), and each language's text costs **+40–56% more tokens** under
      the other language's trim while its own trim costs it +0.5–1.5%. What is still
      open is the *weights* half — scoring one language's compressed model on the other
      language's task.

> **Outcome: the trim ports, and one part of Phase 4's claim does not.**
>
> | | German | Spanish |
> |---|---:|---:|
> | vocabulary kept | 39,399 | **39,886** |
> | parameters removed | 21.9% | **21.8%** |
> | Q4_K_M | 755.4 MB | **756.7 MB** |
> | ΔF0.5 from the trim | −0.0012 | **−0.0038** |
> | 95% CI | [−0.0043, +0.0019] | **[−0.0068, −0.0009]** |
> | p | 0.458 | **0.012** |
>
> Nothing forced those first three rows within 1.3% of each other: the Spanish kept set
> was counted independently, from Spanish corpora and a Spanish task file. The saving is
> dominated by how much of a 128,000-row multilingual vocabulary *any* one European
> language leaves untouched.
>
> **But "the trim is free" was a German finding.** German's loss is a coin flip; Spanish's
> is small and real — under half a percent of its own score, and smaller than the ±0.005
> dev/test spread, but its confidence interval sits entirely below zero. It is precision,
> not recall: overcorrection rises 6.0% → 7.5% while recall is flat. Phase 4's headline is
> restated here as *nearly* free rather than quietly re-used.
>
> A false alarm worth not re-raising: the quantised Spanish model returns bare text under
> `llama-cli`, which looks exactly like the gate's schema failure. The German shipped model
> does the same — `llama-cli` injects a chat template these base-format checkpoints never
> saw. The harness was wrong, not the checkpoint.

**Exit criterion:** if adding Spanish took more than a week, the abstraction was wrong.
Say so honestly in the README rather than hiding it.

> **The abstraction is already one item short and it is worth saying so now.** `errant_es`
> is not a corruptor file; it is 500 lines of linguistics, and Phase 5 listed it nowhere.
> What survives the audit is that **F0.5 never reads our error types** — `m2_scorer.py`
> does not mention them — so a second annotator changes the per-type breakdown and the
> explanations, and leaves 0.7072 and the whole Phase 4 table standing.
>
> **The baseline is weaker in Spanish.** LanguageTool's `es` grammar rules are 2.1 MB
> against German's 4.9 MB, and its style rules 49 KB against 471 KB — roughly 42% of the
> rule mass. Beating LanguageTool will mean less here than it did in German. Phase 1
> already found the honest bar is the few-shot base model; hold Spanish to that.

---

## Phase 6 — Package and ship
*~1 week*

- [x] Quantise to 4-bit — Q4_K_M, on the H200, for both compression variants
- [x] ~~Convert to MLX and~~ GGUF — **MLX dropped.** Quantisation ran on the H200 and
      GGUF already executes on Metal at 62 tok/s; a second Apple-only format would buy
      nothing this project needs to claim, and costs a conversion path to maintain.
- [x] Benchmark tokens/sec and peak memory on the M1 — 62.3 tok/s / 477 MB at 755 MB,
      79.2 tok/s / 379 MB at 593 MB. Numbers and method in the Phase 4 table.
- [ ] Demo app (local, or WASM/browser for the flashier artifact)
- [ ] README
  - [ ] **Lead with the results table**
  - [ ] Tokenizer fertility analysis
  - [ ] Baselines from Phase 1
  - [ ] Compression ablations from Phase 4
  - [ ] Honest limitations section

---

## The two things most likely to sink this

1. **Skipping Phase 1.** Without baselines you can't tell whether the model is good, and the
   project becomes unfalsifiable.
2. **Over-scoping languages.** Two languages fully evaluated beats a framework claiming twelve.

---

## Notes

- LanguageTool core is LGPL-2.1. Running it locally to generate training data is not
  distribution, so no license obligations attach. Check third-party dictionary licenses
  separately if this ever ships commercially.
- Avoid distilling explanation data from OpenAI/Gemini outputs if there is any chance of a
  commercial path — provider terms restrict training competing models. The LT + official
  ruleset route sidesteps this.
- Official German orthography rules (Rat für deutsche Rechtschreibung) are free and numbered,
  so explanations can cite an actual paragraph.
