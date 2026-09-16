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
- [ ] Re-run the Phase 1 baselines on test before any README claim.
      `scripts/run_baselines.py` is hard-wired to dev, so "beats LanguageTool by
      0.29" currently compares a test row against a dev baseline.

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

| Variant | F0.5 | Overcorrection % | Size | tok/s (M1) |
|---|---:|---:|---:|---|
| Uncompressed | 0.7072 | 9.8% | 997 MB | not measured |
| **Vocab-trimmed** | **0.7061** | **9.5%** | **755 MB** | not measured |
| Vocab-trimmed + depth-pruned (−6) | 0.6591 | 16.8% | **593 MB** | not measured |

`tok/s (M1)` is unmeasured because the work ran on the Linux/CUDA box; it belongs with
the outstanding MLX conversion.

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

- [ ] `corruptors/es.py`: ser/estar, subjunctive triggers, gender/number agreement, accents and
      tildes, por/para, preterite vs imperfect, leísmo/laísmo
- [ ] `rules/es.yaml` from Spanish LT rule IDs
- [ ] Eval data: COWS-L2H / CEDEL2
- [ ] Run the full pipeline end to end — **nothing else in the repo should need changing**
- [ ] Cross-lingual pruning experiment: evaluate German-pruned model on Spanish and vice versa
      → how much does language-specific pruning cost cross-lingually?

**Exit criterion:** if adding Spanish took more than a week, the abstraction was wrong.
Say so honestly in the README rather than hiding it.

---

## Phase 6 — Package and ship
*~1 week*

- [ ] Quantise to 4-bit
- [ ] Convert to MLX and GGUF
- [ ] Benchmark tokens/sec and peak memory on the M1
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
