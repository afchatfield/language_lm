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
> One debt: six of the fifteen `reference:` fields in `rules/de.yaml` are `null` because the
> Rat für deutsche Rechtschreibung paragraph has not been checked. A wrong citation in 30k
> examples would teach the model to cite confidently and wrongly, so they are omitted rather
> than guessed. Verify before Phase 3.

---

## Phase 3 — German SFT
*~1 week*

- [ ] LoRA config: r=32, alpha=64, all linear layers, lr 1e-4, cosine, 2–3 epochs
- [ ] **Verify loss masking on the prompt** before launching anything long
- [ ] Structured JSON output schema (span-based edits)
- [ ] Constrained decoding at inference (GBNF / Outlines / XGrammar)
- [ ] Evaluate → read per-error-type breakdown → fix data in Phase 2 → retrain
  - [ ] Iteration 1
  - [ ] Iteration 2
  - [ ] Iteration 3

*Expect 2–4 loops. Quality comes from data iteration, not hyperparameter search.*

**Exit criterion:** beat the LT baseline on F0.5, **or** match it with substantially better
explanations. Overcorrection rate under control.

> ⚠️ If after three data iterations you still can't beat LT, stop grinding. Pivot the framing to
> "small local model matching a mature rule-based system, with explanations, at 600MB offline."
> That is still a good project. Owning it beats pretending on a CV.

---

## Phase 4 — Compression
*~1.5 weeks*

**Check first, before spending GPU hours:**

- [ ] Trim a base model's vocab *without* training, confirm `mlx-lm` and `llama.cpp` still convert it
- [ ] Standard HF format, `vocab_size` correctly updated in config

**Then:**

- [ ] Vocab trim on the base model
  - [ ] Tokenize German+English corpus, frequency threshold
  - [ ] Keep all specials and byte-fallbacks
  - [ ] Slice embedding + LM head rows
  - [ ] Remap token IDs
  - [ ] Rebuild tokenizer merge table consistently with trimmed vocab
- [ ] Heal: short LM-objective finetune on German text
- [ ] Task SFT on trimmed model, same Phase 3 recipe
- [ ] Depth prune (second experiment)
  - [ ] Layer-wise input/output cosine similarity
  - [ ] Sweep 2 / 4 / 6 dropped layers → quality-vs-size *curve*, not a single point
  - [ ] Heal each with LoRA

**Ablation table to report:**

| Variant | F0.5 | Overcorrection % | Size | tok/s (M1) |
|---|---|---|---|---|
| Uncompressed | | | | |
| Vocab-trimmed | | | | |
| Vocab-trimmed + depth-pruned | | | | |

**Exit criterion:** defensible claim, e.g. "25% parameter reduction for X points of F0.5."

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
