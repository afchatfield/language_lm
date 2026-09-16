# language_lm

A small language model that finds grammar and spelling errors in German — and,
more to the point, **explains** them. Built to run offline on a 16GB M1, and
built so that adding a second language (Spanish) is a matter of writing one
corruptor file rather than rebuilding the pipeline.

> **Status: Phase 4 complete.** The model is trained, scored on the held-out test
> split, and compressed: 755 MB for no measurable quality loss, or 593 MB — the
> offline target — for 0.046 of F0.5. The
> [roadmap](roadmap.md) has the full plan and honest phase boundaries, and this
> README reports only what has actually been measured.

## Why this project

Grammar checkers tell you *what* to change. A learner needs to know *why*.
LanguageTool is very good at the first and reasonable at the second, but it is a
150MB Java service with a 3GB language model attached. The goal here is a model
small enough to run on a laptop with no network, that answers with a span, a
correction, and a reason.

Two languages fully evaluated beats a framework claiming twelve, so German comes
first and Spanish is the test of whether the abstraction is real.

## Results

Falko-MERLIN **dev**, 2,503 sentences of real German learner text. Every row is
scored with the same MaxMatch scorer and German ERRANT, so the comparison is
like-for-like. *Overcorrection* is the share of 400 already-correct published
sentences that the system modifies — lower is better, and it is the number that
decides whether a checker is usable.

| System | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| identity (change nothing) | 0.0417 | 0.0002 | 0.0008 | 0.0% |
| LanguageTool 6.8 + n-grams | 0.5653 | 0.1992 | 0.4134 | 28.7% |
| EuroLLM-1.7B zero-shot | 0.4761 | 0.0905 | 0.2571 | 10.0% |
| EuroLLM-1.7B few-shot (8 ex.) | 0.6600 | 0.2675 | 0.5102 | 15.0% |
| **this model** (LoRA SFT) | **0.7339** | **0.6174** | **0.7072** | **9.8%** |

**+0.29 F0.5 over LanguageTool while touching correct German three times less
often.** The recall gap is the wider story: LanguageTool finds 20% of the errors
a learner actually makes, and this model finds 62%.

The bar that mattered was never LanguageTool, though. **The prompted base model
already beat it** — few-shot scores 0.5102 against LanguageTool's 0.4134 — so
Phase 3 was held to 0.5102 instead. Setting the bar at the rule-based system
would have been setting it too low, and that is the kind of thing a project
finds out only by measuring its baselines before it trains anything.

### Held-out test

Frozen on day one of Phase 0, enforced in code, read once at the end of Phase 3
for five candidate models. Full table in [test.md](reports/phase3/test.md).

| | dev | test |
|---|---:|---:|
| F0.5 | 0.7072 | **0.7060** |
| Overcorrection | 9.8% | **9.5%** |

**Six iterations of tuning against dev did not overfit it.** Every candidate
landed within 0.005 of its dev score, mean absolute gap 0.0024 — so the dev
numbers above can be read as honest estimates of held-out performance.

The three best checkpoints turned out to be *one model with three names*: they
span 0.0023 of F0.5 and their ranking **inverts** between dev and test. F0.5
cannot choose between them, so the shipped model was chosen on overcorrection,
where a 4.5-point gap replicates on both splits.

> One bookkeeping caveat: the LanguageTool and few-shot rows are dev-only, since
> `scripts/run_baselines.py` is still hard-wired to the dev split. The two tables
> above are each internally consistent, and the 0.29 gap is far too wide for the
> split to change the conclusion, but the baselines should get a test run before
> any of this is quoted as a single number. Tracked in the roadmap.

## How it works

The model emits JSON: a corrected sentence, then the spans it changed.

```json
{"correction": "Sie hatten keine Möglichkeit , sich auszubilden .",
 "changes": [{"was": "Möglichkeit sich", "now": "Möglichkeit , sich", "type": "M:PUNCT"}]}
```

**The explanation is not generated.** Every explanation in the corpus is a pure
function of `(error type, wrong, right)`, so a model producing prose would be
reciting a template file at 39% of the answer's length. The model emits the
*type*; `langlm.explanations` renders the sentence at serving time from
[`rules/de_types.yaml`](rules/de_types.yaml). That makes the explanation
checkable — the type can be compared against what ERRANT independently derives
from the model's own correction — which prose never was.

Where a citation exists, it names the paragraph of the official German
orthography ruleset:

> German capitalises every noun, not just names. «hund» is a noun here, so it
> needs a capital letter: «Hund».
> *(Rat für deutsche Rechtschreibung, Amtliches Regelwerk 2024, § 55)*

Only 8 of the 25 rules carry one, and that is not laziness: the Regelwerk covers
sound-letter assignment, compound spelling, hyphens, capitalisation, punctuation
and line breaking, and **nothing else**. The other 17 rules are grammar —
agreement, case government, declension, word order — which no orthography
paragraph legislates. Each one records why it cannot be cited, and
`make check-citations` verifies every paragraph that *is* cited against the
official PDF. See [citations.md](reports/phase2/citations.md).

### Training data

29,237 examples: **19,237 real annotated learner sentences** from Falko-MERLIN
train, plus **10,000 synthetically corrupted** clean news and Wikipedia
sentences from 23 hand-written German corruptors covering 17 ERRANT error
types. 22% of examples are already correct and carry an empty edit list — the
ratio measured in the corpus itself, and the main defence against a model that
"fixes" correct German.

The synthetic half alone does not work, and finding that out was the single most
valuable result of Phase 3. See below.

## What did not work

Kept because a negative result that has to be re-derived is one that gets re-run,
and Phase 5 ports this whole repo to Spanish.

**Synthetic data alone made the model worse** — F0.5 0.1891, below even
zero-shot, with recall falling on 14 of 15 error types. The cause was not the
error mix but the *shape*: pristine news German carrying one or two surgical
errors, averaging 1.17 edits per sentence against real learner text's 2.55, with
no sentence over two. The model learned to find a mechanical corruption in
otherwise-perfect prose and overwrote the German competence the base model
already had. Real learner errors as the primary signal took it to 0.6944.

**Widening the corruptors from 8 error types to 17** bought +0.0016, an interval
straddling zero. Coverage was the wrong theory.

**Back-translated error generation** — a generator trained on Falko read
backwards — was predicted at 0.72–0.74 and delivered 0.7056, below the bar. That
prediction was recorded before the run, as was iteration 3's, and both were
wrong. ([headroom.md](reports/phase3/headroom.md))

**Beam search reaches a much better answer than it returns.** The 5-wide beam
contains the annotator's correction on 40.7% of the sentences where the model
does not rank it first — an oracle of 0.8380 against top-1's 0.7148. Nothing
reaches it. Minimum-edit reranking is worth +0.0007, MBR is worth less than
nothing, and a 9-feature learned reranker lands exactly on top-1. What separates
`der meisten Firmen` from `den meisten Firmen` is German case agreement, and a
reranker that cannot read German cannot tell them apart.
([nbest.md](reports/phase3/nbest.md))

**DPO on that same gap fails too**, which is what closes it. Gold-labelled
preference pairs from the model's own beam, 4,289 of them, cost 0.0177 of F0.5
and pushed overcorrection from 9.8% to 15.0%. Tightening the KL anchor walks the
score monotonically back toward the untrained reference and never past it, so the
best available outcome of the procedure is doing nothing. The oracle gap
measures the model's uncertainty; it is not a reserve.
([dpo.md](reports/phase3/dpo.md))

**Constrained decoding was never needed.** The model emitted 1 invalid JSON in
2,903 answers and 0 overlapping spans. The schema was the easy half.

## Base model and the size target

`utter-project/EuroLLM-1.7B`, chosen on measurement with the decision rule
written down before the numbers were collected
([writeup](reports/phase0/base_model_decision.md)).

| Rank | Model | Params | German fertility | Embedding share | Trim headroom @99.9% |
|---:|---|---:|---:|---:|---:|
| 1 | **EuroLLM-1.7B** | 1.66B | **1.693** | **31.6%** | **21.4%** (708 MB) |
| 2 | Qwen3.5-0.8B | 0.87B | 1.736 | 29.1% | 22.2% (388 MB) |
| 3 | Qwen3.5-2B | 2.27B | 1.736 | 22.4% | 17.1% (777 MB) |
| 4 | Qwen3-1.7B | 2.03B | 2.047 | 15.3% | 10.7% (433 MB) |

EuroLLM wins on both axes at once: fewest tokens per German word *and* the most
trimmable vocabulary. Its embeddings are untied, so 524M parameters live in
vocabulary rows and every row removed in Phase 4 pays twice — 86,401 of its
128,000 rows are never touched by German or English text.

| Stage | Params | Predicted (Phase 0) | Measured (Phase 4) |
|---|---:|---:|---:|
| Base, bf16 | 1.66B | 3314 MB | 3163 MB |
| Vocabulary-trimmed, bf16 | 1.29B | 2606 MB | 2469 MB |
| Vocabulary-trimmed, 4-bit | 1.29B | ~782 MB | **755 MB** |
| + 6 layers dropped, 4-bit | 1.01B | — | **593 MB** |

Phase 0 predicted the trim would land near 782 MB and that the depth prune would
have to close the gap to 600 MB. It landed at 755 MB and the prune closed it.
That prediction was recorded before any of it was built, and it held.

## Compression

| Variant | Params | Size | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| Uncompressed | 1.66B | 997 MB | 0.7072 | 9.8% |
| **Vocabulary-trimmed** | 1.29B | **755 MB** | **0.7061** | **9.5%** |
| + 6 layers dropped | 1.01B | **593 MB** | 0.6591 | 16.8% |

**The vocabulary trim is free.** 21.9% of the parameters and 242 MB for 0.0011 of
F0.5 — five times smaller than the dev-to-test spread — with **no healing and no
retraining**. The LoRA targets `q/k/v/o/gate/up/down_proj`, none of which touch
the vocabulary dimension, so an adapter trained on the full vocabulary stays valid
on a trimmed base. The roadmap's healing step turned out to be unnecessary and was
dropped: there was no damage to repair, and healing *cost* 0.0070 of F0.5.

**Depth pruning is not free.** Reaching 600 MB means dropping 6 of 24 layers, for
0.046 of F0.5 — still far above few-shot (0.5102) and LanguageTool (0.4134), but a
real price. And unlike the trim it does not survive without retraining: reusing the
adapter on a depth-pruned base produces a model that emits no schema at all, free
prose at 2 layers dropped and degenerate repetition at 6. It survives losing 69% of
the vocabulary but not 8% of the depth.

Full ablation, including the control that caught a 5pp overcorrection regression
belonging to the training corpus rather than to pruning:
[compression.md](reports/phase4/compression.md), [gate.md](reports/phase4/gate.md).

## Evaluation data, frozen

[Falko-MERLIN](https://aclanthology.org/W18-6111/) (Boyd 2018): 24k sentences of
German learner text, A1–C2, annotated with a German adaptation of ERRANT that
agrees with the corpus at 0.949 span precision.

| Split | Sentences | Edits | Already correct |
|---|---:|---:|---:|
| train | 19,237 | 49,058 | 22.1% |
| dev | 2,503 | 6,385 | 21.3% |
| test | 2,337 | 5,963 | read once, at the end of Phase 3 |

The test split was frozen on day one and its SHA-256 committed to
[`data/splits/manifest.json`](data/splits/manifest.json). This is enforced rather
than remembered: `load_split("falko_merlin", "test")` raises unless the caller
passes `allow_test=True`, so a peek has to be written down in code.

### LanguageTool, with its n-grams actually loaded

Running LanguageTool without its n-gram language model is the quiet failure mode
of this whole project: the server starts, answers every request, and silently
stops firing the confusion-set rules that resolve homophones from context.

`make lt-check` catches it. The probes it uses were chosen empirically, by
diffing a server with n-grams against a control container started without them —
because several classic German confusions (das/dass among them) are caught by
hand-written rules that need no n-grams at all, and would give a false pass.
See [lt_sanity.md](reports/phase0/lt_sanity.md).

## Getting started

```bash
conda env create -f environment.yml     # environment-cuda.yml on a CUDA box
conda activate langlm
make test                # ruff + the offline test suite
```

Then, with Docker Desktop running:

```bash
make ngrams              # ~1.6GB download, 3.1GB on disk
make lt-up               # LanguageTool on :8010
make lt-check            # MUST pass before trusting anything downstream

make train-data          # corpora, splits, overcorrection set, training set
make dry-run             # loss mask + schema + one forward pass, trains nothing
make train               # LoRA SFT — well under an hour on an H200
make evaluate            # score against the Phase 1 baselines
```

`make help` lists every target. `make clean-ngrams` reclaims the 3.1GB.

## Repository layout

```
src/langlm/
├── data/            M2 format, corpus loaders, split freezing and the test guard
├── lt/              LanguageTool client — the single seam to the grammar engine
├── tokenizers/      fertility metrics, trim projection, report rendering
├── corruptors/      synthetic error injection            (Phase 2)
├── train/           schema, loss mask, preference pairs  (Phase 3)
├── eval/            M2 scorer, ERRANT, overcorrection    (Phase 1)
└── compress/        vocabulary trim, depth prune         (Phase 4 — not yet written)

rules/               explanation templates + Regelwerk citations
scripts/             thin CLI entry points, one per Makefile target
docker/              LanguageTool + n-gram mount
configs/             one YAML per phase
reports/             generated tables — committed, so results are reviewable
data/                downloaded and derived artefacts — gitignored,
                     except splits/manifest.json which freezes the evaluation set
tests/               offline; every test runs against a committed fixture
```

## Design notes

**`src/langlm/data/m2.py` is the load-bearing file.** M2 is the interchange
format for grammatical error correction corpora, and every number this project
reports passes through that parser. It round-trips byte-for-byte against the real
corpus and refuses to apply overlapping edits rather than silently producing an
order-dependent result.

**The loss is not uniform over the answer.** The `changes` list is 70% of the
supervised tokens and F0.5 never reads it, so it is down-weighted to 0.25 — worth
+0.0128, the largest single gain after iteration 2 and the only one that came
from the loss function rather than the data. The sweep has since turned over at
0.125, so 0.25 is the optimum.

**Training stack.** PyTorch + `peft` LoRA. Phase 4's vocabulary and depth surgery
operates on standard Hugging Face checkpoints, so the model has to live in that
format anyway. `mlx-lm` is installed for fast local inference, the tokens/sec
benchmark, and the Phase 6 export. Note `bitsandbytes` has no Apple Silicon
build — 4-bit comes from MLX and llama.cpp quantisation, not bnb.

**Reports are generated, not written.** Everything in `reports/` is produced by a
script and committed, so the tables in this README can be regenerated and diffed
rather than trusted.

## Licence and attribution

Code is [MIT](LICENSE). The data it downloads is not:

* **Falko-MERLIN** — Falko is CC BY 3.0, MERLIN is CC BY-SA 4.0. Not
  redistributed here; `make train-data` fetches it from the
  [original release](https://github.com/adrianeboyd/boyd-wnut2018).
* **Leipzig Corpora Collection** — CC BY-NC 4.0, downloaded on demand.
* **LanguageTool** — LGPL-2.1, run locally as a service. Generating training
  data from a local service is not distribution, so no licence obligations
  attach to this repository; third-party dictionary licences would need checking
  separately for any commercial path.
* **Deutsche Rechtschreibung, Amtliches Regelwerk 2024** — the Rat für deutsche
  Rechtschreibung ruleset, free to use, cited by paragraph and not reproduced.

No model output from OpenAI, Gemini or any other provider was used to train
this, at any stage. cLang-8 was declined on those grounds; the preference pairs
in Phase 3 are labelled by the corpus annotation, not by a larger model.
