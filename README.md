# language_lm

A small, compressed language model that finds grammar and spelling errors in
German — and, more to the point, **explains** them. Built to run offline on a
16GB M1, and built so that adding a second language (Spanish) is a matter of
writing one corruptor file rather than rebuilding the pipeline.

> **Status: Phase 0 complete.** Corpus loaded and frozen, LanguageTool running
> locally with its n-gram model verified, base model chosen. No model has been
> trained yet. The [roadmap](roadmap.md) has the full plan and honest phase
> boundaries; this README reports only what has actually been measured.

## Why this project

Grammar checkers tell you *what* to change. A learner needs to know *why*.
LanguageTool is very good at the first and reasonable at the second, but it is a
150MB Java service with a 3GB language model attached. The goal here is a model
small enough to run on a laptop with no network, that answers with a span, a
correction, and a reason.

Two languages fully evaluated beats a framework claiming twelve, so German comes
first and Spanish is the test of whether the abstraction is real.

## Phase 0 results

### Base model: `utter-project/EuroLLM-1.7B`

Chosen on measurement, not vibes. The decision rule was written down before the
numbers were collected — see [the full writeup](reports/phase0/base_model_decision.md).

| Rank | Model | Params | German fertility | Embedding share | Trim headroom @99.9% |
|---:|---|---:|---:|---:|---:|
| 1 | **EuroLLM-1.7B** | 1.66B | **1.693** | **31.6%** | **21.4%** (708 MB) |
| 2 | Qwen3.5-0.8B | 0.87B | 1.736 | 29.1% | 22.2% (388 MB) |
| 3 | Qwen3.5-2B | 2.27B | 1.736 | 22.4% | 17.1% (777 MB) |
| 4 | Qwen3-1.7B | 2.03B | 2.047 | 15.3% | 10.7% (433 MB) |

Two findings worth pulling out:

* **EuroLLM wins on both axes at once.** It spends the fewest tokens per German
  word *and* has the most trimmable vocabulary. Qwen3-1.7B — which is really
  2.03B parameters, since the advertised figure excludes embeddings — needs 21%
  more tokens for the same German text and offers half the compression headroom.
* **EuroLLM's embeddings are untied**, so 524M parameters (31.6% of the model)
  live in vocabulary rows and every row removed in Phase 4 pays twice. 86,401 of
  its 128,000 vocabulary rows are never touched by German or English text.

Full tables: [tokenizer_fertility.md](reports/phase0/tokenizer_fertility.md).

### Where the size target stands

| Stage | Params | Size |
|---|---:|---:|
| Base, bf16 | 1.66B | 3314 MB |
| Vocabulary-trimmed, bf16 | 1.30B | 2606 MB |
| Vocabulary-trimmed, 4-bit | 1.30B | ~782 MB |

Vocabulary trimming plus 4-bit gets to roughly 782MB on its own — short of the
600MB target. The Phase 4 depth-prune has to close the gap, or the target moves.
Recorded now so that it is a prediction rather than a retrofit.

### Evaluation data, frozen

[Falko-MERLIN](https://aclanthology.org/W18-6111/) (Boyd 2018): 24k sentences of
German learner text, A1–C2, automatically annotated with a German adaptation of
ERRANT.

| Split | Sentences | Edits | Already correct |
|---|---:|---:|---:|
| train | 19,237 | 49,058 | 22.1% |
| dev | 2,503 | 6,385 | 21.3% |
| test | 2,337 | 5,963 | *not inspected* |

The test split was frozen on day one and its SHA-256 committed to
[`data/splits/manifest.json`](data/splits/manifest.json). This is enforced rather
than remembered: `load_split("falko_merlin", "test")` raises unless the caller
passes `allow_test=True`, so a peek has to be written down in code.

The [error-type distribution](reports/phase0/error_type_histogram.md) (train+dev
only) confirms the Phase 2 corruptor list is aimed at the right targets:
spelling 13.3%, article form 10.8%, missing punctuation 9.6%, orthography
(largely noun capitalisation) 7.7%, adjective form 4.6%.

That 22% of learner sentences need *no* correction is itself a result: it sets
the ratio of negative examples in Phase 2, which is the main defence against a
model that "fixes" correct German.

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
conda env create -f environment.yml
conda activate langlm
make test                # ruff + the offline test suite
```

Then, with Docker Desktop running:

```bash
make ngrams              # ~1.6GB download, 3.1GB on disk
make lt-up               # LanguageTool on :8010
make lt-check            # MUST pass before trusting anything downstream

make data                # Falko-MERLIN, then freeze the splits
make corpora             # Leipzig samples for the tokenizer analysis
make fertility           # regenerates the tables above
```

`make help` lists every target. `make clean-ngrams` reclaims the 3.1GB.

## Repository layout

```
src/langlm/
├── data/            M2 format, corpus loaders, split freezing and the test guard
├── lt/              LanguageTool client — the single seam to the grammar engine
├── tokenizers/      fertility metrics, trim projection, report rendering
├── corruptors/      synthetic error injection            (Phase 2)
├── eval/            M2 scorer, ERRANT, overcorrection     (Phase 1)
└── compress/        vocabulary trim, depth prune          (Phase 4)

scripts/             thin CLI entry points, one per Makefile target
docker/              LanguageTool + n-gram mount
configs/phase0.yaml  candidates, corpora, coverage thresholds
reports/phase0/      generated tables — committed, so results are reviewable
data/                downloaded and derived artefacts — gitignored,
                     except splits/manifest.json which freezes the evaluation set
tests/               offline; every test runs against a committed fixture
```

## Design notes

**`src/langlm/data/m2.py` is the load-bearing file.** M2 is the interchange
format for grammatical error correction corpora, and every number this project
will ever report passes through that parser. It round-trips byte-for-byte
against the real corpus and refuses to apply overlapping edits rather than
silently producing an order-dependent result.

**Training stack.** PyTorch + `peft` LoRA on the `mps` backend is the canonical
path: Phase 4's vocabulary and depth surgery operates on standard Hugging Face
checkpoints, so the model has to live in that format anyway. `mlx-lm` is
installed for fast local inference, the tokens/sec benchmark, and the Phase 6
export. Note that `bitsandbytes` has no Apple Silicon build — 4-bit comes from
MLX and llama.cpp quantisation, not bnb.

**Reports are generated, not written.** Everything in `reports/phase0/` is
produced by a script and committed, so the tables in this README can be
regenerated and diffed rather than trusted.

## Licence and attribution

Code is [MIT](LICENSE). The data it downloads is not:

* **Falko-MERLIN** — Falko is CC BY 3.0, MERLIN is CC BY-SA 4.0. Not
  redistributed here; `make data` fetches it from the
  [original release](https://github.com/adrianeboyd/boyd-wnut2018).
* **Leipzig Corpora Collection** — CC BY-NC 4.0, downloaded on demand.
* **LanguageTool** — LGPL-2.1, run locally as a service. Generating training
  data from a local service is not distribution, so no licence obligations
  attach to this repository; third-party dictionary licences would need checking
  separately for any commercial path.
