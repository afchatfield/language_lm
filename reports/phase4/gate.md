# Phase 4 gate: does a trimmed checkpoint still convert?

The roadmap puts this before anything expensive, and it is the right order: a
healing run on a checkpoint that no converter will read is a wasted healing run.
Nothing here is trained. The vocabulary is trimmed, the artefact is checked, and
that is all.

**Result: the gate passes on llama.cpp, Phase 0's projection was right to within
half a percent, and one real defect was caught before any GPU time was spent.** The MLX half could not be run here and is still open.

## The gate earned its place: the first trim deleted the output schema

Worth putting before the numbers, because it is the failure the roadmap's
"check first" is for and it would not have been caught by any converter.

The keep-set was chosen by coverage over Leipzig German news, German Wikipedia
and English news. That is natural language, and **this model does not answer in
natural language.** It answers in JSON, and the tokens that structure a JSON
answer are near-absent from news prose. The trim deleted them:

```
▁{   ▁[   }]   :"   ▁.   corre
```

The result loads, converts, passes every structural check in the next section,
and corrects German correctly — as bare text:

> `Ich habe gestern ein Buch gelesen der sehr interessant war .`
> → `Ich habe gestern ein Buch gelesen das sehr interessant war.`

The correction is right. The schema is gone. A checkpoint that silently stops
emitting parseable output is exactly the class of break that a size number and a
successful GGUF conversion will happily sit on top of.

**The fix is to count the task corpus, not only prose** -- prompts and answers
both, since the answers are the only place the output format appears.
`scripts/trim_vocab.py` now does, and `--no-task-data` reproduces the failure on
request. After the fix the 45-token schema sample tokenizes **identically** to
the original, and the Phase 3 adapter parses 8/8 on dev sentences.

Counting the task corpus made the vocabulary *smaller*, not larger — 41,599 →
39,399 — because 76,442 task examples concentrate a great deal of mass on a
narrower set of tokens and move the 99.9% coverage point. The schema tokens are
only a handful of rows; the saving went up, not down.

The general lesson for Phase 5: **a vocabulary trim has to be counted over
everything the deployed model reads or writes.** Porting to Spanish means adding
Spanish prose *and* the Spanish task corpus, and the failure will look like
success if only the first is done.

## What was trimmed

Coverage over Leipzig German news, German Wikipedia and English news — 100,000
sentences each — at a 0.999 occurrence threshold, which is the same counting
Phase 0 used to *predict* this trim rather than a fresh rule invented for it.

| | |
|---|---:|
| Vocabulary | 128,000 → **39,399** rows (69.2% removed) |
| Protected regardless of frequency | 489 (specials + all 256 byte-fallbacks) |
| Parameters | 1,656,850,432 → **1,293,940,736** |
| Share of model removed | **21.9%** |
| bf16 saving | **726 MB** |

Phase 0 predicted 21.4% and 708 MB at this threshold, and the realised trim is
21.9% and 726 MB. That is a *prediction confirmed*, not a number produced and
compared to nothing: the figure was written into the README after Phase 0 and the
trim reproduced it to within half a percent.

The leverage comes from untied embeddings. 69.2% of rows removed buys only 21.9%
of parameters, but it buys it twice — once from `embed_tokens` and once from
`lm_head` — which is why EuroLLM won the Phase 0 comparison on trimmability
rather than on fertility alone.

## The four artefacts agree

A vocabulary trim is four edits that have to land together. Any one of them
missed leaves a checkpoint that loads, generates plausible text, and is wrong.

| Artefact | Expected | Got |
|---|---:|---:|
| `config.json` `vocab_size` | 39,399 | 39,399 |
| `model.embed_tokens.weight` | (39399, 2048) | (39399, 2048) |
| `lm_head.weight` | (39399, 2048) | (39399, 2048) |
| `tokenizer.json` vocab | 39,399 | 39,399 |
| `tokenizer.model` pieces | 39,399 | 39,399 |

Merges fell from 260,662 to **89,212**. A merge survives only if both its parts
*and* the token it produces are still in the vocabulary; without that check the
tokenizer retains rules producing ids that no longer exist.
`tests/test_compress_vocab.py` pins this and the id remapping.

Both tokenizer files are rebuilt because different converters read different
ones — llama.cpp prefers the SentencePiece proto, `tokenizers` uses the JSON.
Rebuilding one and not the other yields two tokenizers that disagree, which fails
loudly nowhere.

`bos_token_id` and `eos_token_id` needed no fixup: ids are kept in ascending
order, so the specials at 0/1/2 stay at 0/1/2.

## It works untrained

| Check | Result |
|---|---|
| Round-trip, German / English / CJK / emoji / Cyrillic | 5/5 exact |
| Fast tokenizer vs slow tokenizer agreement | 5/5 identical ids |
| Highest id emitted | 39,130 < 39,399 |
| Greedy generation | fluent German |

> `Berlin ist die Hauptstadt` → *der Bundesrepublik Deutschland und zugleich die
> größte Stadt Deutschlands. Sie ist die bevölkerungsreichste*

Non-target scripts still round-trip through byte fallback, at 29 tokens for a
short mixed CJK/emoji/Cyrillic string — expensive, which is the correct
trade, and lossless, which is the requirement.

**Segmentation is identical to the untrimmed tokenizer on 95.75% of 2,000 German
news sentences.** The 4.25% that differ are where a trimmed rare token forces a
re-split into pieces the model has seen in other contexts but not in this one.
That is the whole quality risk of the trim, and it is what the healing step
exists to repair — so it is also the number to watch when healing is evaluated.

## Conversion

`convert_hf_to_gguf.py` from llama.cpp (`ggml-org/llama.cpp`, shallow clone),
`gguf` 0.19.0.

| Model | f16 GGUF | Q4_K_M |
|---|---:|---:|
| base (untrimmed) | 3163.3 MB | 996.7 MB |
| **vocabulary-trimmed** | 2469.1 MB | **755.4 MB** |
| saving | 694 MB (21.9%) | **241 MB (24.2%)** |

GGUF metadata reads back consistent: architecture `llama`, 39,399 vocab tokens,
`token_embd.weight` and `output.weight` both (2048, 39399), bos/eos/unk 1/2/0.

Both quantised models run. On 8 chunks of German news at `n_ctx` 512:

| Model | PPL |
|---|---:|
| base Q4_K_M | 18.66 ± 1.20 |
| trimmed Q4_K_M | 19.20 ± 1.24 |

**These two numbers are not strictly comparable and should not be quoted as a
quality delta.** Perplexity is per *token*, the two models tokenize differently
on 4.25% of German sentences, and a model that splits a word into more pieces
faces an easier per-token prediction. The figures are here to show the trimmed
artefact loads and computes sane values, not to measure what the trim cost. The
comparison that decides that is task-level F0.5 after healing and re-SFT, scored
the way every other number in this project is scored.

## Where the size target stands

| | |
|---|---:|
| Vocabulary-trimmed, Q4_K_M | **755 MB** |
| Target | 600 MB |
| Gap | **155 MB** |

Phase 0 predicted ~782 MB and the measured figure is 755 MB, so the projection
was slightly pessimistic and the conclusion it drew is unchanged: **vocabulary
trimming alone does not reach 600 MB.** The depth prune has to find another 155
MB, or the target moves. The roadmap is explicit that swapping in a smaller base
model is the trade this phase exists not to make, even though `headroom.md` names
the base model as the binding constraint on F0.5.

## What is still open

**MLX conversion was not tested.** `mlx` and `mlx-lm` are macOS/Apple-Silicon
only and this box is Linux/CUDA, so half the gate cannot run here. It has to run
on the M1 before the trim is trusted end to end, and it is the half more likely
to break: MLX's Llama loader reads `config.json` and the tokenizer through
different code than llama.cpp does. Until then the gate is passed on one of two
converters.

**Nothing has been healed or re-trained.** The trimmed checkpoint is the *base*
model with rows removed. The Phase 3 adapter has not been applied to it — though
it should apply cleanly, since the LoRA targets `q/k/v/o/gate/up/down_proj` and
none of those touch the vocabulary dimension. That gives a free ablation row:
the shipped model on a trimmed base with no retraining at all, isolating trim
damage from healing effects, and answering whether healing is needed before
paying for it.

## Reproducing

```bash
python scripts/trim_vocab.py --threshold 0.999    # writes checkpoints/base-trimmed
python llama.cpp/convert_hf_to_gguf.py checkpoints/base-trimmed \
    --outfile trimmed-f16.gguf --outtype f16
llama-quantize trimmed-f16.gguf trimmed-q4km.gguf Q4_K_M
```
