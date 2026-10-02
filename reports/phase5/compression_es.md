# Phase 5: what compression costs in Spanish

Phase 4 measured the vocabulary trim in German and called it free: 21.9% of the
parameters for 0.0011 of F0.5, with no healing and no retraining. This repeats
that measurement for Spanish, on COWS-L2H **dev**, scored with the same MaxMatch
and Spanish ERRANT machinery as every other Phase 5 number.

**The result ports, and one part of it does not.** The trim is the same size win
to within a megabyte, and the adapter survives it unchanged for the same reason
it did in German. But where German's F0.5 loss was indistinguishable from zero,
Spanish's is small and real: **−0.0038, 95% CI [−0.0068, −0.0009], p = 0.012**.
"Free" was a German finding, and it is being reported here as a nearly-free one
rather than restated.

## The table

| | German | Spanish |
|---|---:|---:|
| Base model | EuroLLM-1.7B | EuroLLM-1.7B |
| Vocabulary kept | 39,399 / 128,000 | **39,886** / 128,000 |
| Share of rows removed | 69.2% | **68.8%** |
| Parameters | 1.657B → 1.294B | 1.657B → **1.296B** |
| Share of parameters removed | 21.9% | **21.8%** |
| Q4_K_M, uncompressed | 997 MB | **996.7 MB** |
| Q4_K_M, vocabulary-trimmed | 755.4 MB | **756.7 MB** |
| F0.5, uncompressed | 0.7072 | **0.5987** |
| F0.5, trimmed | 0.7061 | **0.5949** |
| ΔF0.5 | −0.0012 | **−0.0038** |
| 95% CI on ΔF0.5 | [−0.0043, +0.0019] | **[−0.0068, −0.0009]** |
| p | 0.458 | **0.012** |
| Overcorrection | 9.8% → 9.5% | **6.0% → 7.5%** |

Sizes are on-disk `llama-quantize` output at Q4_K_M. The *trimmed* models of both
languages were re-quantised with one llama.cpp build for this table, because a
size difference between two toolchain versions would otherwise read as a
difference between two languages — and German's re-quantised 755.4 MB reproduces
Phase 4's published 755 MB, so it does not.

The German **uncompressed** row is Phase 4's published figure carried over, not
re-measured: the roadmap records that the uncompressed row was never converted to
GGUF. The Spanish 996.7 MB is therefore the first actual measurement of that
footprint, and it lands within 0.3 MB of the German figure — which is the
expected result, since the two uncompressed models share a base, an architecture
and all 128,000 vocabulary rows, and differ only in merged LoRA values.

Note that this project's "MB" throughout is MiB — 755.4 MB here is 792,085,568
bytes.

## The trim ports almost exactly

**39,886 rows against German's 39,399**, from an independent count: Spanish and
English prose plus the 97,473-example Spanish task corpus, at the same 0.999
coverage threshold. Nothing forced these two numbers within 1.3% of each other.
Both languages need about 39,600 of EuroLLM's 128,000 rows, and both free about
21.8% of the parameters, because the saving is dominated by how much of a
128,000-row multilingual vocabulary *any* single European language leaves
untouched, not by which language it is.

The adapter was reused unchanged here too — no healing, no re-SFT. The Phase 4
reasoning carries over unaltered: the LoRA targets `q/k/v/o/gate/up/down_proj`
and none of them touch the vocabulary dimension, so an adapter trained on the
full vocabulary is dimension-valid on a trimmed base.

## Where Spanish differs: the trim is not quite free

German's −0.0012 is a coin flip (p = 0.458). Spanish's −0.0038 is not (p = 0.012,
CI entirely below zero). Both are small — Spanish's loss is under half of one
percent of its own score, and smaller than the ±0.005 dev/test spread Phase 3
measured across five German models — but they are different findings and are
written down as different findings.

The loss is not in recall, which is flat (0.4531 → 0.4534). It is precision:
0.6510 → 0.6453, and overcorrection rising 6.0% → 7.5%, which is 6 more of the
400 already-correct sentences touched. The trimmed Spanish model edits very
slightly more freely than the one it came from.

Why Spanish and not German is not established by this measurement, and the
honest answer is that it is not known. One candidate the numbers do support:
Spanish's own trim costs its own text more fertility than German's costs German
(+1.50% against +0.51% tokens per word, table below), so the Spanish kept set is
a marginally worse fit for its language than the German kept set is for its own.
That is a correlation across two points and is offered as the thing to test
next, not as the cause.

## The gate's warning, repeated and passed

Phase 4's gate caught a trim that deleted the JSON structural tokens because it
counted coverage over prose alone: the checkpoint corrected German correctly and
returned it as bare text with no schema. `gate.md` says explicitly that Phase 5
must repeat that check for Spanish, "where the same mistake would again look
like success."

It was repeated. The Spanish trim counts the Spanish task corpus, prompts and
answers both, and the schema survives: **2 answers with no readable correction
and 1 truncated, out of 5,116** — the same rate as the untrimmed model, not a
degradation. The trim did not eat the schema.

One false alarm is worth recording so the next person does not re-raise it.
Driving the quantised Spanish model through `llama-cli` returns a correct
correction as bare text with no JSON, which looks exactly like the gate's
failure. It is not: the German shipped model does the same thing under the same
command, because `llama-cli` injects a chat template these base-format
checkpoints were never trained with, and `<|im_start|>`/`<|im_end|>` leak into
the output. The harness was wrong, not the checkpoint. The authoritative check is
the transformers evaluation above, which reads the prompt the model was trained
on.

## What a language-specific trim costs the other language

The two kept sets are largely but not entirely the same rows:

| | Rows |
|---|---:|
| Kept by German | 39,399 |
| Kept by Spanish | 39,886 |
| Kept by both | **29,061** |
| German only | 10,338 |
| Spanish only | 10,825 |
| Union — the floor for one bilingual model | **50,224** |

So a single checkpoint serving both languages cannot go below 50,224 rows, 27.5%
more than either single-language trim, and would give back roughly a third of the
saving. Two 756 MB models is not obviously worse than one larger one, and this
project ships per-language models anyway, but the number is what that decision
should be made against.

Running each language's text through the other's tokenizer prices the mismatch
without needing a GPU:

| Text | Untrimmed | German trim | Spanish trim |
|---|---:|---:|---:|
| German (Falko dev) | 1.2472 | **1.2536** | 1.9522 |
| Spanish (COWS dev) | 1.1769 | 1.6736 | **1.1945** |

Tokens per whitespace word; bold is the model that language ships on. Its own
trim costs a language almost nothing — **+0.51%** for German, **+1.50%** for
Spanish. The other language's trim costs it **+55.7%** and **+40.1%**
respectively: not a failure, because byte fallback means any text still encodes,
but every Spanish sentence would cost half again as many tokens to read and
write on the German model, at the same tokens per second.

This is the tokenizer half of the roadmap's open cross-lingual pruning question.
The other half — what the *weights* of one language's compressed model do on the
other language's task — is not measured here and remains open.

## Reproducing

```bash
python scripts/trim_vocab.py --threshold 0.999 --language es --languages es en \
    --out checkpoints/base-trimmed-es
python scripts/merge_adapter.py --base checkpoints/base-trimmed-es \
    --adapter checkpoints/phase3-es-cw025 --out checkpoints/ship-es
python scripts/eval_finetuned.py --language es --base checkpoints/base-trimmed-es \
    --adapter checkpoints/phase3-es-cw025 --name trimmed-nohealing-es

python llama.cpp/convert_hf_to_gguf.py checkpoints/ship-es \
    --outfile ship-es-f16.gguf --outtype f16
llama-quantize ship-es-f16.gguf ship-es-q4km.gguf Q4_K_M
```

`--language es` selects the Spanish task corpus, which is what keeps the schema
tokens; `--languages es en` selects the prose corpora. They are separate flags
because they answer separate questions, and the gate's failure was counting only
the second.
