# Base model decision

## The rule, stated before the numbers

1. **Primary: German fertility.** Tokens per German word, averaged over the news and
   Wikipedia samples. Every extra token costs context, training step time, and
   inference latency, and an M1 has none of those to spare.
2. **Tiebreak: vocabulary-trim headroom.** Embedding parameters removable at
   99.9% token coverage, as a fraction of the whole model. This is the Phase 4
   compression budget, and it is decided by the tokenizer before any training happens.
3. **Hard constraints.** At most ~2B parameters, weights openly accessible, and
   convertible by both `mlx-lm` and `llama.cpp` for the Phase 6 export.

## Ranking

| Rank | Model | Params | German fertility | Embed share | Trim @0.999 | Trimmed params |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `EuroLLM-1.7B` | 1.66B | 1.693 | 31.6% | 21.4% (708 MB) | 1.30B |
| 2 | `Qwen3.5-0.8B` | 0.87B | 1.736 | 29.1% | 22.2% (388 MB) | 0.68B |
| 3 | `Qwen3.5-2B` | 2.27B | 1.736 | 22.4% | 17.1% (777 MB) | 1.89B |
| 4 | `Qwen3-1.7B` | 2.03B | 2.047 | 15.3% | 10.7% (433 MB) | 1.82B |

## Decision: `EuroLLM-1.7B`

`EuroLLM-1.7B` wins on the primary metric and the tiebreak at once, which is why this is a short section.

* **Fertility 1.693 tokens/word** on German, the best of the 4 candidates measured. The worst, `Qwen3-1.7B`, needs 21% more tokens for the same German text -- on a fixed context budget that is 21% less material per training example and per correction request.
* **Untied embeddings.** The embedding table and the LM head are separate matrices, so 524M parameters (31.6% of the model) sit in vocabulary rows and every row removed in Phase 4 pays twice. Every tied-embedding candidate saves half as much per row.
* **21.4% of the model is trimmable** at 0.999 coverage: 86,401 of 128,000 vocabulary rows are never touched by German or English text, worth 354M parameters (708 MB at bf16). That is the Phase 4 compression budget, available before any training.
* **1.66B parameters**, inside the size constraint, in standard Hugging Face format.

EU-focused, explicitly trained on German/Spanish/French. Untied embeddings.

## Size budget

Where the roadmap's ~600MB target stands after Phase 0, using only measurements made here:

| Stage | Params | Size |
|---|---:|---:|
| Base, bf16 | 1.66B | 3314 MB |
| Vocabulary-trimmed, bf16 | 1.30B | 2606 MB |
| Vocabulary-trimmed, 4-bit | 1.30B | ~782 MB |

The 4-bit row assumes ~4.8 bits per weight, which is where llama.cpp Q4_K_M and MLX 4-bit actually land once scales and zero-points are counted. Vocabulary trimming plus 4-bit quantisation therefore gets to roughly 782 MB on its own, short of the 600MB target; the Phase 4 depth-prune has to close the remainder, or the target moves.

## Not evaluated

* **Gemma-3-1B** -- google/gemma-3-1b-pt is gated. Accept the licence at https://huggingface.co/google/gemma-3-1b-pt and run `hf auth login`, then re-run the analysis.

The decision above stands on the candidates that could be measured. Re-running `make fertility` after resolving the access issue will regenerate this file.
