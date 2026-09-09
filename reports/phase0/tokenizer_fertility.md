# Tokenizer fertility and vocabulary-trim headroom

Measured over the Leipzig corpora `de_news`, `de_wikipedia`, `en_news`, `es_news`, up to 100,000 sentences each. Only tokenizers and configs were downloaded; no model weights were touched.

Two quantities decide the base model, and they pull against each other. **Fertility** is how many tokens a tokenizer spends per word: a tokenizer that shatters German compounds makes every sequence longer, costing context, training time, and inference latency. **Trim headroom** is the share of the vocabulary that German and English text never touch; those embedding rows can be sliced out in Phase 4 for free. A large multilingual vocabulary tends to score well on the second and badly on the first.

Retained languages for the trim projection: `de`, `en`. English stays because the prompt and instruction format are English.

## Model geometry

| Model | Params | Vocab | Hidden | Layers | Tied? | Embed+head params | Share of model |
|---|---:|---:|---:|---:|:--:|---:|---:|
| `EuroLLM-1.7B` | 1.66B | 128,000 | 2,048 | 24 | **no** | 524M | **31.6%** |
| `Qwen3-1.7B` | 2.03B | 151,936 | 2,048 | 28 | yes | 311M | **15.3%** |
| `Qwen3.5-0.8B` | 0.87B | 248,320 | 1,024 | 24 | yes | 254M | **29.1%** |
| `Qwen3.5-2B` | 2.27B | 248,320 | 2,048 | 24 | yes | 509M | **22.4%** |

## Fertility (tokens per word -- lower is better)

| Model | de_news | de_wikipedia | en_news | es_news | **de mean** |
|---|---:|---:|---:|---:|---:|
| `EuroLLM-1.7B` | 1.654 | 1.731 | 1.396 | 1.404 | **1.693** |
| `Qwen3-1.7B` | 1.994 | 2.100 | 1.307 | 1.612 | 2.047 |
| `Qwen3.5-0.8B` | 1.684 | 1.789 | 1.304 | 1.396 | 1.736 |
| `Qwen3.5-2B` | 1.684 | 1.789 | 1.304 | 1.396 | 1.736 |

## Bytes per token (cross-check -- higher is better)

Fertility divides by whitespace words, which flatters languages that write compounds as separate words. Bytes per token does not depend on how a word is defined, so it is the sanity check on the table above.

| Model | de_news | de_wikipedia | en_news | es_news |
|---|---:|---:|---:|---:|
| `EuroLLM-1.7B` | 4.29 | 4.13 | 4.28 | 4.35 |
| `Qwen3-1.7B` | 3.56 | 3.40 | 4.57 | 3.79 |
| `Qwen3.5-0.8B` | 4.21 | 3.99 | 4.59 | 4.37 |
| `Qwen3.5-2B` | 4.21 | 3.99 | 4.59 | 4.37 |

## Vocabulary-trim projection

`kept` is the smallest vocabulary covering that share of all token *occurrences* in the de+en corpora, plus every special and single-character token, which must survive so the tokenizer can still encode arbitrary text. Reporting a curve rather than a single "every id ever seen" count keeps the estimate honest: a long tail of ids seen once each inflates the vocabulary while carrying almost no probability mass.

`saved` is the resulting reduction in the embedding table plus, where untied, the LM head -- at bf16, and as a fraction of the whole model.

| Model | Coverage | Vocab kept | Vocab removed | Params saved | Saved (bf16) | Saved % of model |
|---|---:|---:|---:|---:|---:|---:|
| `EuroLLM-1.7B` | 0.99 | 29,211 (23%) | 98,789 | 405M | 809 MB | 24.4% |
|  | 0.999 | 41,599 (32%) | 86,401 | 354M | 708 MB | 21.4% |
|  | 0.9999 | 47,321 (37%) | 80,679 | 330M | 661 MB | 19.9% |
|  | 1 | 48,114 (38%) | 79,886 | 327M | 654 MB | 19.7% |
| `Qwen3-1.7B` | 0.99 | 29,195 (19%) | 122,741 | 251M | 503 MB | 12.4% |
|  | 0.999 | 46,278 (30%) | 105,658 | 216M | 433 MB | 10.7% |
|  | 0.9999 | 52,932 (35%) | 99,004 | 203M | 406 MB | 10.0% |
|  | 1 | 53,817 (35%) | 98,119 | 201M | 402 MB | 9.9% |
| `Qwen3.5-0.8B` | 0.99 | 39,001 (16%) | 209,319 | 214M | 429 MB | 24.5% |
|  | 0.999 | 58,651 (24%) | 189,669 | 194M | 388 MB | 22.2% |
|  | 0.9999 | 65,767 (26%) | 182,553 | 187M | 374 MB | 21.4% |
|  | 1 | 66,556 (27%) | 181,764 | 186M | 372 MB | 21.3% |
| `Qwen3.5-2B` | 0.99 | 39,001 (16%) | 209,319 | 429M | 857 MB | 18.9% |
|  | 0.999 | 58,651 (24%) | 189,669 | 388M | 777 MB | 17.1% |
|  | 0.9999 | 65,767 (26%) | 182,553 | 374M | 748 MB | 16.4% |
|  | 1 | 66,556 (27%) | 181,764 | 372M | 745 MB | 16.4% |

## Skipped candidates

* **Gemma-3-1B** -- google/gemma-3-1b-pt is gated. Accept the licence at https://huggingface.co/google/gemma-3-1b-pt and run `hf auth login`, then re-run the analysis.
