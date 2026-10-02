---
license: cc-by-nc-4.0
language:
  - de
  - es
base_model: utter-project/EuroLLM-1.7B
pipeline_tag: text-generation
library_name: gguf
tags:
  - grammatical-error-correction
  - gec
  - language-learning
  - gguf
  - llama.cpp
  - q4_k_m
---

# langlm: grammar correction for learners of German and Spanish

Two small models that correct a learner's sentence and say what kind of error each
fix was. They are fine-tuned from [EuroLLM-1.7B](https://huggingface.co/utter-project/EuroLLM-1.7B),
compressed to about 755 MB each, and fast enough to run offline on a laptop or inside a
browser tab.

| File | Language | Size | Parameters |
|---|---|---:|---:|
| `langlm-de-755mb.gguf` | German | 755 MiB | 1.294B |
| `langlm-es-q4km.gguf` | Spanish | 757 MiB | 1.296B |

Code, training pipeline and every report the numbers below come from:
[github.com/afchatfield/language_lm](https://github.com/afchatfield/language_lm).

## Results

F0.5 is MaxMatch (M²) over corrected sentences, the standard GEC metric, which
weights precision twice as heavily as recall. *Overcorrection* is the share of 400
already-correct published sentences (news and Wikipedia) that the system changed at
all. Lower is better: a tool that "fixes" correct text teaches a learner the wrong thing.

### German: Falko-MERLIN

Held-out **test** split, 2,337 sentences, 5,963 gold edits:

| System | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| LanguageTool (with n-grams) | 0.5890 | 0.2091 | 0.4321 | 28.7% |
| EuroLLM-1.7B zero-shot | 0.4634 | 0.0880 | 0.2501 | 10.0% |
| EuroLLM-1.7B few-shot (8 examples) | 0.6431 | 0.2574 | 0.4948 | 15.2% |
| **langlm-de (before compression)** | **0.7322** | **0.6176** | **0.7060** | **9.5%** |

Compression, on the **dev** split (2,503 sentences):

| Variant | F0.5 | Overcorrection | Size |
|---|---:|---:|---:|
| Uncompressed | 0.7072 | 9.8% | 997 MB |
| **Vocabulary-trimmed (this file)** | **0.7061** | **9.5%** | **755 MB** |

### Spanish: COWS-L2H

Held-out **test** split, 4,284 sentences, 7,333 gold edits:

| System | P | R | F0.5 | Overcorrection |
|---|---:|---:|---:|---:|
| LanguageTool | 0.4547 | 0.1630 | 0.3348 | 27.3% |
| EuroLLM-1.7B zero-shot | 0.4632 | 0.0610 | 0.1997 | 11.2% |
| EuroLLM-1.7B few-shot (8 examples) | 0.6092 | 0.1210 | 0.3371 | 8.0% |
| **langlm-es (before compression)** | **0.6878** | **0.4867** | **0.6353** | **6.0%** |

Compression, on the **dev** split:

| Variant | P | R | F0.5 | Overcorrection | Size |
|---|---:|---:|---:|---:|---:|
| Uncompressed | 0.6510 | 0.4531 | 0.5987 | 6.0% | 997 MB |
| **Vocabulary-trimmed (this file)** | **0.6453** | **0.4534** | **0.5949** | **7.5%** | **757 MB** |

**Which numbers belong to which model.** The test-split rows were measured on each model
*before* compression, and the test split was read once, to choose which model to
compress. The files here are the compressed versions, and they were scored on dev
only. On dev, compression costs German 0.0011 of F0.5 (95% CI straddles zero) and
Spanish 0.0038 (CI [−0.0068, −0.0009]), plus 1.5 points of overcorrection. Across the five German
models that have both scores, dev and test agree to within 0.005.

### Speed

| Setting | Tokens/s | One sentence |
|---|---:|---:|
| German, llama.cpp, Metal, M1 16 GB | 62.3 | ~0.6 s |
| German or Spanish, browser (wllama, WebGPU), M1 | 13–16.5 | 3–5 s |
| Same, browser, CPU only | 2.8 | ~22 s |

Peak memory for the German model under llama.cpp is 477 MB.

## What each model is

Both models share one recipe, run once per language:

1. **Base:** EuroLLM-1.7B. It was chosen over Qwen3-1.7B and Qwen3.5 because it has the
   best German tokenizer fertility (1.69 tokens/word) and the most vocabulary left
   unused by any one language.
2. **Fine-tune:** LoRA (r=32, alpha=64, all linear layers), merged into the base. The loss
   covers the answer only, and the `changes` list is down-weighted to 0.25 because
   F0.5 never reads it. In the German tuning runs that was the largest gain after
   switching to real learner data.
3. **Vocabulary trim:** EuroLLM's 128,000-row vocabulary is cut to the rows that cover
   99.9% of the language's text, with English and the task's own prompts and answers
   counted too, so the JSON schema survives. That keeps 39,399 rows for German and
   39,886 for Spanish and removes about 22% of the parameters. The LoRA touches no
   vocabulary-sized weight, so the adapter carries over with no retraining.
4. **Quantisation:** Q4_K_M with llama.cpp.

**German training data:** the 19,237 real learner sentences in Falko-MERLIN train,
plus errors injected by rule into clean German news and Wikipedia text from the
Leipzig corpora. The injected types are weighted to match the learner corpus's own
error histogram. About 22% of the examples need no correction, which matches
Falko-MERLIN's measured rate and is what teaches the model to leave correct text alone.

**Spanish training data:** 97,473 examples, made up of 37,473 real learner sentences
from COWS-L2H (essays by university learners of Spanish) and 60,000 with rule-injected
errors. 33% need no correction, which is COWS-L2H's own rate.

No output from any other LLM provider was used at any stage.

## How to use it

These are **base-format** checkpoints, not chat models. Send the exact prompt below as
a raw completion. A chat template, which `llama-cli` applies by default, makes the
model answer in prose instead of JSON.

```
Correct the German sentence, then list what changed.

Input: Der Mann hat ein grosse Haus gekauft .
Output:
```

For Spanish, the first line is `Correct the Spanish sentence, then list what changed.`

The input should be **tokenised**, with a space before punctuation, as in the
training data. Decode greedily. The model answers with one line of JSON, the
corrected sentence first:

```json
{"correction": "Der Mann hat ein großes Haus gekauft .", "changes": [{"was": "grosse", "now": "großes", "type": "R:ADJ:FORM"}]}
```

`type` is an [ERRANT](https://github.com/chrisjbryant/errant) error code: an operation
(`M` missing, `R` replaced, `U` unnecessary), a word class, and `:FORM` when the
problem is an inflection. The repository renders these into learner-facing
explanations from `rules/{de,es}_types.yaml`.

With llama.cpp:

```bash
llama-server -m langlm-de-755mb.gguf -c 1024 -ngl 99 --port 8080

curl http://127.0.0.1:8080/completion -d '{
  "prompt": "Correct the German sentence, then list what changed.\n\nInput: Ich habe gestern ein Buch gelest .\nOutput:",
  "n_predict": 200, "temperature": 0
}'
```

In a browser, [wllama](https://github.com/ngxson/wllama) loads these files directly.
The repository's `demo/` is a working example.

## Limitations

- **One sentence at a time.** The models were trained on single sentences, so split paragraphs first.
- **Learner text only.** German was evaluated on Falko-MERLIN essays and Spanish on
  COWS-L2H essays. The models have not been measured on other kinds of writing.
- **Error types are the weak part.** The model's `type` matches the annotator's on 69% of
  German edits. Re-deriving the type from the (input, correction) pair with ERRANT
  gets 89% of the edits it segments the same way, and the repository's CLI does that. Treat `type` as a hint.
- **Some errors are hard for it.** For example, Spanish `R:OTHER` (wrong word choice)
  has 0.23 recall, and the per-type tables are in the repository.
- **The Spanish trim is nearly free, not free.** It touches 6 more of the 400
  already-correct sentences than the uncompressed model does.
- **Language-specific vocabularies.** Each model's vocabulary is trimmed to its own
  language. The other language still encodes through byte fallback, but at 40–56% more
  tokens and with no measured accuracy. Use the matching model.

## Licence and data

The base model, EuroLLM-1.7B, is Apache-2.0. The fine-tuning data carries its own
terms. Falko is CC BY 3.0 and MERLIN is CC BY-SA 4.0. The clean German and Spanish text
that synthetic errors were injected into comes from the Leipzig Corpora Collection,
which is CC BY-NC 4.0, and that is why these weights are released for
**non-commercial** use. COWS-L2H is used under its own release terms.
