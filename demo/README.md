# Browser demo

The Phase 4 GGUFs, running in a browser tab. No inference server: the page
downloads the model once, keeps it in the browser's storage, and runs it with
[wllama](https://github.com/ngxson/wllama), which is llama.cpp compiled to
WebAssembly, with WebGPU when the browser has it.

```bash
cd demo
npm install
npm run dev        # http://localhost:5173, models served from ../checkpoints/gguf
```

Needs Node 18+ (Vite 5 is pinned for that reason; Vite 7 wants 20.19+).

## What it runs

| Language | File | Size |
|---|---|---:|
| Deutsch | `langlm-de-755mb.gguf` (vocab-trimmed, Q4_K_M) | 755 MB |
| Español | `langlm-es-q4km.gguf` | 793 MB |

The German pick is the Phase 4 one: the 593 MB depth-pruned file is faster
but costs 0.047 F0.5 and seven points of overcorrection.

## Speed

Measured on the M1 / 16 GB, Chrome, one sentence at a time:

| Backend | tok/s | One sentence |
|---|---:|---:|
| llama.cpp native, Metal (Phase 4 table) | 62.3 | ~0.6 s |
| wllama, WebGPU (default) | 13–16.5 | ~3–5 s |
| wllama, CPU only, 4 threads (`?gpu=0`) | 2.8 | ~22 s |

WebGPU is what makes this usable. A browser without it (Firefox by default,
older Safari) falls back to the CPU row.

## How it differs from `scripts/correct.py`

The prompt, the answer parser and the explanation templates are the same:
`src/format.js` is a port of `langlm.train.format`, and `src/explain.js` reads
`rules/*_types.yaml` directly, so the templates cannot drift. Two things differ,
both because spaCy does not run in a browser:

- **Tokenisation** uses `correct.py`'s regex fallback, which splits trailing
  punctuation off but disagrees with spaCy on abbreviations and decimals.
- **Explanations** cannot be re-derived with ERRANT. The list follows the
  token diff between input and correction instead, so it explains exactly the
  edits the reader sees struck through. Types come from the model's own
  `changes` list where an entry matches an edit (that list is right on the type
  71% of the time against ERRANT's 89%). Entries that match nothing are
  dropped, and edits the model never claimed get a shape-only explanation
  (missing / extra / "should be X rather than Y").

## Deploying

The build (`npm run build`) is a static folder of about 9 MB, almost all of it
the wasm. The models are not in it: set `VITE_MODEL_BASE` to wherever the
GGUFs are hosted, e.g.

```bash
VITE_MODEL_BASE=https://huggingface.co/<user>/<repo>/resolve/main/ npm run build
```

The host must send these two headers on every page, or wllama runs on one
thread:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The model host must allow cross-origin requests (Hugging Face does).

wllama recommends splitting models into ≤512 MB shards
(`llama-gguf-split --split-max-size 512M`) so the download runs in parallel.
`loadModelFromUrl` takes the first shard's URL and finds the rest.
