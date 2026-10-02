// One model in memory at a time, behind wllama (llama.cpp compiled to wasm).

import { Wllama, LoggerWithoutDebug } from '@wllama/wllama/esm/index.js';
import wasmUrl from '@wllama/wllama/esm/wasm/wllama.wasm?url';
import { buildPrompt } from './format.js';

/** Where the GGUFs live. `/models/` is the dev server's view of `../checkpoints/gguf`. */
const BASE = (import.meta.env.VITE_MODEL_BASE ?? '/models/').replace(/\/?$/, '/');

/**
 * The Phase 4 pick for German is the vocab-trimmed 755 MB file, not the 593 MB
 * depth-pruned one: the prune buys 160 ms and costs 0.047 F0.5 and seven points
 * of overcorrection, and overcorrection is the axis this project turns on.
 */
export const MODELS = {
  de: { file: 'langlm-de-755mb.gguf', label: 'Deutsch', sizeMB: 755 },
  es: { file: 'langlm-es-q4km.gguf', label: 'Español', sizeMB: 793 },
};

export const modelUrl = (language) => BASE + MODELS[language].file;

let wllama = null;
let loaded = null;

function instance() {
  wllama ??= new Wllama({ default: wasmUrl }, { logger: LoggerWithoutDebug, suppressNativeLog: true });
  return wllama;
}

export const loadedLanguage = () => loaded;

export async function isCached(language) {
  try {
    return (await instance().cacheManager.open(modelUrl(language))) !== null;
  } catch {
    return false;
  }
}

/**
 * `?gpu=0` forces CPU-only inference, for comparing backends. Otherwise wllama
 * offloads every layer to WebGPU when the browser has it.
 */
const GPU_LAYERS = new URLSearchParams(location.search).get('gpu') === '0' ? 0 : undefined;

let queue = Promise.resolve();

/**
 * Load a language's model, unloading whatever was there. Calls are queued:
 * two loads interleaving would leave the page holding an instance the other
 * one already tore down.
 */
export function load(language, onProgress) {
  const next = queue.then(() => loadNow(language, onProgress));
  queue = next.catch(() => {});
  return next;
}

async function loadNow(language, onProgress) {
  if (loaded === language) return;
  if (wllama?.isModelLoaded()) {
    await wllama.exit();
    wllama = null;
  }
  loaded = null;
  // Without this the cached model is "best-effort" storage, which Chrome
  // evicts silently when the disk runs low -- and a re-download is 800 MB.
  await navigator.storage?.persist?.().catch(() => false);
  const w = instance();
  await w.loadModelFromUrl(modelUrl(language), {
    // 1024 is what `correct.py` asks llama-server for; a sentence is ~64 tokens.
    n_ctx: 1024,
    n_gpu_layers: GPU_LAYERS,
    progressCallback: ({ loaded: done, total }) => onProgress?.(done, total),
  });
  loaded = language;
}

export function backend() {
  if (!wllama?.isModelLoaded()) return null;
  return {
    threads: wllama.getNumThreads(),
    multithread: wllama.isMultithread(),
    webgpu: wllama.isSupportWebGPU() && GPU_LAYERS !== 0,
  };
}

/**
 * Greedy, like every eval in the repo. Streams the raw text to `onText` so the
 * page can show the answer arriving; resolves to the full text and timings.
 */
export async function generate(source, language, { onText, signal } = {}) {
  let text = '';
  let timings = null;
  await instance().createCompletion({
    prompt: buildPrompt(source, language),
    max_tokens: 200,
    temperature: 0,
    stream: true,
    abortSignal: signal,
    onData: (chunk) => {
      text += chunk.choices?.[0]?.text ?? '';
      if (chunk.timings) timings = chunk.timings;
      onText?.(text);
    },
  });
  return { text, timings };
}
