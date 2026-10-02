import { MODELS, isCached, load, loadedLanguage, backend, generate } from './model.js';
import {
  parseAnswer,
  tokenize,
  detokenize,
  spacesBefore,
  diffTokens,
  editsFromDiff,
  containsTokens,
} from './format.js';
import { explainType, explainUntyped } from './explain.js';

/** The last one in each list is already correct: restraint is half the job. */
const EXAMPLES = {
  de: [
    'Ich habe gestern ein Buch gelest.',
    'Der Mann hat ein grosse Haus gekauft.',
    'Gestern bin ich mit mein Freund ins Kino gegangen.',
    'Ich weiss nicht ob er heute kommt.',
    'Das Wetter ist heute sehr schön.',
  ],
  es: [
    'Ayer yo fui a la playa con mi amigos.',
    'Me gusta mucho las películas de terror.',
    'Ella es muy cansada hoy.',
    'Cuando era niño, yo vivo en Madrid.',
    'Mañana vamos a visitar a mis abuelos.',
  ],
};

const $ = (id) => document.getElementById(id);
const els = {
  languages: document.querySelectorAll('[data-language]'),
  status: $('model-status'),
  load: $('load'),
  progress: $('progress'),
  bar: $('progress-bar'),
  text: $('text'),
  examples: $('examples'),
  correct: $('correct'),
  results: $('results'),
};

let language = 'de';
let busy = false;
let loading = false;
let controller = null;

// --- model ------------------------------------------------------------------

function setStatus(text) {
  els.status.textContent = text;
}

function describeBackend() {
  const b = backend();
  if (!b) return '';
  const cpu = b.multithread ? `${b.threads} threads` : 'single thread';
  return b.webgpu ? `WebGPU · ${cpu}` : `CPU · ${cpu}`;
}

async function refreshModelCard() {
  const model = MODELS[language];
  // A download in flight finishes first; loadModel() comes back here after.
  if (loading) {
    setStatus('Finishing the current download first…');
    return;
  }
  els.progress.hidden = true;
  if (loadedLanguage() === language) {
    setStatus(`Ready · ${describeBackend()}`);
    els.load.hidden = true;
    return;
  }
  setStatus('Checking for a cached model…');
  const cached = await isCached(language);
  if (cached) {
    els.load.hidden = true;
    loadModel();
  } else {
    setStatus(`The ${model.label} model is a one-time ${model.sizeMB} MB download.`);
    els.load.textContent = `Download ${model.sizeMB} MB`;
    els.load.hidden = false;
  }
  updateButton();
}

async function loadModel() {
  const target = language;
  els.load.hidden = true;
  els.progress.hidden = false;
  els.bar.style.width = '0%';
  setStatus('Loading…');
  busy = true;
  loading = true;
  updateButton();
  try {
    await load(target, (done, total) => {
      if (!total) return;
      const pct = (100 * done) / total;
      els.bar.style.width = `${pct}%`;
      setStatus(
        done < total
          ? `Downloading · ${(done / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`
          : 'Loading into memory…',
      );
    });
  } catch (error) {
    console.error(error);
    setStatus(`Could not load the model: ${error.message ?? error}`);
    els.load.textContent = 'Retry';
    els.load.hidden = false;
  } finally {
    busy = false;
    loading = false;
    els.progress.hidden = true;
  }
  // The language may have changed while this was downloading.
  if (target === language) {
    if (loadedLanguage() === language) setStatus(`Ready · ${describeBackend()}`);
  } else {
    refreshModelCard();
  }
  updateButton();
}

function updateButton() {
  const ready = loadedLanguage() === language;
  els.correct.disabled = !ready || (busy && !controller) || !els.text.value.trim();
  els.correct.textContent = controller ? 'Stop' : 'Correct';
  if (controller) els.correct.disabled = false;
}

// --- language and examples --------------------------------------------------

function setLanguage(next) {
  if (next === language) return;
  language = next;
  els.languages.forEach((b) => b.setAttribute('aria-checked', String(b.dataset.language === language)));
  document.documentElement.lang = language;
  els.text.lang = language;
  renderExamples();
  els.results.replaceChildren();
  refreshModelCard();
}

function renderExamples() {
  els.examples.replaceChildren(
    ...EXAMPLES[language].map((sentence) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip';
      chip.textContent = sentence;
      chip.addEventListener('click', () => {
        els.text.value = sentence;
        updateButton();
        els.text.focus();
      });
      return chip;
    }),
  );
}

// --- correction -------------------------------------------------------------

/** One sentence per item: the model was trained on sentences, not paragraphs. */
function splitSentences(text) {
  return text
    .split(/\n+/)
    .flatMap((line) => line.split(/(?<=[.!?…])\s+(?=[¿¡"„«]?\p{Lu})/u))
    .map((s) => s.trim())
    .filter(Boolean);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** The source with deletions struck through and insertions marked, spaced like prose. */
function renderDiff(runs) {
  const flat = runs.flatMap((run) => run.tokens.map((token) => ({ op: run.op, token })));
  const spaces = spacesBefore(flat.map((f) => f.token));
  const line = el('p', 'diff');
  flat.forEach(({ op, token }, i) => {
    if (spaces[i]) line.append(' ');
    line.append(op === 'same' ? token : el(op, null, token));
  });
  return line;
}

const words = (text) => String(text ?? '').split(' ').filter(Boolean);

/**
 * Which of the model's `changes` entries describe this edit. Usually one,
 * whose `was` contains the deleted tokens and whose `now` the inserted ones --
 * the model anchors with context, so containment rather than equality. When
 * none covers the whole edit, the entries that fit *inside* it: adjacent edits
 * like `yo vivo` → `vivía` are one stretch of diff but two claims (U:PRON,
 * R:VERB:FORM), and each deserves its own explanation.
 *
 * @returns {{wrong: string[], right: string[], type?: string}[]}
 */
function claimsFor(edit, changes) {
  const parsed = changes
    .map((change) => ({ was: words(change.was), now: words(change.now), type: change.type }))
    .filter((c) => c.was.join(' ') !== c.now.join(' '));

  const covering = parsed
    .filter((c) => containsTokens(c.was, edit.wrong) && containsTokens(c.now, edit.right))
    // The tightest anchor is the most specific claim.
    .sort((x, y) => x.was.length - y.was.length);
  if (covering.length) return [{ ...edit, type: covering[0].type }];

  const inside = parsed.filter(
    (c) => (c.was.length || c.now.length) && containsTokens(edit.wrong, c.was) && containsTokens(edit.right, c.now),
  );
  if (inside.length > 1) return inside.map((c) => ({ wrong: c.was, right: c.now, type: c.type }));
  return [edit];
}

/**
 * One line per edit in the diff -- not per entry in the model's `changes`.
 *
 * The CLI re-derives edits with ERRANT because the model's list is the weaker
 * signal; the browser cannot run ERRANT, but it can at least explain exactly
 * the edits the reader sees struck through. The model's list contributes only
 * types, and only where its entries match. Entries that match nothing (an
 * anchor that swallowed the sentence, a change never made) are dropped; edits
 * it never claimed are explained from their shape alone.
 */
function renderChanges(edits, changes) {
  const list = el('ol', 'changes');
  for (const part of edits.flatMap((edit) => claimsFor(edit, changes))) {
    const wrong = detokenize(part.wrong);
    const right = detokenize(part.right);
    const item = el('li');
    const head = el('div', 'change-head');
    if (wrong) head.append(el('del', null, wrong));
    if (wrong && right) head.append(el('span', 'arrow', '→'));
    if (right) head.append(el('ins', null, right));
    if (part.type) head.append(el('code', 'type', part.type));
    const why = part.type
      ? explainType(part.type, wrong, right, language)
      : explainUntyped(wrong, right, language, part.wrong, part.right);
    item.append(head, el('p', 'why', why));
    list.append(item);
  }
  return list;
}

function rawOutput(raw) {
  const details = el('details');
  details.append(el('summary', null, 'Raw model output'), el('pre', null, raw.trim()));
  return details;
}

function renderResult(card, sentence, source, raw, timings) {
  card.replaceChildren();
  const answer = parseAnswer(raw);
  if (!answer) {
    card.append(el('p', 'original', sentence), el('p', 'warning', 'The model’s answer could not be read.'));
    card.append(rawOutput(raw));
    return;
  }

  const sourceTokens = source.split(' ');
  // The answer usually comes back tokenised, but nothing guarantees it.
  const correctionTokens = tokenize(answer.correction).split(' ').filter(Boolean);
  const runs = diffTokens(sourceTokens, correctionTokens);
  const edits = editsFromDiff(runs);

  if (!edits.length) {
    card.classList.add('clean');
    card.append(el('p', 'diff', sentence), el('p', 'verdict', 'No changes. This sentence looks correct.'));
  } else {
    card.append(renderDiff(runs));
    const corrected = el('p', 'corrected');
    corrected.append(el('span', 'label', 'Corrected'), detokenize(correctionTokens));
    card.append(corrected);
    const changes = answer.changes.filter((c) => c && typeof c === 'object');
    card.append(renderChanges(edits, changes));
  }

  if (timings) {
    const seconds = (timings.prompt_ms + timings.predicted_ms) / 1000;
    card.append(
      el(
        'p',
        'meta',
        `${timings.predicted_n} tokens · ${timings.predicted_per_second.toFixed(1)} tok/s · ${seconds.toFixed(2)} s`,
      ),
    );
  }
  card.append(rawOutput(raw));
}

async function run() {
  if (controller) {
    controller.abort();
    return;
  }
  const sentences = splitSentences(els.text.value);
  if (!sentences.length || loadedLanguage() !== language) return;

  controller = new AbortController();
  busy = true;
  updateButton();
  els.results.replaceChildren();

  try {
    for (const sentence of sentences) {
      const card = el('article', 'result pending');
      card.append(el('p', 'original', sentence), el('p', 'streaming', '…'));
      els.results.append(card);

      // The model saw tokenised text in training, so it gets tokenised text here.
      const source = tokenize(sentence);
      const { text, timings } = await generate(source, language, {
        signal: controller.signal,
        onText: (partial) => {
          card.querySelector('.streaming').textContent = partial.trim() || '…';
        },
      });
      card.classList.remove('pending');
      renderResult(card, sentence, source, text, timings);
    }
  } catch (error) {
    console.error('[langlm] correction stopped', error);
    if (error.name !== 'AbortError' && !controller.signal.aborted) {
      els.results.append(el('p', 'warning', `Something went wrong: ${error.message ?? error}`));
    }
    els.results.querySelectorAll('.pending').forEach((card) => card.remove());
  } finally {
    controller = null;
    busy = false;
    updateButton();
  }
}

// --- wiring -----------------------------------------------------------------

els.languages.forEach((b) => b.addEventListener('click', () => setLanguage(b.dataset.language)));
els.load.addEventListener('click', loadModel);
els.correct.addEventListener('click', run);
els.text.addEventListener('input', updateButton);
els.text.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    if (!els.correct.disabled) run();
  }
});

if (!crossOriginIsolated) {
  console.warn('Page is not cross-origin isolated: wllama will run single-threaded.');
}

renderExamples();
refreshModelCard();
