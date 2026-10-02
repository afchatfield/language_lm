// The prompt, the answer parser and the tokeniser, ported from Python.
//
// The model was fine-tuned on exactly the string `buildPrompt` makes, so this
// file is a translation of `langlm.train.format` and `scripts/correct.py`, not
// a reinterpretation of them. If either changes, this has to change with it.

/** `langlm.train.format.INSTRUCTION` */
const INSTRUCTION = {
  de: 'Correct the German sentence, then list what changed.',
  es: 'Correct the Spanish sentence, then list what changed.',
};

/** `langlm.train.format.build_prompt` */
export function buildPrompt(source, language) {
  return `${INSTRUCTION[language]}\n\nInput: ${source}\nOutput:`;
}

const CORRECTION = /"correction"\s*:\s*"((?:[^"\\]|\\.)*)"/;

/**
 * `langlm.train.format.parse_answer`: whole-answer JSON first, then the
 * correction field alone. The sentence comes first in the schema so that a
 * cut-off answer still has it, and only the explanations are lost.
 *
 * @returns {{correction: string, changes: object[], complete: boolean} | null}
 */
export function parseAnswer(text) {
  text = text.trim();
  let payload = null;
  try {
    payload = JSON.parse(text);
  } catch {
    // fall through to the field on its own
  }
  if (payload && typeof payload === 'object' && typeof payload.correction === 'string') {
    const changes = Array.isArray(payload.changes) ? payload.changes : [];
    return { correction: payload.correction, changes, complete: true };
  }

  const found = CORRECTION.exec(text);
  if (!found) return null;
  try {
    return { correction: JSON.parse(`"${found[1]}"`), changes: [], complete: false };
  } catch {
    return null;
  }
}

/**
 * The regex fallback from `scripts/correct.py`. The corpus was tokenised with
 * spaCy, which the browser does not have; this splits trailing punctuation off
 * and disagrees with spaCy on abbreviations and decimals.
 */
export function tokenize(text) {
  return text
    .replace(/([.,;:!?…])(\s|$)/g, ' $1$2')
    .split(/\s+/)
    .filter(Boolean)
    .join(' ');
}

const CLOSING = /^[.,;:!?%»“)\]…]$/;
const OPENING = /^[«„([]$/;

/**
 * Whether each token is written with a space before it -- `detokenize` from
 * `scripts/correct.py`, but per token, so a diff of two token sequences can be
 * laid out with the same spacing a reader expects. Straight quotes are their
 * own opening and closing mark and alternate.
 */
export function spacesBefore(tokens) {
  let quotes = 0;
  return tokens.map((token, i) => {
    if (i === 0) {
      if (token === '"') quotes += 1;
      return false;
    }
    const previous = tokens[i - 1];
    if (token === '"') {
      quotes += 1;
      // An opening quote takes a space; a closing one does not.
      return quotes % 2 === 1 && !OPENING.test(previous);
    }
    if (CLOSING.test(token)) return false;
    if (OPENING.test(previous)) return false;
    if (previous === '"' && quotes % 2 === 1) return false;
    return true;
  });
}

export function detokenize(tokens) {
  const spaces = spacesBefore(tokens);
  return tokens.map((t, i) => (spaces[i] ? ' ' : '') + t).join('');
}

/**
 * Token-level diff (LCS), for display only. Returns runs of
 * `{op: 'same'|'del'|'ins', tokens}` in reading order, deletions before the
 * insertions that replace them.
 */
export function diffTokens(a, b) {
  const n = a.length;
  const m = b.length;
  const table = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      table[i][j] = a[i] === b[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }

  const ops = [];
  const push = (op, token) => {
    const last = ops[ops.length - 1];
    if (last && last.op === op) last.tokens.push(token);
    else ops.push({ op, tokens: [token] });
  };
  let i = 0;
  let j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) {
      push('same', a[i]);
      i++;
      j++;
    } else if (j >= m || (i < n && table[i + 1][j] >= table[i][j + 1])) {
      push('del', a[i]);
      i++;
    } else {
      push('ins', b[j]);
      j++;
    }
  }

  // Put each deletion ahead of the insertion it pairs with, so a replacement
  // reads "old new" rather than interleaving.
  for (let k = 0; k + 1 < ops.length; k++) {
    if (ops[k].op === 'ins' && ops[k + 1].op === 'del') {
      [ops[k], ops[k + 1]] = [ops[k + 1], ops[k]];
    }
  }
  return ops;
}

/**
 * The edits in a diff: each maximal stretch of deletions and insertions
 * between unchanged tokens, as `{wrong, right}` token arrays. Either side may
 * be empty -- an insertion has no `wrong`, a deletion no `right`.
 */
export function editsFromDiff(runs) {
  const edits = [];
  let current = null;
  for (const run of runs) {
    if (run.op === 'same') {
      current = null;
      continue;
    }
    if (!current) {
      current = { wrong: [], right: [] };
      edits.push(current);
    }
    (run.op === 'del' ? current.wrong : current.right).push(...run.tokens);
  }
  return edits;
}

/** Whether `needle` occurs as a contiguous run inside `haystack`. */
export function containsTokens(haystack, needle) {
  if (!needle.length) return true;
  for (let i = 0; i + needle.length <= haystack.length; i++) {
    if (needle.every((token, k) => haystack[i + k] === token)) return true;
  }
  return false;
}
