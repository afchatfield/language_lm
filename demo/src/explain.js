// `langlm.explanations.explain_type`, reading the same YAML files.
//
// The Python CLI re-derives each edit with ERRANT before explaining it, which
// is right on the type 89% of the time against 71% for the model's own
// `changes` list. ERRANT needs spaCy, so the browser takes the CLI's fallback
// path instead: the model's list, typed by the model.

import yaml from 'js-yaml';
import deTypes from '../../rules/de_types.yaml?raw';
import esTypes from '../../rules/es_types.yaml?raw';

const TEMPLATES = {
  de: yaml.load(deTypes),
  es: yaml.load(esTypes),
};

/** Python's `str.format` for the named placeholders these files use. */
function format(template, values) {
  return template.replace(/\{(\w+)\}/g, (whole, key) => (key in values ? values[key] : whole)).trim();
}

export function explainType(errorType, wrong, right, language) {
  const templates = TEMPLATES[language];

  const explicit = templates.explicit?.[errorType];
  if (explicit) return format(explicit, { wrong, right });

  const parts = String(errorType ?? '').split(':');
  const [operation, categoryKey] = parts.length > 1 ? [parts[0], parts[1]] : ['R', 'OTHER'];
  const category = templates.categories?.[categoryKey];
  if (category === undefined) return format(templates.fallback, { wrong, right });

  const key = parts.length > 2 && parts[2] === 'FORM' ? 'R:FORM' : operation;
  const template = templates.operations[key] ?? templates.operations.R;
  return format(template, { wrong, right, category });
}

const PUNCT = /^\p{P}+$/u;

/**
 * An edit the model did not type. The operation is plain from its shape, so a
 * missing or extra word still gets the M/U template; a replacement gets the
 * fallback, which claims nothing -- `R:OTHER` would assert "not the right
 * word", which is a type we do not have.
 */
export function explainUntyped(wrong, right, language, wrongTokens, rightTokens) {
  const tokens = [...wrongTokens, ...rightTokens];
  const category = tokens.length && tokens.every((t) => PUNCT.test(t)) ? 'PUNCT' : 'OTHER';
  if (!wrongTokens.length) return explainType(`M:${category}`, wrong, right, language);
  if (!rightTokens.length) return explainType(`U:${category}`, wrong, right, language);
  if (category === 'PUNCT') return explainType('R:PUNCT', wrong, right, language);
  return format(TEMPLATES[language].fallback, { wrong, right });
}
