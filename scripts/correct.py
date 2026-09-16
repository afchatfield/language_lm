"""Send sentences to a running llama-server and print the corrections.

The prompt is built by `langlm.train.format.build_prompt`, imported rather than
retyped: the model was fine-tuned on exactly that string and answers a chat
template with English prose about the instruction instead of German corrections.

Input is re-tokenised the way the corpus is before it reaches the model, and
the answer is detokenised again for display. Without that, `Hund?` against the
model's `Hund ?` scores -- and explains -- as an edit the learner never made.

Explanations come from `explain_correction`, which re-derives the edits with
ERRANT rather than trusting the model's own `changes` list -- the docstring
there has the numbers. When ERRANT is not importable the script falls back to
the model's list, which is worse but still typed.

    llama-server -m checkpoints/gguf/langlm-de-755mb.gguf --port 8080 -c 1024 -ngl 99
    python scripts/correct.py "Ich habe gestern ein Buch gelest ."
    echo "Der Mann hat ein grosse Haus gekauft ." | python scripts/correct.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

from langlm.train.format import build_prompt, parse_answer


#: Space before closing punctuation, and after opening punctuation. Display
#: only: every comparison downstream runs on the tokenised form.
_BEFORE = re.compile(r"\s+([.,;:!?%\u00bb\u201c\)\]\u2026])")
_AFTER = re.compile(r"([\u00ab\u201e\(\[])\s+")


def tokenize(text: str) -> str:
    """Corpus tokenisation, spacy where available.

    The fallback splits trailing punctuation off with a regex. It is not the
    same tokenizer and will disagree on abbreviations and decimals, so it warns
    rather than passing silently for the real thing.
    """
    try:
        from langlm.eval import errant_de

        return errant_de.tokenize(text)
    except Exception:
        spaced = re.sub(r"([.,;:!?\u2026])(\s|$)", r" \1\2", text)
        return " ".join(spaced.split())


def _straight_quotes(text: str) -> str:
    """Close up spacing around `"`, which is its own opening and closing mark.

    The directional marks are handled by the two patterns above; this one can
    only be read by counting, so quotes alternate: odd opens, even closes.
    """
    parts = text.split('"')
    result = parts[0]
    for index, part in enumerate(parts[1:], start=1):
        if index % 2:
            result = f'{result}"{part.lstrip()}'
        else:
            result = f'{result.rstrip()}"{part}'
    return result


def detokenize(text: str) -> str:
    """Put the punctuation back for a human to read."""
    return _straight_quotes(_AFTER.sub(r"\1", _BEFORE.sub(r"\1", text))).strip()


def correct(sentence: str, url: str, n_predict: int) -> dict:
    """One sentence to the server. Greedy, because every eval in the repo was."""
    body = json.dumps(
        {"prompt": build_prompt(sentence), "n_predict": n_predict, "temperature": 0}
    ).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    try:
        payload = json.load(urllib.request.urlopen(request))
    except urllib.error.URLError as error:
        raise SystemExit(f"No server at {url}: {error.reason}. Start llama-server first.")
    if "content" not in payload:
        raise SystemExit(f"Server error: {payload.get('error', payload)}")
    return payload


def explanations(source: str, correction: str, changes: list[dict]) -> list[tuple[str, str, str]]:
    """(wrong, right, why) per edit, ERRANT-derived where possible."""
    try:
        from langlm.explanations import explain_correction, explain_type
    except ImportError:
        return []
    try:
        return [(e.wrong, e.right, e.explanation) for e in explain_correction(source, correction)]
    except Exception:
        # ERRANT needs its German model; the model's own list still carries types.
        return [
            (c.get("was", ""), c.get("now", ""), explain_type(c.get("type", ""), c.get("was", ""), c.get("now", "")))
            for c in changes
            if isinstance(c, dict)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sentences", nargs="*", help="German sentences; omit to read stdin")
    parser.add_argument("--url", default="http://127.0.0.1:8080/completion")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--json", action="store_true", help="Raw model answer, one per line")
    args = parser.parse_args()

    sentences = args.sentences or [line.strip() for line in sys.stdin if line.strip()]
    if not sentences:
        raise SystemExit("Nothing to correct.")

    for sentence in sentences:
        # The model saw tokenised German in training and the scorer counts
        # tokens, so both sides are tokenised before anything compares them.
        source = tokenize(sentence)
        payload = correct(source, args.url, args.max_new_tokens)
        if args.json:
            print(payload["content"].strip())
            continue

        answer = parse_answer(payload["content"])
        if answer is None:
            print(f"  in  {sentence}\n  !!  unparseable: {payload['content'].strip()[:120]}")
            continue

        # The answer usually comes back tokenised, but nothing guarantees it.
        correction = tokenize(answer.correction)

        print(f"  in  {sentence}")
        print(f"  out {detokenize(correction)}")
        if not answer.complete:
            print("  !!  answer truncated; raise --max-new-tokens")
        for wrong, right, why in explanations(source, correction, answer.changes):
            print(f"      {wrong} -> {right}: {why}")
        if correction.split() == source.split():
            print("      (no change)")
        print()


if __name__ == "__main__":
    main()
