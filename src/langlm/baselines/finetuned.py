"""The fine-tuned model, scored the same way as every baseline before it.

The difference from the prompted baselines is the shape of the answer. Those
returned a corrected sentence; this returns a list of span-based edits as JSON,
which has to be parsed and applied before any of the Phase 1 scoring machinery
can see it. Everything downstream of that is identical, deliberately -- the
number this produces is comparable with `reports/phase1/baselines.md` because it
came out of the same scorer, not a new one written for the occasion.

Three ways the model can fail, and they are counted separately because they mean
different things:

* **Invalid JSON.** The answer is not the schema at all. Scored as an unchanged
  sentence, but reported, because an unparseable answer that scores like
  restraint would otherwise be indistinguishable from correctly declining to
  change a correct sentence.
* **Out-of-range spans.** Valid JSON pointing at tokens that do not exist.
  Dropped edit by edit.
* **Overlapping spans.** Two edits claiming the same tokens. The first wins.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from langlm.train.format import build_prompt, parse_answer


@dataclass
class GenerationFaults:
    """How often the model failed to answer in the schema."""

    invalid_json: int = 0
    out_of_range: int = 0
    overlapping: int = 0
    empty: int = 0
    total: int = 0

    def __str__(self) -> str:
        return (
            f"{self.invalid_json:,} invalid JSON, {self.out_of_range:,} out-of-range spans, "
            f"{self.overlapping:,} overlapping, {self.empty:,} empty edit lists"
        )


def apply_edits(tokens: Sequence[str], edits: list[dict], faults: GenerationFaults) -> str:
    """Apply the model's edits to the source, left to right.

    Invalid edits are dropped rather than raising: a model that emits a bad span
    has made a mistake on that edit, not on the sentence, and the rest of its
    answer is still worth scoring.
    """
    usable = []
    for edit in edits:
        try:
            start, end = int(edit["start"]), int(edit["end"])
        except (KeyError, TypeError, ValueError):
            faults.out_of_range += 1
            continue
        if not 0 <= start <= end <= len(tokens):
            faults.out_of_range += 1
            continue
        usable.append((start, end, str(edit.get("correction", ""))))

    usable.sort(key=lambda item: item[0])
    result: list[str] = []
    cursor = 0
    for start, end, correction in usable:
        if start < cursor:
            faults.overlapping += 1
            continue
        result.extend(tokens[cursor:start])
        result.extend(correction.split())
        cursor = end
    result.extend(tokens[cursor:])
    return " ".join(result)


@dataclass
class FineTunedBaseline:
    """A LoRA-adapted model answering in the Phase 3 edit schema."""

    repo_id: str
    adapter_path: str | None = None
    max_new_tokens: int = 320
    batch_size: int = 32
    name: str = "fine-tuned"
    faults: GenerationFaults = field(default_factory=GenerationFaults)
    #: Raw model output, kept so the report can quote explanations.
    answers: list[str] = field(default_factory=list)
    _model: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)

    def _load(self) -> tuple:
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if torch.cuda.is_available():
                device, dtype = "cuda", torch.bfloat16
            elif torch.backends.mps.is_available():
                device, dtype = "mps", torch.bfloat16
            else:
                device, dtype = "cpu", torch.float32

            # The adapter directory normally carries its own tokenizer, saved
            # alongside it by the trainer. Falling back to the base model's is
            # not cosmetic: an adapter from anywhere else -- a checkpoint dir, a
            # colleague, a hub download -- may not have one, and the error when
            # it does not names neither the adapter nor the fix.
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.adapter_path or self.repo_id)
            except (OSError, ValueError):
                tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = AutoModelForCausalLM.from_pretrained(self.repo_id, dtype=dtype)
            if self.adapter_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter_path)
                model = model.merge_and_unload()
            model.to(device).eval()
            self._tokenizer, self._model = tokenizer, model
        return self._tokenizer, self._model

    def correct_many(self, sources: list[str], progress: bool = True) -> list[str]:
        """Correct a corpus, returning the corrected sentences."""
        import torch

        tokenizer, model = self._load()
        corrected: list[str] = []

        for start in range(0, len(sources), self.batch_size):
            batch = sources[start : start + self.batch_size]
            prompts = [build_prompt(source) for source in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            width = encoded["input_ids"].shape[1]
            answers = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
            for answer, source in zip(answers, batch, strict=True):
                corrected.append(self._apply(answer, source))
            if progress:
                print(f"  {min(start + self.batch_size, len(sources))}/{len(sources)}", end="\r")

        if progress:
            print()
        return corrected

    def _apply(self, answer: str, source: str) -> str:
        self.faults.total += 1
        self.answers.append(answer.strip())
        edits = parse_answer(answer.strip())
        if edits is None:
            # Unparseable: treat as "changed nothing", but count it, because a
            # silent failure that scores like restraint is the worst outcome.
            self.faults.invalid_json += 1
            return source
        if not edits:
            self.faults.empty += 1
            return source
        return apply_edits(source.split(), edits, self.faults)

    def correct(self, source: str) -> str:
        return self.correct_many([source], progress=False)[0]
