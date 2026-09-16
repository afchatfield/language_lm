"""The fine-tuned model, scored the same way as every baseline before it.

The model answers with the corrected sentence and a list of what it changed. The
sentence is what the Phase 1 machinery scores, and it needs nothing done to it
first -- the MaxMatch scorer and German ERRANT both take corrected sentences, so
the number this produces is comparable with `reports/phase1/baselines.md`
because it came out of the same scorer, on the same kind of input, not a new one
written for the occasion.

The `changes` list is not on that path. It says which rewrites the model believes
it made and what kind of error each one was, and it is scored separately, against
the edits ERRANT derives from the model's own correction. That separation is the
point: a wrong or missing explanation can no longer damage a correction score,
and a good correction can no longer hide behind an explanation that sounds right.

Two ways the model can fail, counted separately because they mean different
things:

* **No sentence at all.** Nothing in the answer parses as a correction. Scored
  as an unchanged sentence, but reported, because a silent failure that scores
  like restraint is the worst outcome.
* **Cut off.** The generation cap landed mid-answer. The sentence comes first in
  the schema, so it usually survives and only the `changes` list is lost; those
  answers are counted so the explanation metrics can say what they are missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langlm.train.format import CHANGES_MARKER, build_prompt, parse_answer


@dataclass
class GenerationFaults:
    """How often the model failed to answer in the schema."""

    invalid_json: int = 0
    recovered: int = 0
    unchanged: int = 0
    total: int = 0

    def __str__(self) -> str:
        return (
            f"{self.invalid_json:,} with no readable correction, {self.recovered:,} cut off but "
            f"recovered, {self.unchanged:,} left unchanged"
        )


@dataclass
class FineTunedBaseline:
    """A LoRA-adapted model answering in the Phase 3 schema."""

    repo_id: str
    adapter_path: str | None = None
    #: Where the tokenizer comes from, when it must not come from the adapter.
    #: The adapter directory carries the tokenizer it was trained with, which is
    #: the right default and the wrong one for a vocabulary-trimmed base: the
    #: adapter's copy still has all 128,000 rows, and pairing it with trimmed
    #: weights mismatches every id above the trim point without raising.
    tokenizer_path: str | None = None
    #: Long enough for the longest answer the training cap admits. The sentence
    #: is first in the schema, so overrunning this costs the explanations rather
    #: than the correction.
    max_new_tokens: int = 1024
    batch_size: int = 32
    name: str = "fine-tuned"
    #: Stop as soon as the correction is closed. A median answer is 84 tokens
    #: and its correction 28, so this is most of the generation time -- at the
    #: cost of the `changes` list, and so of the explanation metrics.
    scored_only: bool = False
    #: How many times to run the model over its own output. The model enters
    #: 74% of dev's sentences and then under-edits inside them -- only 4.0% of
    #: gold edits sit in a sentence it never touched -- so a second pass is
    #: aimed at the misses that a first pass leaves behind rather than at the
    #: sentences it declined. Round `n` is given round `n-1`'s correction as its
    #: input, and a sentence that stops changing is left alone from then on.
    rounds: int = 1
    #: Per round, how many sentences that round rewrote. Index 0 is round 1.
    rewrites_per_round: list[int] = field(default_factory=list)
    faults: GenerationFaults = field(default_factory=GenerationFaults)
    #: Raw model output, kept so the report can quote it.
    answers: list[str] = field(default_factory=list)
    #: Per sentence, what the model says it changed. Aligned with the input.
    claims: list[list[dict[str, Any]]] = field(default_factory=list)
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
                tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path or self.adapter_path or self.repo_id
                )
            except (OSError, ValueError):
                tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path or self.repo_id)
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
        """Correct a corpus, returning the corrected sentences.

        With :attr:`rounds` above 1 the model is run over its own output. Only
        the last pass over a given sentence fills `claims` and `faults`, so
        those stay aligned with the input and describe the answer that is
        actually returned -- but note that a later round's `changes` list
        describes the hop it made, not the whole edit from the original source.
        For multi-round output the explanation path is ERRANT over
        (original source, final correction), which is what
        `langlm.explanations` already uses.
        """
        if self.rounds <= 1:
            return self._one_pass(sources, progress=progress)

        current = list(sources)
        settled = [False] * len(sources)
        for round_number in range(self.rounds):
            pending = [i for i, done in enumerate(settled) if not done]
            if not pending:
                self.rewrites_per_round.append(0)
                continue
            if progress:
                print(f"  round {round_number + 1}: {len(pending):,} sentences")
            # Counters describe the answer returned, so each round overwrites
            # the last one's rather than appending to it.
            self.faults, self.answers, self.claims = GenerationFaults(), [], []
            answered = self._one_pass([current[i] for i in pending], progress=progress)

            rewritten = 0
            for position, answer in zip(pending, answered, strict=True):
                if answer.strip() == current[position].strip():
                    settled[position] = True
                else:
                    rewritten += 1
                current[position] = answer
            self.rewrites_per_round.append(rewritten)

        # `claims` and `faults` came from the final pass, which only saw the
        # unsettled sentences. Re-align them with the full input so the caller
        # can zip them against it.
        self._realign(sources, settled)
        return current

    def _realign(self, sources: list[str], settled: list[bool]) -> None:
        """Spread the last round's per-sentence records back over every input."""
        pending = [i for i, done in enumerate(settled) if not done]
        if len(self.claims) != len(pending):
            return
        claims: list[list[dict[str, Any]]] = [[] for _ in sources]
        answers: list[str] = [""] * len(sources)
        for slot, position in enumerate(pending):
            claims[position] = self.claims[slot]
            if slot < len(self.answers):
                answers[position] = self.answers[slot]
        self.claims, self.answers = claims, answers
        self.faults.total = len(sources)
        self.faults.unchanged += len(sources) - len(pending)

    def _one_pass(self, sources: list[str], progress: bool = True) -> list[str]:
        """One generation over every sentence given."""
        import torch

        tokenizer, model = self._load()
        stopping = (
            {"stop_strings": [CHANGES_MARKER], "tokenizer": tokenizer} if self.scored_only else {}
        )
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
                    **stopping,
                )
            width = encoded["input_ids"].shape[1]
            answers = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
            for answer, source in zip(answers, batch, strict=True):
                corrected.append(self._read(answer, source))
            if progress:
                print(f"  {min(start + self.batch_size, len(sources))}/{len(sources)}", end="\r")

        if progress:
            print()
        return corrected

    def _read(self, answer: str, source: str) -> str:
        self.faults.total += 1
        self.answers.append(answer.strip())
        parsed = parse_answer(answer.strip())
        if parsed is None:
            # Nothing readable: treat as "changed nothing", but count it,
            # because a silent failure that scores like restraint is the worst
            # outcome.
            self.faults.invalid_json += 1
            self.claims.append([])
            return source
        if not parsed.complete and not self.scored_only:
            self.faults.recovered += 1
        if parsed.correction.strip() == source.strip():
            self.faults.unchanged += 1
        self.claims.append([c for c in parsed.changes if isinstance(c, dict)])
        return parsed.correction

    def correct(self, source: str) -> str:
        return self.correct_many([source], progress=False)[0]
