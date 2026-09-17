"""Prompted base-model baselines: the model before it has learned anything.

Two of the three numbers in the Phase 1 table come from here -- zero-shot and
few-shot EuroLLM-1.7B -- and their job is to say what the fine-tuning in Phase 3
is actually worth. A base model is not an instruction-following model, so the
prompt is written as a completion to be continued rather than a request to be
obeyed, which is the format such a model has actually seen in training.

The few-shot examples are drawn from the *train* split and include correct
sentences in the same proportion the corpus has them. A prompt made only of
corrections teaches the model that every sentence contains one, and the
overcorrection number would then measure the prompt rather than the model.

Decoding is greedy. A baseline whose number moves with a sampling seed is not a
baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import cached_property

from langlm.config import load_config
from langlm.data.m2 import M2Sentence

#: Written in English on purpose, per language. The user of this tool is
#: learning the target language, and Phase 2's explanations are English for
#: the same reason; keeping the prompt in the same language as the
#: explanations means the Phase 5 Spanish port changes the examples and the
#: one word naming the language, not the rest of the instruction.
INSTRUCTION = {
    "de": (
        "Correct the German sentence. Fix grammar, spelling, punctuation and word order. "
        "Change nothing else. If the sentence is already correct, repeat it unchanged."
    ),
    "es": (
        "Correct the Spanish sentence. Fix grammar, spelling, punctuation and word order. "
        "Change nothing else. If the sentence is already correct, repeat it unchanged."
    ),
}

#: The completion format. A base model continues a pattern; it does not take
#: orders, so the pattern has to be unmistakable. Neither marker carries a
#: trailing space: the prompt has to end on a token the model has seen at the end
#: of a token boundary, and a dangling space is a different, much rarer one.
SOURCE_PREFIX = "Input:"
TARGET_PREFIX = "Output:"


@dataclass
class PromptedBaseline:
    """A base model prompted to correct one sentence at a time."""

    repo_id: str
    shots: list[tuple[str, str]] = field(default_factory=list)
    max_new_tokens: int = 96
    #: Big batches matter more than they usually would: on an M1 a single
    #: decoding step costs the same whether it serves one sequence or thirty,
    #: because the time goes on moving the weights, not on the arithmetic.
    batch_size: int = 32
    name: str = "prompted"
    #: Which language's instruction `preamble` uses. Defaults to German, so
    #: every existing caller (`config_name="phase1"`) is unchanged; a Spanish
    #: caller passes `language="es"` alongside `config_name="phase1_es"`.
    language: str = "de"
    _model: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)

    @classmethod
    def from_config(
        cls,
        shots: list[tuple[str, str]] | None = None,
        config_name: str = "phase1",
        language: str = "de",
        **overrides,
    ) -> PromptedBaseline:
        settings = load_config(config_name)["baselines"]["model"]
        return cls(
            repo_id=settings["repo_id"],
            shots=shots or [],
            max_new_tokens=settings["max_new_tokens"],
            name="few-shot" if shots else "zero-shot",
            language=language,
            **overrides,
        )

    # --- prompt ------------------------------------------------------------

    @cached_property
    def preamble(self) -> str:
        """The instruction plus the worked examples, which never change."""
        blocks = [INSTRUCTION[self.language]]
        blocks += [
            f"{SOURCE_PREFIX} {source}\n{TARGET_PREFIX} {target}" for source, target in self.shots
        ]
        return "\n\n".join(blocks)

    def build_prompt(self, source: str) -> str:
        return f"{self.preamble}\n\n{SOURCE_PREFIX} {source}\n{TARGET_PREFIX}"

    @staticmethod
    def parse(completion: str, source: str) -> str:
        """Pull the corrected sentence out of the model's continuation.

        A base model does not stop when it has answered; it starts inventing the
        next example. Everything after the first line is that, and a model that
        answers with nothing at all is treated as having left the sentence alone
        rather than as having deleted it.
        """
        first_line = completion.strip().split("\n")[0].strip()
        # Strip a prefix the model echoed back before answering.
        if first_line.startswith(TARGET_PREFIX):
            first_line = first_line[len(TARGET_PREFIX) :].strip()
        return first_line or source

    # --- generation --------------------------------------------------------

    def _load(self) -> tuple:
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # Left padding: with a batch of prompts of different lengths, the
            # continuation has to start at the last position of every row.
            tokenizer.padding_side = "left"
            # bfloat16 rather than float16: measured at 164ms against 276ms per
            # decoding step on this M1, and with the same range as float32 it
            # cannot overflow the way float16 can on a model trained in bf16.
            model = AutoModelForCausalLM.from_pretrained(
                self.repo_id, dtype=torch.bfloat16 if device == "mps" else torch.float32
            )
            model.to(device).eval()
            self._tokenizer, self._model = tokenizer, model
        return self._tokenizer, self._model

    def correct(self, source: str) -> str:
        return self.correct_many([source])[0]

    def correct_many(self, sources: list[str], progress: bool = True) -> list[str]:
        """Correct a corpus in batches."""
        import torch

        tokenizer, model = self._load()
        outputs: list[str] = []

        for start in range(0, len(sources), self.batch_size):
            batch = sources[start : start + self.batch_size]
            prompts = [self.build_prompt(source) for source in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=[_StopOnNewline(tokenizer, encoded["input_ids"].shape[1])],
                )
            width = encoded["input_ids"].shape[1]
            completions = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
            outputs.extend(
                self.parse(completion, source)
                for completion, source in zip(completions, batch, strict=True)
            )
            if progress:
                print(f"  {min(start + self.batch_size, len(sources))}/{len(sources)}", end="\r")

        if progress:
            print()
        return outputs


class _StopOnNewline:
    """Stop a row once it has finished its line.

    The answer is one sentence, but a base model does not know that: left to
    itself it writes the answer and then invents the next example, and every
    token of that is paid for. Only the corpus-sized cost makes this worth a
    custom criterion -- it is most of the generation budget.
    """

    def __init__(self, tokenizer, prompt_width: int, newline: str = "\n") -> None:
        self.tokenizer = tokenizer
        self.prompt_width = prompt_width
        self.newline = newline

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        completions = input_ids[:, self.prompt_width :]
        if completions.shape[1] == 0:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        # Decoding is cheaper than it looks and far cheaper than the extra
        # decoding steps it saves; comparing token ids would miss a newline that
        # arrives glued to the end of a word piece.
        texts = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        return torch.tensor(
            [self.newline in text for text in texts],
            dtype=torch.bool,
            device=input_ids.device,
        )


def sample_shots(
    sentences: list[M2Sentence],
    count: int,
    seed: int,
    correct_share: float = 0.22,
) -> list[tuple[str, str]]:
    """Pick worked examples from a training split.

    Args:
        sentences: Training sentences to draw from -- never dev or test.
        count: How many examples the prompt should carry.
        seed: Sampling seed, so the prompt is the same on every run.
        correct_share: Fraction of examples that need no correction. The default
            is the measured Falko-MERLIN rate, so the prompt tells the model the
            truth about how often German learner text is already right.

    Returns:
        ``(source, target)`` pairs.
    """
    rng = random.Random(seed)
    correct = [s for s in sentences if s.is_correct()]
    erroneous = [s for s in sentences if not s.is_correct()]

    wanted_correct = round(count * correct_share)
    chosen = rng.sample(correct, wanted_correct) + rng.sample(erroneous, count - wanted_correct)
    rng.shuffle(chosen)
    return [(sentence.source, sentence.target()) for sentence in chosen]
