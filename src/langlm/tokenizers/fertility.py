"""Tokenizer fertility measurement and vocabulary-trim projection.

Phase 0 has to choose a base model with a number behind it. Two numbers matter,
and they pull in opposite directions:

**Fertility** -- tokens emitted per whitespace word. A tokenizer that shatters
German compounds into five pieces makes every sequence longer, which costs
context, training time, and inference latency on an M1. Lower is better.

**Vocabulary headroom** -- what fraction of the vocabulary German and English
text never touches. Those rows can be sliced out of the embedding matrix and the
LM head in Phase 4 at no quality cost, and for a 1-2B model the embeddings are a
startling share of the parameters. A tokenizer with a huge multilingual
vocabulary scores badly on nothing here and well on trimmability.

This module measures both from the tokenizer alone -- no model weights are
downloaded -- and converts the second into megabytes so the two can be traded
off explicitly.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

#: Byte-fallback tokens look like ``<0x41>``. They must survive any trim, or the
#: tokenizer loses its ability to encode arbitrary text.
BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


#: How many texts to hand a fast tokenizer at once. Large enough to amortise the
#: Rust round-trip, small enough that the returned id lists stay in cache.
BATCH_SIZE = 2000


class TokenizerLike(Protocol):
    """The slice of the Hugging Face tokenizer API this module relies on.

    Declared explicitly so the tests can exercise the arithmetic against a stub
    rather than downloading a real tokenizer.
    """

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...
    def get_vocab(self) -> dict[str, int]: ...

    @property
    def all_special_ids(self) -> list[int]: ...


def encode_many(tokenizer: TokenizerLike, texts: list[str]) -> list[list[int]]:
    """Encode a batch of texts, using the fast tokenizer's batch path if present.

    Hugging Face fast tokenizers encode a whole list in one call into Rust,
    which is roughly two orders of magnitude quicker than a Python loop over
    :meth:`encode`. Objects that only implement ``encode`` -- the test stub, for
    one -- fall back to the loop.
    """
    if callable(tokenizer):
        try:
            return tokenizer(texts, add_special_tokens=False)["input_ids"]
        except (TypeError, KeyError):  # pragma: no cover - non-HF tokenizer
            pass
    return [tokenizer.encode(text, add_special_tokens=False) for text in texts]


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to turn a coverage fraction into a size saving.

    Attributes:
        name: Short display name.
        repo_id: Hugging Face repository id.
        params: Total parameter count of the model.
        vocab_size: Vocabulary size from the model config (not the tokenizer --
            configs are frequently padded above the tokenizer's vocabulary, and
            it is the config's rows that actually occupy memory).
        hidden_size: Embedding dimension.
        num_layers: Transformer block count, recorded for the Phase 4 depth-prune.
        tie_word_embeddings: If False, input embeddings and the LM head are
            separate matrices and every trimmed row saves twice as much.
        note: Free-text commentary carried into the report.
    """

    name: str
    repo_id: str
    params: int
    vocab_size: int
    hidden_size: int
    num_layers: int
    tie_word_embeddings: bool
    note: str = ""

    @property
    def embedding_matrices(self) -> int:
        """How many vocab-by-hidden matrices the model carries: 1 if tied, else 2."""
        return 1 if self.tie_word_embeddings else 2

    @property
    def embedding_params(self) -> int:
        """Parameters held in the embedding table plus (if untied) the LM head."""
        return self.vocab_size * self.hidden_size * self.embedding_matrices

    @property
    def embedding_share(self) -> float:
        """Embedding parameters as a fraction of the whole model."""
        return self.embedding_params / self.params


@dataclass
class CorpusStats:
    """Tokenization statistics for one tokenizer over one corpus."""

    corpus: str
    lang: str
    sentences: int = 0
    words: int = 0
    chars: int = 0
    bytes: int = 0
    tokens: int = 0
    multi_token_words: int = 0
    #: token id -> number of occurrences. Drives the coverage curve.
    token_counts: Counter[int] = field(default_factory=Counter)

    @property
    def fertility(self) -> float:
        """Tokens per whitespace-delimited word. The primary metric."""
        return self.tokens / self.words if self.words else 0.0

    @property
    def chars_per_token(self) -> float:
        return self.chars / self.tokens if self.tokens else 0.0

    @property
    def bytes_per_token(self) -> float:
        """Bytes per token: a cross-check that does not depend on how a
        "word" is defined, which matters when comparing across languages."""
        return self.bytes / self.tokens if self.tokens else 0.0

    @property
    def split_word_rate(self) -> float:
        """Fraction of words that the tokenizer breaks into more than one token."""
        return self.multi_token_words / self.words if self.words else 0.0

    @property
    def unique_tokens(self) -> int:
        return len(self.token_counts)


@dataclass(frozen=True)
class TrimProjection:
    """What trimming the vocabulary to a coverage threshold would buy."""

    threshold: float
    kept_vocab: int
    original_vocab: int
    removed_params: int
    removed_bytes: int
    #: Removed embedding parameters as a fraction of the whole model.
    model_fraction: float

    @property
    def removed_vocab(self) -> int:
        return self.original_vocab - self.kept_vocab

    @property
    def kept_fraction(self) -> float:
        return self.kept_vocab / self.original_vocab if self.original_vocab else 0.0

    @property
    def removed_mb(self) -> float:
        return self.removed_bytes / 1e6


@dataclass
class FertilityResult:
    """One model's complete Phase 0 tokenizer analysis."""

    spec: ModelSpec
    corpora: dict[str, CorpusStats] = field(default_factory=dict)
    projections: list[TrimProjection] = field(default_factory=list)

    def fertility_for(self, lang: str) -> float:
        """Mean fertility across every corpus in a language, weighted by words."""
        stats = [s for s in self.corpora.values() if s.lang == lang]
        words = sum(s.words for s in stats)
        tokens = sum(s.tokens for s in stats)
        return tokens / words if words else 0.0


def analyse_corpus(
    tokenizer: TokenizerLike,
    sentences: Iterable[str],
    corpus: str,
    lang: str,
    max_sentences: int | None = None,
) -> CorpusStats:
    """Tokenize a corpus and accumulate fertility and token-usage statistics.

    Sentences are encoded without special tokens: BOS/EOS are a per-sequence
    constant that would distort a per-word ratio.

    The per-word split rate is computed by encoding each *distinct* word once
    and weighting by how often it occurs, rather than re-encoding every
    occurrence -- a 100k-sentence corpus holds around two million word tokens
    but only a few hundred thousand distinct words. Each word is encoded with a
    leading space, because byte-level BPE treats ``" Haus"`` and ``"Haus"`` as
    different tokens and mid-sentence is the overwhelmingly common case.

    Args:
        tokenizer: Any object satisfying :class:`TokenizerLike`.
        sentences: The corpus, one sentence per item.
        corpus: Corpus key, recorded on the result.
        lang: ISO language code, recorded on the result.
        max_sentences: Stop after this many sentences.

    Returns:
        The accumulated :class:`CorpusStats`.
    """
    stats = CorpusStats(corpus=corpus, lang=lang)
    word_counts: Counter[str] = Counter()

    batch: list[str] = []

    def flush() -> None:
        if not batch:
            return
        for sentence, ids in zip(batch, encode_many(tokenizer, batch), strict=True):
            words = sentence.split()
            stats.sentences += 1
            stats.words += len(words)
            stats.chars += len(sentence)
            stats.bytes += len(sentence.encode("utf-8"))
            stats.tokens += len(ids)
            stats.token_counts.update(ids)
            word_counts.update(words)
        batch.clear()

    for index, sentence in enumerate(sentences):
        if max_sentences is not None and index >= max_sentences:
            break
        sentence = sentence.strip()
        if not sentence:
            continue
        batch.append(sentence)
        if len(batch) >= BATCH_SIZE:
            flush()
    flush()

    stats.multi_token_words = _count_split_words(tokenizer, word_counts)
    return stats


def _count_split_words(tokenizer: TokenizerLike, word_counts: Counter[str]) -> int:
    """Total occurrences of words the tokenizer breaks into more than one token."""
    distinct = list(word_counts)
    split_occurrences = 0
    for start in range(0, len(distinct), BATCH_SIZE):
        chunk = distinct[start : start + BATCH_SIZE]
        encoded = encode_many(tokenizer, [f" {word}" for word in chunk])
        for word, ids in zip(chunk, encoded, strict=True):
            if len(ids) > 1:
                split_occurrences += word_counts[word]
    return split_occurrences


def byte_level_alphabet() -> set[str]:
    """The 256 printable stand-ins GPT-2-style byte-level BPE maps raw bytes to.

    Kept behind a function with a fallback so this module stays importable
    without the ``tokenizers`` package, which the test stub does not need.
    """
    try:
        from tokenizers.pre_tokenizers import ByteLevel

        return set(ByteLevel.alphabet())
    except ImportError:  # pragma: no cover - tokenizers ships with transformers
        return set()


def protected_ids(tokenizer: TokenizerLike) -> set[int]:
    """Token ids that must survive any vocabulary trim.

    A trimmed tokenizer must still be able to encode *any* input, including
    characters the target languages never use. Two mechanisms provide that, and
    exactly one of them is present in a given tokenizer:

    * SentencePiece-style ``<0xNN>`` byte-fallback tokens (EuroLLM has all 256).
    * A GPT-2-style byte-level alphabet of 256 single-character tokens standing
      in for raw bytes (Qwen has exactly these and nothing else single-char).

    Protecting *every* single-character token instead would be far too
    generous: EuroLLM's vocabulary holds 8,285 of them, mostly CJK ideographs
    that byte fallback already covers, and keeping those 8,000 rows would
    understate the achievable trim by several percent of the model.
    """
    alphabet = byte_level_alphabet()
    keep = set(getattr(tokenizer, "all_special_ids", []))
    for token, token_id in tokenizer.get_vocab().items():
        if BYTE_FALLBACK_RE.match(token) or (len(token) == 1 and token in alphabet):
            keep.add(token_id)
    return keep


def coverage_vocab_size(
    token_counts: Counter[int],
    threshold: float,
    protected: set[int] | None = None,
) -> int:
    """Smallest vocabulary that covers `threshold` of all token occurrences.

    Tokens are taken in descending frequency until the cumulative share reaches
    the threshold. Reporting this as a curve over several thresholds, rather
    than as a single "every id ever seen" count, keeps the estimate honest:
    a long tail of ids seen once each inflates the vocabulary while carrying
    almost no probability mass.

    Args:
        token_counts: Token id to occurrence count.
        threshold: Target share of occurrences, in ``(0, 1]``.
        protected: Ids that are retained regardless of frequency.

    Returns:
        The number of vocabulary rows that would be kept.
    """
    protected = protected or set()
    total = sum(token_counts.values())
    if total == 0:
        return len(protected)

    target = threshold * total
    cumulative = 0
    kept = set(protected)
    for token_id, count in token_counts.most_common():
        if cumulative >= target:
            break
        cumulative += count
        kept.add(token_id)
    return len(kept)


def project_trim(
    spec: ModelSpec,
    token_counts: Counter[int],
    threshold: float,
    protected: set[int] | None = None,
    bytes_per_param: int = 2,
) -> TrimProjection:
    """Turn a coverage threshold into parameters and megabytes saved.

    Args:
        spec: The model whose embedding geometry sets the exchange rate.
        token_counts: Merged token counts over every retained language.
        threshold: Coverage target passed to :func:`coverage_vocab_size`.
        protected: Ids that survive regardless of frequency.
        bytes_per_param: Storage width of the deployed checkpoint; 2 for bf16.
    """
    kept = min(coverage_vocab_size(token_counts, threshold, protected), spec.vocab_size)
    removed_rows = spec.vocab_size - kept
    removed_params = removed_rows * spec.hidden_size * spec.embedding_matrices
    return TrimProjection(
        threshold=threshold,
        kept_vocab=kept,
        original_vocab=spec.vocab_size,
        removed_params=removed_params,
        removed_bytes=removed_params * bytes_per_param,
        model_fraction=removed_params / spec.params if spec.params else 0.0,
    )


def spec_from_config(
    name: str, repo_id: str, params: int, config: Any, note: str = ""
) -> ModelSpec:
    """Build a :class:`ModelSpec` from a Hugging Face config object.

    Multimodal configs (Gemma 3, for one) nest the language-model settings under
    ``text_config``, so look there first.
    """
    text_config = getattr(config, "text_config", None) or config

    def pick(attribute: str) -> Any:
        return getattr(text_config, attribute, None) or getattr(config, attribute, None)

    return ModelSpec(
        name=name,
        repo_id=repo_id,
        params=params,
        vocab_size=pick("vocab_size"),
        hidden_size=pick("hidden_size"),
        num_layers=pick("num_hidden_layers"),
        tie_word_embeddings=bool(
            getattr(config, "tie_word_embeddings", None)
            or getattr(text_config, "tie_word_embeddings", False)
        ),
        note=note,
    )
