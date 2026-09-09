"""The LanguageTool baseline: a mature rule-based checker, applied greedily.

LanguageTool is the system this project has to justify itself against. It is
twenty years of hand-written German rules plus an n-gram confusion model, it
runs offline, and it is free -- so a neural model that merely matches it has
proved nothing unless it also explains itself or fits in a fraction of the
space.

Turning it into a *corrector* takes one decision: it reports matches, each with
a ranked list of replacements, and a checker is not a corrector until something
picks one. This applies the first replacement of every match that offers one, in
order, skipping any match that overlaps a span already changed. That is the most
favourable reading of LanguageTool's output short of an oracle, which is the
right way to build the number you are trying to beat.

Matches with no replacement are left alone. LanguageTool raises those to say
"something is wrong here" without knowing what to do about it, and inventing a
correction on its behalf would attribute a false positive to the wrong system.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langlm.config import load_config
from langlm.lt.cache import CachedLanguageToolClient
from langlm.lt.client import LanguageToolClient, Match


@dataclass
class LanguageToolBaseline:
    """Applies LanguageTool's own suggestions to a tokenised sentence."""

    client: LanguageToolClient | CachedLanguageToolClient | None = None
    language: str = "de-DE"
    disabled_rules: tuple[str, ...] = ()
    require_replacement: bool = True
    #: Concurrent requests. The server is disk-bound on the n-gram index rather
    #: than CPU-bound, so a few requests in flight help and many do not: at eight
    #: workers a 4GB heap starts timing out individual checks.
    workers: int = 4
    #: Per-request timeout, in seconds. Generous, because a single check against
    #: the n-gram model takes seconds and a slow answer is worth more than a
    #: retry.
    timeout: float = 120.0
    #: Cache responses on disk. A pass over the development split takes tens of
    #: minutes against a server with n-grams loaded, and the answers do not
    #: change between runs, so the default is on.
    cached: bool = True
    name: str = "languagetool"

    @classmethod
    def from_config(cls, config_name: str = "phase1", **overrides) -> LanguageToolBaseline:
        """Build from `configs/<config_name>.yaml`."""
        settings = load_config(config_name)["baselines"]["languagetool"]
        return cls(
            language=settings["language"],
            disabled_rules=tuple(settings["disabled_rules"]),
            require_replacement=settings["require_replacement"],
            **overrides,
        )

    def __post_init__(self) -> None:
        if self.client is None:
            client = LanguageToolClient(language=self.language, timeout=self.timeout)
            self.client = CachedLanguageToolClient(client) if self.cached else client

    def check(self, source: str) -> list[Match]:
        """The raw matches, for callers that want the rule ids rather than a fix."""
        params = {}
        if self.disabled_rules:
            params["disabledRules"] = ",".join(self.disabled_rules)
        return self.client.check(source, language=self.language, **params)

    def correct(self, source: str) -> str:
        """Apply LanguageTool's first suggestion for each non-overlapping match."""
        return apply_matches(source, self.check(source), self.require_replacement)

    def correct_many(self, sources: list[str]) -> list[str]:
        """Correct a corpus. Requests are concurrent; the server is not the bottleneck."""
        if self.workers <= 1:
            return [self.correct(source) for source in sources]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self.correct, sources))


def apply_matches(source: str, matches: list[Match], require_replacement: bool = True) -> str:
    """Rewrite `source` by applying `matches`, later matches yielding to earlier ones.

    Args:
        source: The sentence that was checked.
        matches: LanguageTool's matches for exactly that string.
        require_replacement: Skip matches that suggest nothing.

    Returns:
        The corrected sentence.
    """
    # Right to left, so that an earlier match's offsets stay valid as later
    # spans change length.
    ordered = sorted(matches, key=lambda m: (m.offset, m.end))
    applied: list[Match] = []
    boundary = 0
    for match in ordered:
        if match.offset < boundary:
            continue  # Overlaps something already accepted.
        if require_replacement and not match.best_replacement:
            continue
        if match.excerpt(source) == match.best_replacement:
            continue  # A suggestion that changes nothing.
        applied.append(match)
        boundary = match.end

    text = source
    for match in reversed(applied):
        text = text[: match.offset] + (match.best_replacement or "") + text[match.end :]
    return " ".join(text.split())
