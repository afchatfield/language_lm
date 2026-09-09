"""Client for a self-hosted LanguageTool HTTP server.

LanguageTool supplies two things this project needs: high-precision error
detection to benchmark against (Phase 1), and stable rule ids plus human-written
messages to hang explanation templates off (Phase 2). Both arrive through this
one class, so the decision to run LanguageTool in Docker rather than in-process
never leaks into calling code.

The server itself is defined in ``docker/docker-compose.yml``; start it with
``make lt-up``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from langlm.config import LT_BASE_URL

#: Rule categories that come from plain dictionary lookup. If a check returns
#: *only* these, the n-gram language model almost certainly failed to load --
#: see `scripts/lt_sanity_check.py`.
SPELLING_ONLY_RULES = frozenset({"MORFOLOGIK_RULE_DE_DE", "MORFOLOGIK_RULE_ES", "HUNSPELL_RULE"})


class LanguageToolError(RuntimeError):
    """Raised when the LanguageTool server is unreachable or returns an error."""


@dataclass(frozen=True)
class Match:
    """One error reported by LanguageTool, normalised to what we actually use.

    Attributes:
        offset: Character offset of the error in the submitted text.
        length: Character length of the flagged span.
        rule_id: Stable rule identifier, e.g. ``DE_AGREEMENT``. This is the key
            that Phase 2's explanation templates are indexed by.
        category_id: Coarse grouping, e.g. ``TYPOS``, ``GRAMMAR``, ``PUNCTUATION``.
        message: LanguageTool's human-readable message, in the target language.
        short_message: Terser variant; often empty.
        replacements: Suggested corrections, best first.
        context: The surrounding text LanguageTool echoed back.
    """

    offset: int
    length: int
    rule_id: str
    category_id: str
    message: str
    short_message: str
    replacements: list[str]
    context: str

    @property
    def end(self) -> int:
        return self.offset + self.length

    @property
    def best_replacement(self) -> str | None:
        return self.replacements[0] if self.replacements else None

    def excerpt(self, text: str) -> str:
        """The exact substring of `text` that this match flags."""
        return text[self.offset : self.end]

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> Match:
        """Build a Match from one entry of the ``matches`` array."""
        rule = payload.get("rule", {})
        return cls(
            offset=payload["offset"],
            length=payload["length"],
            rule_id=rule.get("id", ""),
            category_id=rule.get("category", {}).get("id", ""),
            message=payload.get("message", ""),
            short_message=payload.get("shortMessage", ""),
            replacements=[r["value"] for r in payload.get("replacements", [])],
            context=payload.get("context", {}).get("text", ""),
        )


class LanguageToolClient:
    """Thin wrapper over the LanguageTool ``/v2`` HTTP API.

    Example:
        >>> lt = LanguageToolClient()                        # doctest: +SKIP
        >>> for m in lt.check("Ich weiss, das du kommst."):  # doctest: +SKIP
        ...     print(m.rule_id, m.replacements)
    """

    def __init__(
        self,
        base_url: str = LT_BASE_URL,
        language: str = "de-DE",
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.timeout = timeout
        self._session = session or requests.Session()

    def check(self, text: str, language: str | None = None, **params: Any) -> list[Match]:
        """Check one text and return its matches.

        Args:
            text: The text to check.
            language: Override the client's default language code.
            **params: Extra form fields passed straight to the API, e.g.
                ``disabledRules="WHITESPACE_RULE"``.
        """
        return [Match.from_api(m) for m in self.check_raw(text, language, **params)]

    def check_raw(self, text: str, language: str | None = None, **params: Any) -> list[dict]:
        """The ``matches`` array exactly as the server sent it.

        Kept public for the response cache, which stores what the server said
        rather than what this module currently chooses to read out of it: a
        later phase that wants a field :class:`Match` drops today should not
        have to re-run a day of LanguageTool queries to get it.
        """
        payload = {"text": text, "language": language or self.language, **params}
        return self._post("/v2/check", payload).get("matches", [])

    def software(self) -> dict[str, Any]:
        """The server's own description of itself: name, version, build date."""
        payload = {"text": "Test", "language": self.language}
        return self._post("/v2/check", payload).get("software", {})

    def check_many(self, texts: list[str], **kwargs: Any) -> list[list[Match]]:
        """Check several texts. Sequential: the server parallelises internally."""
        return [self.check(text, **kwargs) for text in texts]

    def languages(self) -> list[dict[str, Any]]:
        """List the languages the server supports. Doubles as a health check."""
        response = self._request("GET", "/v2/languages")
        return response.json()

    def is_up(self) -> bool:
        """True if the server answers. Never raises."""
        try:
            self.languages()
        except LanguageToolError:
            return False
        return True

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, data=payload).json()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            response = self._session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LanguageToolError(
                f"{method} {url} failed: {exc}. Is the server running? Try `make lt-up`."
            ) from exc
        return response
