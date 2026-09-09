"""An on-disk cache for LanguageTool responses.

LanguageTool with n-grams loaded is slow -- the index does not fit comfortably
in the container's heap, so every check pays for disk. At roughly half a second
a sentence, one pass over the development split is twenty-five minutes, and
Phase 2 will ask for tens of thousands of sentences.

The results are worth caching because they are deterministic: the same text
against the same server version and the same rule settings gives the same
matches. Keying on all three means a changed rule set or a server upgrade
misses the cache instead of quietly serving stale answers.

The store is a JSON-lines file, appended to as misses occur. That survives an
interrupted run -- important when the run is measured in tens of minutes -- and
stays readable, so a surprising match can be found with grep.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from langlm.config import INTERIM_DIR
from langlm.lt.client import LanguageToolClient, LanguageToolError, Match

#: Default cache location.
CACHE_PATH = INTERIM_DIR / "lt_cache.jsonl"


class CachedLanguageToolClient:
    """A :class:`LanguageToolClient` that remembers what it has already asked.

    Example:
        >>> lt = CachedLanguageToolClient()          # doctest: +SKIP
        >>> lt.check("Ich weiss , das du kommst .")  # doctest: +SKIP
    """

    def __init__(
        self,
        client: LanguageToolClient | None = None,
        path: Path = CACHE_PATH,
        server_version: str | None = None,
    ) -> None:
        self.client = client or LanguageToolClient()
        self.path = Path(path)
        self.hits = 0
        self.misses = 0
        # Corpus-scale callers check in a thread pool, so the in-memory index and
        # the file it is appended to are both guarded.
        self._lock = threading.Lock()
        self._entries = self._read()
        # Asking the server what it is costs one request and makes the key
        # honest about the fact that a LanguageTool upgrade changes the answers.
        self.server_version = server_version or self._server_version()

    @property
    def language(self) -> str:
        return self.client.language

    def check(self, text: str, language: str | None = None, **params: Any) -> list[Match]:
        """Check `text`, consulting the cache first."""
        key = self._key(text, language or self.client.language, params)
        with self._lock:
            cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            return [Match.from_api(match) for match in cached]

        raw = self.client.check_raw(text, language, **params)
        with self._lock:
            self.misses += 1
            self._entries[key] = raw
            self._append(key, text, raw)
        return [Match.from_api(match) for match in raw]

    def check_many(self, texts: list[str], **kwargs: Any) -> list[list[Match]]:
        return [self.check(text, **kwargs) for text in texts]

    def _server_version(self) -> str:
        try:
            return str(self.client.software().get("buildDate", "unknown"))
        except LanguageToolError:
            # A rerun with the server stopped should still be able to read a
            # cache the server filled, so this is not fatal.
            return "unknown"

    def _key(self, text: str, language: str, params: dict) -> str:
        material = json.dumps(
            [self.server_version, language, sorted(params.items()), text],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _read(self) -> dict[str, list[dict]]:
        if not self.path.exists():
            return {}
        entries: dict[str, list[dict]] = {}
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A run killed mid-write; the entry is simply re-fetched.
                entries[record["key"]] = record["matches"]
        return entries

    def _append(self, key: str, text: str, matches: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # `text` is stored alongside the key purely so the file can be read by a
        # human looking for why a sentence was flagged.
        record = {"key": key, "text": text, "matches": matches}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
