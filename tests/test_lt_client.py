"""Tests for the LanguageTool client.

Runs entirely against a recorded response, so CI needs no Docker and no server.
The fixture was captured from a real LanguageTool 6.8 instance with the German
n-gram model loaded -- see `tests/fixtures/lt_response.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from langlm.lt.client import LanguageToolClient, LanguageToolError, Match

FIXTURE = Path(__file__).parent / "fixtures" / "lt_response.json"
PROBE = "Das hat mir fiel geholfen."


@pytest.fixture
def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class FakeSession:
    """Stands in for `requests.Session`, recording what the client sent."""

    def __init__(self, payload: dict | None = None, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        if self._error:
            raise self._error
        return FakeResponse(self._payload)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def test_normalises_a_match(payload):
    client = LanguageToolClient(session=FakeSession(payload))
    (match,) = client.check(PROBE)

    assert match.rule_id == "CONFUSION_RULE_FIEL_VIEL"
    assert match.category_id == "TYPOS"
    assert match.replacements == ["viel"]
    assert match.best_replacement == "viel"
    assert match.offset == 12
    assert match.length == 4
    assert match.end == 16
    # The offsets must index the original string, or every downstream edit is wrong.
    assert match.excerpt(PROBE) == "fiel"


def test_sends_the_configured_language(payload):
    session = FakeSession(payload)
    client = LanguageToolClient(language="de-DE", session=session)
    client.check(PROBE)
    _, url, kwargs = session.calls[0]

    assert url.endswith("/v2/check")
    assert kwargs["data"]["language"] == "de-DE"
    assert kwargs["data"]["text"] == PROBE


def test_language_can_be_overridden_per_call(payload):
    session = FakeSession(payload)
    LanguageToolClient(language="de-DE", session=session).check("Hola", language="es")
    assert session.calls[0][2]["data"]["language"] == "es"


def test_extra_params_reach_the_api(payload):
    session = FakeSession(payload)
    LanguageToolClient(session=session).check(PROBE, disabledRules="WHITESPACE_RULE")
    assert session.calls[0][2]["data"]["disabledRules"] == "WHITESPACE_RULE"


def test_a_match_with_no_suggestion_is_handled():
    match = Match.from_api({"offset": 0, "length": 3, "rule": {"id": "X"}})
    assert match.replacements == []
    assert match.best_replacement is None
    assert match.category_id == ""


def test_connection_failure_names_the_fix():
    session = FakeSession(error=requests.ConnectionError("refused"))
    client = LanguageToolClient(session=session)
    with pytest.raises(LanguageToolError, match="make lt-up"):
        client.check(PROBE)


def test_is_up_reports_false_instead_of_raising():
    session = FakeSession(error=requests.ConnectionError("refused"))
    assert LanguageToolClient(session=session).is_up() is False


def test_is_up_reports_true_when_the_server_answers():
    assert LanguageToolClient(session=FakeSession([])).is_up() is True


def test_check_many_returns_one_list_per_text(payload):
    client = LanguageToolClient(session=FakeSession(payload))
    results = client.check_many([PROBE, PROBE, PROBE])
    assert len(results) == 3
    assert all(len(matches) == 1 for matches in results)
