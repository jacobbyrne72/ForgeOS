"""Provider probing: does it work, or does it merely exist?

`Provider.usable` reports enabled + installed + env-var-present, and on this
machine that scored 13 of 14 providers "ready" while three of them could not
serve a request at all. These tests pin the distinctions that gap costs:
a rejected key is not an exhausted one, a running daemon with no models pulled
is not ready, and a CLI being on PATH proves nothing about its sign-in.
"""

from __future__ import annotations

import json

import pytest

from forgeos.core.probe import (
    ProbeReport,
    ProbeResult,
    ProbeStatus,
    load_report,
    probe_provider,
    save_report,
)
from forgeos.settings import AuthMode, Provider, ProviderKind


def _api(**kw) -> Provider:
    base = dict(name="deepseek", kind=ProviderKind.API, auth=AuthMode.API_KEY,
                env_key="TEST_PROBE_KEY", base_url="https://example.invalid/v1")
    base.update(kw)
    return Provider(**base)


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"data": [{"id": "m"}]}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def http(monkeypatch):
    """Patch httpx.get and record what was requested."""
    import httpx

    calls: list[dict] = []
    box: dict = {"resp": _Resp()}

    def _get(url, **kwargs):
        calls.append({"url": url, **kwargs})
        resp = box["resp"]
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr(httpx, "get", _get)
    return calls, box


# ------------------------------------------------- can't vs didn't


def test_a_working_key_is_verified(monkeypatch, http):
    calls, box = http
    monkeypatch.setenv("TEST_PROBE_KEY", "sk-whatever")
    box["resp"] = _Resp(200, {"data": [{"id": "a"}, {"id": "b"}]})

    result = probe_provider(_api())
    assert result.status is ProbeStatus.READY
    assert result.models_seen == 2
    assert result.routable


def test_a_rejected_key_is_not_an_exhausted_one(monkeypatch, http):
    """401 and 402 must not collapse into one 'broken' state -- one means fix
    your key, the other means top up, and telling a user the wrong one sends
    them to the wrong page."""
    calls, box = http
    monkeypatch.setenv("TEST_PROBE_KEY", "sk-bad")

    box["resp"] = _Resp(401)
    assert probe_provider(_api()).status is ProbeStatus.NO_AUTH

    box["resp"] = _Resp(402)
    assert probe_provider(_api()).status is ProbeStatus.NO_CREDIT


def test_quota_exhaustion_is_reported_as_no_credit(monkeypatch, http):
    """The measured case: a key that is valid and simply out of quota."""
    calls, box = http
    monkeypatch.setenv("TEST_PROBE_KEY", "sk-ok")
    box["resp"] = _Resp(429)
    result = probe_provider(_api())
    assert result.status is ProbeStatus.NO_CREDIT
    assert not result.routable


def test_authenticated_but_serving_nothing_is_not_ready(monkeypatch, http):
    calls, box = http
    monkeypatch.setenv("TEST_PROBE_KEY", "sk-ok")
    box["resp"] = _Resp(200, {"data": []})
    result = probe_provider(_api())
    assert result.status is ProbeStatus.NO_MODEL
    assert not result.routable


def test_a_missing_env_var_is_reported_without_a_network_call(monkeypatch, http):
    calls, _box = http
    monkeypatch.delenv("TEST_PROBE_KEY", raising=False)
    result = probe_provider(_api())
    assert result.status is ProbeStatus.NO_AUTH
    assert calls == [], "probed the network for a provider with no credential"


def test_an_unreachable_host_is_not_a_crash(monkeypatch, http):
    calls, box = http
    monkeypatch.setenv("TEST_PROBE_KEY", "sk-ok")
    box["resp"] = OSError("no route to host")
    result = probe_provider(_api())
    assert result.status is ProbeStatus.UNREACHABLE
    assert not result.routable


def test_a_disabled_provider_is_never_contacted(http):
    calls, _box = http
    result = probe_provider(_api(enabled=False))
    assert result.status is ProbeStatus.DISABLED
    assert calls == []


# ---------------------------------------------------------------- secrets


def test_the_key_value_never_appears_in_the_result(monkeypatch, http):
    """A status object gets printed, logged and written to disk. A credential
    must not be able to ride along in any of those."""
    calls, box = http
    secret = "sk-do-not-leak-me-0123456789"
    monkeypatch.setenv("TEST_PROBE_KEY", secret)
    box["resp"] = _Resp(500)

    result = probe_provider(_api())
    blob = json.dumps(ProbeReport([result]).to_dict()) + result.render()
    assert secret not in blob


def test_an_error_body_is_never_echoed(monkeypatch, http):
    """Some providers reflect request headers in an error body. Echoing it is
    how a key reaches a log file."""
    calls, box = http
    monkeypatch.setenv("TEST_PROBE_KEY", "sk-secret-value")
    box["resp"] = _Resp(500, {"error": "Authorization: Bearer sk-secret-value"})
    result = probe_provider(_api())
    assert "sk-secret-value" not in result.detail


# ------------------------------------------------------------------- CLI


def test_a_cli_on_path_is_not_claimed_as_verified(monkeypatch):
    """The core honesty rule. A binary existing says nothing about whether the
    session behind it is signed in or has quota -- and proving that would cost
    tokens, so the status says 'unverified' rather than lying in either
    direction."""
    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/claude")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: type("P", (), {"returncode": 0, "stdout": b"1.0", "stderr": b""})(),
    )
    result = probe_provider(Provider(name="claude", kind=ProviderKind.CLI, command="claude"))
    assert result.status is ProbeStatus.INSTALLED_UNVERIFIED
    assert result.status is not ProbeStatus.READY
    assert result.routable, "an unverified subscription CLI is still the cheapest capacity"


def test_a_missing_binary_is_not_installed(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _n: None)
    result = probe_provider(Provider(name="kimi", kind=ProviderKind.CLI, command="kimi"))
    assert result.status is ProbeStatus.NOT_INSTALLED
    assert not result.routable


# ---------------------------------------------------------------- report


def test_verified_and_routable_are_counted_separately():
    report = ProbeReport([
        ProbeResult("a", ProbeStatus.READY),
        ProbeResult("b", ProbeStatus.INSTALLED_UNVERIFIED),
        ProbeResult("c", ProbeStatus.NO_CREDIT),
    ])
    assert [r.provider for r in report.verified] == ["a"]
    assert {r.provider for r in report.routable} == {"a", "b"}


def test_a_report_round_trips_through_disk(tmp_path):
    original = ProbeReport([
        ProbeResult("deepseek", ProbeStatus.READY, models_seen=2),
        ProbeResult("kimi", ProbeStatus.NO_CREDIT, "out of balance"),
    ])
    path = tmp_path / "probe.json"
    save_report(original, path)
    loaded = load_report(path)
    assert loaded is not None
    assert {(r.provider, r.status) for r in loaded.results} == {
        ("deepseek", ProbeStatus.READY), ("kimi", ProbeStatus.NO_CREDIT),
    }


def test_a_missing_report_reads_as_never_checked_not_as_nothing_works(tmp_path):
    """None must not be mistaken for an empty report -- one means 'no data',
    the other would mean 'every provider is dead'."""
    assert load_report(tmp_path / "absent.json") is None


def test_one_corrupt_row_does_not_discard_the_rest(tmp_path):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps({
        "schema": "forgeos.provider_probe.v1",
        "results": [
            {"provider": "good", "status": "ready"},
            {"provider": "bad", "status": "not-a-real-status"},
            {"nostatus": True},
        ],
    }), encoding="utf-8")
    loaded = load_report(path)
    assert [r.provider for r in loaded.results] == ["good"]


def test_a_corrupt_file_reads_as_never_checked(tmp_path):
    path = tmp_path / "probe.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_report(path) is None
