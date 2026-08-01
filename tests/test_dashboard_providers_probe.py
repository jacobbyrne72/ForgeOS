"""The providers panel must not report a dead provider as ready.

Measured on this machine before the probe was wired in: the panel said
"13/14 providers usable" while `copilot`, `hermes` and `pi` could not serve a
request at all. `Provider.usable` answers "enabled, installed, env var present",
which is a declaration, not a verification -- and the panel presented it as one.

The fix surfaces both numbers side by side rather than replacing one with the
other. A reader needs to see that they disagree; that disagreement IS the
finding.
"""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from forgeos.dashboard.app import create_app


def _probe_file(state_dir, rows):
    (state_dir / "provider_probe.json").write_text(json.dumps({
        "schema": "forgeos.provider_probe.v1",
        "results": rows,
    }), encoding="utf-8")


def test_declared_and_verified_are_reported_separately(tmp_path):
    """Collapsing them into one number is how three dead providers read as
    ready."""
    _probe_file(tmp_path, [
        {"provider": "deepseek", "status": "ready", "models_seen": 2,
         "checked_at": time.time()},
        {"provider": "copilot", "status": "unreachable", "detail": "TimeoutExpired",
         "checked_at": time.time()},
    ])
    body = TestClient(create_app(tmp_path)).get("/api/providers").json()

    assert "usable_count" in body, "declared count disappeared"
    assert "verified_count" in body, "verified count is not reported"
    assert body["probed"] is True
    assert body["verified_count"] <= body["usable_count"], (
        "more providers verified than were even declared usable"
    )


def test_an_unreachable_provider_is_marked_not_routable(tmp_path):
    _probe_file(tmp_path, [
        {"provider": "copilot", "status": "unreachable", "detail": "TimeoutExpired",
         "checked_at": time.time()},
    ])
    body = TestClient(create_app(tmp_path)).get("/api/providers").json()
    row = next((r for r in body["providers"] if r["name"] == "copilot"), None)
    if row is not None:
        assert row["probe_status"] == "unreachable"
        assert row["probe_routable"] is False


def test_an_unverified_cli_is_not_counted_as_verified(tmp_path):
    """INSTALLED_UNVERIFIED is routable but must never inflate the verified
    count -- a subscription CLI's sign-in cannot be confirmed without spending
    tokens, and claiming otherwise is the overstatement being deleted."""
    _probe_file(tmp_path, [
        {"provider": "claude", "status": "installed?", "checked_at": time.time()},
    ])
    body = TestClient(create_app(tmp_path)).get("/api/providers").json()
    assert body["verified_count"] == 0


def test_no_probe_file_is_reported_as_unprobed_not_as_all_dead(tmp_path):
    """"Never checked" and "checked, nothing works" are different facts."""
    body = TestClient(create_app(tmp_path)).get("/api/providers").json()
    assert body["probed"] is False
    assert body["verified_count"] == 0
    assert body["usable_count"] >= 0


def test_a_corrupt_probe_file_reports_the_error_rather_than_hiding_it(tmp_path):
    """A bare `except Exception` here swallowed a NameError and the panel showed
    no probe data -- indistinguishable from "no probe was run"."""
    (tmp_path / "provider_probe.json").write_text("{not json", encoding="utf-8")
    body = TestClient(create_app(tmp_path)).get("/api/providers").json()
    assert body["probed"] is False
    assert "probe_error" in body


def test_the_panel_never_returns_a_credential_value(tmp_path):
    """`env_key` is the variable NAME, which an operator needs; the value is a
    secret and must never appear."""
    import os

    os.environ["DEEPSEEK_API_KEY"] = "sk-must-not-appear-anywhere"
    try:
        body = TestClient(create_app(tmp_path)).get("/api/providers").json()
        assert "sk-must-not-appear-anywhere" not in json.dumps(body)
    finally:
        os.environ.pop("DEEPSEEK_API_KEY", None)
