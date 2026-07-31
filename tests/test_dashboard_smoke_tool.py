from __future__ import annotations

import json
import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "forgeos_dashboard_smoke_tool", Path(__file__).parents[1] / "tools" / "dashboard_smoke.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_SMOKE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SMOKE)


def test_smoke_report_validation_accepts_the_fixture_rollup() -> None:
    payload = {
        "schema": "forgeos.leaderboard.v1",
        "fleet_rollup": {
            "accepted_count": 2,
            "cost_per_accepted_usd": 0.2,
        },
    }

    result = _SMOKE._validate_payload(payload, expect_fixture=True)

    assert result == {
        "schema": "forgeos.leaderboard.v1",
        "fleet_rollup": payload["fleet_rollup"],
    }
    assert _SMOKE._receipt("smoke/model", 125_000)["model_ref"] == "smoke/model"


def test_smoke_failure_report_is_machine_readable(capsys) -> None:
    _SMOKE._emit_failure("assertion", AssertionError("/api/leaderboard returned HTTP 404"))

    report = json.loads(capsys.readouterr().err)

    assert report == {
        "report_schema": _SMOKE.REPORT_SCHEMA,
        "ok": False,
        "kind": "assertion",
        "error": "/api/leaderboard returned HTTP 404",
    }
