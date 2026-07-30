"""The compact heartbeat is the supervision cost lever, so it gets its own tests.

If `heartbeat()` ever starts carrying transcript-shaped fields, manager cost
grows with worker verbosity and the event-driven design quietly stops paying for
itself. These tests are the guard against that drift.
"""

from __future__ import annotations

import json

from forgeos.contracts import EscalationKind, TaskState, TestResults, Verdict, WorkerReport


def _full_report(**kw) -> WorkerReport:
    base = dict(
        task_id="T-104",
        worker_id="forgeos.executor",
        state=TaskState.RUNNING,
        goal="Implement provider fallback",
        summary="A long narration that the manager must never pay to read." * 40,
        files_touched=["router.py", "tests/test_router.py"],
        commands_run=["pytest tests/test_router.py -q"] * 10,
        evidence="1 failed, 14 passed" * 20,
        tests=TestResults(passed=14, failed=1),
        blocker="Provider 429 response shape differs",
        next_action="Add normalized retry-after parser",
        needs_manager=False,
        tokens_in=12000,
        tokens_out=840,
    )
    base.update(kw)
    return WorkerReport(**base)


def test_heartbeat_carries_the_decision_fields():
    hb = _full_report().heartbeat()
    assert hb["task_id"] == "T-104"
    assert hb["state"] == "running"
    assert hb["goal"] == "Implement provider fallback"
    assert hb["files_changed"] == ["router.py", "tests/test_router.py"]
    assert hb["tests"] == {"passed": 14, "failed": 1}
    assert hb["blocker"] == "Provider 429 response shape differs"
    assert hb["next_action"] == "Add normalized retry-after parser"
    assert hb["needs_manager"] is False
    assert hb["tokens_used"] == 12840


def test_heartbeat_excludes_transcript_shaped_fields():
    """summary/evidence/commands_run belong to the ledger and the human, not the manager."""
    hb = _full_report().heartbeat()
    for leaked in ("summary", "evidence", "commands_run"):
        assert leaked not in hb


def test_heartbeat_of_a_quiet_worker_is_tiny():
    quiet = WorkerReport(task_id="T-1", worker_id="w", state=TaskState.RUNNING)
    hb = quiet.heartbeat()
    assert set(hb) == {"task_id", "state", "needs_manager"}
    assert len(json.dumps(hb)) < 90


def test_heartbeat_is_far_smaller_than_the_full_report():
    r = _full_report()
    assert len(json.dumps(r.heartbeat())) * 10 < len(r.model_dump_json())


def test_heartbeat_surfaces_escalations():
    r = _full_report(
        needs_manager=True,
        escalations=[EscalationKind.SCOPE_REQUEST, EscalationKind.LOW_CONFIDENCE],
    )
    hb = r.heartbeat()
    assert hb["needs_manager"] is True
    assert hb["escalations"] == ["scope_request", "low_confidence"]


def test_heartbeat_is_json_serializable():
    json.dumps(_full_report().heartbeat())  # must not raise


def test_test_results_green_requires_passing_and_no_failures():
    assert TestResults(passed=5, failed=0).green
    assert not TestResults(passed=5, failed=1).green
    assert not TestResults(passed=0, failed=0).green  # nothing ran is not green


def test_test_results_str_is_compact():
    assert str(TestResults(passed=14, failed=1)) == "14p/1f"


def test_report_still_reports_success_correctly_with_new_fields():
    r = _full_report(state=TaskState.DONE, verdict=Verdict.PASS)
    assert r.succeeded
    assert _full_report(state=TaskState.DONE, verdict=Verdict.FAIL).succeeded is False
