"""Dashboard tests: every endpoint on a fresh empty ledger, halt semantics,
the offline-page guarantee, and the localhost bind.

No LLM calls, no network. Each test gets its own `tmp_path` state dir, so
"fresh empty database" is the literal starting point -- this is the case
that breaks naive dashboards (divide-by-zero, absent keys, NaN).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from forgeos.contracts import Budget, JobSpec, TaskSpec, TaskState, WorkerReport
from forgeos.dashboard.app import (
    AVOIDANCE_DB,
    EVENTS_DB,
    HOST,
    LEASES_DB,
    LEDGER_DB,
    PORT,
    QUOTA_FILE,
    STATIC_DIR,
    create_app,
)
from forgeos.core.quota import QuotaTracker
from forgeos.economy.avoidance import AvoidanceLog, AvoidanceMethod
from forgeos.events import EventLog, EventType
from forgeos.ledger import Ledger
from forgeos.leases import LeaseStore, LeaseType


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


@pytest.fixture
def client(state_dir: Path) -> TestClient:
    app = create_app(state_dir)
    return TestClient(app)


# --------------------------------------------------------------- empty db


def test_summary_on_fresh_empty_db_has_documented_keys_and_zeros(client: TestClient):
    res = client.get("/api/summary")
    assert res.status_code == 200
    body = res.json()

    for key in (
        "spend_usd",
        "cap_burn",
        "cache_hit_pct",
        "avoided_tokens",
        "active_jobs",
        "open_escalations",
        "cost_per_accepted_task",
    ):
        assert key in body, key

    assert body["spend_usd"] == 0
    assert body["cache_hit_pct"] == 0
    assert body["avoided_tokens"] == 0
    assert body["active_jobs"] == 0
    assert body["open_escalations"] == 0
    # No accepted tasks yet -> must not divide by zero or render NaN.
    assert body["cost_per_accepted_task"] is None
    # cap_burn is a distinct, honestly-labelled non-dollar figure -- never a
    # bare number silently summed with spend_usd.
    assert isinstance(body["cap_burn"], dict)
    assert body["cap_burn"]["measured"] is False
    assert body["cap_burn"]["value_pct"] is None


def test_jobs_on_fresh_empty_db_is_an_empty_list(client: TestClient):
    res = client.get("/api/jobs")
    assert res.status_code == 200
    assert res.json() == {"jobs": []}


def _write_leaderboard_receipt(path: Path, *, model_ref: str = "local/model") -> None:
    path.write_text(json.dumps({
        "schema": "forgeos.forgebench.v1",
        "mode": "live",
        "provenance": "measured",
        "model_ref": model_ref,
        "suite": {"name": "dashboard-suite", "savings_class": "A", "tasks": ["t"]},
        "totals": {
            "baseline": {"accepted_count": 1, "attempted_count": 1, "usd_micros": 500_000},
            "forgeos": {"accepted_count": 1, "attempted_count": 1, "usd_micros": 125_000},
        },
        "comparison_voided": False,
        "exit_gate_passed": True,
        "proof": {"mission_id": "dashboard-test", "contract_hash": "contract"},
    }), encoding="utf-8")


def test_leaderboard_endpoint_reads_local_receipts_without_provider_calls(state_dir: Path):
    receipt_dir = state_dir / "receipts"
    receipt_dir.mkdir(parents=True)
    _write_leaderboard_receipt(receipt_dir / "run.json")
    client = TestClient(create_app(state_dir, leaderboard_dir=receipt_dir))

    res = client.get("/api/leaderboard")

    assert res.status_code == 200
    body = res.json()
    assert body["schema"] == "forgeos.leaderboard.v1"
    assert body["available"] is True
    assert body["errors"] == []
    assert body["summary"]["eligible_runs"] == 1
    assert body["entries"][0]["label"] == "local/model"
    assert body["entries"][0]["cost_per_accepted_usd"] == 0.125


def test_leaderboard_endpoint_is_empty_and_does_not_create_missing_receipt_dir(state_dir: Path):
    receipt_dir = state_dir / "not-yet-created"
    client = TestClient(create_app(state_dir, leaderboard_dir=receipt_dir))

    res = client.get("/api/leaderboard")

    assert res.status_code == 200
    body = res.json()
    assert body["schema"] == "forgeos.leaderboard.v1"
    assert body["available"] is False
    assert body["entries"] == []
    assert body["summary"]["total_runs"] == 0
    assert body["errors"] == []
    assert not receipt_dir.exists()


def test_queue_status_endpoint_reports_heartbeat_without_provider_calls(state_dir: Path):
    queue = state_dir / "queue"
    queue.mkdir(parents=True)
    (queue / "heartbeat.json").write_text(
        json.dumps({
            "ts": time.time(), "state": "idle", "current_job": None,
            "jobs_done": 4, "jobs_failed": 0,
        }),
        encoding="utf-8",
    )
    client = TestClient(create_app(state_dir, queue_dir=queue))

    res = client.get("/api/queue/status")

    assert res.status_code == 200
    body = res.json()
    assert body["heartbeat_valid"] is True
    assert body["stale"] is False
    assert body["state"] == "idle"
    assert body["jobs_done"] == 4
    assert body["jobs_failed"] == 0


def test_job_detail_missing_job_is_404(client: TestClient):
    res = client.get("/api/jobs/does-not-exist")
    assert res.status_code == 404


def test_economy_on_fresh_empty_db_is_zeroed_not_absent(client: TestClient):
    res = client.get("/api/economy")
    assert res.status_code == 200
    body = res.json()
    assert body["baseline_tokens"] == 0
    assert body["actual_tokens"] == 0
    assert body["saved_tokens"] == 0
    assert body["saved_pct"] == 0
    assert body["by_method"] == {}
    assert body["cache"]["cache_hit_pct"] == 0


def test_workers_on_fresh_empty_db_lists_registered_workers_with_zero_attempts(client: TestClient):
    res = client.get("/api/workers")
    assert res.status_code == 200
    body = res.json()
    assert "workers" in body
    assert isinstance(body["workers"], list)
    # The default fleet is declared even with no measured history -- an
    # operator should see idle workers, not an empty panel.
    assert len(body["workers"]) > 0
    for w in body["workers"]:
        assert w["attempts"] == 0
        assert w["measured"] is False
        assert w["win_rate"] is None


def test_halt_on_missing_job_is_404(client: TestClient):
    res = client.post("/api/jobs/does-not-exist/halt")
    assert res.status_code == 404


# ------------------------------------------------------------- happy path


def _seed_job(state_dir: Path) -> tuple[str, str]:
    """Write directly to the same ledger file the app reads -- this is how
    the real system works: a manager process writes, the dashboard only
    reads.
    """
    ledger = Ledger(state_dir / LEDGER_DB)
    events = EventLog(state_dir / EVENTS_DB)
    try:
        job = JobSpec(objective="prove the dashboard reads real data", cwd=str(state_dir),
                      budget=Budget(max_usd=5.0, max_seconds=600, max_iterations=10))
        ledger.open_job(job)

        task = TaskSpec(job_id=job.id, subject="write a test", description="d",
                         acceptance=["pytest passes"], capabilities=["python"])
        ledger.add_task(task, state=TaskState.RUNNING)

        ledger.record_spend(job.id, "forgeos.executor", "claude-x", 250_000, task_id=task.id,
                            tokens_in=1000, tokens_out=500, tokens_cached_in=200)

        ledger.record_report(
            WorkerReport(
                task_id=task.id,
                worker_id="forgeos.executor",
                state=TaskState.DONE,
                confidence=0.9,
                summary="done",
                usd_micros=250_000,
                tokens_in=1000,
                tokens_out=500,
            )
        )

        events.append(job.id, EventType.TASK_CREATED, task_id=task.id)
        events.append(job.id, EventType.TASK_ACCEPTED, task_id=task.id)
    finally:
        ledger.close()
        events.close()
    return job.id, task.id


def test_summary_reflects_seeded_spend_and_one_accepted_task(state_dir: Path):
    job_id, _task_id = _seed_job(state_dir)
    app = create_app(state_dir)
    client = TestClient(app)

    body = client.get("/api/summary").json()
    assert body["spend_usd"] == pytest.approx(0.25)
    assert body["accepted_tasks"] == 1
    assert body["cost_per_accepted_task"] == pytest.approx(0.25)
    assert body["active_jobs"] == 1


def test_quota_endpoint_reports_persisted_provider_fact_without_network(state_dir: Path):
    quota = QuotaTracker()
    quota.record_report("claude", "Weekly: 75% remaining", at=1_800_000_000)
    quota.save(state_dir / QUOTA_FILE)
    client = TestClient(create_app(state_dir))

    body = client.get("/api/quota").json()
    assert body["available"] is True
    assert body["error"] is None
    assert body["states"][0]["provider"] == "claude"
    assert body["states"][0]["source"] == "reported"
    assert body["states"][0]["pct_remaining"] == 75.0
    assert body["states"][0]["pct_burn"] == 25.0
    assert client.get("/api/summary").json()["cap_burn"] == {
        "measured": True,
        "value_pct": 25.0,
        "currency": "subscription_cap",
        "note": "one provider window reported by the provider",
    }


def test_jobs_list_and_detail_include_documented_fields(state_dir: Path):
    job_id, task_id = _seed_job(state_dir)
    client = TestClient(create_app(state_dir))

    jobs = client.get("/api/jobs").json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["id"] == job_id
    assert jobs[0]["spend_usd"] == pytest.approx(0.25)
    assert jobs[0]["task_counts_by_state"] == {"done": 1}
    assert jobs[0]["halted"] is False

    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["id"] == job_id
    assert len(detail["tasks"]) == 1
    assert detail["tasks"][0]["id"] == task_id
    assert detail["tasks"][0]["spend_usd"] == pytest.approx(0.25)
    assert len(detail["worker_lanes"]) == 1
    assert detail["worker_lanes"][0]["worker_id"] == "forgeos.executor"
    assert len(detail["events"]) == 2


def test_halt_sets_flag_without_deleting_job_or_tasks(state_dir: Path):
    job_id, task_id = _seed_job(state_dir)
    client = TestClient(create_app(state_dir))

    res = client.post(f"/api/jobs/{job_id}/halt", params={"reason": "operator says stop"})
    assert res.status_code == 200
    body = res.json()
    assert body["halted"] is True
    assert body["reason"] == "operator says stop"

    # Nothing was deleted: the job, its task, and its spend are all still there.
    jobs = client.get("/api/jobs").json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["id"] == job_id
    assert jobs[0]["halted"] is True

    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["halted"] is True
    assert len(detail["tasks"]) == 1
    assert detail["tasks"][0]["id"] == task_id


def test_active_leases_are_scoped_to_the_jobs_own_tasks(state_dir: Path):
    job_id, task_id = _seed_job(state_dir)
    leases = LeaseStore(state_dir / LEASES_DB)
    try:
        leases.acquire(task_id, repo_id=str(state_dir), path_pattern="src/**",
                       lease_type=LeaseType.WRITE, ttl_seconds=300)
        leases.acquire("some-other-task", repo_id=str(state_dir), path_pattern="other/**",
                       lease_type=LeaseType.WRITE, ttl_seconds=300)
    finally:
        leases.close()

    client = TestClient(create_app(state_dir))
    detail = client.get(f"/api/jobs/{job_id}").json()
    assert len(detail["active_leases"]) == 1
    assert detail["active_leases"][0]["task_id"] == task_id


def test_activity_on_fresh_empty_db_is_an_empty_list(client: TestClient):
    res = client.get("/api/activity")
    assert res.status_code == 200
    assert res.json() == {"events": []}


def test_activity_reflects_seeded_events_across_jobs_newest_first(state_dir: Path):
    job_id, task_id = _seed_job(state_dir)
    client = TestClient(create_app(state_dir))

    body = client.get("/api/activity").json()
    events = body["events"]
    assert len(events) >= 2  # TASK_CREATED + TASK_ACCEPTED from the seed
    assert all(e["created_at"] >= events[i + 1]["created_at"] for i, e in enumerate(events[:-1]))
    assert {e["job_id"] for e in events} == {job_id}
    assert any(e["kind"] == "task_accepted" for e in events)

    # A halt is an operator action -- it must show up in the same feed, not
    # only in the per-job detail view, since the activity rail is meant to be
    # the one place an operator sees everything happening across every job.
    client.post(f"/api/jobs/{job_id}/halt", params={"reason": "operator says stop"})
    events_after = client.get("/api/activity").json()["events"]
    assert events_after[0]["kind"] == "operator_halt_requested"
    assert events_after[0]["job_id"] == job_id


def test_economy_reports_avoidance_and_cache_totals(state_dir: Path):
    avoidance = AvoidanceLog(state_dir / AVOIDANCE_DB)
    try:
        avoidance.record(
            "job-1", AvoidanceMethod.PACK, baseline_tokens=10_000, actual_tokens=1_000,
            baseline_source="tiktoken count of the full file not read",
        )
    finally:
        avoidance.close()

    client = TestClient(create_app(state_dir))
    body = client.get("/api/economy").json()
    assert body["saved_tokens"] == 9_000
    assert body["by_method"]["pack"]["count"] == 1


# --------------------------------------------------------------- honesty


def test_html_page_has_no_external_http_resource_references():
    html = STATIC_DIR.joinpath("index.html").read_text(encoding="utf-8")
    assert "http://" not in html
    assert "https://" not in html


def test_index_route_serves_the_offline_page(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    assert "forgeos" in res.text.lower()


def test_app_binds_localhost_only():
    assert HOST == "127.0.0.1"
    assert HOST != "0.0.0.0"
    assert PORT == 8899


# -------------------------------------------------------------------- ws


def test_websocket_pushes_a_summary_snapshot_on_connect(client: TestClient):
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "summary"
        assert "cost_per_accepted_task" in msg["data"]


def test_websocket_pushes_jobs_then_activity_after_summary(client: TestClient):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "summary"
        assert ws.receive_json()["type"] == "jobs"
        msg = ws.receive_json()
        assert msg["type"] == "activity"
        assert "events" in msg["data"]


# ------------------------------------------------------------------ honesty


def test_index_html_has_balanced_tags():
    """A structural sanity check for the hand-authored single-file dashboard:
    every opening tag this parser sees must have a matching close. Void
    elements are exempt; nothing else is, so a stray unclosed <div> anywhere
    in a ~1000-line hand-edited file fails loudly instead of silently
    shipping a broken layout.
    """
    from html.parser import HTMLParser

    void_elements = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    class TagBalanceChecker(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.stack: list[str] = []

        def handle_starttag(self, tag: str, attrs) -> None:
            if tag not in void_elements:
                self.stack.append(tag)

        def handle_startendtag(self, tag: str, attrs) -> None:
            pass  # self-closed, e.g. <br/>, never pushed

        def handle_endtag(self, tag: str) -> None:
            assert self.stack, f"</{tag}> with no open tag on the stack"
            assert self.stack[-1] == tag, f"expected </{self.stack[-1]}>, got </{tag}>"
            self.stack.pop()

    html = STATIC_DIR.joinpath("index.html").read_text(encoding="utf-8")
    checker = TagBalanceChecker()
    checker.feed(html)
    checker.close()
    assert checker.stack == [], f"unclosed tags: {checker.stack}"


def test_index_html_surfaces_the_measured_leaderboard_panel():
    html = STATIC_DIR.joinpath("index.html").read_text(encoding="utf-8")
    assert 'id="leaderboard-panel"' in html
    assert 'id="leaderboard-body"' in html
    assert 'id="leaderboard-refresh"' in html
    assert 'api("/api/leaderboard")' in html
    assert "Jump to Leaderboard" in html
    assert "fleet_rollup" in html
    assert "measured live Class-A only" in html


# ===================== event feed cost is bounded by what it returns


def test_replay_limit_returns_the_newest_events_oldest_first(tmp_path):
    """The activity feed asks for "the last N things". Returning the OLDEST N
    would show a permanently frozen feed once history outgrew N."""
    from forgeos.events import EventLog, EventType

    log = EventLog(tmp_path / "ev.db")
    for i in range(25):
        log.append("j", EventType.TASK_CREATED, task_id=f"t{i}")

    tail = log.replay(limit=5)
    assert len(tail) == 5
    assert [e.task_id for e in tail] == ["t20", "t21", "t22", "t23", "t24"]
    seqs = [e.seq for e in tail]
    assert seqs == sorted(seqs), "callers render oldest-first"


def test_replay_without_a_limit_is_unchanged(tmp_path):
    from forgeos.events import EventLog, EventType

    log = EventLog(tmp_path / "ev.db")
    for i in range(7):
        log.append("j", EventType.TASK_CREATED, task_id=f"t{i}")
    assert len(log.replay()) == 7


def test_replay_limit_is_pushed_into_sql_not_applied_in_python(tmp_path):
    """The defect: cost grew with lifetime events while the answer stayed N
    rows, on a 2-second poll. Slicing in Python looks identical from outside,
    so this asserts the bound reached the query."""
    from forgeos.events import EventLog, EventType

    log = EventLog(tmp_path / "ev.db")
    for i in range(200):
        log.append("j", EventType.TASK_CREATED, task_id=f"t{i}")

    seen: list[str] = []
    real_execute = log._conn.execute

    def spy(sql, params=()):
        seen.append(sql)
        return real_execute(sql, params)

    log._conn.execute = spy
    rows = log.replay(limit=10)
    log._conn.execute = real_execute

    assert len(rows) == 10
    selects = [q for q in seen if "event_log" in q and "SELECT *" in q]
    assert selects and all("LIMIT" in q for q in selects), (
        f"replay(limit=...) read without a SQL bound: {selects}"
    )


def test_count_by_type_matches_a_filtered_replay(tmp_path):
    from forgeos.events import EventLog, EventType

    log = EventLog(tmp_path / "ev.db")
    for i in range(4):
        log.append("j", EventType.TASK_CREATED, task_id=f"t{i}")
    for i in range(3):
        log.append("j", EventType.TASK_ACCEPTED, task_id=f"t{i}")

    assert log.count_by_type(EventType.TASK_ACCEPTED) == 3
    assert log.count_by_type(EventType.TASK_ACCEPTED) == sum(
        1 for e in log.replay() if e.type is EventType.TASK_ACCEPTED
    )
    assert log.count() == 7
