"""Dashboard tests: every endpoint on a fresh empty ledger, halt semantics,
the offline-page guarantee, and the localhost bind.

No LLM calls, no network. Each test gets its own `tmp_path` state dir, so
"fresh empty database" is the literal starting point -- this is the case
that breaks naive dashboards (divide-by-zero, absent keys, NaN).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hive.contracts import Budget, JobSpec, TaskSpec, TaskState, WorkerReport
from hive.dashboard.app import (
    AVOIDANCE_DB,
    EVENTS_DB,
    HOST,
    LEASES_DB,
    LEDGER_DB,
    PORT,
    STATIC_DIR,
    create_app,
)
from hive.economy.avoidance import AvoidanceLog, AvoidanceMethod
from hive.events import EventLog, EventType
from hive.ledger import Ledger
from hive.leases import LeaseStore, LeaseType


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

        ledger.record_spend(job.id, "omc.executor", "claude-x", 250_000, task_id=task.id,
                            tokens_in=1000, tokens_out=500, tokens_cached_in=200)

        ledger.record_report(
            WorkerReport(
                task_id=task.id,
                worker_id="omc.executor",
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
    assert detail["worker_lanes"][0]["worker_id"] == "omc.executor"
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
    assert "hive" in res.text.lower()


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
