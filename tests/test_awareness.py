"""Teammate-awareness board tests.

`TeamBoard` is read-only by contract: it must never call `LeaseStore.acquire`,
`.release`, or `.expire_stale`. These tests build real `LeaseStore`/`EventLog`
instances on ":memory:" and drive them the way `forgeos.core.scheduler.Scheduler`
does (event sequence, then a lease), so the board is exercised against realistic
event/lease shapes rather than hand-waved stand-ins. No network, no LLM calls.
"""

from __future__ import annotations

import pytest

from forgeos.core.awareness import BOARD_MAX_WORKERS, TeamBoard
from forgeos.events import EventLog, EventType
from forgeos.leases import LeaseStore, LeaseType

# --------------------------------------------------------------------- rig


@pytest.fixture()
def rig():
    events = EventLog(":memory:")
    leases = LeaseStore(":memory:")
    yield events, leases
    events.close()
    leases.close()


def _activate(
    events: EventLog,
    job_id: str,
    task_id: str,
    worker_id: str,
    subject: str,
    *,
    goal: str = "",
    tests: dict | None = None,
) -> None:
    """Replay the event sequence `Scheduler.assign()` + a checkpoint produce."""
    events.append(job_id, EventType.TASK_CREATED, task_id=task_id, subject=subject)
    events.append(job_id, EventType.WORKER_ASSIGNED, task_id=task_id, worker=worker_id)
    events.append(job_id, EventType.SESSION_STARTED, task_id=task_id, worker=worker_id)
    payload: dict = {}
    if goal:
        payload["goal"] = goal
    if tests:
        payload["tests"] = tests
    if payload:
        events.append(job_id, EventType.CHECKPOINT_CREATED, task_id=task_id, **payload)


def _finish(events: EventLog, job_id: str, task_id: str) -> None:
    events.append(job_id, EventType.TASK_ACCEPTED, task_id=task_id)


def _lease_rows(leases: LeaseStore) -> list[tuple]:
    return [tuple(r) for r in leases._conn.execute("SELECT * FROM path_leases ORDER BY id").fetchall()]


# -------------------------------------------------------------- activities


def test_activities_lists_only_active_workers_not_finished(rig):
    events, leases = rig
    board = TeamBoard(leases, events)

    _activate(events, "job1", "T-1", "forgeos.executor", "add retry logic", goal="wire up retries")
    _activate(events, "job1", "T-2", "cheap.local", "fix typo")
    _finish(events, "job1", "T-2")

    acts = board.activities("job1")

    assert [a.task_id for a in acts] == ["T-1"]
    assert acts[0].worker_id == "forgeos.executor"
    assert acts[0].subject == "add retry logic"
    assert acts[0].current_goal == "wire up retries"


def test_activities_carries_test_results_when_reported(rig):
    events, leases = rig
    board = TeamBoard(leases, events)

    _activate(
        events, "job1", "T-8", "forgeos.executor", "wire tests",
        goal="get suite green", tests={"passed": 3, "failed": 1},
    )

    acts = board.activities("job1")

    assert len(acts) == 1
    assert acts[0].tests is not None
    assert (acts[0].tests.passed, acts[0].tests.failed) == (3, 1)


# ---------------------------------------------------------------- who_owns


def test_who_owns_finds_holder_under_directory_lease(rig):
    events, leases = rig
    board = TeamBoard(leases, events)

    _activate(events, "job1", "T-3", "forgeos.executor", "normalise parsing")
    leases.acquire("T-3", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=1800)

    owner = board.who_owns("src/router/handler.py", "repo1")

    assert owner is not None
    assert owner.task_id == "T-3"
    assert owner.worker_id == "forgeos.executor"


def test_who_owns_returns_none_for_a_free_path(rig):
    events, leases = rig
    board = TeamBoard(leases, events)

    _activate(events, "job1", "T-3", "forgeos.executor", "normalise parsing")
    leases.acquire("T-3", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=1800)

    assert board.who_owns("src/unrelated/file.py", "repo1") is None


# ------------------------------------------------------------ would_collide


def test_would_collide_names_holder_and_goal(rig):
    events, leases = rig
    board = TeamBoard(leases, events)

    _activate(
        events, "job1", "T-12", "forgeos.executor", "normalise retries",
        goal="normalise retry parsing",
    )
    leases.acquire("T-12", "repo1", "src/b/**", LeaseType.WRITE, ttl_seconds=1800)

    report = board.would_collide(["src/a/", "src/b/config.py"], "repo1", task_id="T-99")

    assert report.collides is True
    assert len(report.overlapping) == 1
    hit = report.overlapping[0]
    assert hit.path == "src/b/config.py"
    assert hit.holder_worker == "forgeos.executor"
    assert hit.holder_task == "T-12"
    assert hit.holder_goal == "normalise retry parsing"
    assert "src/a/" in report.suggestion
    assert "src/b/config.py" in report.suggestion
    assert "forgeos.executor" in report.suggestion


def test_would_collide_excludes_the_asking_tasks_own_lease(rig):
    events, leases = rig
    board = TeamBoard(leases, events)

    _activate(events, "job1", "T-20", "forgeos.executor", "own work")
    leases.acquire("T-20", "repo1", "src/mine/**", LeaseType.WRITE, ttl_seconds=1800)

    report = board.would_collide(["src/mine/x.py"], "repo1", task_id="T-20")

    assert report.collides is False
    assert report.overlapping == []


def test_expired_lease_is_not_a_collision(rig):
    events, leases = rig
    board = TeamBoard(leases, events)

    _activate(events, "job1", "T-4", "ghost.worker", "dead task")
    leases.acquire("T-4", "repo1", "src/dead/**", LeaseType.WRITE, ttl_seconds=-1000)

    assert board.who_owns("src/dead/file.py", "repo1") is None

    report = board.would_collide(["src/dead/file.py"], "repo1", task_id="T-99")
    assert report.collides is False
    assert report.overlapping == []


# -------------------------------------------------------------- idle_paths


def test_idle_paths_returns_free_subset(rig):
    events, leases = rig
    board = TeamBoard(leases, events)

    _activate(events, "job1", "T-5", "forgeos.executor", "held work")
    leases.acquire("T-5", "repo1", "src/held/**", LeaseType.WRITE, ttl_seconds=1800)

    free = board.idle_paths(["src/held/x.py", "src/free/y.py"], "repo1")

    assert free == ["src/free/y.py"]


# -------------------------------------------------------------------- board


def test_board_is_bounded_and_reports_elided_count(rig):
    events, leases = rig
    board = TeamBoard(leases, events)

    n = BOARD_MAX_WORKERS + 4
    for i in range(n):
        _activate(events, "job1", f"T-{i:02d}", f"worker.{i}", f"subject {i}")

    text = board.board("job1")
    lines = text.splitlines()

    assert len(lines) == BOARD_MAX_WORKERS + 1
    assert lines[-1] == f"...+{n - BOARD_MAX_WORKERS} more teammate(s) not shown"


def test_board_is_deterministic_given_the_same_inputs(rig):
    events, leases = rig
    board = TeamBoard(leases, events)

    _activate(events, "job1", "T-6", "forgeos.executor", "steady work", goal="steady")
    leases.acquire("T-6", "repo1", "src/steady/**", LeaseType.WRITE, ttl_seconds=1800)

    first = board.board("job1")
    second = board.board("job1")

    assert first == second


def test_empty_job_yields_empty_board_without_crashing(rig):
    events, leases = rig
    board = TeamBoard(leases, events)

    assert board.board("no-such-job") == ""
    assert board.activities("no-such-job") == []


# ------------------------------------------------------------------ read-only


def test_read_only_enforced_across_all_query_methods(rig):
    events, leases = rig
    board = TeamBoard(leases, events)

    _activate(events, "job1", "T-7", "forgeos.executor", "steady work", goal="hold the line")
    leases.acquire("T-7", "repo1", "src/steady/**", LeaseType.WRITE, ttl_seconds=1800)
    leases.acquire("T-7", "repo1", "src/other/**", LeaseType.WRITE, ttl_seconds=-1000)  # already expired

    before = _lease_rows(leases)

    board.activities("job1")
    board.who_owns("src/steady/x.py", "repo1")
    board.would_collide(["src/steady/x.py", "src/free/y.py"], "repo1", task_id="T-99")
    board.idle_paths(["src/steady/x.py", "src/free/y.py"], "repo1")
    board.board("job1")

    after = _lease_rows(leases)

    assert before == after
