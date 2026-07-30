"""Scheduler tests — the deterministic spine.

The crash-recovery and lease-conflict tests are the ones that matter: they prove
work is never redone and never done twice concurrently. Both are token-waste bugs
before they are correctness bugs.
"""

from __future__ import annotations

import pytest

from forgeos.contracts import (
    Budget,
    JobSpec,
    Scope,
    TaskSpec,
    TaskState,
    Verdict,
    WorkerReport,
    to_micros,
)
from forgeos.core.governor import Action, Governor
from forgeos.core.scheduler import Scheduler
from forgeos.events import EventLog, EventType
from forgeos.leases import LeaseStore
from forgeos.ledger import Ledger
from forgeos.registry import Adapter, CostTier, Registry, WorkerProfile


@pytest.fixture()
def rig():
    led, ev, ls = Ledger(":memory:"), EventLog(":memory:"), LeaseStore(":memory:")
    reg = Registry(
        [
            WorkerProfile(
                worker_id="cheap.local",
                adapter=Adapter.OLLAMA,
                tier=CostTier.FREE,
                capabilities={"edit", "python"},
                can_edit_files=True,
                prior_win_rate=0.6,
            ),
            WorkerProfile(
                worker_id="pricey.cloud",
                adapter=Adapter.CLI_TEAM,
                tier=CostTier.PREMIUM,
                capabilities={"edit", "python", "security"},
                can_edit_files=True,
                prior_win_rate=0.9,
            ),
        ]
    )
    gov = Governor(led, ev, loop_threshold=3)
    sch = Scheduler(led, ev, ls, reg, gov, max_parallel=2)
    yield sch, led, ev, ls
    led.close()
    ev.close()
    ls.close()


def _job(sch, led, **kw):
    j = JobSpec(objective="build the thing", cwd=".", budget=Budget(max_usd=10.0, **kw))
    return j


def _task(job, subject, paths=(), caps=("edit", "python")):
    return TaskSpec(
        job_id=job.id,
        subject=subject,
        description="d",
        capabilities=list(caps),
        scope=Scope(paths=list(paths)),
    )


# ------------------------------------------------------------------ submit


def test_submit_records_job_and_tasks_and_events(rig):
    sch, led, ev, _ = rig
    j = _job(sch, led)
    a = _task(j, "a")
    sch.submit(j, [a])
    assert led.job(j.id)["objective"] == "build the thing"
    kinds = [e.type for e in ev.replay(j.id)]
    assert EventType.MISSION_CREATED in kinds
    assert EventType.TASK_CREATED in kinds


# ------------------------------------------------------------------- queue


def test_ready_tasks_excludes_finished_and_running(rig):
    sch, led, _, _ = rig
    j = _job(sch, led)
    a, b = _task(j, "a"), _task(j, "b")
    sch.submit(j, [a, b])
    led.set_task_state(a.id, TaskState.DONE)
    assert sch.ready_tasks(j.id) == [b.id]


def test_dependency_order_is_respected(rig):
    sch, led, _, _ = rig
    j = _job(sch, led)
    a, b = _task(j, "a"), _task(j, "b")
    sch.submit(j, [a, b])
    deps = {b.id: [a.id]}
    assert sch.ready_tasks(j.id, deps) == [a.id]
    led.set_task_state(a.id, TaskState.DONE)
    assert sch.ready_tasks(j.id, deps) == [b.id]


def test_task_whose_dependency_failed_never_becomes_ready(rig):
    """Building on known-broken work spends tokens to produce garbage."""
    sch, led, _, _ = rig
    j = _job(sch, led)
    a, b = _task(j, "a"), _task(j, "b")
    sch.submit(j, [a, b])
    led.set_task_state(a.id, TaskState.FAILED)
    deps = {b.id: [a.id]}
    assert b.id not in sch.ready_tasks(j.id, deps)
    assert "dependency failed" in sch.blocked_tasks(j.id, deps)[b.id]


def test_blocked_tasks_explains_what_it_waits_on(rig):
    sch, led, _, _ = rig
    j = _job(sch, led)
    a, b = _task(j, "a"), _task(j, "b")
    sch.submit(j, [a, b])
    assert "waiting on" in sch.blocked_tasks(j.id, {b.id: [a.id]})[b.id]


# ------------------------------------------------------------------ assign


def test_assign_prefers_the_cheapest_capable_worker(rig):
    """Free-and-capable must beat premium-and-capable, or the cost model is theatre."""
    sch, led, _, _ = rig
    j = _job(sch, led)
    a = _task(j, "a", paths=["src/"])
    sch.submit(j, [a])
    asn = sch.assign(j.id, a.id)
    assert asn is not None
    assert asn.worker_id == "cheap.local"


def test_assign_escalates_when_only_the_premium_worker_qualifies(rig):
    sch, led, _, _ = rig
    j = _job(sch, led)
    a = _task(j, "a", paths=["src/"], caps=("edit", "security"))
    sch.submit(j, [a])
    assert sch.assign(j.id, a.id).worker_id == "pricey.cloud"


def test_assign_acquires_write_leases_and_marks_running(rig):
    sch, led, _, ls = rig
    j = _job(sch, led)
    a = _task(j, "a", paths=["src/router/"])
    sch.submit(j, [a])
    asn = sch.assign(j.id, a.id)
    assert len(asn.lease_ids) == 1
    assert led.task(a.id)["state"] == "running"
    assert ls.holder("default", "src/router/x.py").task_id == a.id


def test_two_tasks_cannot_hold_overlapping_write_leases(rig):
    """The invariant that stops two workers corrupting the same file."""
    sch, led, _, _ = rig
    j = _job(sch, led)
    a, b = _task(j, "a", paths=["src/"]), _task(j, "b", paths=["src/router/"])
    sch.submit(j, [a, b])
    assert sch.assign(j.id, a.id) is not None
    assert sch.assign(j.id, b.id) is None  # overlaps a's lease


def test_disjoint_paths_run_in_parallel(rig):
    sch, led, _, _ = rig
    j = _job(sch, led)
    a, b = _task(j, "a", paths=["src/a/"]), _task(j, "b", paths=["src/b/"])
    sch.submit(j, [a, b])
    assert sch.assign(j.id, a.id) is not None
    assert sch.assign(j.id, b.id) is not None
    assert sch.active_count == 2


def test_max_parallel_is_enforced(rig):
    sch, led, _, _ = rig
    j = _job(sch, led)
    tasks = [_task(j, f"t{i}", paths=[f"src/{i}/"]) for i in range(3)]
    sch.submit(j, tasks)
    got = [sch.assign(j.id, t.id) for t in tasks]
    assert sum(1 for g in got if g is not None) == 2  # max_parallel=2
    assert sch.at_capacity()


def test_partial_lease_failure_grants_nothing(rig):
    """A worker holding some of its paths would edit files it does not own."""
    sch, led, _, ls = rig
    j = _job(sch, led)
    blocker = _task(j, "blocker", paths=["src/b/"])
    wide = _task(j, "wide", paths=["src/a/", "src/b/"])
    sch.submit(j, [blocker, wide])
    sch.assign(j.id, blocker.id)
    assert sch.assign(j.id, wide.id) is None
    # src/a/ must NOT have been left leased to the rejected task.
    assert ls.holder("default", "src/a/x.py") is None


def test_assign_returns_none_when_no_worker_has_the_capability(rig):
    sch, led, _, _ = rig
    j = _job(sch, led)
    a = _task(j, "a", paths=["src/"], caps=("kubernetes",))
    sch.submit(j, [a])
    assert sch.assign(j.id, a.id) is None


def test_assign_next_skips_a_blocked_task_and_takes_the_next(rig):
    sch, led, _, _ = rig
    j = _job(sch, led)
    a, b = _task(j, "a", paths=["src/x/"]), _task(j, "b", paths=["src/y/"])
    sch.submit(j, [a, b])
    sch.assign(j.id, a.id)
    nxt = sch.assign_next(j.id)
    assert nxt is not None and nxt.task_id == b.id


# ------------------------------------------------------------------ report


def test_accepted_task_releases_its_leases(rig):
    sch, led, _, ls = rig
    j = _job(sch, led)
    a = _task(j, "a", paths=["src/"])
    sch.submit(j, [a])
    sch.assign(j.id, a.id)
    sch.report(
        WorkerReport(task_id=a.id, worker_id="cheap.local", state=TaskState.DONE, verdict=Verdict.PASS),
        job_id=j.id,
    )
    assert ls.holder("default", "src/x.py") is None
    assert sch.active_count == 0


def test_report_writes_only_the_compact_heartbeat_to_the_event_log(rig):
    """Transcript-shaped fields must never reach the log the manager reads."""
    sch, led, ev, _ = rig
    j = _job(sch, led)
    a = _task(j, "a", paths=["src/"])
    sch.submit(j, [a])
    sch.assign(j.id, a.id)
    sch.report(
        WorkerReport(
            task_id=a.id,
            worker_id="cheap.local",
            state=TaskState.RUNNING,
            summary="a very long narration " * 50,
            evidence="lots of raw output " * 50,
            goal="fix the router",
        ),
        job_id=j.id,
    )
    cp = [e for e in ev.for_task(a.id) if e.type is EventType.CHECKPOINT_CREATED][-1]
    assert cp.payload.get("goal") == "fix the router"
    assert "summary" not in cp.payload
    assert "evidence" not in cp.payload


def test_governor_trip_on_report_pauses_and_frees_the_path(rig):
    sch, led, _, ls = rig
    j = _job(sch, led)
    a = _task(j, "a", paths=["src/"])
    sch.submit(j, [a])
    sch.assign(j.id, a.id)
    led.record_spend(j.id, "cheap.local", "m", to_micros(9.0), task_id=a.id)
    d = sch.report(
        WorkerReport(task_id=a.id, worker_id="cheap.local", state=TaskState.RUNNING),
        job_id=j.id,
        budget=Budget(max_usd=1.0),
    )
    assert d.action is Action.TRIP
    assert led.task(a.id)["state"] == "paused"
    assert ls.holder("default", "src/x.py") is None


def test_blocked_report_records_the_blocker(rig):
    sch, led, ev, _ = rig
    j = _job(sch, led)
    a = _task(j, "a", paths=["src/"])
    sch.submit(j, [a])
    sch.assign(j.id, a.id)
    sch.report(
        WorkerReport(
            task_id=a.id, worker_id="cheap.local", state=TaskState.BLOCKED,
            blocker="retry-after shape differs", needs_manager=True,
        ),
        job_id=j.id,
    )
    blocked = [e for e in ev.for_task(a.id) if e.type is EventType.TASK_BLOCKED]
    assert blocked and blocked[-1].payload["blocker"] == "retry-after shape differs"


# --------------------------------------------------------- liveness/recovery


def test_silent_worker_is_reclaimed_and_its_path_freed(rig):
    sch, led, _, ls = rig
    j = _job(sch, led)
    a = _task(j, "a", paths=["src/"])
    sch.submit(j, [a])
    asn = sch.assign(j.id, a.id)
    reclaimed = sch.expire_heartbeats(at=asn.last_heartbeat + 10_000)
    assert reclaimed == [a.id]
    assert led.task(a.id)["state"] == "queued"
    assert ls.holder("default", "src/x.py") is None


def test_a_live_worker_is_not_reclaimed(rig):
    sch, led, _, _ = rig
    j = _job(sch, led)
    a = _task(j, "a", paths=["src/"])
    sch.submit(j, [a])
    sch.assign(j.id, a.id)
    sch.heartbeat(a.id)
    assert sch.expire_heartbeats() == []


def test_resume_after_crash_does_not_redo_accepted_work(rig):
    """The entire return on event sourcing: finished work stays finished."""
    sch, led, ev, _ = rig
    j = _job(sch, led)
    done, mid = _task(j, "done", paths=["src/a/"]), _task(j, "mid", paths=["src/b/"])
    sch.submit(j, [done, mid])

    sch.assign(j.id, done.id)
    sch.report(
        WorkerReport(task_id=done.id, worker_id="cheap.local", state=TaskState.DONE,
                     verdict=Verdict.PASS),
        job_id=j.id,
    )
    sch.assign(j.id, mid.id)  # still running when the process dies

    resumed = sch.resume(j.id)
    assert mid.id in resumed
    assert done.id not in resumed


def test_resume_frees_leases_held_by_interrupted_tasks(rig):
    sch, led, _, ls = rig
    j = _job(sch, led)
    a = _task(j, "a", paths=["src/"])
    sch.submit(j, [a])
    sch.assign(j.id, a.id)
    sch.resume(j.id)
    assert ls.holder("default", "src/x.py") is None
    assert sch.active_count == 0


def test_scheduler_makes_no_model_calls(rig):
    """Structural guarantee: the deterministic layer must not import a model client.

    If this ever fails, a scheduling decision has been moved into a model and the
    zero-token property of this layer is gone.
    """
    import pathlib

    src = pathlib.Path("forgeos/core/scheduler.py").read_text(encoding="utf-8")
    for banned in ("litellm", "openai", "anthropic", "requests.post", "httpx"):
        assert banned not in src
