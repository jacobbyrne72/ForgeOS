"""Generation fencing — closes the orphaned-worker-report race.

The Forge runs tasks in a thread pool. When a task's heartbeat expires, the
scheduler reclaims it: the lease goes back to the pool and the task returns to
the queue so a replacement worker can pick it up. But reclaiming an assignment
does not kill the thread behind it — nothing stops the orphaned-yet-still-alive
worker from the previous attempt calling `report()` afterwards, writing a
result over a task that has already moved on to a different attempt (possibly
a different worker entirely).

Each task carries a monotonically increasing generation counter, persisted in
the ledger (`Ledger.task_generation` / `Ledger.bump_generation`). Reclaiming a
task bumps it BEFORE the lease is freed or the task returns to QUEUED, so the
fence is up before any replacement worker could exist. Every `WorkerReport`
carries the generation it was issued under (`Assignment.generation`), and
`Scheduler.report` refuses to act on one that does not match the task's
current generation — it records the rejection as a `REPORT_FENCED` event
(evidence an operator can see) rather than silently dropping it, and touches
nothing else: no state change, no lease release, no spend, no governor call.

`Scheduler.heartbeat` is fenced by the same counter. An orphan keeps ticking
after it is reclaimed, and an unfenced tick would refresh the liveness of
whichever attempt holds that task_id NOW — making a genuinely stuck
replacement look healthy and postponing the very reclaim that expiry exists
to perform. Heartbeat fencing differs from report fencing in two deliberate
ways, each pinned by a test below: it checks the in-memory assignment rather
than the ledger, and it accepts an unstamped call even after a bump.
"""

from __future__ import annotations

import threading

import pytest

from forgeos.contracts import (
    Budget,
    JobSpec,
    Scope,
    TaskSpec,
    TaskState,
    Verdict,
    WorkerReport,
    now,
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
        ]
    )
    gov = Governor(led, ev, loop_threshold=3)
    sch = Scheduler(led, ev, ls, reg, gov, max_parallel=2)
    yield sch, led, ev, ls
    led.close()
    ev.close()
    ls.close()


def _job(**kw):
    return JobSpec(objective="build the thing", cwd=".", budget=Budget(max_usd=10.0, **kw))


def _task(job, subject, paths=("src/",), caps=("edit", "python")):
    return TaskSpec(
        job_id=job.id,
        subject=subject,
        description="d",
        capabilities=list(caps),
        scope=Scope(paths=list(paths)),
    )


def _reclaim(sch, asn):
    """Force a reclaim: jump the clock past the heartbeat timeout."""
    reclaimed = sch.expire_heartbeats(at=asn.last_heartbeat + 10_000)
    assert reclaimed == [asn.task_id]


# ------------------------------------------------------------ assignment


def test_assignment_carries_the_current_generation(rig):
    """A fresh, never-reclaimed task is issued generation 0."""
    sch, led, _, _ = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])
    asn = sch.assign(j.id, a.id)
    assert asn.generation == 0
    assert led.task_generation(a.id) == 0


# ------------------------------------------------------------ reclaim bumps


def test_reclaim_bumps_the_task_generation(rig):
    sch, led, _, _ = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])
    asn = sch.assign(j.id, a.id)
    assert led.task_generation(a.id) == 0

    _reclaim(sch, asn)

    assert led.task_generation(a.id) == 1


def test_reclaim_then_reassign_issues_the_bumped_generation(rig):
    sch, led, _, _ = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])
    asn = sch.assign(j.id, a.id)
    _reclaim(sch, asn)

    asn2 = sch.assign(j.id, a.id)
    assert asn2.generation == 1
    assert asn2.generation != asn.generation


# ------------------------------------------------------------ accept path


def test_current_generation_report_is_accepted(rig):
    """The ordinary, never-reclaimed path must keep working unchanged."""
    sch, led, ev, ls = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])
    asn = sch.assign(j.id, a.id)

    decision = sch.report(
        WorkerReport(
            task_id=a.id, worker_id=asn.worker_id, state=TaskState.DONE,
            verdict=Verdict.PASS, generation=asn.generation,
        ),
        job_id=j.id,
    )

    assert decision.action is not Action.TRIP
    assert led.task(a.id)["state"] == "done"
    assert len(led.reports_for_task(a.id)) == 1
    assert ls.holder("default", "src/x.py") is None  # lease released normally
    assert not [e for e in ev.for_task(a.id) if e.type is EventType.REPORT_FENCED]


# ------------------------------------------------------------ fencing itself


def test_stale_generation_report_is_rejected_and_changes_nothing(rig):
    """The core bug: an orphaned worker's late report must be a no-op."""
    sch, led, ev, ls = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])

    orphan = sch.assign(j.id, a.id)          # attempt 1, generation 0
    _reclaim(sch, orphan)                    # heartbeat expires, generation -> 1
    current = sch.assign(j.id, a.id)         # attempt 2, generation 1, holds the lease

    state_before = led.task(a.id)["state"]
    spend_before = led.task_spend_micros(a.id)
    holder_before = ls.holder("default", "src/x.py")
    assert holder_before is not None and holder_before.task_id == a.id

    # The zombie from attempt 1 finally calls in, carrying the OLD generation.
    decision = sch.report(
        WorkerReport(
            task_id=a.id, worker_id=orphan.worker_id, state=TaskState.DONE,
            verdict=Verdict.PASS, generation=orphan.generation,
        ),
        job_id=j.id,
    )

    # Must not look like a governor trip -- it is a fence, not a budget event.
    assert decision.action is Action.CONTINUE
    assert "fenced" in decision.reason.lower()

    # Nothing about the task changed: same state, no spend, no report row,
    # and the CURRENT attempt's lease is still held (not freed, not taken).
    assert led.task(a.id)["state"] == state_before
    assert led.task_spend_micros(a.id) == spend_before
    assert led.reports_for_task(a.id) == []
    still_holding = ls.holder("default", "src/x.py")
    assert still_holding is not None
    assert still_holding.task_id == a.id
    assert still_holding.id == holder_before.id  # exact same lease, untouched

    # The assignment map still shows the CURRENT attempt, not the orphan's.
    assert sch._active[a.id].worker_id == current.worker_id
    assert sch._active[a.id].generation == current.generation


def test_fenced_report_is_visible_to_an_operator(rig):
    """A fenced report is still evidence, not a silent drop."""
    sch, led, ev, _ = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])

    orphan = sch.assign(j.id, a.id)
    _reclaim(sch, orphan)
    sch.assign(j.id, a.id)

    sch.report(
        WorkerReport(
            task_id=a.id, worker_id=orphan.worker_id, state=TaskState.DONE,
            generation=orphan.generation,
        ),
        job_id=j.id,
    )

    fenced = [e for e in ev.for_task(a.id) if e.type is EventType.REPORT_FENCED]
    assert len(fenced) == 1
    assert fenced[0].payload["worker"] == orphan.worker_id
    assert fenced[0].payload["report_generation"] == 0
    assert fenced[0].payload["current_generation"] == 1


# ------------------------------------------------------------ backwards compat


def test_unstamped_report_is_accepted_when_task_was_never_reclaimed(rig):
    """A caller that predates generation fencing must not be broken by it."""
    sch, led, _, _ = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])
    sch.assign(j.id, a.id)

    decision = sch.report(
        WorkerReport(task_id=a.id, worker_id="cheap.local", state=TaskState.DONE,
                     verdict=Verdict.PASS),  # generation left unset (None)
        job_id=j.id,
    )

    assert decision.action is not Action.TRIP
    assert led.task(a.id)["state"] == "done"
    assert len(led.reports_for_task(a.id)) == 1


def test_unstamped_report_is_fenced_once_a_bump_has_happened(rig):
    """Once reclaim has occurred, an unstamped report can no longer prove
    which attempt it belongs to, so it is fenced like any stale one."""
    sch, led, ev, _ = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])
    orphan = sch.assign(j.id, a.id)
    _reclaim(sch, orphan)
    sch.assign(j.id, a.id)

    decision = sch.report(
        WorkerReport(task_id=a.id, worker_id=orphan.worker_id, state=TaskState.DONE),
        job_id=j.id,  # generation left unset (None), same as a pre-fencing caller
    )

    assert decision.action is Action.CONTINUE
    assert "fenced" in decision.reason.lower()
    assert led.reports_for_task(a.id) == []
    assert [e for e in ev.for_task(a.id) if e.type is EventType.REPORT_FENCED]


# ------------------------------------------------------------ concurrency


def test_concurrent_reports_from_two_generations_resolve_to_exactly_one_winner(rig):
    """Two attempts report at (as close to) the same instant. Exactly one
    must win, regardless of thread scheduling, and neither call may raise."""
    sch, led, ev, _ = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])

    orphan = sch.assign(j.id, a.id)      # generation 0
    _reclaim(sch, orphan)                # generation -> 1
    current = sch.assign(j.id, a.id)     # generation 1

    stale_report = WorkerReport(
        task_id=a.id, worker_id=orphan.worker_id, state=TaskState.DONE,
        verdict=Verdict.PASS, generation=orphan.generation, summary="stale-attempt",
    )
    winning_report = WorkerReport(
        task_id=a.id, worker_id=current.worker_id, state=TaskState.DONE,
        verdict=Verdict.PASS, generation=current.generation, summary="current-attempt",
    )

    barrier = threading.Barrier(2)
    results: dict[str, object] = {}
    errors: list[BaseException] = []

    def _call(name: str, report: WorkerReport) -> None:
        try:
            barrier.wait(timeout=5)
            results[name] = sch.report(report, job_id=j.id)
        except BaseException as exc:  # noqa: BLE001 — surfaced via `errors`, not swallowed
            errors.append(exc)

    threads = [
        threading.Thread(target=_call, args=("stale", stale_report)),
        threading.Thread(target=_call, args=("current", winning_report)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"scheduler.report() raised under concurrency: {errors}"
    assert len(results) == 2

    rows = led.reports_for_task(a.id)
    assert len(rows) == 1
    assert rows[0]["summary"] == "current-attempt"
    assert led.task(a.id)["state"] == "done"

    fenced = [e for e in ev.for_task(a.id) if e.type is EventType.REPORT_FENCED]
    assert len(fenced) == 1
    assert fenced[0].payload["report_generation"] == 0


# ------------------------------------------------------- heartbeat fencing


def _overdue(sch, asn) -> float:
    """Backdate an assignment so it is already past the reclaim deadline.

    Lets a test assert on the observable consequence -- whether
    `expire_heartbeats` reclaims -- instead of on a wall-clock delta, which
    two `now()` calls in the same millisecond would make flaky.
    """
    asn.last_heartbeat = now() - (sch.heartbeat_timeout * 2)
    return asn.last_heartbeat


def test_matching_heartbeat_refreshes_the_current_assignment(rig):
    """The holder of the current assignment can still prove it is alive."""
    sch, _, _, _ = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])
    asn = sch.assign(j.id, a.id)
    _overdue(sch, asn)

    accepted = sch.heartbeat(a.id, generation=asn.generation, worker_id=asn.worker_id)

    assert accepted is True
    assert sch.expire_heartbeats() == []  # refreshed, no longer stale


def test_orphan_heartbeat_cannot_refresh_a_newer_attempt(rig):
    """The core bug: a reclaimed worker must not keep a LATER attempt alive.

    Attempt 1 is reclaimed and attempt 2 takes the slot. The orphaned thread
    from attempt 1 is still running and still heartbeating. If those
    heartbeats land on attempt 2's assignment, a genuinely stuck attempt 2
    looks healthy and its own reclaim is postponed indefinitely.
    """
    sch, _, _, _ = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])

    orphan = sch.assign(j.id, a.id)      # attempt 1, generation 0
    _reclaim(sch, orphan)                # heartbeat expires, generation -> 1
    current = sch.assign(j.id, a.id)     # attempt 2, generation 1, holds the slot

    stamp_before = current.last_heartbeat

    accepted = sch.heartbeat(
        a.id, generation=orphan.generation, worker_id=orphan.worker_id
    )

    assert accepted is False
    # Attempt 2's liveness is exactly as it was -- the orphan touched nothing.
    assert sch._active[a.id] is current
    assert sch._active[a.id].last_heartbeat == stamp_before
    # And attempt 2 is still reclaimable on ITS OWN schedule, which is the
    # consequence that matters: the orphan bought the stuck worker no time.
    assert sch.expire_heartbeats(at=stamp_before + sch.heartbeat_timeout + 1) == [a.id]


def test_heartbeat_from_a_different_worker_is_fenced(rig):
    """Right generation, wrong worker. Two assignments can share a generation
    (a task assigned twice with no reclaim between), so the worker identity is
    checked as well rather than trusting the counter alone."""
    sch, _, _, _ = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])
    asn = sch.assign(j.id, a.id)
    stamp_before = _overdue(sch, asn)

    accepted = sch.heartbeat(a.id, generation=asn.generation, worker_id="someone.else")

    assert accepted is False
    assert sch._active[a.id].last_heartbeat == stamp_before
    assert sch.expire_heartbeats() == [a.id]  # still stale, still reclaimed


def test_heartbeat_for_an_unknown_task_is_a_no_op(rig):
    """No assignment to refresh is a False, not a KeyError."""
    sch, _, _, _ = rig
    assert sch.heartbeat("tsk-does-not-exist") is False


# ------------------------------------------- heartbeat backwards compat


def test_unstamped_heartbeat_still_refreshes(rig):
    """A caller that predates heartbeat fencing must not be broken by it."""
    sch, _, _, _ = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])
    asn = sch.assign(j.id, a.id)
    _overdue(sch, asn)

    assert sch.heartbeat(a.id) is True  # no generation, no worker_id
    assert sch.expire_heartbeats() == []


def test_unstamped_heartbeat_is_not_fenced_after_a_bump(rig):
    """Deliberately ASYMMETRIC with `report()`, which fences unstamped calls
    once a bump has happened.

    The two cases have opposite cost profiles. An unstamped report accepted
    after a bump writes a wrong result into the ledger permanently; refusing
    it costs one retry. An unstamped heartbeat refused after a bump starves
    the CURRENT, healthy worker of any way to prove liveness -- it gets
    reclaimed, reassigned, refused again, forever. Accepting it costs only a
    late reclaim of a stuck worker, which is the mild condition this fencing
    exists to improve, not a corruption. So the strict rule is right for
    reports and wrong for heartbeats.
    """
    sch, _, _, _ = rig
    j = _job()
    a = _task(j, "a")
    sch.submit(j, [a])
    orphan = sch.assign(j.id, a.id)
    _reclaim(sch, orphan)
    current = sch.assign(j.id, a.id)
    _overdue(sch, current)

    assert sch.heartbeat(a.id) is True
    assert sch.expire_heartbeats() == []
