"""A task that keeps failing must not be resumed forever.

An audit of other harnesses reported this as MISSING and recommended building a
POISONED state. It is not missing -- checking before building found the whole
mechanism already present and correct, spread across three modules that never
mention each other:

    Governor.check_task()   trips on attempts > budget.max_iterations
    EventType.GOVERNOR_TRIPPED  projects to TaskState.PAUSED
    EventLog.resumable_tasks()  returns RUNNING/QUEUED/VERIFYING/BLOCKED only

so a tripped task is PAUSED and a restarted kernel does not pick it back up.

These tests exist because that property was load-bearing and unpinned. Nothing
asserted the connection between the three, so any one of them could have been
changed independently -- adding PAUSED to the resumable set would have looked
harmless and quietly created an infinite retry loop that spends real money.
"""

from __future__ import annotations

from forgeos.events import EventLog, EventType
from forgeos.contracts import TaskState


def _log(tmp_path) -> EventLog:
    return EventLog(tmp_path / "ev.db")


def test_a_governor_trip_pauses_the_task(tmp_path):
    log = _log(tmp_path)
    log.append("j", EventType.TASK_CREATED, task_id="t1")
    log.append("j", EventType.GOVERNOR_TRIPPED, task_id="t1",
               reason="iteration ceiling exceeded", limit="3", observed="4")
    assert log.project_task_states("j")["t1"] is TaskState.PAUSED


def test_a_tripped_task_is_not_resumed(tmp_path):
    """THE property. If PAUSED were resumable, a task that keeps failing would
    be retried forever, and every retry costs real money."""
    log = _log(tmp_path)
    log.append("j", EventType.TASK_CREATED, task_id="t1")
    log.append("j", EventType.SESSION_STARTED, task_id="t1")
    log.append("j", EventType.GOVERNOR_TRIPPED, task_id="t1",
               reason="iteration ceiling exceeded", limit="3", observed="4")
    assert "t1" not in log.resumable_tasks("j")


def test_a_genuinely_mid_flight_task_is_still_resumed(tmp_path):
    """The guard must not swallow the case resume exists for."""
    log = _log(tmp_path)
    log.append("j", EventType.TASK_CREATED, task_id="t2")
    log.append("j", EventType.SESSION_STARTED, task_id="t2")
    assert "t2" in log.resumable_tasks("j")


def test_attempts_are_counted_from_session_starts(tmp_path):
    """What the governor's iteration ceiling reads."""
    log = _log(tmp_path)
    log.append("j", EventType.TASK_CREATED, task_id="t3")
    for _ in range(4):
        log.append("j", EventType.SESSION_STARTED, task_id="t3")
    assert log.attempt_count("t3") == 4


def test_paused_is_absent_from_the_resumable_set():
    """Pinned directly against the source, not inferred from a scenario: adding
    PAUSED here would look harmless in review and create an unbounded retry
    loop."""
    import inspect

    from forgeos.events import EventLog as EL

    src = inspect.getsource(EL.resumable_tasks)
    assert "PAUSED" not in src, (
        "PAUSED became resumable -- a governor-tripped task would be retried "
        "forever, spending real money each time"
    )
