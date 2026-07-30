"""Governor tests.

The loop-detection and failure-classification tests here encode the two most
expensive mistakes an agent harness can make: burning a whole budget on a worker
that is producing nothing, and escalating to a premium model for a failure no model
can fix.
"""

from __future__ import annotations

import pytest

from forgeos.contracts import (
    Budget,
    FailureClass,
    JobSpec,
    TaskSpec,
    TaskState,
    WorkerReport,
    to_micros,
)
from forgeos.core.governor import Action, Governor, Remedy
from forgeos.events import EventLog, EventType
from forgeos.ledger import Ledger


@pytest.fixture()
def led():
    x = Ledger(":memory:")
    yield x
    x.close()


@pytest.fixture()
def events():
    e = EventLog(":memory:")
    yield e
    e.close()


@pytest.fixture()
def job(led):
    j = JobSpec(objective="ship it", cwd=".", budget=Budget(max_usd=1.0, max_seconds=3600))
    led.open_job(j)
    return j


@pytest.fixture()
def gov(led, events):
    return Governor(led, events, loop_threshold=3)


def _task(led, job, **kw):
    t = TaskSpec(job_id=job.id, subject="s", description="d", **kw)
    led.add_task(t)
    return t


def _report(led, task_id, *, files, evidence, state=TaskState.RUNNING):
    led.record_report(
        WorkerReport(
            task_id=task_id,
            worker_id="w1",
            state=state,
            files_touched=files,
            evidence=evidence,
        )
    )


# ------------------------------------------------------------------ ceilings


def test_fresh_job_continues(gov, job):
    d = gov.check_job(job.id)
    assert d.action is Action.CONTINUE
    assert not d.stopped


def test_unknown_job_trips_rather_than_crashing(gov):
    assert gov.check_job("job_nope").action is Action.TRIP


def test_spending_exactly_to_the_cap_is_allowed(gov, led, job):
    led.record_spend(job.id, "w", "m", to_micros(1.0))
    assert gov.check_job(job.id).action is not Action.TRIP


def test_one_microdollar_over_the_cap_trips(gov, led, job):
    led.record_spend(job.id, "w", "m", to_micros(1.0) + 1)
    d = gov.check_job(job.id)
    assert d.action is Action.TRIP
    assert "usd" in d.reason


def test_trip_is_recorded_for_the_human(gov, led, job):
    led.record_spend(job.id, "w", "m", to_micros(5.0))
    gov.check_job(job.id)
    trips = led.trips_for_job(job.id)
    assert len(trips) == 1
    assert "1.0000" in trips[0]["limit_desc"]


def test_trip_emits_an_event_so_state_survives_restart(gov, led, events, job):
    led.record_spend(job.id, "w", "m", to_micros(5.0))
    gov.check_job(job.id)
    kinds = [e.type for e in events.replay(job.id)]
    assert EventType.GOVERNOR_TRIPPED in kinds


def test_sixty_percent_burn_narrows_context(gov, led, job):
    led.record_spend(job.id, "w", "m", to_micros(0.65))
    d = gov.check_job(job.id)
    assert d.action is Action.NARROW
    assert d.burn_fraction == pytest.approx(0.65)


def test_eighty_percent_burn_forces_focus(gov, led, job):
    led.record_spend(job.id, "w", "m", to_micros(0.85))
    assert gov.check_job(job.id).action is Action.FOCUS


def test_wall_clock_ceiling_trips(led, events):
    j = JobSpec(objective="slow", cwd=".", budget=Budget(max_seconds=1))
    led.open_job(j)
    # Backdate creation rather than sleeping — a test must not cost wall-clock either.
    led._conn.execute(  # noqa: SLF001 - deliberate, no public setter for this
        "UPDATE jobs SET created_at = created_at - 100 WHERE id=?", (j.id,)
    )
    led._conn.commit()  # noqa: SLF001
    d = Governor(led, events).check_job(j.id)
    assert d.action is Action.TRIP
    assert "wall-clock" in d.reason


def test_task_usd_ceiling_trips(gov, led, job):
    t = _task(led, job)
    led.record_spend(job.id, "w", "m", to_micros(3.0), task_id=t.id)
    d = gov.check_task(t.id, job.id, Budget(max_usd=2.0))
    assert d.action is Action.TRIP
    assert "task usd" in d.reason


def test_iteration_ceiling_trips(gov, led, events, job):
    t = _task(led, job)
    for _ in range(4):
        events.append(job.id, EventType.SESSION_STARTED, task_id=t.id)
    d = gov.check_task(t.id, job.id, Budget(max_iterations=3))
    assert d.action is Action.TRIP
    assert "iteration" in d.reason


# ------------------------------------------------------------ loop detection


def test_churn_with_identical_evidence_is_a_loop(gov, led, job):
    """Three rewrites of one file, same failure every time. Inside every budget,
    producing nothing — exactly what the ceilings cannot see."""
    t = _task(led, job)
    for _ in range(3):
        _report(led, t.id, files=["router.py"], evidence="1 failed, 14 passed")
    d = gov.check_task(t.id, job.id, Budget())
    assert d.action is Action.TRIP
    assert d.loop is not None
    assert d.loop.file == "router.py"
    assert d.loop.touches == 3


def test_repeated_touches_with_changing_evidence_is_progress_not_a_loop(gov, led, job):
    """Iterating toward green must never be mistaken for spinning."""
    t = _task(led, job)
    for ev in ("3 failed", "2 failed", "1 failed"):
        _report(led, t.id, files=["router.py"], evidence=ev)
    assert gov.check_task(t.id, job.id, Budget()).action is Action.CONTINUE


def test_below_threshold_touches_are_not_a_loop(gov, led, job):
    t = _task(led, job)
    for _ in range(2):
        _report(led, t.id, files=["router.py"], evidence="same")
    assert gov.detect_loop(t.id) is None


def test_task_with_no_reports_has_no_loop(gov, led, job):
    t = _task(led, job)
    assert gov.detect_loop(t.id) is None


def test_loop_detector_reports_the_worst_offending_file(gov, led, job):
    t = _task(led, job)
    for _ in range(4):
        _report(led, t.id, files=["hot.py", "cold.py"], evidence="stuck")
    loop = gov.detect_loop(t.id)
    assert loop is not None
    assert loop.touches == 4


# -------------------------------------------------- failure classification


@pytest.mark.parametrize(
    "failure,expected",
    [
        (FailureClass.ENVIRONMENT, Remedy.REPAIR_ENVIRONMENT),
        (FailureClass.CONTEXT, Remedy.ADD_CONTEXT),
        (FailureClass.MODEL, Remedy.ESCALATE_MODEL),
        (FailureClass.SPECIFICATION, Remedy.RETURN_TO_MANAGER),
        (FailureClass.POLICY, Remedy.ASK_HUMAN),
        (FailureClass.TRANSIENT, Remedy.BACKOFF_RETRY),
    ],
)
def test_each_failure_class_maps_to_its_cheapest_real_fix(gov, failure, expected):
    assert gov.remedy_for(failure, attempts=0, max_attempts=3) is expected


def test_only_model_failures_justify_paying_more_per_token(gov):
    assert gov.should_escalate_model(FailureClass.MODEL) is True
    for fc in (
        FailureClass.ENVIRONMENT,
        FailureClass.CONTEXT,
        FailureClass.SPECIFICATION,
        FailureClass.POLICY,
        FailureClass.TRANSIENT,
    ):
        assert gov.should_escalate_model(fc) is False


def test_escalating_a_broken_environment_is_flagged_as_pure_waste(gov):
    """A broken venv fails identically at every price tier."""
    assert gov.wasted_escalation_would_have_cost(FailureClass.ENVIRONMENT) is True
    assert gov.wasted_escalation_would_have_cost(FailureClass.MODEL) is False


def test_exhausted_attempts_hand_policy_failures_to_a_human(gov):
    assert gov.remedy_for(FailureClass.POLICY, attempts=3, max_attempts=3) is Remedy.ASK_HUMAN


def test_exhausted_attempts_stop_rather_than_escalate_tiers(gov):
    """Out of attempts means stop, not "try the expensive model once more"."""
    assert gov.remedy_for(FailureClass.MODEL, attempts=5, max_attempts=3) is Remedy.GIVE_UP


def test_governor_works_without_an_event_log(led, job):
    """Events are optional; the ledger alone must still enforce ceilings."""
    g = Governor(led)
    led.record_spend(job.id, "w", "m", to_micros(9.0))
    assert g.check_job(job.id).action is Action.TRIP
