"""Tests for Auto Memory — mining ledger/event history into unverified claims.

Every scenario below either fires a pattern (>= min_occurrences) or doesn't
(below threshold), and every filed claim must land at the bottom tier of
`knowledge.claims` — UNVERIFIED, never PROMOTED — per AGENTS.md rule 10.
"""

from __future__ import annotations

import pytest

from forgeos.contracts import Budget, JobSpec, Scope, TaskSpec, TaskState, Verdict, WorkerReport
from forgeos.events import EventLog, EventType
from forgeos.knowledge.automemory import (
    PatternKind,
    file_candidates,
    mine_lessons,
)
from forgeos.knowledge.claims import ClaimStore, VerificationStatus
from forgeos.ledger import Ledger


@pytest.fixture()
def rig():
    ledger = Ledger(":memory:")
    events = EventLog(":memory:")
    claims = ClaimStore(":memory:")
    job = JobSpec(objective="mine some lessons", cwd=".", budget=Budget(max_usd=10.0))
    ledger.open_job(job)
    yield ledger, events, claims, job
    ledger.close()
    events.close()
    claims.close()


def _task(job, subject, *, capabilities=("edit",), paths=("src/",), acceptance=("1. tests pass",)):
    return TaskSpec(
        job_id=job.id,
        subject=subject,
        description="d",
        capabilities=list(capabilities),
        scope=Scope(paths=list(paths)),
        acceptance=list(acceptance),
    )


def _settle(ledger, events, job, task, *, worker_id, accepted, reasons=(), usd_micros=0):
    """Mirror `forge.py`'s exact wiring: record spend, record the merge-gate's
    own report, then append TASK_ACCEPTED/TASK_REJECTED with the joined
    reasons as the event's `reason` payload.
    """
    task_id = ledger.add_task(task)
    if usd_micros:
        ledger.record_spend(job.id, worker_id, "worker-model", usd_micros, task_id=task_id)
    ledger.record_report(
        WorkerReport(
            task_id=task_id,
            worker_id=worker_id,
            state=TaskState.DONE if accepted else TaskState.FAILED,
            verdict=Verdict.PASS if accepted else Verdict.FAIL,
            summary="merge gate ruling",
            blocker="" if accepted else "; ".join(reasons),
            usd_micros=0,
        )
    )
    events.append(
        job.id,
        EventType.TASK_ACCEPTED if accepted else EventType.TASK_REJECTED,
        task_id=task_id,
        reason="merge gate allowed" if accepted else "; ".join(reasons),
    )
    return task_id


# ------------------------------------------------------- pattern 1: merge refusals


def test_repeated_merge_refusal_reason_is_mined(rig):
    ledger, events, claims, job = rig
    for i in range(2):
        _settle(
            ledger, events, job, _task(job, f"fix bug {i}"),
            worker_id="w1", accepted=False, reasons=["no independent review"],
        )

    candidates = mine_lessons(ledger, events)
    hits = [c for c in candidates if c.kind is PatternKind.MERGE_REFUSAL_REPEATED]
    assert len(hits) == 1
    assert hits[0].subject == "no independent review"
    assert hits[0].occurrences == 2
    assert {r.task_id for r in hits[0].receipts} == {
        r["id"] for r in ledger.tasks_for_job(job.id)
    }


def test_single_merge_refusal_reason_is_below_threshold(rig):
    ledger, events, claims, job = rig
    _settle(
        ledger, events, job, _task(job, "fix bug once"),
        worker_id="w1", accepted=False, reasons=["no tests passed"],
    )

    candidates = mine_lessons(ledger, events)
    assert not [c for c in candidates if c.kind is PatternKind.MERGE_REFUSAL_REPEATED]


def test_one_flaky_task_retried_several_times_is_one_task_not_several(rig):
    """A single task rejected across multiple attempts must count as ONE
    occurrence, not one per TASK_REJECTED event -- forge.py emits one event
    per attempt on the SAME task_id.
    """
    ledger, events, claims, job = rig
    task = _task(job, "retry me")
    task_id = ledger.add_task(task)
    for _ in range(3):
        events.append(job.id, EventType.TASK_REJECTED, task_id=task_id, reason="no independent review")
    ledger.record_report(
        WorkerReport(
            task_id=task_id, worker_id="w1", state=TaskState.FAILED, verdict=Verdict.FAIL,
            summary="merge gate ruling", blocker="no independent review", usd_micros=0,
        )
    )

    candidates = mine_lessons(ledger, events)
    hits = [c for c in candidates if c.kind is PatternKind.MERGE_REFUSAL_REPEATED]
    assert not hits  # only 1 distinct task, below the default threshold of 2


# ------------------------------------------------------ pattern 2: worker miscast


def test_worker_never_winning_a_capability_is_mined(rig):
    ledger, events, claims, job = rig
    for i in range(2):
        _settle(
            ledger, events, job, _task(job, f"py task {i}", capabilities=("python",)),
            worker_id="flaky.local", accepted=False, reasons=["no tests passed"],
        )

    candidates = mine_lessons(ledger, events)
    hits = [c for c in candidates if c.kind is PatternKind.CAPABILITY_MISCAST]
    assert len(hits) == 1
    assert hits[0].subject == "flaky.local / python"
    assert hits[0].occurrences == 2


def test_worker_with_one_win_is_not_miscast(rig):
    ledger, events, claims, job = rig
    _settle(
        ledger, events, job, _task(job, "py task a", capabilities=("python",)),
        worker_id="good.local", accepted=False, reasons=["no tests passed"],
    )
    _settle(
        ledger, events, job, _task(job, "py task b", capabilities=("python",)),
        worker_id="good.local", accepted=True,
    )

    candidates = mine_lessons(ledger, events)
    hits = [c for c in candidates if c.kind is PatternKind.CAPABILITY_MISCAST]
    assert not [h for h in hits if h.subject.startswith("good.local")]


# ----------------------------------------------------- pattern 3: cost outliers


def test_expensive_worker_relative_to_fleet_median_is_mined(rig):
    ledger, events, claims, job = rig
    cheap_workers = ["cheap-a.local", "cheap-b.local"]
    for w in cheap_workers:
        for i in range(2):
            _settle(
                ledger, events, job, _task(job, f"{w} task {i}", capabilities=("python",)),
                worker_id=w, accepted=True, usd_micros=100_000,
            )
    for i in range(2):
        _settle(
            ledger, events, job, _task(job, f"pricey task {i}", capabilities=("python",)),
            worker_id="pricey.local", accepted=True, usd_micros=1_000_000,
        )

    candidates = mine_lessons(ledger, events)
    hits = [c for c in candidates if c.kind is PatternKind.CAPABILITY_EXPENSIVE]
    assert len(hits) == 1
    assert hits[0].subject == "pricey.local / python"
    assert hits[0].occurrences == 2


def test_uniform_cost_across_fleet_produces_no_outlier(rig):
    ledger, events, claims, job = rig
    for w in ["a.local", "b.local"]:
        for i in range(2):
            _settle(
                ledger, events, job, _task(job, f"{w} task {i}", capabilities=("python",)),
                worker_id=w, accepted=True, usd_micros=100_000,
            )

    candidates = mine_lessons(ledger, events)
    assert not [c for c in candidates if c.kind is PatternKind.CAPABILITY_EXPENSIVE]


# -------------------------------------------------- pattern 4: duplicate contracts


def test_identical_contract_submitted_twice_is_mined(rig):
    ledger, events, claims, job = rig
    for _ in range(2):
        _settle(
            ledger, events, job, _task(job, "always the same task"),
            worker_id="w1", accepted=True,
        )
    # A genuinely different contract must not be swept in.
    _settle(ledger, events, job, _task(job, "a different task"), worker_id="w1", accepted=True)

    candidates = mine_lessons(ledger, events)
    hits = [c for c in candidates if c.kind is PatternKind.DUPLICATE_CONTRACT_RESUBMITTED]
    assert len(hits) == 1
    assert hits[0].subject == "always the same task"
    assert hits[0].occurrences == 2


def test_single_submission_is_not_a_duplicate(rig):
    ledger, events, claims, job = rig
    _settle(ledger, events, job, _task(job, "only once"), worker_id="w1", accepted=True)

    candidates = mine_lessons(ledger, events)
    assert not [c for c in candidates if c.kind is PatternKind.DUPLICATE_CONTRACT_RESUBMITTED]


# --------------------------------------------------------------- thresholds


def test_min_occurrences_is_configurable(rig):
    ledger, events, claims, job = rig
    for i in range(2):
        _settle(
            ledger, events, job, _task(job, f"fix bug {i}"),
            worker_id="w1", accepted=False, reasons=["no independent review"],
        )

    def _refusal_hits(min_occurrences):
        return [
            c for c in mine_lessons(ledger, events, min_occurrences=min_occurrences)
            if c.kind is PatternKind.MERGE_REFUSAL_REPEATED
        ]

    assert _refusal_hits(3) == []
    assert len(_refusal_hits(2)) == 1


# ------------------------------------------------------------------- filing


def test_filed_candidates_land_only_in_unverified_tier(rig):
    ledger, events, claims, job = rig
    for i in range(2):
        _settle(
            ledger, events, job, _task(job, f"fix bug {i}"),
            worker_id="w1", accepted=False, reasons=["no independent review"],
        )

    candidates = mine_lessons(ledger, events)
    filed = file_candidates(claims, candidates)

    assert filed
    for claim in filed:
        assert claim.status is VerificationStatus.UNVERIFIED
    assert claims.instructions() == []  # nothing promoted


def test_below_threshold_patterns_file_nothing(rig):
    ledger, events, claims, job = rig
    _settle(
        ledger, events, job, _task(job, "solo failure"),
        worker_id="w1", accepted=False, reasons=["no tests passed"],
    )

    candidates = mine_lessons(ledger, events)
    assert candidates == []
    assert file_candidates(claims, candidates) == []
    assert claims.stats() == {}


def test_remining_is_idempotent(rig):
    """Mining and filing the same unchanged ledger twice must not create a
    second claim or a second source -- `ClaimStore.add` is itself idempotent
    per (text, source_ref), and mining a stable history must reproduce the
    same text and source_ref every time.
    """
    ledger, events, claims, job = rig
    for i in range(2):
        _settle(
            ledger, events, job, _task(job, f"fix bug {i}"),
            worker_id="w1", accepted=False, reasons=["no independent review"],
        )

    first_candidates = mine_lessons(ledger, events)
    second_candidates = mine_lessons(ledger, events)
    assert first_candidates == second_candidates

    first_filed = file_candidates(claims, first_candidates)
    stats_after_first = claims.stats()
    second_filed = file_candidates(claims, second_candidates)
    stats_after_second = claims.stats()

    assert stats_after_first == stats_after_second
    assert {c.id for c in first_filed} == {c.id for c in second_filed}
    for claim in first_filed:
        assert claims.source_count(claim.id) == 1
