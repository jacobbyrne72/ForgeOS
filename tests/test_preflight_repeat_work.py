"""Receipt-aware preflight: never pay for the same task twice.

Covers `forgeos.economy.preflight.task_fingerprint` / `prior_outcome` /
`check_repeat_work` — an exact-match ledger lookup, never fuzzy similarity.
No LLM calls, no network; `Ledger(":memory:")` throughout.
"""

from __future__ import annotations

import pytest

from forgeos.contracts import JobSpec, Scope, TaskSpec, TaskState, Verdict, WorkerReport, new_id
from forgeos.economy.preflight import (
    Decision,
    check_repeat_work,
    matching_tasks,
    prior_outcome,
    task_fingerprint,
)
from forgeos.ledger import Ledger


@pytest.fixture()
def ledger():
    led = Ledger(":memory:")
    yield led
    led.close()


def _spec(**kw) -> TaskSpec:
    # Each call gets its own fresh job_id by default -- these tests seed one
    # job per task, and a fixed placeholder would collide (jobs.id is a
    # PRIMARY KEY) the moment a test seeds more than one task.
    base = dict(
        subject="Add retry logic to the gateway client",
        description="whatever prose happens to be here this time",
        acceptance=["python -m pytest tests/test_gateway.py -q passes"],
        scope=Scope(paths=["forgeos/gateway/client.py"]),
        capabilities=["python"],
    )
    base.setdefault("job_id", new_id("job"))
    base.update(kw)
    return TaskSpec(**base)


def _seed(ledger: Ledger, spec: TaskSpec, *, state: TaskState, verdict: Verdict | None,
           usd_micros: int = 0, evidence: str = "") -> TaskSpec:
    """Submit `spec` to the ledger and record one settling report for it."""
    job = JobSpec(id=spec.job_id, objective="obj", cwd=".")
    ledger.open_job(job)
    ledger.add_task(spec)
    if usd_micros:
        ledger.record_spend(job.id, "w1", "free/model", usd_micros, task_id=spec.id)
    ledger.record_report(WorkerReport(
        task_id=spec.id, worker_id="w1", state=state, verdict=verdict, evidence=evidence,
    ))
    return spec


# --------------------------------------------------------------- fingerprint


def test_fingerprint_stable_across_list_order_and_path_style():
    a = _spec(
        capabilities=["python", "git"],
        scope=Scope(paths=["b/two.py", "./a/one.py"]),
    )
    b = _spec(
        capabilities=["git", "python"],
        scope=Scope(paths=["a/one.py", "b\\two.py"]),
    )
    assert task_fingerprint(a) == task_fingerprint(b)


def test_fingerprint_ignores_description_and_id():
    a = _spec(description="first draft of the prose")
    b = _spec(description="totally rewritten prose, same task")
    assert task_fingerprint(a) == task_fingerprint(b)
    assert a.id != b.id  # sanity: these really are two distinct submissions


def test_fingerprint_changes_when_scope_changes():
    a = _spec(scope=Scope(paths=["forgeos/gateway/client.py"]))
    b = _spec(scope=Scope(paths=["forgeos/gateway/client.py", "forgeos/gateway/dead_models.py"]))
    assert task_fingerprint(a) != task_fingerprint(b)


def test_fingerprint_changes_when_capabilities_change():
    a = _spec(capabilities=["python"])
    b = _spec(capabilities=["python", "rust"])
    assert task_fingerprint(a) != task_fingerprint(b)


def test_fingerprint_changes_when_acceptance_changes():
    a = _spec(acceptance=["a passes"])
    b = _spec(acceptance=["a passes", "b passes"])
    assert task_fingerprint(a) != task_fingerprint(b)


# --------------------------------------------------------------- prior_outcome


def test_empty_ledger_has_no_prior_outcome(ledger):
    assert prior_outcome(ledger, task_fingerprint(_spec())) is None
    assert matching_tasks(ledger, task_fingerprint(_spec())) == []


def test_prior_outcome_finds_the_settled_match(ledger):
    spec = _spec()
    _seed(ledger, spec, state=TaskState.DONE, verdict=Verdict.PASS,
          usd_micros=1_500_000, evidence="merge gate ruling")
    prior = prior_outcome(ledger, task_fingerprint(spec))
    assert prior is not None
    assert prior.task_id == spec.id
    assert prior.state == "done"
    assert prior.usd_micros == 1_500_000


def test_prior_outcome_ignores_a_still_running_task(ledger):
    # A submitted-but-unsettled task is not an "outcome" yet -- must not surface.
    spec = _spec()
    job = JobSpec(id=spec.job_id, objective="obj", cwd=".")
    ledger.open_job(job)
    ledger.add_task(spec)  # defaults to PENDING; never reported
    assert prior_outcome(ledger, task_fingerprint(spec)) is None


# ----------------------------------------------------------- check_repeat_work


def test_clean_pass_when_ledger_is_empty(ledger):
    v = check_repeat_work(ledger, _spec())
    assert v.decision is Decision.ALLOW
    assert v.allowed
    assert v.prior is None


def test_accepted_prior_refuses_with_receipt(ledger):
    prior_spec = _spec()
    _seed(ledger, prior_spec, state=TaskState.DONE, verdict=Verdict.PASS,
          usd_micros=2_500_000, evidence="merge gate ruling")

    new_spec = _spec(description="resubmitted later, same contract")
    v = check_repeat_work(ledger, new_spec)

    assert v.decision is Decision.REFUSE_DUPLICATE
    assert not v.allowed
    assert prior_spec.id in v.reason
    assert "$2.5000" in v.reason
    assert v.prior is not None and v.prior.task_id == prior_spec.id


def test_raise_if_refused_raises_on_a_duplicate(ledger):
    prior_spec = _spec()
    _seed(ledger, prior_spec, state=TaskState.DONE, verdict=Verdict.PASS, usd_micros=1)
    v = check_repeat_work(ledger, _spec())
    with pytest.raises(Exception):
        v.raise_if_refused()


def test_skip_opts_out_of_a_duplicate_refusal(ledger):
    prior_spec = _spec()
    _seed(ledger, prior_spec, state=TaskState.DONE, verdict=Verdict.PASS, usd_micros=1)
    v = check_repeat_work(ledger, _spec(), skip=True)
    assert v.decision is Decision.ALLOW
    assert v.allowed


def test_two_unreviewed_failures_are_advisory_not_a_refusal(ledger):
    # Explicit, strictly-increasing created_at: which of two same-fingerprint
    # failures is "most recent" must not depend on real-clock tie-breaking.
    first = _spec(created_at=2_000.0)
    second = _spec(description="second attempt, same contract", created_at=2_001.0)
    _seed(ledger, first, state=TaskState.FAILED, verdict=None,
          evidence="ModuleNotFoundError: no module named forgeos_test_dep")
    _seed(ledger, second, state=TaskState.FAILED, verdict=None,
          evidence="ModuleNotFoundError: no module named forgeos_test_dep")

    v = check_repeat_work(ledger, _spec(description="third attempt"))

    assert v.decision is Decision.ALLOW  # advice, never a hard refusal
    assert v.allowed
    assert "2 prior attempts" in v.reason
    assert second.id in v.reason  # most recent failure named


def test_one_unreviewed_failure_is_not_enough_for_the_advisory(ledger):
    spec = _spec()
    _seed(ledger, spec, state=TaskState.FAILED, verdict=None, evidence="pip install failed")
    v = check_repeat_work(ledger, _spec(description="retry"))
    assert v.decision is Decision.ALLOW
    assert "prior attempts" not in v.reason


def test_merge_gate_rejections_do_not_count_toward_the_advisory(ledger):
    # verdict='fail' means a patch WAS reviewed and rejected -- not an
    # environment-shaped failure -- so it must not feed the failure count.
    for i in range(3):
        spec = _spec(description=f"attempt {i}")
        _seed(ledger, spec, state=TaskState.FAILED, verdict=Verdict.FAIL,
              evidence="merge gate ruling: no tests run")
    v = check_repeat_work(ledger, _spec(description="latest attempt"))
    assert v.decision is Decision.ALLOW
    assert "prior attempts" not in v.reason


def test_custom_environment_failure_threshold_is_honoured(ledger):
    spec = _spec()
    _seed(ledger, spec, state=TaskState.FAILED, verdict=None, evidence="venv broken")
    v = check_repeat_work(ledger, _spec(description="retry"), environment_failure_threshold=1)
    assert v.decision is Decision.ALLOW
    assert "1 prior attempts" in v.reason


def test_scan_limit_bounds_recall_honestly(ledger):
    # A match older than `scan_limit` tasks back is honestly not found, not
    # silently promoted -- recall is bounded on purpose (see docstrings).
    # Explicit, strictly-increasing created_at: `recent_tasks`' DESC-then-LIMIT
    # window must not depend on real-clock tie-breaking to exclude the oldest.
    prior_spec = _spec(created_at=1_000.0)
    _seed(ledger, prior_spec, state=TaskState.DONE, verdict=Verdict.PASS, usd_micros=1)

    # Bury it under enough newer, unrelated, settled tasks.
    for i in range(5):
        filler = _spec(subject=f"unrelated filler task {i}", description="filler",
                        created_at=1_001.0 + i)
        _seed(ledger, filler, state=TaskState.DONE, verdict=Verdict.PASS)

    v = check_repeat_work(ledger, _spec(description="resubmitted"), scan_limit=3)
    assert v.decision is Decision.ALLOW  # the real match is outside the 3-task window
