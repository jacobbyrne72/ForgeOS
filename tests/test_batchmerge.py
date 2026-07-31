"""Bors-style batch-merge tests.

The theme: batching must never trust more than it checked. All-pass is the cheap
path (one check instead of N). A failure isolates exactly the guilty candidates
while innocents still land. A check that can't run (raises) blames no one and
lands no one in that subset. Same inputs must bisect the same way every time.
"""

from __future__ import annotations


from forgeos.core.batchmerge import (
    BatchCandidate,
    BatchPlan,
    CheckOutcome,
    verify_batch,
)


def _plan(*ids: str) -> BatchPlan:
    return BatchPlan(candidates=[BatchCandidate(id=i) for i in ids])


def _fails_if_any(*bad_ids: str):
    """A monotonic checker: a subset fails iff it contains any of `bad_ids`."""
    bad = set(bad_ids)

    def check(subset: list[BatchCandidate]) -> CheckOutcome:
        hit = bad & {c.id for c in subset}
        if hit:
            return CheckOutcome(passed=False, evidence=f"failing due to {sorted(hit)}")
        return CheckOutcome(passed=True, evidence="clean")

    return check


# ------------------------------------------------------------- all pass


def test_all_pass_is_one_check():
    plan = _plan("a", "b", "c", "d")
    result = verify_batch(plan, _fails_if_any())

    assert result.checks_run == 1
    assert result.naive_checks == 4
    assert result.landed == ["a", "b", "c", "d"]
    assert result.culprits == []
    assert result.requeued_unknown == []


# ------------------------------------------------------------- single culprit


def test_single_culprit_is_isolated_and_innocents_land():
    plan = _plan("a", "b", "c", "d")
    result = verify_batch(plan, _fails_if_any("c"))

    # full batch FAIL, [a,b] PASS, [c,d] FAIL, [c] FAIL, [d] PASS = 5 checks
    assert result.checks_run == 5
    assert result.landed == ["a", "b", "d"]
    assert [c.candidate_id for c in result.culprits] == ["c"]
    assert result.culprits[0].evidence == "failing due to ['c']"
    assert result.requeued_unknown == []


def test_single_culprit_evidence_is_from_the_check_that_convicted_it():
    plan = _plan("a", "b")
    result = verify_batch(plan, _fails_if_any("b"))

    assert result.landed == ["a"]
    assert len(result.culprits) == 1
    assert result.culprits[0].candidate_id == "b"
    assert "b" in result.culprits[0].evidence


# ------------------------------------------------------------- two culprits


def test_two_culprits_both_isolated_innocents_still_land():
    plan = _plan("a", "b", "c", "d")
    result = verify_batch(plan, _fails_if_any("b", "d"))

    # full FAIL, [a,b] FAIL, [a] PASS, [b] FAIL, [c,d] FAIL, [c] PASS, [d] FAIL = 7 checks
    assert result.checks_run == 7
    assert result.landed == ["a", "c"]
    assert sorted(c.candidate_id for c in result.culprits) == ["b", "d"]
    assert result.requeued_unknown == []


# ------------------------------------------------------------- check raises -> UNKNOWN


def test_raising_subset_is_unknown_and_requeued_not_blamed():
    def check(subset: list[BatchCandidate]) -> CheckOutcome:
        ids = {c.id for c in subset}
        if ids == {"a", "b", "c", "d"}:
            return CheckOutcome(passed=False, evidence="batch failed")
        if "c" in ids:
            raise RuntimeError("infra crashed running this subset")
        return CheckOutcome(passed=True, evidence="clean")

    plan = _plan("a", "b", "c", "d")
    result = verify_batch(plan, check)

    # full FAIL (1), [a,b] PASS (2), [c,d] raises -> UNKNOWN (3) — no further bisection
    assert result.checks_run == 3
    assert result.landed == ["a", "b"]
    assert result.culprits == []
    assert result.requeued_unknown == ["c", "d"]


def test_whole_batch_raising_requeues_everyone_and_lands_no_one():
    def always_raises(subset: list[BatchCandidate]) -> CheckOutcome:
        raise RuntimeError("runner unavailable")

    plan = _plan("a", "b", "c")
    result = verify_batch(plan, always_raises)

    assert result.checks_run == 1
    assert result.landed == []
    assert result.culprits == []
    assert result.requeued_unknown == ["a", "b", "c"]


# ------------------------------------------------------------- empty batch


def test_empty_batch_is_a_no_op_not_an_error():
    def never_called(subset: list[BatchCandidate]) -> CheckOutcome:
        raise AssertionError("run_batch_check must not be called for an empty plan")

    result = verify_batch(BatchPlan(candidates=[]), never_called)

    assert result.checks_run == 0
    assert result.naive_checks == 0
    assert result.landed == []
    assert result.culprits == []
    assert result.requeued_unknown == []
    assert result.checks == []


# ------------------------------------------------------------- determinism


def test_same_inputs_bisect_the_same_way_every_time():
    plan = _plan("a", "b", "c", "d", "e")

    def run_once():
        result = verify_batch(plan, _fails_if_any("b", "e"))
        return [(r.candidate_ids, r.status.value) for r in result.checks], result.landed, sorted(
            c.candidate_id for c in result.culprits
        )

    first = run_once()
    second = run_once()

    assert first == second
