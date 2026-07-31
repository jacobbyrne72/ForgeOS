"""Bors-style bisecting batch verification, wired around the per-candidate `MergeGate`.

`MergeGate.evaluate()` (verify.py) already decides whether one candidate, on its own,
is good enough to merge. This module never reopens that question — every candidate
handed to `verify_batch` has already passed it. What this module adds is the *next*
question: once several gate-accepted candidates finish near-simultaneously (the whole
point of one worktree per worker), do they land together in one check, or does
serializing N individual checks throw away the parallelism the worktrees already
bought? Per Gas Town (docs/research/harness-workflows.md): batch first, verify once,
and only pay for more checks when the batch actually fails.

Pure logic. No git, no subprocess, no filesystem. `run_batch_check` is injected by the
caller — it is the only thing that knows how to actually check a subset of candidates
together (run tests in a combined worktree, whatever). That keeps this module testable
without touching a real repo, and keeps the wiring to forge/worktrees a separate task.

Bisection, once a batch fails: split it in half, recurse on each half independently.
A half that passes lands as-is (its members were never the problem). A half that
fails and has more than one member splits again. A singleton that fails is a culprit,
named with the check's own evidence. Worst case for k failing candidates among n is
O(k log n) checks — each halving pays for itself once, and only the failing branches
keep splitting.

Honesty over cleverness: if `run_batch_check` raises for a subset, that subset's
status is UNKNOWN, never a pass, and it is not bisected further — bisecting an UNKNOWN
would mean trusting splits derived from a check that never actually ran. The whole
subset re-queues for individual, separate verification instead. A candidate lands only
on a recorded PASS that covered it.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from pydantic import BaseModel, Field


class BatchCandidate(BaseModel):
    """One MergeGate-accepted change waiting to land, in submission order.

    Opaque beyond `id` on purpose: this module doesn't know or care what a candidate
    *is* (a task id, a commit sha, a worktree branch) — only `run_batch_check` needs
    to resolve `id` into something it can actually verify.
    """

    id: str


class CheckOutcome(BaseModel):
    """What `run_batch_check` reports for one subset it was actually able to run.

    Raising instead of returning is how a subset communicates "I couldn't tell" —
    see `CheckStatus.UNKNOWN` below. This model is only for the calls that completed.
    """

    passed: bool
    evidence: str = ""


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"  # run_batch_check raised — never treated as a pass


class CheckRecord(BaseModel):
    """One call to `run_batch_check`, in call order.

    The full sequence is `BatchResult.checks` — the audit trail that makes bisection's
    determinism checkable: same candidates, same check behavior, same sequence.
    """

    candidate_ids: list[str]
    status: CheckStatus
    evidence: str = ""


class CulpritReport(BaseModel):
    """A candidate bisection isolated as failing on its own, with the exact evidence
    from the check that convicted it — never a bare id with no reason attached."""

    candidate_id: str
    evidence: str = ""


class BatchPlan(BaseModel):
    """Accepted-by-gate candidates, in submission order, about to be checked as one batch."""

    candidates: list[BatchCandidate] = Field(default_factory=list)

    @property
    def naive_checks(self) -> int:
        """What checking every candidate individually would have cost — the baseline
        the caller compares `BatchResult.checks_run` against for the savings receipt.
        Reported, never asserted as a saving by this module."""
        return len(self.candidates)


class BatchResult(BaseModel):
    """The outcome of one `verify_batch` call.

    `landed` preserves submission order regardless of the bisection path taken.
    `culprits` and `requeued_unknown` are disjoint from `landed` and from each other:
    a culprit was proven guilty by a recorded FAIL; an unknown was never proven
    anything and re-queues for individual verification, not blame.
    """

    landed: list[str] = Field(default_factory=list)
    culprits: list[CulpritReport] = Field(default_factory=list)
    requeued_unknown: list[str] = Field(default_factory=list)
    checks: list[CheckRecord] = Field(default_factory=list)
    checks_run: int = 0
    naive_checks: int = 0


RunBatchCheck = Callable[[list[BatchCandidate]], CheckOutcome]


def verify_batch(plan: BatchPlan, run_batch_check: RunBatchCheck) -> BatchResult:
    """Check `plan.candidates` together, bisecting only on failure.

    An empty plan is a no-op result, not an error — there is nothing to check and
    nothing to blame, so `run_batch_check` is never called.
    """
    candidates = plan.candidates
    if not candidates:
        return BatchResult(naive_checks=0)

    checks: list[CheckRecord] = []
    landed: list[str] = []
    culprits: list[CulpritReport] = []
    requeued_unknown: list[str] = []

    def check(subset: list[BatchCandidate]) -> CheckRecord:
        ids = [c.id for c in subset]
        try:
            outcome = run_batch_check(subset)
        except Exception as exc:  # noqa: BLE001 - any failure to run the check is UNKNOWN, not a pass
            record = CheckRecord(candidate_ids=ids, status=CheckStatus.UNKNOWN, evidence=str(exc))
        else:
            status = CheckStatus.PASS if outcome.passed else CheckStatus.FAIL
            record = CheckRecord(candidate_ids=ids, status=status, evidence=outcome.evidence)
        checks.append(record)
        return record

    def bisect(subset: list[BatchCandidate]) -> None:
        record = check(subset)
        if record.status is CheckStatus.PASS:
            landed.extend(c.id for c in subset)
        elif record.status is CheckStatus.UNKNOWN:
            # House rule: an unrun check proves nothing. Re-queue the whole subset
            # individually rather than bisect splits derived from a non-result.
            requeued_unknown.extend(c.id for c in subset)
        elif len(subset) == 1:
            culprits.append(CulpritReport(candidate_id=subset[0].id, evidence=record.evidence))
        else:
            mid = len(subset) // 2
            bisect(subset[:mid])
            bisect(subset[mid:])

    bisect(candidates)

    return BatchResult(
        landed=landed,
        culprits=culprits,
        requeued_unknown=requeued_unknown,
        checks=checks,
        checks_run=len(checks),
        naive_checks=len(candidates),
    )


__all__ = [
    "BatchCandidate",
    "BatchPlan",
    "BatchResult",
    "CheckOutcome",
    "CheckRecord",
    "CheckStatus",
    "CulpritReport",
    "RunBatchCheck",
    "verify_batch",
]
