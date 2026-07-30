"""The governor — the brake, and the thing that decides what a failure means.

Two jobs, both about not wasting money:

1. **Stop runaway work.** Hard USD / wall-clock / iteration ceilings, plus a loop
   detector for the case the ceilings miss: a worker that is spending steadily and
   producing nothing new. Without that second check a spinning worker burns the
   entire cap before anything notices.

2. **Classify the failure before reacting to it.** The reflex on a red test is
   "escalate to a stronger model", but that only helps a MODEL failure. Escalating
   an ENVIRONMENT failure buys premium tokens for a broken venv and fails again
   identically at every tier. This is the largest avoidable waste in an agent
   harness, and it is a routing decision, not a model decision.

A tripped governor is a signal to escalate to the human. It is never a number to
edit — see AGENTS.md rule 1.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from ..contracts import Budget, FailureClass, GovernorTrip, from_micros, now
from ..events import EventLog, EventType
from ..ledger import Ledger

# Same file rewritten this many times with no new evidence == churn, not progress.
DEFAULT_LOOP_THRESHOLD = 3

# Budget fractions that change behaviour rather than stopping it.
NARROW_AT = 0.60   # stop broadening context, ask for a checkpoint
FOCUS_AT = 0.80    # exploratory work ends; targeted plan only


class Action(str, Enum):
    CONTINUE = "continue"
    NARROW = "narrow"      # 60% burn: stop expanding context
    FOCUS = "focus"        # 80% burn: targeted work only
    TRIP = "trip"          # stop, record, escalate to human


class Remedy(str, Enum):
    """What to actually do about a failure. Chosen by class, never by reflex."""

    REPAIR_ENVIRONMENT = "repair_environment"
    ADD_CONTEXT = "add_context"
    ESCALATE_MODEL = "escalate_model"
    RETURN_TO_MANAGER = "return_to_manager"
    ASK_HUMAN = "ask_human"
    BACKOFF_RETRY = "backoff_retry"
    GIVE_UP = "give_up"


_REMEDY_BY_CLASS: dict[FailureClass, Remedy] = {
    FailureClass.ENVIRONMENT: Remedy.REPAIR_ENVIRONMENT,
    FailureClass.CONTEXT: Remedy.ADD_CONTEXT,
    FailureClass.MODEL: Remedy.ESCALATE_MODEL,
    FailureClass.SPECIFICATION: Remedy.RETURN_TO_MANAGER,
    FailureClass.POLICY: Remedy.ASK_HUMAN,
    FailureClass.TRANSIENT: Remedy.BACKOFF_RETRY,
}


class LoopSignal(BaseModel):
    """Evidence that a task is churning rather than progressing."""

    task_id: str
    file: str
    touches: int
    distinct_evidence: int
    detail: str


class Decision(BaseModel):
    action: Action
    reason: str = ""
    trip: GovernorTrip | None = None
    loop: LoopSignal | None = None
    burn_fraction: float = 0.0

    @property
    def stopped(self) -> bool:
        return self.action is Action.TRIP


class Governor:
    def __init__(
        self,
        ledger: Ledger,
        events: EventLog | None = None,
        *,
        loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
    ):
        self.ledger = ledger
        self.events = events
        self.loop_threshold = loop_threshold

    # ------------------------------------------------------------ ceilings

    def check_job(self, job_id: str) -> Decision:
        """Hard ceilings for a whole job, read from the ledger."""
        row = self.ledger.job(job_id)
        if row is None:
            return Decision(action=Action.TRIP, reason="unknown job")

        spent = self.ledger.job_spend_micros(job_id)
        cap = int(row["max_usd_micros"])
        # Spending exactly to the cap is allowed; one microdollar past it is not.
        if spent > cap:
            return self._trip(
                job_id,
                reason="usd ceiling exceeded",
                limit=f"${from_micros(cap):.4f}",
                observed=f"${from_micros(spent):.4f}",
            )

        elapsed = now() - float(row["created_at"])
        max_secs = int(row["max_seconds"])
        if elapsed > max_secs:
            return self._trip(
                job_id,
                reason="wall-clock ceiling exceeded",
                limit=f"{max_secs}s",
                observed=f"{elapsed:.0f}s",
            )

        frac = (spent / cap) if cap else 0.0
        if frac >= FOCUS_AT:
            return Decision(
                action=Action.FOCUS,
                reason=f"{frac:.0%} of budget spent — targeted work only",
                burn_fraction=frac,
            )
        if frac >= NARROW_AT:
            return Decision(
                action=Action.NARROW,
                reason=f"{frac:.0%} of budget spent — stop expanding context",
                burn_fraction=frac,
            )
        return Decision(action=Action.CONTINUE, burn_fraction=frac)

    def check_task(self, task_id: str, job_id: str, budget: Budget) -> Decision:
        """Per-task ceilings plus the loop detector."""
        spent = self.ledger.task_spend_micros(task_id)
        if spent > budget.max_usd_micros:
            return self._trip(
                job_id,
                task_id=task_id,
                reason="task usd ceiling exceeded",
                limit=f"${budget.max_usd:.4f}",
                observed=f"${from_micros(spent):.4f}",
            )

        attempts = self.attempt_count(task_id)
        if attempts > budget.max_iterations:
            return self._trip(
                job_id,
                task_id=task_id,
                reason="iteration ceiling exceeded",
                limit=str(budget.max_iterations),
                observed=str(attempts),
            )

        loop = self.detect_loop(task_id)
        if loop is not None:
            d = self._trip(
                job_id,
                task_id=task_id,
                reason="loop detected",
                limit=f"{self.loop_threshold} touches without new evidence",
                observed=loop.detail,
            )
            d.loop = loop
            return d

        return Decision(action=Action.CONTINUE)

    # ------------------------------------------------------- loop detection

    def detect_loop(self, task_id: str) -> LoopSignal | None:
        """Spending with nothing new to show for it.

        The ceilings alone cannot catch this: a worker rewriting the same file with
        identical test output is inside every budget right up to the moment it
        exhausts them.
        """
        touches = self.ledger.file_touch_counts(task_id)
        if not touches:
            return None

        evidence = self.ledger.evidence_fingerprints(task_id)
        distinct = len({e for e in evidence if e})

        worst_file, worst_n = max(touches.items(), key=lambda kv: kv[1])
        if worst_n < self.loop_threshold:
            return None

        # Repeated touches are fine when the evidence keeps changing — that is
        # iteration. Repeated touches with one unchanging result is churn.
        if distinct > 1:
            return None

        return LoopSignal(
            task_id=task_id,
            file=worst_file,
            touches=worst_n,
            distinct_evidence=distinct,
            detail=f"{worst_file} touched {worst_n}x, {distinct} distinct evidence",
        )

    def attempt_count(self, task_id: str) -> int:
        if self.events is not None:
            return self.events.attempt_count(task_id)
        return len(self.ledger.reports_for_task(task_id))

    # -------------------------------------------------- failure classification

    def remedy_for(self, failure: FailureClass, attempts: int, max_attempts: int) -> Remedy:
        """Map a failure class to the cheapest action that could actually fix it."""
        if attempts >= max_attempts:
            # Out of attempts: a human decides, rather than silently escalating tiers.
            return Remedy.ASK_HUMAN if failure.needs_human else Remedy.GIVE_UP
        return _REMEDY_BY_CLASS[failure]

    def should_escalate_model(self, failure: FailureClass) -> bool:
        """Only a MODEL failure justifies paying more per token."""
        return failure.escalate_model

    def wasted_escalation_would_have_cost(self, failure: FailureClass) -> bool:
        """True when the naive reflex (escalate) would have burned tokens pointlessly."""
        return not failure.escalate_model

    # ------------------------------------------------------------- internals

    def _trip(
        self,
        job_id: str,
        *,
        reason: str,
        limit: str,
        observed: str,
        task_id: str | None = None,
    ) -> Decision:
        trip = GovernorTrip(
            job_id=job_id, task_id=task_id, reason=reason, limit=limit, observed=observed
        )
        self.ledger.record_trip(trip)
        if self.events is not None:
            self.events.append(
                job_id,
                EventType.GOVERNOR_TRIPPED,
                task_id=task_id,
                reason=reason,
                limit=limit,
                observed=observed,
            )
        return Decision(action=Action.TRIP, reason=reason, trip=trip)


__all__ = [
    "Action",
    "DEFAULT_LOOP_THRESHOLD",
    "Decision",
    "FOCUS_AT",
    "Governor",
    "LoopSignal",
    "NARROW_AT",
    "Remedy",
]
