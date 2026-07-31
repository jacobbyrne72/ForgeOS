"""Cost-aware retry with exponential backoff, a cumulative spend cap, and
fallback-cost accounting.

Consolidates cost_retry + cost_aware_retry + retry_budget (2026-07-31):
three independent retry-on-failure implementations covering the same job —
should we retry, and what does it cost — from three angles: per-attempt
budget halving (this module, unchanged by default), a session-wide spend
ceiling (from retry_budget.py's RetryBudget, now opt-in via
max_total_retry_spend), and charging retries at a cheaper fallback cost
(from cost_aware_retry.py's CostAwareRetry, now opt-in via run_with_retry's
fallback_cost). Both donor modules are gone; neither was wired anywhere or
used by any other module.

Prevents wasteful re-runs that exceed a cost budget.
Each retry doubles the wait time and halves the remaining budget.
Stops when the next retry would overspend.
"""
from __future__ import annotations

from .contracts import FailureClass
from .cost_tracker import CostTracker

_RETRY_COST = 0.03  # typical API call

# LAST-RESORT ONLY. `FailureClass` is the real answer and a caller that knows
# why the attempt failed should pass it; these substrings are what remains for
# callers that only have an error string.
#
# Two of these were actively wrong before `FailureClass` was wired in. "rate
# limit" and "quota exceeded" are TRANSIENT — the definitively retryable class,
# the one backoff exists for — and treating them as unretryable waste meant
# giving up on the failures most likely to succeed on the next attempt. They
# are gone. What is left maps to SPECIFICATION or POLICY: a wrong prompt or a
# denied permission does not improve by being asked again.
#
# Substring matching on an error message is a guess about someone else's
# wording. It is kept only because deleting it would silently change behaviour
# for callers that pass no failure class.
_WASTE_PATTERNS = [
    "invalid prompt", "permission denied", "not found",
]


class CostRetry:
    def __init__(
        self,
        max_retries: int = 3,
        base_cost_budget: float = 0.20,
        max_total_retry_spend: float | None = None,
    ):
        self.max_retries = max_retries
        self.base_cost_budget = base_cost_budget
        self.max_total_retry_spend = max_total_retry_spend
        self.total_retries = 0
        self.total_retries_saved = 0.0
        self.retry_spend = 0.0
        self.blocked_by_budget = 0
        self.tracker = CostTracker()

    def should_retry(
        self, attempt: int, last_error: str = "", retry_cost: float = _RETRY_COST,
        failure_class: FailureClass | None = None,
    ) -> tuple[bool, dict]:
        """Decide if retrying makes financial sense.

        Returns (should_retry, info dict).

        `failure_class` is the answer when the caller has it. This module used
        to decide entirely on budget arithmetic plus substring-matching the
        error text, while `contracts.FailureClass` -- a considered taxonomy that
        `circuit_breaker.py` already consumes correctly -- sat unused beside it.
        Two classification systems in one package, and the weaker one was the
        one wired up.

        Worse than redundant: its waste list contained "rate limit" and "quota
        exceeded", which are TRANSIENT, the definitively retryable class. It was
        giving up precisely on the failures most likely to succeed next attempt.
        """
        if attempt >= self.max_retries:
            return False, {"reason": "max_retries_reached", "attempt": attempt}

        # Checked BEFORE the budget arithmetic: whether retrying can possibly
        # work is a different question from whether it is affordable, and the
        # first one answers "no" for free.
        if failure_class is not None and failure_class.needs_human:
            return False, {
                "reason": "needs_human",
                "attempt": attempt,
                "failure_class": failure_class.value,
                "detail": "a SPECIFICATION or POLICY failure does not improve by "
                          "being retried -- the task or the permission is wrong",
            }

        # Exponential backoff: budget halves each retry
        remaining_budget = self.base_cost_budget / (2 ** attempt)

        if retry_cost > remaining_budget:
            return False, {
                "reason": "budget_exceeded",
                "attempt": attempt,
                "remaining_budget": round(remaining_budget, 4),
                "retry_cost": retry_cost,
            }

        # Cumulative spend cap across the retry's whole lifetime, independent
        # of the per-attempt budget above (from retry_budget.py).
        if (
            self.max_total_retry_spend is not None
            and self.retry_spend + retry_cost > self.max_total_retry_spend
        ):
            self.blocked_by_budget += 1
            return False, {
                "reason": "total_retry_budget_exceeded",
                "attempt": attempt,
                "retry_spend": round(self.retry_spend, 4),
                "max_total_retry_spend": self.max_total_retry_spend,
            }

        # Substring matching runs ONLY when the caller supplied no failure
        # class. A caller that knows why the attempt failed has already been
        # answered above, and guessing from someone else's error wording on top
        # of a definite answer is how the two systems disagreed.
        if failure_class is None:
            for pattern in _WASTE_PATTERNS:
                if pattern.lower() in last_error.lower():
                    return False, {
                        "reason": "waste_error",
                        "attempt": attempt,
                        "match": pattern,
                        "detail": "matched on error text; pass a FailureClass for a "
                                  "decision that does not depend on wording",
                    }

        self.total_retries += 1
        self.retry_spend += retry_cost
        return True, {
            "reason": "retry_ok",
            "attempt": attempt + 1,
            "remaining_budget": round(remaining_budget, 4),
            "wait_seconds": 2 ** attempt,
        }

    def run_with_retry(
        self,
        task_fn,
        task_type: str = "code_gen",
        fallback_cost: float | None = None,
    ) -> dict:
        """Run a task with cost-aware retry logic.

        task_fn returns (success, error_message, output). By default every
        attempt costs the same _RETRY_COST (identical to the pre-merge
        behavior). Pass fallback_cost to price retries after the first
        attempt at a cheaper rate — e.g. routing to a cheaper model on retry,
        the idea cost_aware_retry.py contributed.
        """
        last_result = None
        attempt = -1
        total_cost = 0.0
        for attempt in range(self.max_retries + 1):
            cost = _RETRY_COST if (attempt == 0 or fallback_cost is None) else fallback_cost
            should, info = self.should_retry(attempt, last_result or "", retry_cost=cost)
            if not should and attempt > 0:
                break

            try:
                success, error, output = task_fn()
                total_cost += cost
                if success:
                    return {
                        "success": True,
                        "attempts": attempt + 1,
                        "output": output,
                        "total_cost": round(total_cost, 6),
                    }
                last_result = error or "unknown error"
            except Exception as e:
                total_cost += cost
                last_result = str(e)

        return {
            "success": False,
            "attempts": attempt + 1,
            "output": None,
            "last_error": last_result,
            "total_cost": round(total_cost, 6),
        }

    def savings_report(self) -> dict:
        return {
            "total_retries": self.total_retries,
            "max_saved_per_waste_retry": round(_RETRY_COST * self.max_retries, 4),
        }

    def budget_status(self) -> dict:
        """Cumulative retry-spend cap status (from retry_budget.py)."""
        remaining = (
            None
            if self.max_total_retry_spend is None
            else round(self.max_total_retry_spend - self.retry_spend, 4)
        )
        return dict(
            retry_spend=round(self.retry_spend, 4),
            blocked_by_budget=self.blocked_by_budget,
            max_total_retry_spend=self.max_total_retry_spend,
            remaining=remaining,
        )
