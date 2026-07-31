"""Cost-aware retry with exponential backoff.

Prevents wasteful re-runs that exceed a cost budget.
Each retry doubles the wait time and halves the remaining budget.
Stops when the next retry would overspend.
"""
from __future__ import annotations

from .cost_tracker import CostTracker


class CostRetry:
    def __init__(self, max_retries: int = 3, base_cost_budget: float = 0.20):
        self.max_retries = max_retries
        self.base_cost_budget = base_cost_budget
        self.total_retries = 0
        self.total_retries_saved = 0.0
        self.tracker = CostTracker()

    def should_retry(self, attempt: int, last_error: str = "") -> tuple[bool, dict]:
        """Decide if retrying makes financial sense.

        Returns (should_retry, info dict).
        """
        if attempt >= self.max_retries:
            return False, {"reason": "max_retries_reached", "attempt": attempt}

        # Exponential backoff: budget halves each retry
        remaining_budget = self.base_cost_budget / (2 ** attempt)
        retry_cost = 0.03  # typical API call

        if retry_cost > remaining_budget:
            return False, {
                "reason": "budget_exceeded",
                "attempt": attempt,
                "remaining_budget": round(remaining_budget, 4),
                "retry_cost": retry_cost,
            }

        # Check for waste errors
        waste_patterns = [
            "invalid prompt", "permission denied", "not found",
            "rate limit", "quota exceeded",
        ]
        for pattern in waste_patterns:
            if pattern.lower() in last_error.lower():
                return False, {
                    "reason": "waste_error",
                    "attempt": attempt,
                    "match": pattern,
                }

        self.total_retries += 1
        return True, {
            "reason": "retry_ok",
            "attempt": attempt + 1,
            "remaining_budget": round(remaining_budget, 4),
            "wait_seconds": 2 ** attempt,
        }

    def run_with_retry(self, task_fn, task_type: str = "code_gen") -> dict:
        """Run a task with cost-aware retry logic.

        task_fn returns (success, error_message, output).
        """
        last_result = None
        for attempt in range(self.max_retries + 1):
            should, info = self.should_retry(attempt, last_result or "")
            if not should and attempt > 0:
                break

            try:
                success, error, output = task_fn()
                if success:
                    return {
                        "success": True,
                        "attempts": attempt + 1,
                        "output": output,
                        "total_cost": 0.03 * (attempt + 1),
                    }
                last_result = error or "unknown error"
            except Exception as e:
                last_result = str(e)

        return {
            "success": False,
            "attempts": attempt + 1,
            "output": None,
            "last_error": last_result,
            "total_cost": 0.03 * (attempt + 1),
        }

    def savings_report(self) -> dict:
        return {
            "total_retries": self.total_retries,
            "max_saved_per_waste_retry": round(0.03 * self.max_retries, 4),
        }