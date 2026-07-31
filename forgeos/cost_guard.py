"""Cost guard - hard budget cap per session.

Enforces a cost ceiling so no session can ever
exceed the target budget regardless of usage.
Like a circuit breaker but for spend, not failures.
"""
from __future__ import annotations

from .cost_tracker import CostTracker


class CostGuard:
    def __init__(self, budget_usd: float = 10.0):
        self.budget_usd = budget_usd
        self.tracker = CostTracker()
        self.breaches = 0
        self.blocked_calls = 0

    def check(self, estimated_cost: float = 0.03) -> tuple[bool, dict]:
        """Check if this call would exceed the budget."""
        remaining = self.budget_usd - self.tracker.total_saved().get("total_usd", 0)

        if estimated_cost > remaining:
            self.breaches += 1
            return False, {
                "allowed": False,
                "reason": "budget_exceeded",
                "remaining_budget": round(remaining, 4),
                "estimated_cost": estimated_cost,
                "breach_count": self.breaches,
            }

        return True, {
            "allowed": True,
            "remaining_budget": round(remaining, 4),
            "estimated_cost": estimated_cost,
        }

    def reset(self, new_budget: float | None = None) -> dict:
        """Reset the budget for a new session."""
        if new_budget is not None:
            self.budget_usd = new_budget
        self.breaches = 0
        return {
            "budget_usd": self.budget_usd,
            "reset": True,
        }

    def report(self) -> dict:
        return {
            "budget_usd": self.budget_usd,
            "breaches": self.breaches,
            "blocked_calls": self.blocked_calls,
        }
