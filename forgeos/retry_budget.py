"""Retry budget caps total retry cost."""
from __future__ import annotations

class RetryBudget:
    def __init__(self, max_retry_spend=5.0):
        self.max_retry_spend = max_retry_spend
        self.retry_spend = 0.0
        self.blocked = 0

    def allow_retry(self, estimated_cost):
        if self.retry_spend + estimated_cost > self.max_retry_spend:
            self.blocked += 1
            return False
        self.retry_spend += estimated_cost
        return True

    def stats(self):
        return dict(
            retry_spend=round(self.retry_spend, 4),
            blocked=self.blocked,
            remaining=round(self.max_retry_spend - self.retry_spend, 4),
        )
