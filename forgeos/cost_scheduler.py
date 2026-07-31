"""Cost-aware task scheduler - batches similar tasks together."""
from __future__ import annotations

from collections import defaultdict
from .cost_retry import CostRetry
from .cost_router import CostRouter, Route


class CostScheduler:
    """Schedule tasks for minimum cost execution."""

    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.router = CostRouter()
        self.retryer = CostRetry(max_retries=2, base_cost_budget=0.10)
        self.scheduled = defaultdict(list)
        self.executed = 0
        self.skipped = 0

    def add_task(self, task_type: str, payload: dict) -> None:
        """Queue a task for batch execution."""
        self.scheduled[task_type].append(payload)

    def schedule(self) -> dict:
        """Group tasks into batches and return execution plan."""
        plan = []
        for task_type, tasks in self.scheduled.items():
            route = self.router.route(task_type)
            for i in range(0, len(tasks), self.batch_size):
                batch = tasks[i:i + self.batch_size]
                plan.append(dict(
                    task_type=task_type,
                    route=route.value,
                    batch_size=len(batch),
                    batch_index=i // self.batch_size + 1,
                    cost=0.03 if route == Route.FULL_MODEL else 0.001,
                ))

        return dict(
            total_tasks=sum(len(v) for v in self.scheduled.values()),
            total_batches=len(plan),
            batches=plan,
            total_cost=round(sum(b["cost"] for b in plan), 4),
        )

    def reset(self) -> None:
        """Clear all scheduled tasks."""
        self.scheduled.clear()
        self.executed = 0
        self.skipped = 0
