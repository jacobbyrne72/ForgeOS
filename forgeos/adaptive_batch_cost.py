"""Adaptive batch scheduling."""
from __future__ import annotations
import time


class AdaptiveBatchCostOptimizer:
    def __init__(self):
        self._history = []

    def recommend_batch_size(self, task_type, avg_tokens=500):
        cost_per = avg_tokens * 0.03 / 1000.0
        optimal = max(1, min(50, int(1.0 / max(cost_per, 0.00001))))
        self._history.append(dict(
            task_type=task_type, avg_tokens=avg_tokens,
            cost_per_task=cost_per, optimal_batch=optimal,
        ))
        return dict(
            task_type=task_type, optimal_batch_size=optimal,
            estimated_cost_per_task=cost_per,
            estimated_batch_cost=round(cost_per * optimal, 6),
            savings_vs_individual=round(cost_per * optimal * 0.3, 6),
        )

    def get_savings_trend(self):
        return dict(
            total_optimized_tasks=len(self._history),
            trend='optimizing' if len(self._history) > 5 else 'establishing',
        )
