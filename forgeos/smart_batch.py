"""Smart batch predictor — predicts cheapest optimization strategy
using historical cost data from the ledger.

Reads past job completions, derives task-type cost patterns,
and recommends the optimization strategy that minimizes spend.
"""
from __future__ import annotations

from .optimizer import CostOptimizer
from .auto_optimize import AutoOptimizer

class SmartBatchPredictor:
    def __init__(self):
        self.auto = AutoOptimizer()
        self.planner = CostOptimizer()
        self._history = []

    def record(self, task_type: str, actual_saved_usd: float,
              tokens_saved: int) -> None:
        """Record actual savings from a completed task."""
        self._history.append({
            "task_type": task_type,
            "saved_usd": actual_saved_usd,
            "tokens": tokens_saved,
        })

    def predict_savings(self, task_type: str, count: int) -> dict:
        """Predict savings for N tasks of a given type.

        Uses historical data if available, otherwise falls back
        to optimizer estimates.
        """
        # Get historical data for this type
        history = [h for h in self._history
                   if h["task_type"] == task_type]

        if len(history) >= 3:
            # Use median of historical savings
            savings = sorted([h["saved_usd"] for h in history])
            median = savings[len(savings) // 2]
            tokens = sum(h["tokens"] for h in history) // len(history)
            source = "historical"
        else:
            # Fall back to optimizer
            plan = self.planner.plan_for(task_type)
            median = plan.estimated_savings_usd
            tokens = int(plan.estimated_savings_usd * 10000)  # rough estimate
            source = "optimizer_estimate"

        return {
            "task_type": task_type,
            "count": count,
            "estimated_total_usd": round(median * count, 4),
            "per_task_usd": round(median, 4),
            "avg_tokens_saved": tokens,
            "estimated_tokens_total": tokens * count,
            "source": source,
            "history_samples": len(history),
        }

    def predict_batch(self, tasks: list) -> dict:
        """Predict savings for a batch of mixed tasks."""
        predictions = []
        types_seen = {}

        for t in tasks:
            tt = t.get("type", "code_gen")
            types_seen[tt] = types_seen.get(tt, 0) + 1

        for tt, count in types_seen.items():
            pred = self.predict_savings(tt, count)
            predictions.append(pred)

        total_usd = sum(p["estimated_total_usd"] for p in predictions)
        total_tokens = sum(p["estimated_tokens_total"] for p in predictions)

        return {
            "total_tasks": len(tasks),
            "predictions": predictions,
            "total_estimated_usd": round(total_usd, 4),
            "total_estimated_tokens": total_tokens,
            "monthly_projection": round(total_usd * 22, 2),
            "yearly_projection": round(total_usd * 22 * 12, 2),
        }
