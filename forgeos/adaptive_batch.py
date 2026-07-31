"""Adaptive batch optimizer — picks cheapest strategy for any workload, and
recommends a per-call batch size.

Analyzes task characteristics and automatically selects
the optimization strategy that cuts cost the most.

Consolidates adaptive_batch + adaptive_batch_cost (2026-07-31):
recommend_batch_size()/get_savings_trend() below are adaptive_batch_cost.py's
AdaptiveBatchCostOptimizer, folded in verbatim — strategy selection and
batch-size recommendation were split across two classes that nothing ever
used together. adaptive_batch_cost.py is gone.

(smart_batching.py, a third "batching" module, was deleted outright rather
than merged: it was unwired anywhere, and its form_batch() returned the same
[calls] result regardless of its own should_batch() check — i.e. it did not
actually batch anything.)
"""
from __future__ import annotations

from .optimizer import CostOptimizer
from .auto_optimize import AutoOptimizer
from .batch_optimize import BatchOptimizer
from .token_budget import TokenBudget

class AdaptiveBatch:
    def __init__(self):
        self.auto = AutoOptimizer()
        self.planner = CostOptimizer()
        self.budget = TokenBudget()
        self._size_history = []

    def analyze(self, tasks: list) -> dict:
        """Analyze workload and recommend strategy."""
        types = {}
        for t in tasks:
            tt = t.get("type", "unknown")
            types[tt] = types.get(tt, 0) + 1

        # Strategy selection based on workload shape
        if len(types) == 1:
            rationale = "Single task type — batch with same optimizer"
        elif len(types) <= 3:
            rationale = "Few types — group same types, apply per-type optimizer"
        else:
            rationale = "Many types — let auto-optimizer pick per-task"

        # Calculate savings for each strategy
        total_tasks = len(tasks)
        savings = {}
        for s in ["bulk_single", "grouped", "auto_pipeline"]:
            savings[s] = self._estimate_strategy_savings(tasks, s)

        best = max(savings, key=lambda strategy: savings[strategy])
        return {
            "total_tasks": total_tasks,
            "task_types": types,
            "recommended_strategy": best,
            "strategy_rationale": rationale,
            "strategy_savings": savings[best],
            "all_strategies": savings,
        }

    def _estimate_strategy_savings(self, tasks, strategy):
        """Estimate savings for a given strategy."""
        total = 0.0
        for t in tasks:
            plan = self.planner.plan_for(t.get("type", "code_gen"))
            if strategy == "bulk_single":
                # Bulk discount
                total += plan.estimated_savings_usd * 1.2
            elif strategy == "grouped":
                # Group savings
                total += plan.estimated_savings_usd * 1.1
            else:
                total += plan.estimated_savings_usd
        return round(total, 4)

    def run(self, tasks):
        """Analyze and apply best strategy."""
        analysis = self.analyze(tasks)
        batch = BatchOptimizer()
        s = batch.run_batch(tasks)
        return {
            "analysis": analysis,
            "actual_savings": {
                "total_saved_usd": s["total_saved_usd"],
                "total_tokens": s["total_tokens_saved"],
            },
        }

    def recommend_batch_size(self, task_type: str, avg_tokens: int = 500) -> dict:
        """Recommend an optimal batch size from a per-token cost heuristic.

        (from adaptive_batch_cost.py's AdaptiveBatchCostOptimizer.)
        """
        cost_per = avg_tokens * 0.03 / 1000.0
        optimal = max(1, min(50, int(1.0 / max(cost_per, 0.00001))))
        self._size_history.append(dict(
            task_type=task_type, avg_tokens=avg_tokens,
            cost_per_task=cost_per, optimal_batch=optimal,
        ))
        return dict(
            task_type=task_type, optimal_batch_size=optimal,
            estimated_cost_per_task=cost_per,
            estimated_batch_cost=round(cost_per * optimal, 6),
            savings_vs_individual=round(cost_per * optimal * 0.3, 6),
        )

    def get_savings_trend(self) -> dict:
        """Trend across recommend_batch_size() calls (from adaptive_batch_cost.py)."""
        return dict(
            total_optimized_tasks=len(self._size_history),
            trend='optimizing' if len(self._size_history) > 5 else 'establishing',
        )
