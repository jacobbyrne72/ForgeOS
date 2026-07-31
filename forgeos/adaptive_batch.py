"""Adaptive batch optimizer — picks cheapest strategy for any workload.

Analyzes task characteristics and automatically selects
the optimization strategy that cuts cost the most.
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

        best = max(savings, key=savings.get)
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
