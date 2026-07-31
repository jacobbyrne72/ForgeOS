"""Batch optimizer - run many tasks through auto-optimizer.

Demonstrates provable cost-cutting across an entire task batch.
"""
from __future__ import annotations
from forgeos.auto_optimize import AutoOptimizer
from forgeos.optimizer import CostOptimizer

class BatchOptimizer:
    def __init__(self):
        self.auto = AutoOptimizer()
        self.planner = CostOptimizer()

    def run_batch(self, tasks: list) -> dict:
        """Run auto-optimizer across a batch of tasks."""
        results = []
        total_saved = 0.0
        total_tokens = 0

        for task in tasks:
            r = self.auto.apply(task["type"], task.get("name", ""))
            results.append(r)
            total_saved += r.saved_usd
            total_tokens += r.saved_tokens

        by_type: dict = {}
        for r in results:
            t = r.task_type
            if t not in by_type:
                by_type[t] = {"count": 0, "saved_usd": 0.0, "tokens": 0, "steps": r.steps_applied}
            by_type[t]["count"] += 1
            by_type[t]["saved_usd"] += r.saved_usd
            by_type[t]["tokens"] += r.saved_tokens

        return {
            "total_tasks": len(results),
            "total_saved_usd": round(total_saved, 4),
            "total_tokens_saved": total_tokens,
            "by_task_type": by_type,
            "monthly_projection": round(total_saved * 22, 2),
            "yearly_projection": round(total_saved * 22 * 12, 2),
        }

    def print_report(self, summary: dict) -> str:
        lines = [
            "=== BATCH SAVINGS REPORT ===",
            "Total tasks: " + str(summary["total_tasks"]),
            "Total per-task savings: $" + str(round(summary["total_saved_usd"], 4)),
            "Total tokens saved: " + str(summary["total_tokens_saved"]),
            "",
        ]
        lines.append("Breakdown by task type:")
        for t, d in sorted(summary["by_task_type"].items(), key=lambda x: -x[1]["saved_usd"]):
            lines.append("  " + t + ": " + str(d["count"]) + " task(s), $" + str(round(d["saved_usd"], 4)) + ", " + str(d["tokens"]) + " tokens")
        lines.append("")
        lines.append("Monthly projection: $" + str(summary["monthly_projection"]))
        lines.append("Yearly projection:  $" + str(summary["yearly_projection"]))
        return chr(10).join(lines)
