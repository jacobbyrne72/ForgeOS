"""Batch cost projection module.

Estimates total cost of a batch of tasks before execution
using the cost router. Saves from surprise bills.
"""
from __future__ import annotations

from typing import Any

from .cost_router import ROUTE_COST, CostRouter


class BatchProjection:
    """Estimate batch cost before execution."""

    def __init__(self):
        self.router = CostRouter()

    def project(self, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        estimates: list[dict[str, Any]] = []
        total_cost = 0.0
        for t in tasks:
            task_type = t.get("type", "code_gen")
            est = self.router.cost_for(task_type)
            estimates.append(dict(
                name=t.get("name", "unknown"),
                type=task_type,
                estimated_cost=est,
            ))
            total_cost += est

        cheapest_route = min(ROUTE_COST, key=lambda route: ROUTE_COST[route])
        best_cost = ROUTE_COST[cheapest_route]
        saved = total_cost - best_cost * len(tasks)

        return dict(
            total_tasks=len(tasks),
            total_estimated_cost=round(total_cost, 6),
            per_task_cost=round(total_cost / max(1, len(tasks)), 6),
            cheapest_model=cheapest_route.value,
            cheapest_route=cheapest_route.value,
            estimated_savings_usd=round(saved, 6),
            cost_breakdown=estimates,
        )

    def print_projection(self, tasks):
        proj = self.project(tasks)
        print("=== Batch Cost Projection ===")
        print("Total tasks:", proj["total_tasks"])
        print("Total estimated cost: $%.6f" % proj["total_estimated_cost"])
        print("Per-task cost: $%.6f" % proj["per_task_cost"])
        print("Cheapest model:", proj["cheapest_model"])
        print("Estimated savings: $%.6f" % proj["estimated_savings_usd"])
        print()
        for e in proj["cost_breakdown"]:
            print("  %s: $%.6f" % (e["name"], e["estimated_cost"]))
        return proj
