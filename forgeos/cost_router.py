"""Cost router — dispatches tasks to cheapest capable handler.

Routes each task to the cheapest execution path:
- Local execution for simple tasks (regex, lint, format)
- Cheap model for medium tasks (summarize, extract)
- Full model for complex tasks (debug, architect)
"""
from __future__ import annotations

from enum import Enum
from .circuit_breaker import CircuitBreaker, BreakerState


class Route(Enum):
    LOCAL = "local"      
    CHEAP_MODEL = "cheap"
    FULL_MODEL = "full"  
    DEFERRED = "deferred"

# Routing rules: task type -> cheapest route
ROUTING_RULES = {
    "format": Route.LOCAL,
    "lint": Route.LOCAL,
    "syntax_check": Route.LOCAL,
    "grep": Route.LOCAL,
    "summarize": Route.CHEAP_MODEL,
    "extract": Route.CHEAP_MODEL,
    "translate": Route.CHEAP_MODEL,
    "classify": Route.CHEAP_MODEL,
    "code_gen": Route.FULL_MODEL,
    "debug": Route.FULL_MODEL,
    "review": Route.FULL_MODEL,
    "refactor": Route.FULL_MODEL,
    "architect": Route.FULL_MODEL,
    "security_scan": Route.FULL_MODEL,
    "test_gen": Route.FULL_MODEL,
}

ROUTE_COST = {
    Route.LOCAL: 0.0001,    # $0.01 per 100 tasks
    Route.CHEAP_MODEL: 0.001,  # $0.10 per task
    Route.FULL_MODEL: 0.03,    # $3.00 per task
    Route.DEFERRED: 0.0,       # Free (offline)
}

class CostRouter:
    def __init__(self):
        self.routes_taken: dict[str, int] = {}
        self.total_saved = 0.0
        self._breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=30)


    def route(self, task_type: str) -> Route:
        """Determine cheapest route for a task type.

        If the circuit breaker is OPEN for full-model tasks,
        downgrade to deferred or cheap alternatives when possible.
        """
        route = ROUTING_RULES.get(task_type, Route.FULL_MODEL)

        # If breaker is OPEN and task would use full model,
        # check for cheaper alternatives
        if self._breaker.is_open() and route == Route.FULL_MODEL:
            # Downgrade to cheap model if available
            route = Route.CHEAP_MODEL

        key = route.value
        self.routes_taken[key] = self.routes_taken.get(key, 0) + 1
        return route

    def cost_for(self, task_type: str) -> float:
        """Get cost estimate for a task type."""
        route = self.route(task_type)
        return ROUTE_COST[route]

    def route_many(self, tasks: list) -> dict:
        """Route a batch of tasks and return cost summary."""
        routes = {}
        total_cost = 0.0
        for t in tasks:
            tt = t.get("type", "code_gen")
            route = self.route(tt)
            cost = ROUTE_COST[route]
            total_cost += cost
            if route.value not in routes:
                routes[route.value] = {"count": 0, "cost": 0.0}
            routes[route.value]["count"] += 1
            routes[route.value]["cost"] += cost
        return {
            "total_tasks": len(tasks),
            "total_cost_usd": round(total_cost, 4),
            "routes": routes,
            "savings_vs_full_model": round(
                len(tasks) * ROUTE_COST[Route.FULL_MODEL] - total_cost, 4
            ),
        }
