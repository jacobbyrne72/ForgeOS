"""Cost router — dispatches tasks to cheapest capable handler.

Routes each task to the cheapest execution path:
- Local execution for simple tasks (regex, lint, format)
- Cheap model for medium tasks (summarize, extract)
- Full model for complex tasks (debug, architect)
"""
from __future__ import annotations

from enum import Enum
from .circuit_breaker import CircuitBreaker
from .diagnostics import record_degradation


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

        # If the router detects that full-model calls are failing
        # (tracked via circuit breaker), downgrade to cheaper alternatives
        # Check any worker's breaker state for OPEN
        try:
            breaker_open = False
            # QUARANTINED counts here too: it means a worker's local setup is
            # broken, which is an even more certain "don't send it work" than
            # a plain OPEN trip -- both are reasons to downgrade the route.
            breaker_open = any(
                state.value in ("open", "quarantined")
                for state in self._breaker.get_all_states().values()
            )
            if breaker_open and route == Route.FULL_MODEL:
                route = Route.CHEAP_MODEL
        except Exception as exc:
            record_degradation(
                "cost_router", "circuit-breaker inspection failed", exc,
                consequence="routing used the original tier; a failing full-model lane may not have been downgraded",
            )

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
