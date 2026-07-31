"""Cost-router degradation and circuit-breaker integration."""

from __future__ import annotations

import pytest

from forgeos.cost_router import CostRouter, Route
from forgeos.diagnostics import clear, degradations


@pytest.fixture(autouse=True)
def _clean_diagnostics():
    clear()
    yield
    clear()


def test_open_full_model_breaker_downgrades_without_touching_private_state():
    router = CostRouter()
    for _ in range(3):
        router._breaker.record_failure("full-model")

    assert router.route("debug") is Route.CHEAP_MODEL
    assert degradations() == []


def test_breaker_inspection_failure_is_visible_but_preserves_route():
    class BrokenBreaker:
        def get_all_states(self):
            raise RuntimeError("breaker state unavailable")

    router = CostRouter()
    router._breaker = BrokenBreaker()

    assert router.route("debug") is Route.FULL_MODEL
    [degradation] = degradations()
    assert degradation.subsystem == "cost_router"
    assert "breaker state unavailable" in degradation.reason
    assert "original tier" in degradation.consequence


def test_unknown_task_still_uses_full_model_route():
    assert CostRouter().route("new_task_type") is Route.FULL_MODEL
