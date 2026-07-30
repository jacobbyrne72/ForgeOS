"""Router tests — the escalation ladder.

The refusal tests are the important ones. Escalating on a failure no model can fix
is the most expensive reflex in an agent harness, and `escalate()` returning None is
the behaviour that prevents it.
"""

from __future__ import annotations

import pytest

from hive.contracts import FailureClass
from hive.core.router import (
    DEFAULT_EFFORT,
    ReasoningEffort,
    Route,
    Router,
    Tier,
)
from hive.registry import Adapter, CostTier, Registry, WorkerProfile


def _w(wid: str, tier: CostTier, caps: set[str], win: float = 0.8, **kw) -> WorkerProfile:
    return WorkerProfile(
        worker_id=wid,
        adapter=Adapter.OMC_TEAM,
        tier=tier,
        capabilities=caps,
        prior_win_rate=win,
        **kw,
    )


@pytest.fixture()
def ladder() -> Router:
    return Router(
        Registry(
            [
                _w("local.free", CostTier.FREE, {"edit", "mechanical"}, win=0.55, can_edit_files=True),
                _w("cheap.api", CostTier.CHEAP, {"edit", "mechanical", "review"}, win=0.70,
                   can_edit_files=True),
                _w("sub.cli", CostTier.STANDARD, {"edit", "mechanical", "review", "debug"},
                   win=0.85, can_edit_files=True),
                _w("strong.model", CostTier.PREMIUM,
                   {"edit", "review", "debug", "architecture", "security"}, win=0.93,
                   can_edit_files=True),
            ]
        )
    )


# ------------------------------------------------------------------- effort


def test_effort_follows_the_task_not_the_model(ladder):
    assert ladder.effort_for(["trivial_edit"]) == ReasoningEffort.NONE
    assert ladder.effort_for(["mechanical"]) == ReasoningEffort.LOW
    assert ladder.effort_for(["narrow_bugfix"]) == ReasoningEffort.MEDIUM
    assert ladder.effort_for(["architecture"]) == ReasoningEffort.MAX


def test_effort_takes_the_highest_demanded_by_any_capability(ladder):
    assert ladder.effort_for(["mechanical", "security"]) == ReasoningEffort.MAX


def test_unknown_capability_gets_the_default_effort(ladder):
    assert ladder.effort_for(["kubernetes"]) == DEFAULT_EFFORT


# -------------------------------------------------------------------- tier 0


def test_deterministic_work_never_reaches_a_model(ladder):
    """The largest saving available: don't call a model at all."""
    r = ladder.route(["edit"], deterministic_available=True)
    assert r.tier is Tier.DETERMINISTIC
    assert r.deterministic is True
    assert r.costs_tokens is False
    assert r.effort == ReasoningEffort.NONE


# -------------------------------------------------------------- cheapest-first


def test_low_risk_work_goes_to_the_cheapest_tier_that_clears_the_bar(ladder):
    r = ladder.route(["mechanical"], risk="low")
    assert r.tier is Tier.LOCAL
    assert r.worker_id == "local.free"


def test_high_risk_work_skips_tiers_that_cannot_clear_the_bar(ladder):
    """0.75 threshold: local (0.55) and cheap (0.70) are not eligible at all."""
    r = ladder.route(["mechanical"], risk="high")
    assert r.tier is Tier.SUBSCRIPTION
    assert r.worker_id == "sub.cli"


def test_medium_risk_lands_between_the_two(ladder):
    r = ladder.route(["mechanical"], risk="medium")
    assert r.worker_id == "cheap.api"


def test_capability_still_gates_the_ladder(ladder):
    """Only the premium worker knows security, so cheapness is irrelevant."""
    r = ladder.route(["security"], risk="low")
    assert r.worker_id == "strong.model"
    assert r.tier is Tier.STRONG


def test_no_capable_worker_returns_none(ladder):
    assert ladder.route(["kubernetes"]) is None


def test_when_nothing_clears_the_bar_take_the_best_not_the_cheapest():
    """A cheap likely-failure costs its own tokens plus the retry plus escalation."""
    r = Router(
        Registry(
            [
                _w("weak.free", CostTier.FREE, {"edit"}, win=0.10, can_edit_files=True),
                _w("less.weak", CostTier.PREMIUM, {"edit"}, win=0.45, can_edit_files=True),
            ]
        )
    ).route(["edit"], risk="high")
    assert r.worker_id == "less.weak"
    assert "best available" in r.reason


def test_measured_history_overrides_the_declared_prior(ladder):
    stats = {
        "local.free": {"attempts": 10, "win_rate": 0.95, "avg_usd_micros": 0, "avg_seconds": 5},
    }
    r = ladder.route(["mechanical"], stats=stats, risk="high")
    assert r.worker_id == "local.free"
    assert r.measured is True
    assert r.tier is Tier.LOCAL


# ------------------------------------------------------------------ escalate


def test_model_failure_escalates_one_rung(ladder):
    current = Route(tier=Tier.LOCAL, worker_id="local.free")
    nxt = ladder.escalate(current, FailureClass.MODEL, ["mechanical"])
    assert nxt is not None
    assert nxt.tier > current.tier
    assert nxt.escalated_from is Tier.LOCAL


@pytest.mark.parametrize(
    "failure",
    [
        FailureClass.ENVIRONMENT,
        FailureClass.CONTEXT,
        FailureClass.SPECIFICATION,
        FailureClass.POLICY,
        FailureClass.TRANSIENT,
    ],
)
def test_non_model_failures_refuse_to_escalate(ladder, failure):
    """A broken venv, a missing symbol, or a bad spec fails identically at every
    price tier. Refusing here is the whole point of classifying failures."""
    current = Route(tier=Tier.LOCAL, worker_id="local.free")
    assert ladder.escalate(current, failure, ["mechanical"]) is None


def test_escalation_waits_for_a_phase_boundary(ladder):
    """Per-turn escalation pays premium rates for a whole task over one bad turn."""
    current = Route(tier=Tier.LOCAL, worker_id="local.free")
    assert ladder.escalate(current, FailureClass.MODEL, ["mechanical"],
                           at_phase_boundary=False) is None


def test_escalation_stops_at_the_top_of_the_ladder(ladder):
    current = Route(tier=Tier.COUNCIL, worker_id="strong.model")
    assert ladder.escalate(current, FailureClass.MODEL, ["mechanical"]) is None


def test_escalation_from_strong_does_not_silently_convene_a_council(ladder):
    current = Route(tier=Tier.STRONG, worker_id="strong.model")
    nxt = ladder.escalate(current, FailureClass.MODEL, ["architecture"])
    assert nxt is None or nxt.tier <= Tier.STRONG


# ------------------------------------------------------------------- council


def test_council_is_refused_when_tests_can_decide(ladder):
    """Run both and let the suite adjudicate — cheaper and more reliable than
    paying several models to argue, because argument converges on confidence."""
    assert ladder.should_convene_council("high", disagreement=True, tests_can_decide=True) is False


def test_council_only_for_high_risk_irreducible_disagreement(ladder):
    assert ladder.should_convene_council("high", True, False) is True
    assert ladder.should_convene_council("low", True, False) is False
    assert ladder.should_convene_council("high", False, False) is False


def test_tier_order_is_cost_ordered():
    assert Tier.DETERMINISTIC < Tier.LOCAL < Tier.CHEAP_API < Tier.SUBSCRIPTION
    assert Tier.SUBSCRIPTION < Tier.STRONG < Tier.COUNCIL


# ------------------------------------------------- output-budget routing (R2)


def test_route_sets_an_output_budget_matched_to_effort(ladder):
    """Routing the output length is as important as routing the model — output
    tokens are the expensive side of most price sheets."""
    easy = ladder.route(["mechanical"], risk="low")
    hard = ladder.route(["architecture"], risk="low")
    assert easy.effort == ReasoningEffort.LOW
    assert hard.effort == ReasoningEffort.MAX
    assert easy.max_output_tokens < hard.max_output_tokens


def test_effort_policy_keys_are_not_required_to_be_worker_capabilities(ladder):
    """Effort is a property of the work; capability is a property of the worker.
    They are separate namespaces, so an effort-only tag must not gate routing."""
    assert ladder.effort_for(["trivial_edit"]) == ReasoningEffort.NONE
    # ...but a worker still has to declare the capability to be eligible.
    assert ladder.route(["trivial_edit"]) is None
