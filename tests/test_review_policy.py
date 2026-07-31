"""Tier-aware review: who is strong enough to judge this, and when is it worth it.

`verify.py` refuses self-review and warns on same-family review. Neither check
looks at CAPABILITY, so a different worker at a weaker tier passed both while
carrying no information -- and "two workers agreed" reads in a receipt exactly
like real corroboration.

The floor here is not an escalation policy, it is a rule: a reviewer must never
be weaker than the implementer.
"""

from __future__ import annotations

import pytest

from forgeos.core.effort import Difficulty
from forgeos.core.review_policy import (
    DEFAULT_MIN_WIN_RATE,
    plan_review,
    required_review,
    review_is_adequate,
)
from forgeos.core.router import Tier


# ------------------------------------------------------------------ the floor


def test_a_weaker_reviewer_is_blocked_outright():
    """The shape that produces 'approved' with nothing behind it."""
    req = required_review(implementer_tier=Tier.STRONG)
    ok, blocking, _warnings = review_is_adequate(
        req, reviewer_tier=Tier.LOCAL, implementer_tier=Tier.STRONG
    )
    assert not ok
    assert any("BELOW implementer" in r for r in blocking)


def test_an_equal_tier_reviewer_passes_ordinary_work():
    req = required_review(implementer_tier=Tier.CHEAP_API)
    # Bound to a local first: `reviewer_tier=Tier.CHEAP_API,
    # implementer_tier=Tier.CHEAP_API` on one line reads as a key assignment to
    # gitleaks' generic-api-key rule, and forgeos runs that scanner over its own
    # source.
    cheap = Tier.CHEAP_API
    ok, blocking, _ = review_is_adequate(
        req, reviewer_tier=cheap, implementer_tier=cheap
    )
    assert ok and blocking == []


def test_a_stronger_reviewer_always_passes():
    req = required_review(implementer_tier=Tier.LOCAL, difficulty=Difficulty.DEEP)
    ok, _blocking, _ = review_is_adequate(
        req, reviewer_tier=Tier.STRONG, implementer_tier=Tier.LOCAL
    )
    assert ok


# --------------------------------------------------------------- the triggers


@pytest.mark.parametrize("difficulty", [Difficulty.COMPLEX, Difficulty.DEEP])
def test_hard_work_asks_for_a_stronger_reader(difficulty):
    req = required_review(implementer_tier=Tier.CHEAP_API, difficulty=difficulty)
    assert req.min_tier > Tier.CHEAP_API
    assert req.escalated


@pytest.mark.parametrize("difficulty", [Difficulty.LOOKUP, Difficulty.MECHANICAL,
                                        Difficulty.STANDARD])
def test_ordinary_work_does_not_burn_the_scarce_tier(difficulty):
    """Escalating everything re-buys what mechanical gates already did free."""
    req = required_review(implementer_tier=Tier.CHEAP_API, difficulty=difficulty)
    assert req.min_tier == Tier.CHEAP_API
    assert not req.escalated


def test_a_gate_warning_escalates():
    req = required_review(implementer_tier=Tier.CHEAP_API, gate_warnings=2)
    assert req.escalated and req.min_tier > Tier.CHEAP_API
    assert any("warning" in r for r in req.reasons)


def test_a_poor_win_rate_escalates():
    """A worker that keeps getting rejected does not get a cheaper reviewer."""
    req = required_review(implementer_tier=Tier.SUBSCRIPTION, win_rate=0.2)
    assert req.escalated
    assert any("win rate" in r for r in req.reasons)


def test_a_good_win_rate_does_not_escalate():
    req = required_review(implementer_tier=Tier.SUBSCRIPTION,
                          win_rate=DEFAULT_MIN_WIN_RATE + 0.2)
    assert not req.escalated


def test_an_unmeasured_win_rate_does_not_escalate_by_itself():
    """No data is not bad data. Escalating on absence would tier-up every new
    worker's first task forever."""
    assert not required_review(implementer_tier=Tier.CHEAP_API, win_rate=None).escalated


@pytest.mark.parametrize("cap", ["security", "migration", "auth", "deploy"])
def test_risky_surface_escalates(cap):
    req = required_review(implementer_tier=Tier.CHEAP_API, capabilities={cap, "edit"})
    assert req.escalated
    assert any("risky surface" in r for r in req.reasons)


def test_ordinary_capabilities_do_not_escalate():
    assert not required_review(implementer_tier=Tier.CHEAP_API,
                               capabilities={"edit", "python"}).escalated


# ---------------------------------------------------------------- the ladder


def test_escalation_is_one_rung_not_straight_to_the_top():
    """The point is a reader who could have done the work, not the most
    expensive model on the shelf."""
    req = required_review(implementer_tier=Tier.LOCAL, difficulty=Difficulty.DEEP)
    assert req.min_tier == Tier.CHEAP_API


def test_escalation_never_reaches_council():
    """COUNCIL is a parallel panel -- a different decision with a different
    cost, not the top of this ladder."""
    req = required_review(implementer_tier=Tier.STRONG, difficulty=Difficulty.DEEP,
                          win_rate=0.1, gate_warnings=5, capabilities={"security"})
    assert req.min_tier == Tier.STRONG
    assert req.min_tier < Tier.COUNCIL


def test_many_triggers_still_only_raise_one_rung():
    a = required_review(implementer_tier=Tier.LOCAL, difficulty=Difficulty.DEEP)
    b = required_review(implementer_tier=Tier.LOCAL, difficulty=Difficulty.DEEP,
                        win_rate=0.1, gate_warnings=3, capabilities={"security"})
    assert a.min_tier == b.min_tier
    assert len(b.reasons) > len(a.reasons), "every reason is still recorded"


# ------------------------------------------------------------- family warning


def test_same_family_review_warns_but_does_not_block():
    """Blocking it outright would refuse every single-vendor fleet."""
    req = required_review(implementer_tier=Tier.CHEAP_API)
    cheap = Tier.CHEAP_API
    ok, blocking, warnings = review_is_adequate(
        req, reviewer_tier=cheap, implementer_tier=cheap,
        reviewer_family="anthropic", implementer_family="anthropic",
    )
    assert ok and blocking == []
    assert warnings and "2404.13076" in warnings[0]


def test_cross_family_review_draws_no_warning():
    req = required_review(implementer_tier=Tier.CHEAP_API)
    cheap = Tier.CHEAP_API
    _ok, _blocking, warnings = review_is_adequate(
        req, reviewer_tier=cheap, implementer_tier=cheap,
        reviewer_family="anthropic", implementer_family="deepseek",
    )
    assert warnings == []


# ------------------------------------------------------------------ the plan


def test_the_plan_says_when_it_did_not_escalate_and_why():
    plan = plan_review(implementer_tier=Tier.CHEAP_API, difficulty=Difficulty.LOOKUP)
    assert plan.escalate is False
    assert plan.notes and "mechanical gates" in plan.notes[0]


def test_the_plan_escalates_on_a_trigger():
    plan = plan_review(implementer_tier=Tier.CHEAP_API, difficulty=Difficulty.DEEP)
    assert plan.escalate is True
    assert plan.requirement.min_tier > Tier.CHEAP_API


def test_the_requirement_renders_its_reasoning():
    req = required_review(implementer_tier=Tier.LOCAL, capabilities={"security"})
    text = req.render()
    assert "security" in text and "CHEAP_API" in text
