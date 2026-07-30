"""Capacity market tests.

The tests that matter are the negative-price ones. Every cost-minimising router can
conserve a scarce resource; almost none can recognise that an unused allowance is
about to expire and should be spent FIRST. Unused quota at reset is a total loss,
and pricing it at zero makes that loss invisible.
"""

from __future__ import annotations

import pytest

from hive.core.market import (
    MIN_CONFIDENCE_TO_PRICE,
    URGENCY_WINDOW_SECONDS,
    CapacityMarket,
    Entitlement,
    Forecast,
    TelemetrySource,
)

T0 = 1_800_000_000.0
HOUR = 3600.0


def _sub(rid="claude.sub", remaining=0.5, resets_in=24 * HOUR,
         source=TelemetrySource.OFFICIAL_CLI, cash=0.0) -> Entitlement:
    return Entitlement(resource_id=rid, remaining=remaining,
                       resets_at=T0 + resets_in, source=source,
                       measured_at=T0, cash_cost=cash)


def _api(rid="openrouter", cash=0.5) -> Entitlement:
    # A metered API has no window to expire, so no reset time.
    return Entitlement(resource_id=rid, remaining=1.0, resets_at=None,
                       source=TelemetrySource.OFFICIAL_API, measured_at=T0,
                       cash_cost=cash)


# ------------------------------------------------------------- provenance


def test_official_telemetry_is_a_fact_and_estimates_are_not():
    assert _sub(source=TelemetrySource.OFFICIAL_API).is_measured
    assert _sub(source=TelemetrySource.OFFICIAL_CLI).is_measured
    assert not _sub(source=TelemetrySource.ESTIMATED).is_measured
    assert not _sub(source=TelemetrySource.USER_ENTERED).is_measured


def test_rendering_marks_estimates_visibly():
    """A confident-looking number derived from a guess is how an optimiser makes
    bad decisions on bad data."""
    measured = _sub(remaining=0.64, source=TelemetrySource.OFFICIAL_API)
    assert measured.render() == "64% measured"

    est = Entitlement(resource_id="x", remaining=0.64, source=TelemetrySource.ESTIMATED,
                      error_bound=0.08)
    assert "estimated" in est.render()
    assert "±8%" in est.render()


def test_confidence_ordering_matches_trustworthiness():
    assert (TelemetrySource.OFFICIAL_API.confidence
            > TelemetrySource.LOCAL_USAGE_LOG.confidence
            > TelemetrySource.ESTIMATED.confidence)


# ---------------------------------------------------------------- urgency


def test_urgency_is_zero_far_from_reset_and_rises_near_it():
    ent = _sub(resets_in=48 * HOUR)
    assert ent.urgency(at=T0) == 0.0

    soon = _sub(resets_in=URGENCY_WINDOW_SECONDS / 2)
    assert soon.urgency(at=T0) == pytest.approx(0.5)

    imminent = _sub(resets_in=60)
    assert imminent.urgency(at=T0) > 0.9


def test_a_resource_with_no_reset_has_no_urgency():
    """A metered API cannot expire, so it can never be use-it-or-lose-it."""
    assert _api().urgency(at=T0) == 0.0


# ------------------------------------------------------------ shadow price


def test_scarce_quota_prices_positive_and_advises_conserving():
    m = CapacityMarket()
    m.record(_sub(remaining=0.10))
    m.record(_api(cash=0.5))
    m.forecast(Forecast(resource_id="claude.sub", expected_demand=0.9, confidence=0.9))

    p = m.shadow_price("claude.sub", at=T0)
    assert p.price > 0
    assert p.scarce
    assert "reserve" in p.advice


def test_expiring_surplus_prices_NEGATIVE_and_advises_spending_first():
    """THE case a conventional router cannot express. 80% left, almost no demand,
    reset imminent — that allowance is about to become worthless."""
    m = CapacityMarket()
    m.record(_sub(remaining=0.8, resets_in=HOUR))
    m.record(_api(cash=0.5))
    m.forecast(Forecast(resource_id="claude.sub", expected_demand=0.05, confidence=0.9))

    p = m.shadow_price("claude.sub", at=T0)
    assert p.price < 0, "an about-to-expire surplus must price negative"
    assert p.expiring_unused
    assert "FIRST" in p.advice


def test_the_same_surplus_far_from_reset_is_not_urgent():
    """Surplus alone is not a reason to rush — demand may still materialise."""
    m = CapacityMarket()
    m.record(_sub(remaining=0.8, resets_in=72 * HOUR))
    m.record(_api(cash=0.5))
    m.forecast(Forecast(resource_id="claude.sub", expected_demand=0.05, confidence=0.9))

    p = m.shadow_price("claude.sub", at=T0)
    assert p.urgency == 0.0
    assert not p.expiring_unused


def test_balanced_supply_and_demand_prices_near_zero():
    m = CapacityMarket()
    m.record(_sub(remaining=0.5, resets_in=HOUR))
    m.record(_api(cash=0.5))
    m.forecast(Forecast(resource_id="claude.sub", expected_demand=0.5, confidence=0.9))

    p = m.shadow_price("claude.sub", at=T0)
    assert abs(p.price) <= 0.05
    assert "normally" in p.advice


def test_low_confidence_telemetry_refuses_to_price_aggressively():
    """Acting decisively on a guess is worse than acting normally."""
    m = CapacityMarket()
    m.record(_sub(remaining=0.9, resets_in=HOUR, source=TelemetrySource.ESTIMATED))
    m.record(_api(cash=0.5))
    m.forecast(Forecast(resource_id="claude.sub", expected_demand=0.0, confidence=0.2))

    p = m.shadow_price("claude.sub", at=T0)
    assert p.price == 0.0
    assert p.confidence < MIN_CONFIDENCE_TO_PRICE
    assert "too low to price" in p.advice


def test_conserving_is_worthless_when_the_alternative_is_free():
    """Hoarding a subscription buys nothing if the fallback costs nothing."""
    m = CapacityMarket()
    m.record(_sub(remaining=0.05))
    m.record(Entitlement(resource_id="local", remaining=1.0,
                         source=TelemetrySource.OFFICIAL_API, cash_cost=0.0))
    m.forecast(Forecast(resource_id="claude.sub", expected_demand=1.0, confidence=0.9))

    assert m.shadow_price("claude.sub", at=T0).price == 0.0


def test_unknown_resource_prices_neutral_rather_than_crashing():
    p = CapacityMarket().shadow_price("nope", at=T0)
    assert p.price == 0.0
    assert "unknown" in p.advice


def test_missing_forecast_assumes_balanced_rather_than_panicking():
    m = CapacityMarket()
    m.record(_sub(remaining=0.4, resets_in=HOUR))
    m.record(_api(cash=0.5))
    p = m.shadow_price("claude.sub", at=T0)
    assert abs(p.price) <= 0.05


# ------------------------------------------------------------- allocation


def test_effective_cost_makes_a_free_but_scarce_subscription_expensive():
    """Cash price says 'free'; scarcity says otherwise. That gap is the whole point."""
    m = CapacityMarket()
    m.record(_sub(remaining=0.05, cash=0.0))
    m.record(_api(cash=0.5))
    m.forecast(Forecast(resource_id="claude.sub", expected_demand=1.0, confidence=0.9))

    assert m.effective_cost("claude.sub", at=T0) > 0.0


def test_effective_cost_makes_an_expiring_subscription_cheaper_than_free():
    m = CapacityMarket()
    m.record(_sub(remaining=0.9, resets_in=HOUR, cash=0.0))
    m.record(_api(cash=0.6))
    m.forecast(Forecast(resource_id="claude.sub", expected_demand=0.0, confidence=0.9))

    assert m.effective_cost("claude.sub", at=T0) < 0.0


def test_ranking_prefers_the_expiring_allowance_over_a_metered_api():
    m = CapacityMarket()
    m.record(_sub(remaining=0.9, resets_in=HOUR, cash=0.0))
    m.record(_api(cash=0.6))
    m.forecast(Forecast(resource_id="claude.sub", expected_demand=0.0, confidence=0.9))

    assert m.prefer(["openrouter", "claude.sub"], at=T0) == "claude.sub"


def test_ranking_avoids_a_scarce_subscription_in_favour_of_the_api():
    m = CapacityMarket()
    m.record(_sub(remaining=0.02, cash=0.0))
    m.record(_api(cash=0.1))
    m.forecast(Forecast(resource_id="claude.sub", expected_demand=1.0, confidence=0.95))

    assert m.prefer(["claude.sub", "openrouter"], at=T0) == "openrouter"


def test_use_it_or_lose_it_lists_the_most_urgent_first():
    m = CapacityMarket()
    m.record(_sub(rid="a.sub", remaining=0.9, resets_in=HOUR))
    m.record(_sub(rid="b.sub", remaining=0.6, resets_in=5 * HOUR))
    m.record(_api(cash=0.5))
    m.forecast(Forecast(resource_id="a.sub", expected_demand=0.0, confidence=0.9))
    m.forecast(Forecast(resource_id="b.sub", expected_demand=0.0, confidence=0.9))

    expiring = m.use_it_or_lose_it(at=T0)
    assert [p.resource_id for p in expiring][0] == "a.sub"


def test_nothing_is_expiring_when_everything_is_in_demand():
    m = CapacityMarket()
    m.record(_sub(remaining=0.2, resets_in=HOUR))
    m.record(_api(cash=0.5))
    m.forecast(Forecast(resource_id="claude.sub", expected_demand=0.9, confidence=0.9))
    assert m.use_it_or_lose_it(at=T0) == []


def test_ranking_ignores_candidates_the_market_has_never_seen():
    m = CapacityMarket()
    m.record(_api(cash=0.5))
    assert m.rank(["openrouter", "ghost"], at=T0) == [("openrouter", 0.5)]


def test_report_renders_every_resource_with_provenance():
    m = CapacityMarket()
    m.record(_sub(remaining=0.9, resets_in=HOUR))
    m.record(_api(cash=0.5))
    m.forecast(Forecast(resource_id="claude.sub", expected_demand=0.0, confidence=0.9))

    text = m.report(at=T0)
    assert "claude.sub" in text and "openrouter" in text
    assert "measured" in text
    assert "FIRST" in text


def test_empty_market_reports_without_crashing():
    assert "resource" in CapacityMarket().report(at=T0)
    assert CapacityMarket().prices(at=T0) == []
