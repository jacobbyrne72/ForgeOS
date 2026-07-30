"""Quota tracking tests.

Deterministic: every time is passed explicitly via `at=`, never slept for. Two
themes — reset times must come from the provider (never invented), and a finite
banked reset must never be spent by accident.
"""

from __future__ import annotations

import pytest

from hive.core.quota import (
    CONFIRM_AFTER_RESET_SECONDS,
    PROBE_BACKOFF_SECONDS,
    QuotaSource,
    QuotaState,
    QuotaTracker,
    QuotaWindow,
    parse_usage_report,
)

T0 = 1_800_000_000.0
HOUR = 3600.0


# ------------------------------------------------------------------ parsing


def test_parses_hours_and_minutes_until_reset():
    facts = parse_usage_report("Usage limit reached. Resets in 2h 15m.", at=T0)
    assert facts["exhausted"] is True
    assert facts["resets_at"] == pytest.approx(T0 + 2 * HOUR + 15 * 60)


def test_parses_minutes_only():
    facts = parse_usage_report("Rate limit hit. Resets in 45 minutes.", at=T0)
    assert facts["resets_at"] == pytest.approx(T0 + 45 * 60)


def test_parses_retry_after_seconds():
    facts = parse_usage_report("429 Too Many Requests. Retry-After: 120", at=T0)
    assert facts["resets_at"] == pytest.approx(T0 + 120)


def test_parses_percentage_remaining():
    assert parse_usage_report("Weekly: 37% remaining")["pct_remaining"] == 37.0


def test_parses_banked_reset_count():
    assert parse_usage_report("You have 1 banked reset available")["banked_resets"] == 1


def test_distinguishes_weekly_from_five_hour_window():
    assert parse_usage_report("weekly limit reached")["window"] is QuotaWindow.WEEKLY
    assert parse_usage_report("5-hour limit reached")["window"] is QuotaWindow.FIVE_HOUR


def test_absent_fields_stay_absent_rather_than_defaulted():
    """A fabricated reset time is worse than none — the scheduler would trust it."""
    facts = parse_usage_report("Something went wrong.", at=T0)
    assert "resets_at" not in facts
    assert "pct_remaining" not in facts
    assert "exhausted" not in facts


def test_empty_text_does_not_crash():
    assert parse_usage_report("") == {}


# ------------------------------------------------------------- availability


def test_unknown_provider_is_treated_as_available():
    """Never observed is not the same as exhausted — refusing here would make an
    unmeasured provider permanently useless."""
    assert QuotaTracker().available("codex") is True


def test_exhausted_provider_is_unavailable_until_its_reset():
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    assert q.available("codex", at=T0) is False
    assert q.available("codex", at=T0 + HOUR - 1) is False
    assert q.available("codex", at=T0 + HOUR) is True


def test_exhausted_with_no_known_reset_stays_unavailable():
    """Without a reported reset there is nothing to wait for, so do not guess one."""
    q = QuotaTracker()
    st = q.record_exhaustion("codex", at=T0)
    assert st.source is QuotaSource.ESTIMATED
    assert q.available("codex", at=T0 + 10 * HOUR) is False


def test_reported_reset_is_marked_as_fact_not_estimate():
    q = QuotaTracker()
    st = q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    assert st.source is QuotaSource.REPORTED
    assert st.known


def test_available_providers_filters_the_candidate_list():
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    assert q.available_providers(["codex", "claude", "ollama"], at=T0) == ["claude", "ollama"]


def test_a_healthy_usage_report_clears_exhaustion():
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    q.record_report("codex", "Weekly: 82% remaining", at=T0 + HOUR + 60)
    assert q.available("codex", at=T0 + HOUR + 60) is True


# ------------------------------------------------------------------- probing


def test_probe_clusters_just_after_a_reported_reset_not_throughout():
    """Each probe against a closed window is a real failed request. Polling
    'constantly' spends the very resource being protected."""
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    nxt = q.next_probe_at("codex", at=T0)
    assert nxt == pytest.approx(T0 + HOUR + CONFIRM_AFTER_RESET_SECONDS)


def test_probe_is_due_immediately_once_the_reset_has_passed():
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    at = T0 + HOUR + 5
    assert q.next_probe_at("codex", at=at) == at


def test_probe_backs_off_when_no_reset_time_is_known():
    q = QuotaTracker()
    q.record_exhaustion("codex", at=T0)
    first = q.next_probe_at("codex", at=T0)
    assert first == pytest.approx(T0 + PROBE_BACKOFF_SECONDS[0])
    for _ in range(4):
        q.record_probe("codex")
    later = q.next_probe_at("codex", at=T0)
    assert later > first, "backoff must widen, not stay fixed"


def test_backoff_is_capped_not_unbounded():
    q = QuotaTracker()
    q.record_exhaustion("codex", at=T0)
    for _ in range(50):
        q.record_probe("codex")
    assert q.next_probe_at("codex", at=T0) == pytest.approx(T0 + PROBE_BACKOFF_SECONDS[-1])


def test_healthy_provider_needs_no_probe():
    assert QuotaTracker().next_probe_at("codex") is None


# -------------------------------------------------------------------- park


def test_parked_work_resumes_automatically_when_the_window_reopens(monkeypatch):
    """The point of the whole module: work waits, then resumes by itself."""
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    q.park("job_1", "codex", task_id="T-1", reason="weekly cap")

    assert q.resume_ready(at=T0) == []
    ready = q.resume_ready(at=T0 + HOUR)
    assert [p.job_id for p in ready] == ["job_1"]
    # Drained — a job must not be resumed twice.
    assert q.resume_ready(at=T0 + HOUR) == []


def test_parked_work_for_a_healthy_provider_is_immediately_ready():
    q = QuotaTracker()
    q.park("job_1", "claude")
    assert len(q.resume_ready(at=T0)) == 1


def test_parking_is_per_provider():
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    q.park("j1", "codex")
    q.park("j2", "claude")
    ready = q.resume_ready(at=T0)
    assert [p.job_id for p in ready] == ["j2"]
    assert len(q.parked_for("codex")) == 1


# ------------------------------------------------------------------ banked


def _exhausted_with_bank(**kw) -> QuotaTracker:
    q = QuotaTracker(**kw)
    q.record_exhaustion("codex", resets_at=T0 + 6 * HOUR, banked_resets=1, at=T0)
    return q


def test_banked_offer_is_surfaced_with_the_context_to_decide():
    q = _exhausted_with_bank()
    q.park("j1", "codex", value=0.9)
    offer = q.banked_offer("codex", at=T0)
    assert offer is not None
    assert offer.banked_available == 1
    assert offer.parked_jobs == 1
    assert offer.seconds_until_natural_reset == pytest.approx(6 * HOUR)
    assert "spend" in offer.recommendation


def test_no_offer_when_nothing_is_banked():
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    assert q.banked_offer("codex", at=T0) is None


def test_offer_recommends_holding_when_nothing_is_waiting():
    q = _exhausted_with_bank()
    assert "hold" in q.banked_offer("codex", at=T0).recommendation


def test_offer_recommends_holding_when_the_natural_reset_is_imminent():
    """Spending a finite reset to save ten minutes is waste."""
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + 600, banked_resets=1, at=T0)
    q.park("j1", "codex", value=1.0)
    assert "hold" in q.banked_offer("codex", at=T0).recommendation


def test_banked_reset_is_never_auto_consumed_by_default():
    """THE default that matters: a finite, non-refundable benefit is not spent
    unattended. Consuming one at 3am on a trivial task cannot be undone."""
    q = _exhausted_with_bank()
    q.park("j1", "codex", value=1.0)
    assert q.auto_consume_banked is False
    assert q.should_auto_consume("codex", at=T0) is False


def test_opt_in_auto_consume_fires_for_high_value_parked_work():
    q = _exhausted_with_bank(auto_consume_banked=True, banked_value_threshold=0.8)
    q.park("j1", "codex", value=0.95)
    assert q.should_auto_consume("codex", at=T0) is True


def test_opt_in_auto_consume_refuses_low_value_work():
    q = _exhausted_with_bank(auto_consume_banked=True, banked_value_threshold=0.8)
    q.park("j1", "codex", value=0.2)
    assert q.should_auto_consume("codex", at=T0) is False


def test_opt_in_auto_consume_refuses_when_reset_is_imminent():
    q = QuotaTracker(auto_consume_banked=True, banked_value_threshold=0.5)
    q.record_exhaustion("codex", resets_at=T0 + 300, banked_resets=1, at=T0)
    q.park("j1", "codex", value=1.0)
    assert q.should_auto_consume("codex", at=T0) is False


def test_opt_in_auto_consume_refuses_when_nothing_is_parked():
    q = _exhausted_with_bank(auto_consume_banked=True)
    assert q.should_auto_consume("codex", at=T0) is False


def test_consuming_a_reset_reopens_the_window_and_decrements_the_bank():
    q = _exhausted_with_bank()
    st = q.mark_banked_consumed("codex", at=T0)
    assert st is not None
    assert st.exhausted is False
    assert st.banked_resets == 0
    assert q.available("codex", at=T0) is True
    assert q.banked_consumed == [("codex", T0)]


def test_consumed_state_is_estimated_until_usage_confirms_it():
    """We believe the reset worked; we have not yet verified it. Mark it as such."""
    q = _exhausted_with_bank()
    st = q.mark_banked_consumed("codex", at=T0)
    assert st.source is QuotaSource.ESTIMATED
    assert "awaiting" in st.detail


def test_consuming_with_an_empty_bank_is_a_no_op():
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    assert q.mark_banked_consumed("codex", at=T0) is None
    assert q.available("codex", at=T0) is False


def test_spending_the_bank_frees_parked_work():
    q = _exhausted_with_bank()
    q.park("j1", "codex", value=0.9)
    q.mark_banked_consumed("codex", at=T0)
    assert [p.job_id for p in q.resume_ready(at=T0)] == ["j1"]


def test_quota_state_reports_time_remaining():
    st = QuotaState(provider="codex", exhausted=True, resets_at=T0 + 90)
    assert st.seconds_until_reset(at=T0) == pytest.approx(90)
    assert st.seconds_until_reset(at=T0 + 200) == 0.0
    assert QuotaState(provider="x").seconds_until_reset(at=T0) is None
