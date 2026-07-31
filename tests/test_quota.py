"""Quota tracking tests.

Deterministic: every time is passed explicitly via `at=`, never slept for. Two
themes — reset times must come from the provider (never invented), and a finite
banked reset must never be spent by accident.
"""

from __future__ import annotations

import pytest

from forgeos.core.quota import (
    CONFIRM_AFTER_RESET_SECONDS,
    PROBE_BACKOFF_SECONDS,
    ExhaustionSignal,
    QuotaSource,
    QuotaState,
    QuotaTracker,
    QuotaWindow,
    parse_usage_report,
    should_rotate,
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


# --------------------------------------------------------------- rotation rule


def test_should_rotate_only_on_vendor_exhausted():
    assert should_rotate(ExhaustionSignal.VENDOR_EXHAUSTED) is True
    assert should_rotate(ExhaustionSignal.RATE_LIMITED) is False
    assert should_rotate(ExhaustionSignal.NETWORK_ERROR) is False
    assert should_rotate(ExhaustionSignal.OK) is False


def test_record_signal_vendor_exhausted_blocks_until_its_reset():
    q = QuotaTracker()
    q.record_signal("codex", ExhaustionSignal.VENDOR_EXHAUSTED, resets_at=T0 + HOUR, at=T0)
    assert q.available("codex", at=T0) is False
    assert q.available("codex", at=T0 + HOUR) is True


def test_record_signal_rate_limited_pauses_using_retry_after():
    q = QuotaTracker()
    q.record_signal("codex", ExhaustionSignal.RATE_LIMITED, retry_after=120, at=T0)
    assert q.available("codex", at=T0) is False
    assert q.available("codex", at=T0 + 119) is False
    assert q.available("codex", at=T0 + 120) is True


def test_record_signal_rate_limited_requires_retry_after():
    q = QuotaTracker()
    with pytest.raises(ValueError):
        q.record_signal("codex", ExhaustionSignal.RATE_LIMITED, at=T0)


def test_record_signal_network_error_leaves_a_healthy_seat_untouched():
    """The seat is fine; the network isn't. Recording this must not make a
    healthy seat look exhausted."""
    q = QuotaTracker()
    q.record_signal("codex", ExhaustionSignal.NETWORK_ERROR, at=T0)
    assert q.available("codex", at=T0) is True
    assert q.state("codex") is None


def test_record_signal_network_error_does_not_clear_an_existing_exhaustion():
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    q.record_signal("codex", ExhaustionSignal.NETWORK_ERROR, at=T0 + 10)
    assert q.available("codex", at=T0 + 10) is False


def test_record_signal_ok_clears_a_prior_exhaustion():
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    q.record_signal("codex", ExhaustionSignal.OK, at=T0 + 10)
    assert q.available("codex", at=T0 + 10) is True


# ------------------------------------------------------- per (provider, model)


def test_quota_is_tracked_per_model_not_just_per_provider():
    """An exhausted Fable quota on an account must not block Opus/Sonnet on
    that same account."""
    q = QuotaTracker()
    q.record_exhaustion("claude", model="fable", resets_at=T0 + HOUR, at=T0)
    assert q.available("claude", at=T0, model="fable") is False
    assert q.available("claude", at=T0, model="opus") is True


def test_omitting_model_matches_the_legacy_no_model_key():
    """Backward compatibility: calls that never pass `model` land on the same
    key as before per-model tracking existed."""
    q = QuotaTracker()
    q.record_exhaustion("codex", resets_at=T0 + HOUR, at=T0)
    assert q.available("codex", at=T0, model="") is False
    assert q.state("codex").provider == q.state("codex", model="").provider


# ------------------------------------------------------------------ select_seat


def test_select_seat_picks_the_only_available_candidate():
    q = QuotaTracker()
    q.record_exhaustion("codex", model="gpt5", resets_at=T0 + HOUR, at=T0)
    choice = q.select_seat("gpt5", ["codex", "claude"], at=T0)
    assert choice.ready is True
    assert choice.seat == "claude"


def test_select_seat_prefers_the_seat_whose_window_resets_soonest():
    """Spend the seat closest to its own reset first; save the others' headroom."""
    q = QuotaTracker()
    q.record_report("seat_a", "Weekly: 50% remaining, resets in 1h", at=T0)
    q.record_report("seat_b", "Weekly: 50% remaining, resets in 5h", at=T0)
    choice = q.select_seat("sonnet", ["seat_a", "seat_b"], at=T0)
    assert choice.ready is True
    assert choice.seat == "seat_a"


def test_select_seat_reports_the_honest_wait_when_all_candidates_are_blocked():
    q = QuotaTracker()
    q.record_exhaustion("codex", model="gpt5", resets_at=T0 + HOUR, at=T0)
    q.record_exhaustion("claude", model="gpt5", resets_at=T0 + 2 * HOUR, at=T0)
    choice = q.select_seat("gpt5", ["codex", "claude"], at=T0)
    assert choice.ready is False
    assert choice.seat is None
    assert choice.wait_seconds == pytest.approx(HOUR)


def test_select_seat_does_not_guess_a_wait_when_no_reset_is_known():
    q = QuotaTracker()
    q.record_exhaustion("codex", model="gpt5", at=T0)  # no resets_at reported
    choice = q.select_seat("gpt5", ["codex"], at=T0)
    assert choice.ready is False
    assert choice.wait_seconds is None


def test_select_seat_with_no_candidates_is_not_ready():
    choice = QuotaTracker().select_seat("gpt5", [], at=T0)
    assert choice.ready is False
    assert choice.seat is None


# ------------------------------------------------------------ persistence


def test_quota_snapshot_round_trips_provider_facts_and_parked_work(tmp_path):
    path = tmp_path / "quota.json"
    original = QuotaTracker()
    original.record_exhaustion(
        "claude", model="sonnet", window=QuotaWindow.WEEKLY,
        resets_at=T0 + 2 * HOUR, banked_resets=1, at=T0,
    )
    original.park(
        "job-1", "claude", model="sonnet", task_id="task-1", value=0.9, reason="weekly cap"
    )
    original.record_probe("claude", model="sonnet")
    original.mark_banked_consumed("claude", model="sonnet", at=T0)
    original.save(path)

    restored = QuotaTracker.load(path)

    assert restored.state("claude", model="sonnet").banked_resets == 0
    assert restored.state("claude", model="sonnet").source is QuotaSource.ESTIMATED
    assert restored.parked_for("claude")[0].task_id == "task-1"
    assert restored.next_probe_at("claude", model="sonnet", at=T0) is None
    assert restored.snapshot()["schema"] == "forgeos.quota.v1"


def test_corrupt_quota_snapshot_is_rejected(tmp_path):
    path = tmp_path / "quota.json"
    path.write_text('{"schema":"wrong"}', encoding="utf-8")

    with pytest.raises(ValueError, match="expected quota snapshot schema"):
        QuotaTracker.load(path)
