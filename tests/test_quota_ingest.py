from __future__ import annotations

import pytest

from forgeos.core.quota import ExhaustionSignal, QuotaTracker, QuotaWindow
from forgeos.core.quota_ingest import QuotaIngestor

T0 = 1_800_000_000.0


def test_headers_normalize_utilization_reset_and_window_without_network():
    observation = QuotaIngestor.from_headers(
        "claude",
        {
            "anthropic-ratelimit-unified-5h-utilization": "0.25",
            "anthropic-ratelimit-unified-5h-reset": "120",
            "anthropic-ratelimit-unified-status": "allowed",
        },
        model="sonnet",
        at=T0,
    )

    assert observation.signal is ExhaustionSignal.OK
    assert observation.pct_remaining == pytest.approx(75.0)
    assert observation.resets_at == pytest.approx(T0 + 120)
    assert observation.window is QuotaWindow.FIVE_HOUR

    tracker = QuotaTracker()
    state = QuotaIngestor.apply(tracker, observation)
    assert state.pct_remaining == pytest.approx(75.0)
    assert tracker.available("claude", at=T0, model="sonnet") is True


def test_standard_headers_produce_typed_rate_limit_pause():
    observation = QuotaIngestor.from_headers(
        "openai",
        {
            "x-ratelimit-remaining-tokens": "25",
            "x-ratelimit-limit-tokens": "100",
            "retry-after": "30",
        },
        at=T0,
    )

    assert observation.signal is ExhaustionSignal.RATE_LIMITED
    assert observation.pct_remaining == pytest.approx(25.0)
    tracker = QuotaTracker()
    state = QuotaIngestor.apply(tracker, observation)
    assert state.exhausted is True
    assert tracker.available("openai", at=T0 + 29) is False
    assert tracker.available("openai", at=T0 + 30) is True


def test_rejected_header_is_vendor_exhaustion_and_can_be_parked():
    observation = QuotaIngestor.from_headers(
        "claude",
        {
            "anthropic-ratelimit-unified-status": "rejected",
            "anthropic-ratelimit-unified-7d-reset": "3600",
        },
        model="opus",
        at=T0,
    )
    tracker = QuotaTracker()
    QuotaIngestor.apply(tracker, observation)
    tracker.park("job-1", "claude", model="opus")

    assert observation.signal is ExhaustionSignal.VENDOR_EXHAUSTED
    assert tracker.available("claude", at=T0, model="opus") is False
    assert [p.job_id for p in tracker.resume_ready(at=T0 + 3600)] == ["job-1"]


def test_cli_report_uses_the_existing_fact_parser():
    observation = QuotaIngestor.from_report(
        "codex", "Weekly: 37% remaining. Resets in 45 minutes.", at=T0
    )

    assert observation.source == "cli_report"
    assert observation.pct_remaining == 37.0
    assert observation.resets_at == pytest.approx(T0 + 45 * 60)
    assert observation.window is QuotaWindow.WEEKLY
