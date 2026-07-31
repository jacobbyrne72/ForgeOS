"""Tests for the per-provider error-text -> exhaustion-signal corpus.

Table-driven over real error strings per provider (mined from
`vendor/9router` and each vendor's own documented error format -- see
`forgeos/gateway/signals.py`'s module docstring for receipts). Three themes:

- unrecognized text is UNKNOWN (`None`), never a guessed rotation or pause
- retry-after is read from the body only when actually present, never invented
- feeding a classified signal through the existing `QuotaTracker` produces
  exactly the behavior `test_quota.py` already documents for that signal --
  this module adds a corpus, not a new rule
"""

from __future__ import annotations

import httpx
import pytest

from forgeos.core.quota import ExhaustionSignal, QuotaTracker, should_rotate
from forgeos.gateway.client import HttpTransport
from forgeos.gateway.signals import classify_provider_error, extract_retry_after

T0 = 1_800_000_000.0


# ------------------------------------------------------------- classification


@pytest.mark.parametrize(
    "provider, status_code, text, expected",
    [
        # --- Anthropic ---
        ("anthropic", 429, "Claude AI usage limit reached. Resets in 2h.", ExhaustionSignal.VENDOR_EXHAUSTED),
        (
            "anthropic",
            529,
            '{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
            ExhaustionSignal.RATE_LIMITED,
        ),
        (
            "anthropic",
            429,
            '{"type":"error","error":{"type":"rate_limit_error","message":"..."}}',
            ExhaustionSignal.RATE_LIMITED,
        ),
        ("anthropic", 429, "", ExhaustionSignal.RATE_LIMITED),
        # --- OpenAI ---
        (
            "openai",
            429,
            '{"error":{"message":"...","code":"insufficient_quota"}}',
            ExhaustionSignal.VENDOR_EXHAUSTED,
        ),
        ("openai", 403, "You exceeded your current quota", ExhaustionSignal.VENDOR_EXHAUSTED),
        (
            "openai",
            429,
            '{"error":{"message":"...","code":"rate_limit_exceeded"}}',
            ExhaustionSignal.RATE_LIMITED,
        ),
        ("openai", 429, "", ExhaustionSignal.RATE_LIMITED),
        # --- DeepSeek ---
        ("deepseek", 402, "Insufficient Balance", ExhaustionSignal.VENDOR_EXHAUSTED),
        ("deepseek", 402, "", ExhaustionSignal.VENDOR_EXHAUSTED),
        ("deepseek", 429, "Rate limit reached for requests", ExhaustionSignal.RATE_LIMITED),
        # --- OpenRouter ---
        ("openrouter", 402, "You have run out of credits", ExhaustionSignal.VENDOR_EXHAUSTED),
        ("openrouter", 402, "", ExhaustionSignal.VENDOR_EXHAUSTED),
        ("openrouter", 429, "Rate limit exceeded", ExhaustionSignal.RATE_LIMITED),
        # --- Google ---
        (
            "google",
            429,
            '{"error":{"status":"RESOURCE_EXHAUSTED","message":"Quota exceeded for quota metric X per day"}}',
            ExhaustionSignal.VENDOR_EXHAUSTED,
        ),
        (
            "google",
            429,
            '{"error":{"status":"RESOURCE_EXHAUSTED","message":"Resource has been exhausted"}}',
            ExhaustionSignal.RATE_LIMITED,
        ),
        ("google", 429, "", ExhaustionSignal.RATE_LIMITED),
        # --- generic / unregistered provider falls back to the shared table ---
        ("some-new-vendor", 402, "Quota exceeded for this billing period", ExhaustionSignal.VENDOR_EXHAUSTED),
        ("some-new-vendor", 500, "Too many requests, please slow down", ExhaustionSignal.RATE_LIMITED),
        ("some-new-vendor", 503, "Server overloaded, capacity exceeded", ExhaustionSignal.RATE_LIMITED),
        ("some-new-vendor", 429, "", ExhaustionSignal.RATE_LIMITED),
    ],
)
def test_classifies_real_provider_error_text(provider, status_code, text, expected):
    assert classify_provider_error(provider, status_code, text) is expected


@pytest.mark.parametrize(
    "provider, status_code, text",
    [
        ("anthropic", None, "Something went wrong."),
        ("anthropic", 400, "Invalid request: missing 'messages' field"),
        ("openai", 401, "Incorrect API key provided"),
        ("openrouter", 404, "No endpoints found for model"),
        ("some-new-vendor", 500, "Internal server error"),
        ("unheard-of-vendor", None, ""),
    ],
)
def test_unrecognized_text_is_unknown_not_a_guess(provider, status_code, text):
    """The safety property: text that matches nothing must return `None`, not
    a default signal. A wrong VENDOR_EXHAUSTED rotates off a good seat; a
    wrong RATE_LIMITED parks it -- neither is safe to guess."""
    assert classify_provider_error(provider, status_code, text) is None


def test_unknown_never_looks_like_ok_either():
    """`None` must be distinguishable from `ExhaustionSignal.OK` -- a caller
    that can't tell "we learned nothing" from "the call succeeded" would be
    tempted to clear an existing exhaustion on unrecognized text, which is
    the same class of bug as guessing a rotation."""
    result = classify_provider_error("anthropic", 400, "Invalid request")
    assert result is None
    assert result is not ExhaustionSignal.OK


# ------------------------------------------------------------------ retry-after


@pytest.mark.parametrize(
    "text, expected_seconds",
    [
        ('{"error":{"details":[{"retryDelay":"41s"}]}}', 41.0),
        ('{"error":{"details":[{"retryDelay": "12.5s"}]}}', 12.5),
        ("429 Too Many Requests. Retry-After: 90", 90.0),
        ("Rate limited. Please try again in 20s.", 20.0),
        ("Rate limited. Please retry after 15 seconds.", 15.0),
    ],
)
def test_extracts_retry_after_present_in_body_text(text, expected_seconds):
    assert extract_retry_after(text) == pytest.approx(expected_seconds)


@pytest.mark.parametrize("text", ["", "Something went wrong.", "Invalid API key provided"])
def test_extract_retry_after_is_none_when_absent_never_invented(text):
    assert extract_retry_after(text) is None


# ------------------------------------------------------ HttpTransport wiring


def test_httptransport_uses_header_retry_after_when_present(monkeypatch):
    fake_response = httpx.Response(429, headers={"Retry-After": "77"}, text="rate limited")
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: fake_response)

    transport = HttpTransport(base_url="https://example.invalid", name="fakehttp")
    from forgeos.gateway.client import RateLimitError

    with pytest.raises(RateLimitError) as exc_info:
        transport.complete(
            model_id="test-model", prompt="hi", max_output_tokens=10, reasoning_effort="none", tools_schema=None
        )
    assert exc_info.value.retry_after == pytest.approx(77.0)


def test_httptransport_falls_back_to_body_text_retry_after_when_header_absent(monkeypatch):
    """The sharpening this module adds: today's flat 30s guess only applies
    when NEITHER the header nor the body says anything."""
    fake_response = httpx.Response(
        429,
        json={"error": {"details": [{"retryDelay": "41s"}]}},
    )
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: fake_response)

    transport = HttpTransport(base_url="https://example.invalid", name="fakehttp")
    from forgeos.gateway.client import RateLimitError

    with pytest.raises(RateLimitError) as exc_info:
        transport.complete(
            model_id="test-model", prompt="hi", max_output_tokens=10, reasoning_effort="none", tools_schema=None
        )
    assert exc_info.value.retry_after == pytest.approx(41.0)


def test_httptransport_uses_flat_default_when_neither_header_nor_body_has_a_hint(monkeypatch):
    fake_response = httpx.Response(429, text="Too many requests.")
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: fake_response)

    transport = HttpTransport(base_url="https://example.invalid", name="fakehttp")
    from forgeos.gateway.client import RateLimitError

    with pytest.raises(RateLimitError) as exc_info:
        transport.complete(
            model_id="test-model", prompt="hi", max_output_tokens=10, reasoning_effort="none", tools_schema=None
        )
    assert exc_info.value.retry_after == pytest.approx(30.0)


# --------------------------------------------------- compatibility with QuotaTracker
#
# This module adds a corpus, not a new rule: feeding a classified signal
# through `QuotaTracker.record_signal` must behave exactly as `test_quota.py`
# already documents for that signal.


def test_classified_vendor_exhausted_rotates_and_blocks_until_reset():
    signal = classify_provider_error("openai", 429, '{"error":{"code":"insufficient_quota"}}')
    assert should_rotate(signal) is True

    q = QuotaTracker()
    q.record_signal("openai", signal, resets_at=T0 + 3600, at=T0)
    assert q.available("openai", at=T0) is False
    assert q.available("openai", at=T0 + 3600) is True


def test_classified_rate_limited_pauses_without_rotating():
    signal = classify_provider_error("anthropic", 429, "rate_limit_error")
    assert should_rotate(signal) is False

    q = QuotaTracker()
    retry_after = extract_retry_after("Retry-After: 60") or 30.0
    q.record_signal("anthropic", signal, retry_after=retry_after, at=T0)
    assert q.available("anthropic", at=T0) is False
    assert q.available("anthropic", at=T0 + 60) is True


def test_unknown_classification_is_never_fed_to_record_signal():
    """The contract a real caller must honor: `None` means skip the call
    entirely, leaving whatever state `QuotaTracker` already had."""
    q = QuotaTracker()
    q.record_exhaustion("anthropic", resets_at=T0 + 3600, at=T0)

    signal = classify_provider_error("anthropic", 400, "Invalid request: bad JSON")
    assert signal is None
    if signal is not None:  # pragma: no cover - documents the required guard
        q.record_signal("anthropic", signal, at=T0)

    # Untouched: still exhausted exactly as it was before the unrecognized error.
    assert q.available("anthropic", at=T0) is False
    assert q.available("anthropic", at=T0 + 3600) is True
