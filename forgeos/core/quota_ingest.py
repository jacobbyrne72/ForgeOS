"""Vendor-neutral quota telemetry normalization.

Adapters hand this module headers or CLI text they already received. It never
opens a network connection, reads credentials, or guesses a reset from a clock.
The normalized observation can then be applied to :class:`QuotaTracker` and
persisted by the owning ``Forge`` instance.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from ..contracts import now
from .quota import (
    ExhaustionSignal,
    QuotaSource,
    QuotaTracker,
    QuotaWindow,
    parse_usage_report,
)


class QuotaObservation(BaseModel):
    provider: str
    model: str = ""
    pct_remaining: float | None = Field(default=None, ge=0.0, le=100.0)
    resets_at: float | None = None
    retry_after: float | None = Field(default=None, ge=0.0)
    window: QuotaWindow = QuotaWindow.UNKNOWN
    banked_resets: int = Field(default=0, ge=0)
    signal: ExhaustionSignal = ExhaustionSignal.OK
    source: str = "provider_telemetry"
    observed_at: float = Field(default_factory=now)
    detail: str = ""


def _header_map(headers: Mapping[str, Any]) -> dict[str, str]:
    return {str(key).lower().replace("_", "-"): str(value).strip() for key, value in headers.items()}


def _first(headers: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = headers.get(name)
        if value not in (None, ""):
            return value
    return None


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else None


def _duration(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip().lower()
    number = _number(text)
    if number is None:
        return None
    if re.search(r"\bd\b|day", text):
        return number * 86_400
    if re.search(r"\bh\b|hour", text):
        return number * 3_600
    if re.search(r"\bm\b|min", text):
        return number * 60
    return number


def _reset_at(value: str | None, at: float) -> float | None:
    if value is None:
        return None
    number = _number(value)
    if number is None:
        return None
    # Common APIs use either epoch seconds or a duration such as ``2m``. A
    # large bare number is unambiguously epoch time; a small one is a delay.
    if number >= 1_000_000_000:
        return number
    delay = _duration(value)
    return at + delay if delay is not None and delay > 0 else None


def _window_from_keys(keys: list[str]) -> QuotaWindow:
    joined = " ".join(keys)
    if re.search(r"5[-_ ]?h|five[-_ ]?hour", joined):
        return QuotaWindow.FIVE_HOUR
    if re.search(r"7[-_ ]?d|weekly|seven[-_ ]?day", joined):
        return QuotaWindow.WEEKLY
    return QuotaWindow.UNKNOWN


def _remaining_pct(headers: dict[str, str]) -> float | None:
    # Anthropic's unified headers expose utilization, not remaining headroom.
    for key, value in headers.items():
        if "utilization" in key:
            utilization = _number(value)
            if utilization is not None:
                utilization = utilization / 100 if utilization > 1 else utilization
                return max(0.0, min(100.0, (1.0 - utilization) * 100.0))

    remaining = _first(
        headers,
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-remaining-requests",
        "anthropic-ratelimit-unified-remaining",
    )
    if remaining is None:
        return None
    remaining_number = _number(remaining)
    if remaining_number is None:
        return None
    limit = _first(
        headers,
        "x-ratelimit-limit-tokens",
        "x-ratelimit-limit-requests",
        "anthropic-ratelimit-unified-limit",
    )
    limit_number = _number(limit)
    if limit_number and limit_number > 0:
        return max(0.0, min(100.0, remaining_number / limit_number * 100.0))
    # A fractional remaining value is already a normalized ratio. Do not
    # pretend that an absolute count is a percentage without its denominator.
    if 0.0 <= remaining_number <= 1.0:
        return remaining_number * 100.0
    if 0.0 <= remaining_number <= 100.0 and "%" in remaining:
        return remaining_number
    return None


class QuotaIngestor:
    """Pure normalization/apply facade for gateway and CLI adapters."""

    @staticmethod
    def from_headers(
        provider: str,
        headers: Mapping[str, Any],
        *,
        model: str = "",
        at: float | None = None,
    ) -> QuotaObservation:
        at = at or now()
        normalized = _header_map(headers)
        status = _first(
            normalized,
            "anthropic-ratelimit-unified-status",
            "x-ratelimit-status",
            "ratelimit-status",
        )
        status = status.lower() if status else ""
        retry = _duration(_first(normalized, "retry-after", "x-ratelimit-retry-after"))
        signal = (
            ExhaustionSignal.VENDOR_EXHAUSTED
            if status in {"rejected", "exhausted", "blocked"}
            else ExhaustionSignal.RATE_LIMITED
            if retry is not None and status in {"", "rate_limited", "limited"}
            else ExhaustionSignal.OK
        )
        reset_keys = [key for key in normalized if "reset" in key]
        reset_value = _first(normalized, *reset_keys) if reset_keys else None
        banked = _number(_first(normalized, "x-ratelimit-banked-resets", "banked-resets")) or 0
        window = _window_from_keys(list(normalized))
        return QuotaObservation(
            provider=provider,
            model=model,
            pct_remaining=_remaining_pct(normalized),
            resets_at=_reset_at(reset_value, at),
            retry_after=retry,
            window=window,
            banked_resets=int(banked),
            signal=signal,
            source="response_headers",
            observed_at=at,
            detail="; ".join(f"{key}: {value}" for key, value in normalized.items() if "rate" in key or "reset" in key),
        )

    @staticmethod
    def from_report(
        provider: str,
        text: str,
        *,
        model: str = "",
        at: float | None = None,
    ) -> QuotaObservation:
        at = at or now()
        facts = parse_usage_report(text, at=at)
        return QuotaObservation(
            provider=provider,
            model=model,
            pct_remaining=(facts.get("pct_remaining") * 1.0 if "pct_remaining" in facts else None),
            resets_at=facts.get("resets_at"),
            window=facts.get("window", QuotaWindow.UNKNOWN),
            banked_resets=facts.get("banked_resets", 0),
            signal=(
                ExhaustionSignal.VENDOR_EXHAUSTED
                if facts.get("exhausted") else ExhaustionSignal.OK
            ),
            source="cli_report",
            observed_at=at,
            detail=text.strip()[:300],
        )

    @staticmethod
    def apply(tracker: QuotaTracker, observation: QuotaObservation):
        """Apply one normalized observation using the tracker’s typed rules."""
        if observation.signal is ExhaustionSignal.RATE_LIMITED:
            return tracker.record_signal(
                observation.provider,
                observation.signal,
                model=observation.model,
                retry_after=observation.retry_after,
                at=observation.observed_at,
                detail=observation.detail,
            )
        if observation.signal is ExhaustionSignal.VENDOR_EXHAUSTED:
            return tracker.record_signal(
                observation.provider,
                observation.signal,
                model=observation.model,
                resets_at=observation.resets_at,
                window=observation.window,
                banked_resets=observation.banked_resets,
                at=observation.observed_at,
                detail=observation.detail,
            )
        if any(
            value is not None
            for value in (observation.pct_remaining, observation.resets_at)
        ):
            return tracker.record_facts(
                observation.provider,
                model=observation.model,
                pct_remaining=observation.pct_remaining,
                resets_at=observation.resets_at,
                window=observation.window,
                banked_resets=observation.banked_resets,
                source=QuotaSource.REPORTED,
                observed_at=observation.observed_at,
                detail=observation.detail,
            )
        return tracker.record_ok(
            observation.provider, model=observation.model, at=observation.observed_at
        )


__all__ = ["QuotaIngestor", "QuotaObservation"]
