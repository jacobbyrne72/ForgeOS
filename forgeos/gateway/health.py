"""Per-provider health, so a bad or rate-limited provider is skipped instead of
hammered.

Retry storms — hitting the same 429 in a tight loop — are one of the biggest
token-waste mechanisms this harness exists to eliminate. A rate limit records
an `available_at` timestamp rather than a boolean flag, so the provider comes
back automatically once the window passes instead of needing a manual reset
or, worse, being retried every call in the meantime.

A plain failure (not a rate limit) has no automatic recovery: it stays
unhealthy until an explicit `record_success` proves otherwise. Only rate
limits are known to be time-bounded; an ordinary failure might not be.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, Field

# A saturation reading older than this is stale, not evidence -- a provider's
# quota window resets (RPM/TPM are windowed), so a number from several
# minutes ago says nothing about right now. Stale reads back off to "unknown"
# via `saturation()` returning None, the same as never having observed one.
_SATURATION_TTL_SECONDS = 120.0


class ProviderHealth(BaseModel):
    """One provider's last-observed state. A snapshot, not a history."""

    name: str
    reachable: bool = True
    latency_ms: float = 0.0
    last_error: str = ""
    rate_limited: bool = False
    checked_at: float = Field(default_factory=time.time)


class HealthTracker:
    """Tracks reachability and rate-limit windows per provider name.

    `clock` is injectable so a rate-limit window can be tested by advancing a
    fake clock instead of sleeping — sleeping in a test is how tests get
    skipped for being slow, and a governor invariant like this must never be
    the one that gets skipped.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._health: dict[str, ProviderHealth] = {}
        self._available_at: dict[str, float] = {}
        # name -> (saturation 0..1, observed_at). Separate from `_health`
        # because saturation is a proactive, continuous signal present on
        # successful responses too, not a reachability verdict -- a provider
        # can be perfectly healthy and 95% saturated at the same time.
        self._saturation: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------- recording

    def record_success(self, name: str, latency_ms: float = 0.0) -> None:
        self._available_at.pop(name, None)
        self._health[name] = ProviderHealth(
            name=name,
            reachable=True,
            latency_ms=latency_ms,
            last_error="",
            rate_limited=False,
            checked_at=self._clock(),
        )

    def record_failure(self, name: str, error: str) -> None:
        self._health[name] = ProviderHealth(
            name=name,
            reachable=False,
            latency_ms=0.0,
            last_error=error,
            rate_limited=False,
            checked_at=self._clock(),
        )

    def record_rate_limit(self, name: str, retry_after_seconds: float = 30.0) -> None:
        checked = self._clock()
        self._available_at[name] = checked + retry_after_seconds
        self._health[name] = ProviderHealth(
            name=name,
            reachable=False,
            latency_ms=0.0,
            last_error="rate limited",
            rate_limited=True,
            checked_at=checked,
        )

    def record_saturation(self, name: str, saturation: float | None) -> None:
        """Store a proactive 0..1 rate-limit-header saturation reading for `name`.

        `None` means the response carried no usable rate-limit headers at
        all, and is a no-op -- recorded as "still don't know", never folded
        in as a worst-case value. A provider that simply never publishes
        these headers must not be sorted into permanent last place by
        `saturation()` defaulting to "fully saturated" instead of "unknown".
        """
        if saturation is None:
            return
        self._saturation[name] = (min(1.0, max(0.0, saturation)), self._clock())

    # ---------------------------------------------------------------- query

    def saturation(self, name: str) -> float | None:
        """The freshest saturation reading for `name`.

        None when never observed, or when the last observation is older
        than `_SATURATION_TTL_SECONDS` -- a stale reading is worse than no
        reading, because it looks like evidence.
        """
        entry = self._saturation.get(name)
        if entry is None:
            return None
        value, observed_at = entry
        if self._clock() - observed_at > _SATURATION_TTL_SECONDS:
            return None
        return value

    def available_at(self, name: str) -> float | None:
        """When a rate-limited provider becomes eligible again. None if it isn't blocked."""
        return self._available_at.get(name)

    def snapshot(self, name: str) -> ProviderHealth | None:
        return self._health.get(name)

    def healthy(self, name: str) -> bool:
        """Whether `name` may be tried right now.

        A provider still inside its rate-limit window is never healthy,
        regardless of anything else recorded about it. Once the window has
        passed, a rate-limited provider is healthy again automatically; a
        plain failure is not — it needs a real `record_success`.
        """
        until = self._available_at.get(name)
        if until is not None and self._clock() < until:
            return False
        record = self._health.get(name)
        if record is None:
            return True  # never observed: innocent until proven otherwise
        return record.reachable or record.rate_limited

    def pick_healthy(self, candidates: list[str]) -> list[str]:
        """Healthy candidates only, ordered best first: measured-healthy
        before never-tried, least rate-limit-saturated before most, lowest
        measured latency before highest.

        Providers with no measurement yet sort after measured-healthy ones —
        a provider with a real good track record should not be bumped by one
        nobody has tried yet. Saturation is the tiebreaker inside that: a
        nearly-exhausted provider is pushed later so it gets tried only
        after fresher options, without ever being excluded outright the way
        an actual rate limit excludes one -- this is deprioritisation before
        the 429, not a health verdict. A provider with no saturation reading
        (never observed, or missing headers) sorts as 0.0 -- the best
        possible value -- so silence is never punished into permanent last
        place.
        """
        healthy = [c for c in candidates if self.healthy(c)]

        def key(name: str) -> tuple[int, float, float]:
            record = self._health.get(name)
            bucket = 0 if record is not None else 1
            latency = record.latency_ms if record is not None else 0.0
            observed = self.saturation(name)
            saturation = observed if observed is not None else 0.0
            return (bucket, saturation, latency)

        return sorted(healthy, key=key)


__all__ = ["HealthTracker", "ProviderHealth"]
