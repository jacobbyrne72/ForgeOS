"""Subscription quota tracking — the scarce resource that is not dollars.

A subscription is not billed per token; it is capped per window. Codex enforces two
simultaneously (a rolling 5-hour window and a 7-day weekly one), and Claude is
similarly window-capped. So the currency to conserve is *quota headroom*, and the
most valuable thing this module does is know when a window reopens so parked work
resumes the moment it can rather than idling or paying API rates in the meantime.

Three deliberate design positions:

1. **Reset times are read, never guessed.** They come from what the provider itself
   reports (its `/usage` panel, or the reset time in its rate-limit message). When
   nothing is reported, the state is marked `estimated` and treated conservatively —
   an invented reset time causes a thundering probe against a still-closed window.

2. **A banked reset is not spent automatically by default.** Codex ≥ v0.135 can bank
   a rate-limit reset, but it is a finite, non-refundable account benefit and
   applying it goes through a vendor confirmation screen. Consuming one unattended
   could burn it on a trivial 3am task. `auto_consume_banked` exists for operators
   who want that, gated by a value threshold — but the default is to surface the
   option and let a human decide.

3. **Rotation branches on the typed signal, not on "it failed."** Three independent
   shipped tools (Claudexor, teamclaude, claude-swap) converged on the same rule:
   a confirmed vendor exhaustion is the only signal that justifies looking for a
   different seat (`should_rotate`); an ordinary rate limit just pauses that exact
   seat for its `retry_after` window, because the seat itself is fine; a network
   error touches nothing, because it says nothing about quota at all — recording it
   would make a flaky connection look like a dead seat. Quota is tracked per
   (provider, model) so an exhausted Fable window on one account doesn't block
   Opus/Sonnet on that same account.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field

from ..contracts import now

# Probe backoff. Polling a closed window costs a real failed request each time and
# tells you nothing new, so checks get sparser the longer the wait, and cluster
# around the moment a reported reset is due.
PROBE_BACKOFF_SECONDS = (60, 300, 900, 1800, 3600)
# How soon after a reported reset to confirm it actually reopened.
CONFIRM_AFTER_RESET_SECONDS = 30


class QuotaWindow(str, Enum):
    FIVE_HOUR = "five_hour"   # rolling
    WEEKLY = "weekly"         # 7-day cycle
    UNKNOWN = "unknown"


class QuotaSource(str, Enum):
    REPORTED = "reported"    # the provider told us — a fact
    ESTIMATED = "estimated"  # inferred — must never be presented as a fact


class ExhaustionSignal(str, Enum):
    """What a worker call actually told us. The rotation rule branches on this,
    not on a bare success/failure, because conflating them is the exact bug that
    burns a seat rotation on a problem rotating won't fix."""

    OK = "ok"
    VENDOR_EXHAUSTED = "vendor_exhausted"  # quota gone until reset — rotate
    RATE_LIMITED = "rate_limited"          # ordinary 429 — pause this seat, don't rotate
    NETWORK_ERROR = "network_error"        # transient — neither rotate nor pause


def should_rotate(signal: ExhaustionSignal) -> bool:
    """Whether this signal alone justifies looking for a different seat.

    Only a confirmed vendor exhaustion does. A rate limit self-heals on the same
    seat it hit; a network error says nothing about the seat's quota at all.
    """
    return signal is ExhaustionSignal.VENDOR_EXHAUSTED


class QuotaState(BaseModel):
    provider: str
    model: str = ""  # "" = provider-wide, the key every call site used before
    window: QuotaWindow = QuotaWindow.UNKNOWN
    pct_remaining: float | None = None
    resets_at: float | None = None
    banked_resets: int = 0
    exhausted: bool = False
    source: QuotaSource = QuotaSource.ESTIMATED
    observed_at: float = Field(default_factory=now)
    detail: str = ""

    @property
    def known(self) -> bool:
        return self.source is QuotaSource.REPORTED

    def available(self, at: float | None = None) -> bool:
        """Whether the window is open. Unknown state is treated as available.

        Defaulting to available is deliberate: refusing to work because we have not
        yet observed a quota report would make an unmeasured provider useless.
        """
        if not self.exhausted:
            return True
        if self.resets_at is None:
            return False
        return (at or now()) >= self.resets_at

    def seconds_until_reset(self, at: float | None = None) -> float | None:
        if self.resets_at is None:
            return None
        return max(0.0, self.resets_at - (at or now()))


class ParkedJob(BaseModel):
    """Work held until a provider's window reopens, rather than failed or downgraded."""

    job_id: str
    task_id: str | None = None
    provider: str
    parked_at: float = Field(default_factory=now)
    reason: str = ""
    value: float = 0.5  # 0..1; gates whether a banked reset is worth spending


class BankedResetOffer(BaseModel):
    """A consumable reset the operator may choose to spend. Never spent silently."""

    provider: str
    banked_available: int
    parked_jobs: int
    highest_parked_value: float
    seconds_until_natural_reset: float | None
    command: str
    recommendation: str


class SeatChoice(BaseModel):
    """What `QuotaTracker.select_seat` decided for one model across candidate seats.

    `ready=False` with `seat=None` is the honest "nothing is usable right now"
    answer — it never names a winner it can't back up. `wait_seconds` is filled in
    only when a real reset time is known for at least one blocked candidate.
    """

    model: str
    seat: str | None = None
    ready: bool = False
    wait_seconds: float | None = None


# Parsed from what the provider actually printed. NAMED groups on purpose — the
# first version of this branched on substrings of the regex source itself, which
# silently read "2h 15m" as 2 minutes because "hour" is not contiguous inside
# `h(?:ours?)?`. Named groups make the unit explicit instead of inferred.
_RESET_PATTERNS = (
    # "resets in 2h 15m" / "resets in 2 hours 15 minutes"
    re.compile(
        r"resets?\s+(?:in|after)\s+(?P<hours>\d+)\s*h(?:ours?|rs?)?"
        r"(?:\s*(?P<minutes>\d+)\s*m(?:in(?:utes?)?)?)?",
        re.I,
    ),
    # "resets in 45 minutes"
    re.compile(r"resets?\s+(?:in|after)\s+(?P<minutes>\d+)\s*m(?:in(?:utes?)?)?\b", re.I),
    # "Retry-After: 120" (seconds, per HTTP)
    re.compile(r"retry[- ]after[:\s]+(?P<seconds>\d+)", re.I),
)
_PCT_PATTERN = re.compile(r"(\d{1,3})\s*%\s*(?:remaining|left)", re.I)
_BANKED_PATTERN = re.compile(r"(\d+)\s+(?:banked|saved)\s+reset", re.I)


def parse_usage_report(text: str, *, at: float | None = None) -> dict:
    """Extract quota facts from a provider's own usage/rate-limit output.

    Returns only what the text actually stated. A field absent from the output stays
    absent rather than being filled with a plausible default — a fabricated reset
    time is worse than no reset time, because the scheduler would trust it.
    """
    at = at or now()
    out: dict = {}

    m = _PCT_PATTERN.search(text)
    if m:
        out["pct_remaining"] = float(m.group(1))

    m = _BANKED_PATTERN.search(text)
    if m:
        out["banked_resets"] = int(m.group(1))

    for pat in _RESET_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        found = m.groupdict()
        seconds = (
            int(found.get("hours") or 0) * 3600
            + int(found.get("minutes") or 0) * 60
            + int(found.get("seconds") or 0)
        )
        if seconds > 0:
            out["resets_at"] = at + seconds
            break

    low = text.lower()
    if any(s in low for s in ("usage limit", "rate limit", "quota exceeded", "429")):
        out["exhausted"] = True
    if "weekly" in low:
        out["window"] = QuotaWindow.WEEKLY
    elif "5-hour" in low or "five hour" in low or "5 hour" in low:
        out["window"] = QuotaWindow.FIVE_HOUR

    return out


class QuotaTracker:
    def __init__(
        self,
        *,
        auto_consume_banked: bool = False,
        banked_value_threshold: float = 0.8,
    ):
        # Off by default. A banked reset is finite and non-refundable; spending one
        # unattended on low-value work is a loss the operator cannot undo.
        self.auto_consume_banked = auto_consume_banked
        self.banked_value_threshold = banked_value_threshold
        self._states: dict[str, QuotaState] = {}
        self._parked: list[ParkedJob] = []
        self._probe_attempts: dict[str, int] = {}
        self.banked_consumed: list[tuple[str, float]] = []

    @staticmethod
    def _key(provider: str, model: str) -> str:
        # model="" collapses to the bare provider string — exactly the key every
        # call site used before per-model tracking existed, so nothing that never
        # passes `model` can observe a difference.
        return provider if not model else f"{provider}::{model}"

    # ------------------------------------------------------------- observe

    def record_report(
        self, provider: str, text: str, *, model: str = "", at: float | None = None
    ) -> QuotaState:
        """Ingest a provider's own usage output — the authoritative path."""
        at = at or now()
        facts = parse_usage_report(text, at=at)
        state = QuotaState(
            provider=provider,
            model=model,
            observed_at=at,
            # Reported only if the text actually carried a reset time or a percentage.
            source=(
                QuotaSource.REPORTED
                if ("resets_at" in facts or "pct_remaining" in facts)
                else QuotaSource.ESTIMATED
            ),
            detail=text.strip()[:300],
            **{k: v for k, v in facts.items() if k in QuotaState.model_fields},
        )
        key = self._key(provider, model)
        self._states[key] = state
        if not state.exhausted:
            self._probe_attempts.pop(key, None)
        return state

    def record_exhaustion(
        self,
        provider: str,
        *,
        model: str = "",
        resets_at: float | None = None,
        window: QuotaWindow = QuotaWindow.UNKNOWN,
        banked_resets: int = 0,
        at: float | None = None,
        detail: str = "",
    ) -> QuotaState:
        state = QuotaState(
            provider=provider,
            model=model,
            window=window,
            exhausted=True,
            resets_at=resets_at,
            banked_resets=banked_resets,
            pct_remaining=0.0,
            source=QuotaSource.REPORTED if resets_at is not None else QuotaSource.ESTIMATED,
            observed_at=at or now(),
            detail=detail,
        )
        self._states[self._key(provider, model)] = state
        return state

    def record_rate_limit(
        self,
        provider: str,
        retry_after: float,
        *,
        model: str = "",
        at: float | None = None,
        detail: str = "",
    ) -> QuotaState:
        """An ordinary 429: pause this exact (provider, model) seat for
        `retry_after` seconds. This is never a rotation signal — the seat itself
        is fine, it just needs to wait out its own window — so `should_rotate`
        returns False for `RATE_LIMITED` even though this makes `available()`
        False just like a vendor exhaustion would.
        """
        at = at or now()
        state = QuotaState(
            provider=provider,
            model=model,
            exhausted=True,
            resets_at=at + retry_after,
            pct_remaining=0.0,
            source=QuotaSource.REPORTED,
            observed_at=at,
            detail=detail or f"rate limited; retry after {retry_after:.0f}s",
        )
        self._states[self._key(provider, model)] = state
        return state

    def record_ok(self, provider: str, *, model: str = "", at: float | None = None) -> QuotaState:
        """A call succeeded: clear whatever exhaustion or pause this (provider,
        model) seat was carrying. Does not fabricate a headroom figure — absence
        of a usage report still means we don't know pct_remaining, only that the
        seat answered.
        """
        at = at or now()
        state = QuotaState(provider=provider, model=model, observed_at=at)
        key = self._key(provider, model)
        self._states[key] = state
        self._probe_attempts.pop(key, None)
        return state

    def record_signal(
        self,
        provider: str,
        signal: ExhaustionSignal,
        *,
        model: str = "",
        resets_at: float | None = None,
        retry_after: float | None = None,
        window: QuotaWindow = QuotaWindow.UNKNOWN,
        banked_resets: int = 0,
        at: float | None = None,
        detail: str = "",
    ) -> QuotaState | None:
        """Apply one typed call outcome to (provider, model) — the single entry
        point the three-way rotation rule lives behind (see the module docstring).
        """
        at = at or now()
        if signal is ExhaustionSignal.NETWORK_ERROR:
            # Says nothing about quota. Recording it would make a flaky
            # connection look like a dead seat, so this is a deliberate no-op.
            return self.state(provider, model=model)
        if signal is ExhaustionSignal.RATE_LIMITED:
            if retry_after is None:
                raise ValueError("RATE_LIMITED requires retry_after")
            return self.record_rate_limit(provider, retry_after, model=model, at=at, detail=detail)
        if signal is ExhaustionSignal.VENDOR_EXHAUSTED:
            return self.record_exhaustion(
                provider,
                model=model,
                resets_at=resets_at,
                window=window,
                banked_resets=banked_resets,
                at=at,
                detail=detail,
            )
        return self.record_ok(provider, model=model, at=at)

    def state(self, provider: str, *, model: str = "") -> QuotaState | None:
        return self._states.get(self._key(provider, model))

    def available(self, provider: str, at: float | None = None, *, model: str = "") -> bool:
        st = self._states.get(self._key(provider, model))
        return True if st is None else st.available(at)

    def available_providers(self, candidates: list[str], at: float | None = None) -> list[str]:
        return [c for c in candidates if self.available(c, at)]

    # --------------------------------------------------------------- probe

    def next_probe_at(self, provider: str, at: float | None = None, *, model: str = "") -> float | None:
        """When to next check a closed window.

        Clusters just after a reported reset rather than polling throughout. Each
        probe against a closed window is a real request that fails, so polling
        "constantly" spends the resource it is trying to protect.
        """
        at = at or now()
        key = self._key(provider, model)
        st = self._states.get(key)
        if st is None or not st.exhausted:
            return None

        if st.resets_at is not None:
            if at >= st.resets_at:
                return at  # due now
            return st.resets_at + CONFIRM_AFTER_RESET_SECONDS

        n = self._probe_attempts.get(key, 0)
        delay = PROBE_BACKOFF_SECONDS[min(n, len(PROBE_BACKOFF_SECONDS) - 1)]
        return at + delay

    def record_probe(self, provider: str, *, model: str = "") -> int:
        key = self._key(provider, model)
        self._probe_attempts[key] = self._probe_attempts.get(key, 0) + 1
        return self._probe_attempts[key]

    # ------------------------------------------------------------ rotation

    def select_seat(self, model: str, candidates: list[str], *, at: float | None = None) -> SeatChoice:
        """Choose which seat should take the next call for `model`.

        An available seat always wins outright. Among several, prefer the one
        whose current window closes soonest — spending a seat that is about to
        reset anyway preserves the other seats' remaining headroom for later.
        If every candidate is currently blocked (vendor-exhausted or
        rate-limited — `select_seat` doesn't need to know which), this does not
        invent a winner: it reports that nothing is ready and, only when a real
        reset time is known for at least one candidate, how long the actual
        wait is.
        """
        at = at or now()
        if not candidates:
            return SeatChoice(model=model)

        states = {c: self._states.get(self._key(c, model)) for c in candidates}
        available = [c for c in candidates if states[c] is None or states[c].available(at)]

        if available:
            def soonest_reset(c: str) -> tuple[bool, float]:
                st = states[c]
                r = st.resets_at if st else None
                return (r is None, r if r is not None else float("inf"))

            return SeatChoice(model=model, seat=min(available, key=soonest_reset), ready=True)

        waits = [
            st.seconds_until_reset(at)
            for st in states.values()
            if st is not None and st.seconds_until_reset(at) is not None
        ]
        return SeatChoice(model=model, wait_seconds=min(waits) if waits else None)

    # ---------------------------------------------------------------- park

    def park(self, job_id: str, provider: str, *, task_id: str | None = None,
             value: float = 0.5, reason: str = "") -> ParkedJob:
        p = ParkedJob(job_id=job_id, task_id=task_id, provider=provider,
                      value=value, reason=reason)
        self._parked.append(p)
        return p

    def parked_for(self, provider: str) -> list[ParkedJob]:
        return [p for p in self._parked if p.provider == provider]

    def resume_ready(self, at: float | None = None) -> list[ParkedJob]:
        """Parked work whose provider window has reopened. Removed from the park."""
        at = at or now()
        ready = [p for p in self._parked if self.available(p.provider, at)]
        if ready:
            ids = {(p.job_id, p.task_id) for p in ready}
            self._parked = [p for p in self._parked
                            if (p.job_id, p.task_id) not in ids]
        return ready

    # -------------------------------------------------------------- banked

    def banked_offer(self, provider: str, at: float | None = None) -> BankedResetOffer | None:
        """Surface a banked reset as a decision, with the context to make it.

        Returned when one exists and work is waiting — the operator sees how long the
        natural reset is and how valuable the parked work is, rather than a bare
        'use a reset?' prompt.
        """
        st = self._states.get(provider)
        if st is None or not st.exhausted or st.banked_resets <= 0:
            return None

        parked = self.parked_for(provider)
        top = max((p.value for p in parked), default=0.0)
        wait = st.seconds_until_reset(at)

        if not parked:
            rec = "hold — nothing is waiting on this provider"
        elif wait is not None and wait < 900:
            rec = f"hold — natural reset is only {int(wait // 60)}m away"
        elif top >= self.banked_value_threshold:
            rec = f"spend — {len(parked)} job(s) waiting, top value {top:.2f}"
        else:
            rec = f"hold — parked work value {top:.2f} is below the {self.banked_value_threshold:.2f} threshold"

        return BankedResetOffer(
            provider=provider,
            banked_available=st.banked_resets,
            parked_jobs=len(parked),
            highest_parked_value=top,
            seconds_until_natural_reset=wait,
            # The vendor flow owns the actual confirmation screen.
            command=f"{provider} → /usage → apply banked reset",
            recommendation=rec,
        )

    def should_auto_consume(self, provider: str, at: float | None = None) -> bool:
        """Whether policy permits spending a banked reset without asking."""
        if not self.auto_consume_banked:
            return False
        offer = self.banked_offer(provider, at)
        if offer is None or offer.parked_jobs == 0:
            return False
        if offer.highest_parked_value < self.banked_value_threshold:
            return False
        # If the window reopens shortly anyway, spending a finite reset is waste.
        if offer.seconds_until_natural_reset is not None and offer.seconds_until_natural_reset < 900:
            return False
        return True

    def mark_banked_consumed(self, provider: str, *, at: float | None = None) -> QuotaState | None:
        """Record that a reset was applied — by the operator or by policy.

        Recorded either way so the dashboard can show a finite benefit being spent.
        """
        st = self._states.get(provider)
        if st is None or st.banked_resets <= 0:
            return None
        at = at or now()
        self.banked_consumed.append((provider, at))
        refreshed = QuotaState(
            provider=provider,
            window=st.window,
            exhausted=False,
            pct_remaining=100.0,
            banked_resets=st.banked_resets - 1,
            source=QuotaSource.ESTIMATED,  # assumed until /usage confirms it
            observed_at=at,
            detail="banked reset applied; awaiting /usage confirmation",
        )
        self._states[provider] = refreshed
        self._probe_attempts.pop(provider, None)
        return refreshed


__all__ = [
    "BankedResetOffer",
    "CONFIRM_AFTER_RESET_SECONDS",
    "PROBE_BACKOFF_SECONDS",
    "ExhaustionSignal",
    "ParkedJob",
    "QuotaSource",
    "QuotaState",
    "QuotaTracker",
    "QuotaWindow",
    "SeatChoice",
    "parse_usage_report",
    "should_rotate",
]
