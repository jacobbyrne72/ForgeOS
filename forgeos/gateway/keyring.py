"""Per-provider key ring: many keys behind one provider name, one dead one does
not take the provider down.

`Provider.env_key` (in `forgeos/settings.py`) names exactly one env var. That is
fine for a single account, but a user with several free-tier accounts, or a
personal key and a work key, has more than one credential for the same
provider -- and today, the moment the one key on file hits its quota, the
whole provider goes dark even though a second key would have served the call.

Each key gets tracked independently:

    HEALTHY       -- usable right now.
    RATE_LIMITED  -- busy, not broken. Comes back on its own once a reset
                     time passes (provider-given, or exponential backoff).
    EXHAUSTED     -- out of credit (HTTP 402). Also self-heals, but on a much
                     longer clock -- a balance does not refill in seconds the
                     way a rate-limit window does, and treating the two the
                     same means hammering a still-empty account every 30s.
    INVALID       -- HTTP 401/403. The credential itself is wrong or revoked.
                     Quarantined PERMANENTLY for the process lifetime: no
                     amount of waiting turns a rejected key into an accepted
                     one, so retrying it is pure waste, forever.

Selection is a fixed scan of keys in DISCOVERY order, returning the first one
that is available right now (see `KeyRing.select`). No randomness anywhere --
picking `random.choice` among healthy keys would make a cost regression
unattributable, because the same state could route to a different key on the
next call for no visible reason. Deterministic order also means "rotate to
the next healthy key on failure" falls out for free: quarantining key A just
removes it from the scan, so the same fixed scan now stops at key B.

If every key for a provider is quarantined, `KeyRing.usable` is False and
`KeyRing.select`/`reveal` return None -- the provider is reported unusable,
not silently retried on a key everyone already knows is dead. That decision
belongs to the caller (typically the same place `HealthTracker.healthy` /
`DeadModelStore.is_dead` already gate a call), so this module never raises to
signal it.

SECRET HYGIENE -- read this before touching this file:
Every field on `KeyRecord`, and everything `snapshot()` / `status()` / `KeyRing.
__repr__` can produce, is a NAME and a STATE. Never a VALUE. The one place a
credential's actual value is reachable is `KeyRing.reveal`, and its docstring
says what it says for a reason: the return value must go straight into a
request header and nowhere else -- not a log line, not an exception message,
not a dict that later gets `json.dumps`-ed for a report. This mirrors
`forgeos/core/probe.py`'s rule ("Credential values are never read, logged,
printed, or returned. ... what comes back here is a status, never a secret")
at the one point in this module where that rule is load-bearing instead of
free.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from enum import Enum

from pydantic import BaseModel

from ..settings import Provider

# A misconfigured environment (or a typo'd base name that happens to collide
# with an unrelated _2/_3 var elsewhere) must not scan forever looking for
# the next suffix. 20 keys for one provider is already an implausible amount
# of quota-splitting.
MAX_KEY_SUFFIX = 20

# Rate limits are windowed and expected to clear within minutes; exhaustion
# (out of credit) is not expected to clear within a session at all, only
# eventually (a plan renewal, a top-up). The two backoff ladders are kept far
# apart so an exhausted key is not re-probed on the same cheap cadence as a
# merely-busy one.
DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 30.0
MAX_RATE_LIMIT_BACKOFF_SECONDS = 900.0  # 15 min
DEFAULT_EXHAUSTED_BACKOFF_SECONDS = 3600.0  # 1 hour
MAX_EXHAUSTED_BACKOFF_SECONDS = 86_400.0  # 24 hours


class KeyState(str, Enum):
    HEALTHY = "healthy"
    RATE_LIMITED = "rate_limited"
    EXHAUSTED = "exhausted"
    INVALID = "invalid"


# HTTP status -> what it means for the KEY specifically, as opposed to the
# transport or the model. Anything not in this table (5xx, network errors,
# 404/400 on the model) is deliberately left unclassified: it says nothing
# about whether THIS credential is good, so `KeyRing.record_failure` leaves
# the key's state untouched for it. Conflating "the key is bad" with "the
# call failed for some other reason" is the exact mistake `dead_models.py`
# already documents fixing once for models vs. transports; the same split
# applies here for keys vs. everything else that can go wrong in a request.
_STATUS_TO_STATE: dict[int, KeyState] = {
    401: KeyState.INVALID,
    403: KeyState.INVALID,
    429: KeyState.RATE_LIMITED,
    402: KeyState.EXHAUSTED,
}


class KeyRecord(BaseModel):
    """One key slot's public state. `name` is an env-var name (or a synthetic
    label for a comma-split fragment, see `_discover`) -- never a secret."""

    name: str
    state: KeyState = KeyState.HEALTHY
    # Epoch seconds this key becomes eligible again. None means either "never
    # quarantined" (state is HEALTHY) or "quarantined permanently" (state is
    # INVALID) -- the two are told apart by `state`, not by this field alone.
    quarantined_until: float | None = None
    reason: str = ""
    consecutive_failures: int = 0
    # The backoff window used for the CURRENT quarantine, so a repeat failure
    # can double it instead of resetting to the base every time. Cleared on
    # `record_success` and on an explicit `retry_after` from the provider,
    # since a provider-given number is not something to escalate away from.
    backoff_seconds: float = 0.0
    checked_at: float = 0.0

    def available(self, now: float) -> bool:
        """Whether this key may be tried right now."""
        if self.state is KeyState.HEALTHY:
            return True
        if self.state is KeyState.INVALID:
            return False  # terminal -- see module docstring
        return self.quarantined_until is not None and now >= self.quarantined_until


def _discover(base_env_key: str, environ: Mapping[str, str]) -> list[tuple[str, str]]:
    """Ordered (name, value) pairs for every key configured under `base_env_key`.

    Two conventions, both additive:

    - `base_env_key` itself may hold ONE key, or a COMMA-SEPARATED LIST of
      keys. A list has no individual env-var name to point back at per
      fragment, so each fragment gets a synthetic label (`BASE[0]`,
      `BASE[1]`, ...) -- still just a name, never a value.
    - `BASE_2`, `BASE_3`, ... are separate env vars, one key each, scanned
      contiguously starting at 2 and stopping at the first missing or blank
      one. Contiguous-from-2 (rather than "scan all N and skip gaps") means
      removing `BASE_3` to retire a key can't leave `BASE_4` silently still
      active with a hole in between -- the numbering is a priority order,
      not just a set of names.
    """
    pairs: list[tuple[str, str]] = []

    raw = (environ.get(base_env_key) or "").strip()
    if raw:
        fragments = [f.strip() for f in raw.split(",") if f.strip()]
        if len(fragments) > 1:
            pairs.extend((f"{base_env_key}[{i}]", frag) for i, frag in enumerate(fragments))
        else:
            pairs.append((base_env_key, fragments[0]))

    suffix = 2
    while suffix <= MAX_KEY_SUFFIX:
        name = f"{base_env_key}_{suffix}"
        value = (environ.get(name) or "").strip()
        if not value:
            break
        pairs.append((name, value))
        suffix += 1

    return pairs


class KeyRing:
    """A provider's pool of keys, with independent health and quarantine per key.

    `environ` and `clock` are injectable for the same reason `HealthTracker`'s
    and `DeadModelStore`'s are: discovery from a real environ is one call, but
    a quarantine window has to be testable by advancing a fake clock instead
    of sleeping in a test.
    """

    def __init__(
        self,
        provider: str,
        base_env_key: str,
        *,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.provider = provider
        self.base_env_key = base_env_key
        self._clock = clock
        discovered = _discover(base_env_key, environ if environ is not None else os.environ)
        # Discovery order IS priority order -- see `select`. Kept as a plain
        # list (not a dict's insertion order relied on implicitly) so the
        # order is a visible, deliberate property of this object.
        self._order: list[str] = [name for name, _ in discovered]
        self._values: dict[str, str] = dict(discovered)  # PRIVATE. See module docstring.
        self._records: dict[str, KeyRecord] = {name: KeyRecord(name=name) for name in self._order}

    @classmethod
    def for_provider(
        cls,
        provider: Provider,
        *,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> KeyRing:
        """Build a `KeyRing` from a `settings.Provider`'s own `env_key` -- the
        single-var name `Provider.authenticated`/`installed` already treat as
        the source of truth for this provider's credential, used here as the
        BASE var the `BASE`, `BASE_2`, `BASE_3`, ... convention discovers from.
        """
        if not provider.env_key:
            raise ValueError(f"provider {provider.name!r} has no env_key to discover keys from")
        return cls(provider.name, provider.env_key, environ=environ, clock=clock)

    def __len__(self) -> int:
        return len(self._order)

    def __repr__(self) -> str:
        # Names and states only -- this is exactly the kind of object that
        # ends up in a log line by accident, so it must be safe by construction.
        return f"KeyRing({self.provider!r}, keys={self.status()!r})"

    # --------------------------------------------------------------- names

    def names(self) -> list[str]:
        """Env-var names (or synthetic labels) in priority order. Never values."""
        return list(self._order)

    def status(self) -> dict[str, KeyState]:
        """name -> current state. Safe to log, print, or serialise."""
        return {name: record.state for name, record in self._records.items()}

    def snapshot(self) -> list[KeyRecord]:
        """Full per-key state in priority order. Safe to log, print, or serialise
        -- `KeyRecord` has no field that can hold a credential value."""
        return [self._records[name] for name in self._order]

    # ------------------------------------------------------------- recording

    def record_success(self, name: str) -> None:
        """A call using `name` succeeded: clear any quarantine, reset backoff."""
        record = self._records[name]
        record.state = KeyState.HEALTHY
        record.quarantined_until = None
        record.reason = ""
        record.consecutive_failures = 0
        record.backoff_seconds = 0.0
        record.checked_at = self._clock()

    def record_failure(
        self,
        name: str,
        status_code: int,
        *,
        retry_after_seconds: float | None = None,
        reason: str = "",
    ) -> KeyState:
        """A call using `name` failed with `status_code`. Returns the key's
        state afterward (unchanged if `status_code` isn't one this module
        classifies as a key problem -- see `_STATUS_TO_STATE`).

        `retry_after_seconds`, when the provider supplied one (a `Retry-After`
        header, a body-embedded reset hint), is honoured directly and is NOT
        escalated on a repeat failure -- the provider is telling us exactly
        when it resets, and doubling a number the provider already gave us
        would just make the wait longer than the provider asked for. Without
        one, this module picks its own window and doubles it (capped) on each
        consecutive failure of the same kind, the same escalation shape
        `DeadModelStore.report_probe` uses for models.
        """
        state = _STATUS_TO_STATE.get(status_code)
        if state is None:
            return self._records[name].state  # not evidence about this key

        now = self._clock()
        record = self._records[name]
        record.checked_at = now
        record.consecutive_failures += 1
        record.state = state
        record.reason = reason or f"HTTP {status_code}"

        if state is KeyState.INVALID:
            # Permanent for the process lifetime. No self-heal path exists on
            # purpose -- see module docstring.
            record.quarantined_until = None
            record.backoff_seconds = 0.0
            return state

        base, cap = (
            (DEFAULT_RATE_LIMIT_BACKOFF_SECONDS, MAX_RATE_LIMIT_BACKOFF_SECONDS)
            if state is KeyState.RATE_LIMITED
            else (DEFAULT_EXHAUSTED_BACKOFF_SECONDS, MAX_EXHAUSTED_BACKOFF_SECONDS)
        )
        if retry_after_seconds is not None:
            window = max(0.0, retry_after_seconds)
            record.backoff_seconds = window  # provider-given: not an escalation base
        else:
            prior = record.backoff_seconds if record.backoff_seconds > 0 else base
            window = min(prior * 2 if record.backoff_seconds > 0 else base, cap)
            record.backoff_seconds = window
        record.quarantined_until = now + window
        return state

    # ---------------------------------------------------------------- select

    def select(self) -> str | None:
        """The key to use right now, deterministically: the first name (in
        fixed discovery order) whose record is available. None if every key
        is quarantined -- the caller must treat that as "this provider is
        unusable right now", not retry a key already known to be down.
        """
        now = self._clock()
        for name in self._order:
            if self._records[name].available(now):
                return name
        return None

    @property
    def usable(self) -> bool:
        """Whether ANY key can serve a call right now. Mirrors `Provider.usable`
        in `forgeos/settings.py` -- a gate, checked before routing, not a hint."""
        return self.select() is not None

    def reveal(self, name: str) -> str:
        """The credential VALUE for `name`.

        This is the only method in this module that returns a secret. The
        return value must be used immediately to build a request header (the
        way `HttpTransport.complete` in `client.py` uses `os.environ.get(...)`
        today) and then discarded -- never logged, printed, stored on an
        object that outlives the call, or included in an exception message.
        """
        return self._values[name]

    def select_and_reveal(self) -> tuple[str, str] | None:
        """Convenience: `(name, value)` for the key `select()` would choose,
        or None if none is available. Same handling rule as `reveal` applies
        to the value half of the tuple -- see its docstring.
        """
        name = self.select()
        if name is None:
            return None
        return name, self.reveal(name)


__all__ = [
    "DEFAULT_EXHAUSTED_BACKOFF_SECONDS",
    "DEFAULT_RATE_LIMIT_BACKOFF_SECONDS",
    "MAX_EXHAUSTED_BACKOFF_SECONDS",
    "MAX_KEY_SUFFIX",
    "MAX_RATE_LIMIT_BACKOFF_SECONDS",
    "KeyRecord",
    "KeyRing",
    "KeyState",
]
