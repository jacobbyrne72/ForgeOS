"""Per-worker circuit breaker: trip/stay-tripped/auto-recover/quarantine.

A worker can fail for two fundamentally different reasons, and treating them
the same is a bug in both directions:

- The **provider** is having a bad moment (429/503/timeout). This is
  transient and external -- the worker itself is fine, and the right move is
  to back off for a while and probe again, because it will very likely
  recover on its own.
- The worker's **local** setup is broken (missing binary, invalid auth, bad
  config file). No amount of waiting fixes this -- the next call fails
  identically to the last one, so a timer-based cooldown just burns calls
  re-discovering the same broken environment. The only fix is a human (or an
  automated repair step) actually changing the local state, so the breaker
  must stay tripped until something explicitly says "go check now"
  (`clear_quarantine`), not "enough time passed".

Rather than invent a parallel classification for this, the breaker reuses
`contracts.FailureClass` -- the taxonomy the rest of the harness already
uses to decide "escalate the model" vs "retry same tier" vs "ask a human"
(see `contracts.FailureClass`'s own docstring). Of its six members, exactly
one -- `ENVIRONMENT` -- means "this worker's own machine is broken"; that is
the LOCAL bucket that routes to QUARANTINED. Everything else (`TRANSIENT`,
and also `CONTEXT`/`MODEL`/`SPECIFICATION`/`POLICY`, none of which assert
anything about the worker's local health) keeps the original
trip-after-N-failures / exponential-backoff / HALF_OPEN-probe behavior,
because none of them is proven to be a permanent local defect that repeating
the call would just rediscover. Callers that don't classify at all
(`failure_class=None`, the default) get that same UPSTREAM-style handling --
this is what keeps every pre-existing caller of `record_failure(worker_id)`
working unchanged.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock

from .contracts import FailureClass


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    QUARANTINED = "quarantined"


# The one FailureClass that means "the worker's local setup is broken", not
# "something upstream/ambiguous happened". Kept as a set (rather than one
# equality check inline) so the LOCAL/UPSTREAM line is a single readable
# statement a reviewer can audit against `contracts.FailureClass`'s docstring,
# instead of logic implicit in an if/else.
_LOCAL_FAILURE_CLASSES = frozenset({FailureClass.ENVIRONMENT})


@dataclass
class BreakerRecord:
    worker_id: str
    total_calls: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state: BreakerState = BreakerState.CLOSED
    cooldown_until: float = 0.0
    circuit_opened_at: float = 0.0
    # How many consecutive times OPEN has been (re-)entered since the last
    # full CLOSE. Drives exponential backoff: a worker that keeps failing its
    # probe gets a longer and longer timeout instead of being re-tested every
    # `cooldown_seconds` forever.
    open_count: int = 0
    # Probe calls currently let through while HALF_OPEN and not yet resolved
    # by a record_success/record_failure. This is what makes the allowance a
    # real concurrency limit rather than just a naming convention -- without
    # it, two threads racing `is_available()` would both read HALF_OPEN and
    # both get waved through.
    half_open_probes_in_flight: int = 0
    last_failure_class: FailureClass | None = None
    quarantine_reason: str = ""
    quarantined_at: float = 0.0

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failures / self.total_calls

    @property
    def is_open(self) -> bool:
        return self.state is BreakerState.OPEN

    @property
    def is_quarantined(self) -> bool:
        return self.state is BreakerState.QUARANTINED


class CircuitBreaker:
    DEFAULT_FAILURE_THRESHOLD = 3
    DEFAULT_COOLDOWN_SECONDS = 60.0
    DEFAULT_MAX_COOLDOWN_SECONDS = 600.0
    DEFAULT_HALF_OPEN_ALLOWANCE = 1

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        max_cooldown_seconds: float = DEFAULT_MAX_COOLDOWN_SECONDS,
        half_open_allowance: int = DEFAULT_HALF_OPEN_ALLOWANCE,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_cooldown_seconds = max_cooldown_seconds
        self.half_open_allowance = half_open_allowance
        self._records: dict[str, BreakerRecord] = {}
        self._lock = Lock()

    def _next_cooldown(self, open_count: int) -> float:
        """Exponential backoff, capped. `open_count` is 1 on the first trip,
        2 on the trip right after a failed probe, and so on -- each failed
        probe means the outage is lasting longer than last guessed, so the
        next guess should too, up to `max_cooldown_seconds` so a chronically
        bad upstream doesn't push the wait to something absurd."""
        return min(self.cooldown_seconds * (2 ** max(0, open_count - 1)), self.max_cooldown_seconds)

    def record_success(self, worker_id: str) -> None:
        with self._lock:
            record = self._records.setdefault(worker_id, BreakerRecord(worker_id=worker_id))
            record.total_calls += 1
            record.consecutive_failures = 0
            record.last_success_time = time.time()
            if record.state is BreakerState.HALF_OPEN:
                # The probe passed: the upstream (or transient condition) has
                # actually recovered, not just "enough time passed", so drop
                # the backoff counter back to zero rather than leaving the
                # next trip inheriting an inflated open_count.
                record.state = BreakerState.CLOSED
                record.cooldown_until = 0.0
                record.open_count = 0
                record.half_open_probes_in_flight = 0
            elif record.state is BreakerState.OPEN:
                # A success arriving while nominally OPEN means a caller
                # bypassed is_available() (or a stale in-flight call landed
                # after the trip). Either way it is real evidence of health,
                # so start probing rather than ignoring it.
                record.state = BreakerState.HALF_OPEN
            # QUARANTINED intentionally falls through and does nothing: a
            # local fault does not get waived by one lucky call slipping
            # through outside is_available()'s gate -- only clear_quarantine()
            # exits quarantine. See module docstring.

    def record_failure(
        self,
        worker_id: str,
        failure_class: FailureClass | None = None,
        *,
        reason: str = "",
    ) -> None:
        """Record a failed call. `failure_class` is the harness's own
        `contracts.FailureClass` verdict for *why* it failed (see
        `adapters.executor.classify_failure` for one way to produce it);
        omitting it (the default) preserves the original trip-after-N
        behavior for callers that predate this classification.
        """
        with self._lock:
            record = self._records.setdefault(worker_id, BreakerRecord(worker_id=worker_id))
            record.total_calls += 1
            record.failures += 1
            record.consecutive_failures += 1
            record.last_failure_time = time.time()
            record.last_failure_class = failure_class

            if failure_class in _LOCAL_FAILURE_CLASSES:
                # Quarantine on the FIRST local failure, not after threshold
                # consecutive ones: a missing binary or invalid credential is
                # deterministic, so the next `failure_threshold - 1` calls
                # would just fail the exact same way for the exact same
                # reason. Waiting for a streak here only burns calls
                # re-discovering what call #1 already proved.
                record.state = BreakerState.QUARANTINED
                record.quarantine_reason = reason or f"{failure_class.value} failure"
                record.quarantined_at = record.last_failure_time
                record.half_open_probes_in_flight = 0
                return

            if record.state is BreakerState.HALF_OPEN:
                # The probe failed: whatever tripped it originally is still
                # going on, so re-open with a LONGER wait (see _next_cooldown)
                # instead of retrying at the same cadence that just failed.
                record.half_open_probes_in_flight = max(0, record.half_open_probes_in_flight - 1)
                record.open_count += 1
                record.state = BreakerState.OPEN
                record.circuit_opened_at = record.last_failure_time
                record.cooldown_until = record.last_failure_time + self._next_cooldown(record.open_count)
                return

            if record.state is not BreakerState.OPEN and record.consecutive_failures >= self.failure_threshold:
                record.open_count += 1
                record.state = BreakerState.OPEN
                record.circuit_opened_at = record.last_failure_time
                record.cooldown_until = record.last_failure_time + self._next_cooldown(record.open_count)

    def is_available(self, worker_id: str) -> bool:
        """Whether a call to `worker_id` should be attempted right now.

        Holds `self._lock` for the whole decision, including the
        OPEN-cooldown-expired -> HALF_OPEN transition and the HALF_OPEN slot
        check, so two threads racing this call cannot both observe "cooldown
        just expired" or both observe "a probe slot is free" and both get
        waved through -- exactly the race a probe allowance exists to
        prevent.
        """
        with self._lock:
            record = self._records.get(worker_id)
            if record is None:
                return True
            if record.state is BreakerState.CLOSED:
                return True
            if record.state is BreakerState.QUARANTINED:
                # No timer check here on purpose -- see module docstring.
                # This is the entire reason QUARANTINED exists as a state
                # distinct from OPEN.
                return False
            if record.state is BreakerState.OPEN:
                if time.time() < record.cooldown_until:
                    return False
                # Cooldown elapsed: enter probing. Falls through to the
                # HALF_OPEN slot check below so this first probe is counted
                # against the allowance too, not given a free pass.
                record.state = BreakerState.HALF_OPEN
                record.half_open_probes_in_flight = 0
            # HALF_OPEN: only let `half_open_allowance` unresolved probes
            # through at once. A probe that gets a slot must eventually
            # resolve via record_success/record_failure to free it again.
            if record.half_open_probes_in_flight >= self.half_open_allowance:
                return False
            record.half_open_probes_in_flight += 1
            return True

    def get_state(self, worker_id: str) -> BreakerState:
        with self._lock:
            record = self._records.get(worker_id)
            if record is None:
                return BreakerState.CLOSED
            return record.state

    def get_all_states(self) -> dict[str, BreakerState]:
        with self._lock:
            return {wid: rec.state for wid, rec in self._records.items()}

    def clear_quarantine(self, worker_id: str) -> bool:
        """The only way out of QUARANTINED. Call this after actually fixing
        the local fault (reinstalling a binary, rotating a credential,
        repairing a config) -- never on a schedule. Returns False (no-op) if
        the worker isn't quarantined, so callers can call it defensively
        without checking state first.
        """
        with self._lock:
            record = self._records.get(worker_id)
            if record is None or record.state is not BreakerState.QUARANTINED:
                return False
            record.state = BreakerState.CLOSED
            record.consecutive_failures = 0
            record.open_count = 0
            record.cooldown_until = 0.0
            record.quarantine_reason = ""
            record.quarantined_at = 0.0
            return True

    def reset(self, worker_id: str | None = None) -> None:
        with self._lock:
            if worker_id is None:
                self._records.clear()
            else:
                self._records.pop(worker_id, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                wid: {
                    "state": rec.state.value,
                    "total_calls": rec.total_calls,
                    "failures": rec.failures,
                    "failure_rate": round(rec.failure_rate, 3),
                    "consecutive_failures": rec.consecutive_failures,
                    "cooldown_until": rec.cooldown_until,
                    "open_count": rec.open_count,
                    "last_failure_class": rec.last_failure_class.value if rec.last_failure_class else None,
                    "quarantine_reason": rec.quarantine_reason,
                }
                for wid, rec in self._records.items()
            }
