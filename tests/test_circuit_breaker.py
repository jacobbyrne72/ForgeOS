"""Circuit breaker: LOCAL (ENVIRONMENT) faults quarantine and stay quarantined
until explicitly cleared; UPSTREAM/TRANSIENT faults trip with backoff and
auto-recover through a bounded HALF_OPEN probe window.
"""

from __future__ import annotations

import threading

import pytest

from forgeos import circuit_breaker as cb_module
from forgeos.circuit_breaker import BreakerState, CircuitBreaker
from forgeos.contracts import FailureClass


def _freeze(monkeypatch: pytest.MonkeyPatch, at: float) -> None:
    monkeypatch.setattr(cb_module.time, "time", lambda: at)


# --------------------------------------------------------------------- CLOSED


def test_starts_closed_and_available():
    cb = CircuitBreaker()
    assert cb.get_state("w") is BreakerState.CLOSED
    assert cb.is_available("w") is True


def test_unclassified_failures_trip_after_threshold_like_before():
    """No failure_class given (the pre-upgrade call shape) must keep tripping
    a normal OPEN circuit after `failure_threshold` consecutive failures --
    this is the exact shape `cost_router.py` / existing tests call it with."""
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=30)
    cb.record_failure("w")
    cb.record_failure("w")
    assert cb.get_state("w") is BreakerState.CLOSED
    cb.record_failure("w")
    assert cb.get_state("w") is BreakerState.OPEN
    assert cb.is_available("w") is False


def test_success_resets_consecutive_failure_count():
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure("w")
    cb.record_failure("w")
    cb.record_success("w")
    cb.record_failure("w")
    cb.record_failure("w")
    # Only 2 consecutive since the success reset the streak.
    assert cb.get_state("w") is BreakerState.CLOSED


# ----------------------------------------------------------- UPSTREAM/TRANSIENT


def test_transient_failure_trips_open_with_cooldown(monkeypatch):
    _freeze(monkeypatch, 1_000.0)
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=10)
    cb.record_failure("w", FailureClass.TRANSIENT)
    cb.record_failure("w", FailureClass.TRANSIENT)
    assert cb.get_state("w") is BreakerState.OPEN
    assert cb.is_available("w") is False  # still inside the cooldown window


def test_open_auto_recovers_to_half_open_after_cooldown(monkeypatch):
    """This is the DOES-auto-recover half of the LOCAL/UPSTREAM split: a
    provider outage is expected to pass, so the breaker probes again once the
    timer elapses -- no human action required."""
    _freeze(monkeypatch, 1_000.0)
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10)
    cb.record_failure("w", FailureClass.TRANSIENT)
    assert cb.get_state("w") is BreakerState.OPEN

    _freeze(monkeypatch, 1_011.0)  # past cooldown_until
    assert cb.is_available("w") is True
    assert cb.get_state("w") is BreakerState.HALF_OPEN


def test_half_open_success_closes_and_resets_backoff(monkeypatch):
    _freeze(monkeypatch, 1_000.0)
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10)
    cb.record_failure("w", FailureClass.TRANSIENT)
    _freeze(monkeypatch, 1_011.0)
    assert cb.is_available("w") is True  # consumes the probe slot

    cb.record_success("w")
    assert cb.get_state("w") is BreakerState.CLOSED
    assert cb.is_available("w") is True


def test_half_open_failure_reopens_with_longer_backoff(monkeypatch):
    """A failed probe must not just repeat the same wait -- it re-opens with
    a strictly longer cooldown than the first trip (exponential backoff)."""
    _freeze(monkeypatch, 1_000.0)
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10, max_cooldown_seconds=1_000)
    cb.record_failure("w", FailureClass.TRANSIENT)
    first_cooldown = cb._records["w"].cooldown_until - 1_000.0
    assert first_cooldown == 10

    _freeze(monkeypatch, 1_011.0)
    assert cb.is_available("w") is True  # -> HALF_OPEN, probe slot consumed

    cb.record_failure("w", FailureClass.TRANSIENT)  # probe fails
    assert cb.get_state("w") is BreakerState.OPEN
    second_cooldown = cb._records["w"].cooldown_until - 1_011.0
    assert second_cooldown > first_cooldown
    assert second_cooldown == 20  # 10 * 2**(2-1)


def test_backoff_caps_at_max_cooldown(monkeypatch):
    """Each failed probe doubles the wait (10 -> 20 -> 40 -> ...) but never
    past `max_cooldown_seconds` -- otherwise a chronically bad upstream would
    push the retry interval to something absurd instead of settling."""
    t = 0.0
    _freeze(monkeypatch, t)
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10, max_cooldown_seconds=25)
    cb.record_failure("w", FailureClass.TRANSIENT)  # open_count=1, span=10

    spans: list[float] = []
    for _ in range(6):
        record = cb._records["w"]
        spans.append(record.cooldown_until - record.last_failure_time)
        t = record.cooldown_until + 1
        _freeze(monkeypatch, t)
        assert cb.is_available("w") is True  # cooldown elapsed -> HALF_OPEN, probe consumed
        cb.record_failure("w", FailureClass.TRANSIENT)  # probe fails -> re-open

    assert spans[0] == 10
    assert spans[1] == 20
    assert all(s == 25 for s in spans[2:]), spans  # capped from here on


# -------------------------------------------------------------- HALF_OPEN cap


def test_half_open_allowance_limits_probe_count(monkeypatch):
    _freeze(monkeypatch, 1_000.0)
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10, half_open_allowance=2)
    cb.record_failure("w", FailureClass.TRANSIENT)
    _freeze(monkeypatch, 1_011.0)

    assert cb.is_available("w") is True   # probe 1
    assert cb.is_available("w") is True   # probe 2
    assert cb.is_available("w") is False  # allowance exhausted, third caller blocked


def test_half_open_slot_frees_once_a_probe_resolves(monkeypatch):
    _freeze(monkeypatch, 1_000.0)
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10, half_open_allowance=1)
    cb.record_failure("w", FailureClass.TRANSIENT)
    _freeze(monkeypatch, 1_011.0)

    assert cb.is_available("w") is True   # the one allowed probe
    assert cb.is_available("w") is False  # slot occupied

    cb.record_failure("w", FailureClass.TRANSIENT)  # that probe resolves (fails)
    # Re-opened with backoff -- still not available until the new cooldown passes.
    assert cb.is_available("w") is False


def test_concurrent_callers_cannot_both_take_the_same_half_open_slot(monkeypatch):
    """The property this guards: with an allowance of 1, out of N threads
    racing `is_available()` at the exact moment HALF_OPEN opens up, exactly
    one may get True. Any more than that means the probe limit is not
    actually exclusive under concurrency."""
    _freeze(monkeypatch, 1_000.0)
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10, half_open_allowance=1)
    cb.record_failure("w", FailureClass.TRANSIENT)
    _freeze(monkeypatch, 1_011.0)  # cooldown has elapsed for every thread below

    results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()  # line every thread up so they hit is_available() together
        got = cb.is_available("w")
        with results_lock:
            results.append(got)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1, "more than one caller slipped through the same probe slot"
    assert results.count(False) == 19


# ------------------------------------------------------------------ LOCAL/ENV


def test_environment_failure_quarantines_immediately():
    """Unlike TRANSIENT, a single ENVIRONMENT failure is enough -- a missing
    binary is deterministic, so waiting for a streak just repeats the same
    proven failure `failure_threshold - 1` more times."""
    cb = CircuitBreaker(failure_threshold=5)
    cb.record_failure("w", FailureClass.ENVIRONMENT, reason="binary not found")
    assert cb.get_state("w") is BreakerState.QUARANTINED
    assert cb.is_available("w") is False
    assert cb.stats()["w"]["quarantine_reason"] == "binary not found"


def test_quarantine_does_not_auto_recover_on_a_timer(monkeypatch):
    """The core LOCAL-vs-UPSTREAM behavioral difference: no matter how much
    time passes, a QUARANTINED worker never becomes available on its own."""
    _freeze(monkeypatch, 1_000.0)
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=1)
    cb.record_failure("w", FailureClass.ENVIRONMENT)
    assert cb.is_available("w") is False

    _freeze(monkeypatch, 1_000.0 + 365 * 24 * 3600)  # a full year later
    assert cb.get_state("w") is BreakerState.QUARANTINED
    assert cb.is_available("w") is False


def test_quarantine_requires_explicit_clear():
    cb = CircuitBreaker()
    cb.record_failure("w", FailureClass.ENVIRONMENT)
    assert cb.is_available("w") is False

    cleared = cb.clear_quarantine("w")
    assert cleared is True
    assert cb.get_state("w") is BreakerState.CLOSED
    assert cb.is_available("w") is True


def test_clear_quarantine_is_a_no_op_when_not_quarantined():
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure("w", FailureClass.TRANSIENT)  # OPEN, not QUARANTINED
    assert cb.clear_quarantine("w") is False
    assert cb.get_state("w") is BreakerState.OPEN

    assert cb.clear_quarantine("unknown-worker") is False


def test_success_does_not_lift_quarantine():
    """A stray success (e.g. a health-check call that happens to work despite
    the broken config) must not silently exit quarantine -- only an explicit
    clear does. Otherwise a flaky one-off success would mask a real local
    fault the next real call hits again."""
    cb = CircuitBreaker()
    cb.record_failure("w", FailureClass.ENVIRONMENT)
    cb.record_success("w")
    assert cb.get_state("w") is BreakerState.QUARANTINED


def test_other_failure_classes_use_upstream_trip_not_quarantine():
    """CONTEXT/MODEL/SPECIFICATION/POLICY are about the *task*, not proof the
    worker's local environment is broken, so none of them should quarantine."""
    cb = CircuitBreaker(failure_threshold=1)
    for failure_class in (
        FailureClass.CONTEXT,
        FailureClass.MODEL,
        FailureClass.SPECIFICATION,
        FailureClass.POLICY,
    ):
        cb.reset()
        cb.record_failure("w", failure_class)
        assert cb.get_state("w") is BreakerState.OPEN, failure_class


# ------------------------------------------------------------------- plumbing


def test_reset_clears_quarantine_too():
    cb = CircuitBreaker()
    cb.record_failure("w", FailureClass.ENVIRONMENT)
    cb.reset("w")
    assert cb.get_state("w") is BreakerState.CLOSED
    assert cb.is_available("w") is True


def test_get_all_states_reports_quarantine():
    cb = CircuitBreaker()
    cb.record_failure("a", FailureClass.ENVIRONMENT)
    cb.record_success("b")
    states = cb.get_all_states()
    assert states == {"a": BreakerState.QUARANTINED, "b": BreakerState.CLOSED}


def test_stats_reports_last_failure_class():
    cb = CircuitBreaker(failure_threshold=5)
    cb.record_failure("w", FailureClass.TRANSIENT)
    assert cb.stats()["w"]["last_failure_class"] == "transient"
