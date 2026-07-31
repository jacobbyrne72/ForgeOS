"""Half-open probe recovery for `forgeos/gateway/dead_models.py`'s
TEMPORARY dead-model entries -- the circuit-breaker HALF_OPEN state layered
on top of the plain clock-based self-heal `test_gateway_resilience.py`
already covers.

`claim_probe`/`report_probe` are purely additive: no existing `is_dead`/
`mark_dead`/`clear` behavior changes, which is why those are re-tested here
only where a probe interacts with them (`mark_dead` clearing a stale claim,
`report_probe` reusing `mark_dead` under the hood), not duplicated wholesale
from `test_gateway_resilience.py`.
"""

from __future__ import annotations

import sqlite3

from forgeos.gateway.dead_models import (
    DEFAULT_PROBE_BACKOFF_SECONDS,
    MAX_PROBE_BACKOFF_SECONDS,
    PROBE_CLAIM_TIMEOUT_SECONDS,
    DeadModelStore,
)


def _store(t: float = 1_000.0):
    clock = {"t": t}
    store = DeadModelStore(clock=lambda: clock["t"])
    return store, clock


# ===================================================== claim_probe: gating


def test_claim_probe_false_for_an_unknown_pair():
    store, _ = _store()
    assert store.claim_probe("openrouter", "openrouter/never-seen", now=1_000.0) is False


def test_claim_probe_false_for_a_terminal_entry_even_long_after():
    store, clock = _store()
    store.mark_dead("openrouter", "openrouter/gone", reason="404")

    assert store.claim_probe("openrouter", "openrouter/gone", now=clock["t"]) is False
    assert store.claim_probe("openrouter", "openrouter/gone", now=clock["t"] + 10_000_000) is False


def test_claim_probe_false_while_still_fully_open():
    store, clock = _store()
    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0)

    # now has not reached retry_after yet -- still OPEN, no trial call due.
    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_059.0) is False


def test_claim_probe_true_for_the_first_caller_once_retry_after_has_passed():
    store, _ = _store()
    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0)

    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is True


def test_claim_probe_false_for_a_concurrent_second_caller_after_the_first_claims():
    store, _ = _store()
    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0)

    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is True
    # Same instant, a second caller -- must not also get the trial.
    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is False
    # Still no, a little later but within the claim timeout.
    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0 + 30.0) is False


def test_claim_probe_is_reclaimable_after_the_claim_timeout_elapses():
    """A claimant that crashed or hung without ever calling `report_probe`
    must not wedge the HALF_OPEN slot shut forever."""
    store, _ = _store()
    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0)

    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is True
    still_within = 1_060.0 + PROBE_CLAIM_TIMEOUT_SECONDS - 1
    assert store.claim_probe("openrouter", "openrouter/flaky", now=still_within) is False

    past_timeout = 1_060.0 + PROBE_CLAIM_TIMEOUT_SECONDS + 1
    assert store.claim_probe("openrouter", "openrouter/flaky", now=past_timeout) is True


def test_claim_probe_is_scoped_per_transport():
    store, _ = _store()
    store.mark_dead("openrouter", "shared/model", reason="glitch", retry_after=1_060.0)

    assert store.claim_probe("openrouter", "shared/model", now=1_060.0) is True
    # A different transport was never marked dead at all -- nothing to claim.
    assert store.claim_probe("omniroute", "shared/model", now=1_060.0) is False


# =================================================== report_probe: success


def test_report_probe_ok_clears_the_entry():
    store, _ = _store()
    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0)
    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is True

    store.report_probe("openrouter", "openrouter/flaky", True, now=1_060.5)

    assert store.get("openrouter", "openrouter/flaky") is None
    assert store.is_dead("openrouter", "openrouter/flaky") is False


def test_report_probe_ok_true_clears_even_without_a_prior_claim():
    """Defensive: resolving success should not require re-deriving the claim
    state, since the only caller that legitimately reaches this already won
    the claim."""
    store, _ = _store()
    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0)

    store.report_probe("openrouter", "openrouter/flaky", True, now=1_060.0)

    assert store.get("openrouter", "openrouter/flaky") is None


# =================================================== report_probe: failure


def test_report_probe_failure_with_explicit_retry_after_reopens_temporary():
    store, _ = _store()
    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0)
    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is True

    store.report_probe("openrouter", "openrouter/flaky", False, now=1_060.0, retry_after=2_000.0)

    entry = store.get("openrouter", "openrouter/flaky")
    assert entry is not None
    assert entry.terminal is False
    assert entry.retry_after == 2_000.0
    assert store.is_dead("openrouter", "openrouter/flaky") is True


def test_report_probe_failure_without_retry_after_backs_off_from_the_prior_window():
    store, _ = _store()
    # Original window: recorded_at=1_000.0, retry_after=1_060.0 -> 60s window.
    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0, now=1_000.0)
    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is True

    store.report_probe("openrouter", "openrouter/flaky", False, now=1_060.0)

    entry = store.get("openrouter", "openrouter/flaky")
    assert entry is not None
    # Doubled window (60s -> 120s) from `now`.
    assert entry.retry_after == 1_060.0 + 120.0


def test_report_probe_failure_backoff_is_capped_at_the_max():
    store, _ = _store()
    huge_window = MAX_PROBE_BACKOFF_SECONDS * 10
    store.mark_dead(
        "openrouter", "openrouter/flaky", reason="glitch",
        retry_after=1_000.0 + huge_window, now=1_000.0,
    )
    now = 1_000.0 + huge_window
    assert store.claim_probe("openrouter", "openrouter/flaky", now=now) is True

    store.report_probe("openrouter", "openrouter/flaky", False, now=now)

    entry = store.get("openrouter", "openrouter/flaky")
    assert entry is not None
    assert entry.retry_after == now + MAX_PROBE_BACKOFF_SECONDS


def test_report_probe_failure_with_no_prior_measured_window_uses_the_default_backoff():
    store, _ = _store()
    # No prior entry at all -- defensive path, e.g. a claim resolved after
    # something else already cleared the row out from under it.
    store.report_probe("openrouter", "openrouter/never-marked", False, now=1_000.0)

    entry = store.get("openrouter", "openrouter/never-marked")
    assert entry is not None
    assert entry.retry_after == 1_000.0 + DEFAULT_PROBE_BACKOFF_SECONDS * 2


def test_report_probe_failure_reuses_the_prior_reason():
    store, _ = _store()
    store.mark_dead("openrouter", "openrouter/flaky", reason="502 upstream", retry_after=1_060.0)
    store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0)

    store.report_probe("openrouter", "openrouter/flaky", False, now=1_060.0)

    assert store.get("openrouter", "openrouter/flaky").reason == "502 upstream"


def test_report_probe_failure_on_a_terminal_entry_leaves_it_terminal():
    """Defensive: `claim_probe` never grants a claim for a TERMINAL entry, so
    a well-behaved caller can't reach this -- but a misused/forged claim
    must not be able to downgrade a permanent death into a self-healing
    one."""
    store, _ = _store()
    store.mark_dead("openrouter", "openrouter/gone", reason="404")

    store.report_probe("openrouter", "openrouter/gone", False, now=1_000.0, retry_after=2_000.0)

    entry = store.get("openrouter", "openrouter/gone")
    assert entry is not None
    assert entry.terminal is True
    assert store.is_dead("openrouter", "openrouter/gone") is True


# ========================================================= mark_dead + claim


def test_mark_dead_clears_a_stale_probe_claim():
    """A fresh dead-mark (e.g. the next real job hitting `ModelUnavailableError`
    again) must not leave a claim from a previous, now-irrelevant trial
    dangling -- otherwise a legitimate new claimant is blocked by a claim
    nobody will ever resolve."""
    store, _ = _store()
    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0)
    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is True
    assert store.get("openrouter", "openrouter/flaky").probe_claimed_at == 1_060.0

    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch again", retry_after=2_000.0)

    entry = store.get("openrouter", "openrouter/flaky")
    assert entry.probe_claimed_at is None
    # And the new window is independently claimable once it opens.
    assert store.claim_probe("openrouter", "openrouter/flaky", now=2_000.0) is True


# ============================================================= end to end


def test_full_half_open_cycle_success_then_reuse():
    store, _ = _store()
    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0)

    assert store.is_dead("openrouter", "openrouter/flaky") is True  # still OPEN
    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_059.0) is False  # too early

    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is True  # HALF_OPEN, claimed
    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is False  # already claimed

    store.report_probe("openrouter", "openrouter/flaky", True, now=1_060.2)  # trial call succeeded

    assert store.is_dead("openrouter", "openrouter/flaky") is False
    assert store.get("openrouter", "openrouter/flaky") is None


def test_full_half_open_cycle_failure_reopens_for_a_later_probe():
    store, _ = _store()
    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0, now=1_000.0)

    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is True
    store.report_probe("openrouter", "openrouter/flaky", False, now=1_060.0)  # trial call failed too

    new_retry_after = store.get("openrouter", "openrouter/flaky").retry_after
    assert new_retry_after > 1_060.0
    assert store.is_dead("openrouter", "openrouter/flaky") is True
    # No claim is grantable again until the new window opens.
    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.1) is False
    assert store.claim_probe("openrouter", "openrouter/flaky", now=new_retry_after) is True


# ========================================================== schema migration


def _create_legacy_db(path) -> None:
    """A `dead_models` table as it existed before `probe_claimed_at` --
    hand-built with raw sqlite3, not `DeadModelStore`, so the migration path
    in `DeadModelStore.__init__` is actually exercised rather than assumed."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE dead_models (
                transport TEXT NOT NULL,
                model_ref TEXT NOT NULL,
                reason TEXT NOT NULL,
                retry_after REAL,
                recorded_at REAL NOT NULL,
                PRIMARY KEY (transport, model_ref)
            )
            """
        )
        conn.execute(
            "INSERT INTO dead_models (transport, model_ref, reason, retry_after, recorded_at)"
            " VALUES (?,?,?,?,?)",
            ("openrouter", "openrouter/gone", "404", None, 900.0),
        )
        conn.commit()
    finally:
        conn.close()


def test_opening_a_pre_migration_database_adds_the_probe_column_without_crashing(tmp_path):
    db_path = tmp_path / "dead_models.db"
    _create_legacy_db(db_path)

    store = DeadModelStore(db_path)
    try:
        entry = store.get("openrouter", "openrouter/gone")
        assert entry is not None
        assert entry.terminal is True
        assert entry.probe_claimed_at is None

        # The migrated table is fully usable by the new probe API too.
        store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0)
        assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is True
    finally:
        store.close()


def test_probe_claim_persists_across_reopening_the_same_file(tmp_path):
    db_path = tmp_path / "dead_models.db"
    store = DeadModelStore(db_path)
    store.mark_dead("openrouter", "openrouter/flaky", reason="glitch", retry_after=1_060.0)
    assert store.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is True
    store.close()

    reopened = DeadModelStore(db_path)
    try:
        entry = reopened.get("openrouter", "openrouter/flaky")
        assert entry is not None
        assert entry.probe_claimed_at == 1_060.0
        # The claim from before the restart is still respected.
        assert reopened.claim_probe("openrouter", "openrouter/flaky", now=1_060.0) is False
    finally:
        reopened.close()
