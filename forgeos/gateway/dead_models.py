"""Persistent memory of models that stopped answering, so the gateway does
not pay to rediscover the same dead `model_ref` on every run.

Ports the terminal-vs-temporary distinction from OmniRoute's connection
status model (`vendor/OmniRoute/src/lib/freeProviderRankings.ts::
isProviderUsable`, `TERMINAL_CONNECTION_STATUSES`), moved down to the
granularity forgeos actually has -- `(transport_name, model_ref)`, not a whole
provider connection:

- TERMINAL (`retry_after=None`): the model is gone for good as far as this
  transport is concerned -- a retired free-tier slug, credits exhausted, a
  banned or expired key. Excluded until something *explicitly* reconfigures
  it via `clear()`. Nothing about calling again turns a 404 into a 200, so
  there is no self-heal and no background sweep needed.
- TEMPORARY (`retry_after=<epoch seconds>`): self-heals the moment `now`
  passes that timestamp. `is_dead` just compares against the clock whenever
  it's asked -- no sweep needed there either.

`is_dead` alone is optimistic about that self-heal: the instant `now` passes
`retry_after`, *every* caller sees the pair as alive again and is free to
hit the transport, so concurrent callers can all retry the same
still-broken model at once. `claim_probe`/`report_probe` layer an opt-in
HALF_OPEN discipline on top for callers that want the stricter
circuit-breaker behaviour instead -- one trial call, not a stampede --
without changing what `is_dead` means for callers that don't use them.
`claim_probe` grants that trial to exactly one caller per TEMPORARY entry
once its `retry_after` has passed; `report_probe` resolves it by clearing
the entry (probe succeeded) or reopening it with a new `retry_after` (probe
failed). TERMINAL entries never grant a claim, for the same reason they
never self-heal: nothing about calling again turns a 404 into a 200.

`forgeos/gateway/client.py` is the only writer today: it calls `mark_dead`
(terminal) when a transport raises `ModelUnavailableError` for a specific
model, and consults `is_dead` before ever handing that (transport, model_ref)
pair to a transport again. This is deliberately independent of
`HealthTracker`: a dead *model* on one transport says nothing about that
transport's own health, and conflating the two is exactly the bug forgeos
already fixed once -- see `ModelUnavailableError`'s docstring in client.py --
treating a model problem as a transport problem took an entire provider
offline over one stale slug.

Backed by its own SQLite table opened through `forgeos/_sqlite.py`'s guarded
connection -- the pattern every store in this codebase follows (`Ledger`,
`AvoidanceLog`, `LeaseStore`, ...) -- so this survives a process restart
without touching `Ledger`'s schema or internals.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from .._sqlite import connect as _sql_connect

DEAD_MODELS_SCHEMA = """
CREATE TABLE IF NOT EXISTS dead_models (
    transport TEXT NOT NULL,
    model_ref TEXT NOT NULL,
    reason TEXT NOT NULL,
    retry_after REAL,
    recorded_at REAL NOT NULL,
    probe_claimed_at REAL,
    PRIMARY KEY (transport, model_ref)
);
"""

# How long a `claim_probe` grant blocks other claimants before it counts as
# abandoned. Long enough to comfortably outlast `HttpTransport`'s own 60s
# default call timeout -- the same margin `Gateway`'s `dedup_wait_timeout`
# (90s) gives its leader-wait -- so a probe that is still genuinely in
# flight is never mistaken for a crashed one. Short enough that a claimant
# that really did crash or hang doesn't wedge the HALF_OPEN slot shut
# indefinitely; the next caller after this window reclaims it.
PROBE_CLAIM_TIMEOUT_SECONDS = 90.0

# Fallback and ceiling for the `retry_after` a failed probe picks when the
# caller doesn't supply one. Doubling the interval that produced the
# previous `retry_after` (capped at an hour) means a model that keeps
# failing its trial calls gets probed less and less often instead of taking
# a fresh network hit on every access past its original `retry_after` --
# the exponential-backoff half of the circuit-breaker pattern.
DEFAULT_PROBE_BACKOFF_SECONDS = 30.0
MAX_PROBE_BACKOFF_SECONDS = 3600.0


class DeadModel(BaseModel):
    """One (transport, model_ref) pair's recorded death."""

    transport: str
    model_ref: str
    reason: str
    retry_after: float | None = None
    recorded_at: float
    probe_claimed_at: float | None = None

    @property
    def terminal(self) -> bool:
        return self.retry_after is None


class DeadModelStore:
    """SQLite-backed memory of (transport, model_ref) pairs that stopped working.

    `clock` is injectable for the same reason `HealthTracker`'s is: a
    temporary entry's self-heal has to be testable by advancing a fake clock
    instead of sleeping in a test.
    """

    def __init__(self, path: str | Path = ":memory:", *, clock: Callable[[], float] = time.time) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._conn = _sql_connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(DEAD_MODELS_SCHEMA)
        # Upgrade path for a `dead_models` table persisted before half-open
        # probe claiming existed. `CREATE TABLE IF NOT EXISTS` above never
        # adds a column to an already-existing table, so a pre-existing
        # on-disk store needs this explicitly. A fresh (or already-migrated)
        # database raises "duplicate column name", the expected, harmless
        # case -- same pattern as `Ledger`'s `generation` column upgrade.
        try:
            self._conn.execute("ALTER TABLE dead_models ADD COLUMN probe_claimed_at REAL")
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------------------------------------------------------------- write

    def mark_dead(
        self,
        transport: str,
        model_ref: str,
        *,
        reason: str,
        retry_after: float | None = None,
        now: float | None = None,
    ) -> None:
        """Record that `model_ref` stopped working on `transport`.

        `retry_after=None` (the default) is TERMINAL: permanent until
        `clear()`. A finite `retry_after` (epoch seconds, same clock as
        `clock=`) is TEMPORARY and self-heals once that time passes.

        `now` overrides `recorded_at` (defaulting to `self._clock()` as
        before) so `report_probe` can stamp a fresh dead-mark with the same
        `now` its caller passed in, instead of drifting from whatever the
        store's own clock says at the exact instant this runs.

        Always clears any outstanding `probe_claimed_at`: a fresh dead-mark
        -- whether the first one or a re-mark on top of an existing entry
        -- means the old claim (if any) no longer refers to a trial that
        makes sense to resolve.
        """
        recorded_at = self._clock() if now is None else now
        with self._conn:
            self._conn.execute(
                "INSERT INTO dead_models"
                " (transport, model_ref, reason, retry_after, recorded_at, probe_claimed_at)"
                " VALUES (?,?,?,?,?,NULL)"
                " ON CONFLICT(transport, model_ref) DO UPDATE SET"
                " reason=excluded.reason, retry_after=excluded.retry_after,"
                " recorded_at=excluded.recorded_at, probe_claimed_at=NULL",
                (transport, model_ref, reason, retry_after, recorded_at),
            )

    def clear(self, transport: str, model_ref: str) -> None:
        """Explicit reconfiguration: forget this pair, terminal or not.

        The only way a TERMINAL entry ever goes away -- there is no
        automatic path, by design.
        """
        with self._conn:
            self._conn.execute(
                "DELETE FROM dead_models WHERE transport=? AND model_ref=?",
                (transport, model_ref),
            )

    # ----------------------------------------------------------------- read

    def get(self, transport: str, model_ref: str) -> DeadModel | None:
        row = self._conn.execute(
            "SELECT * FROM dead_models WHERE transport=? AND model_ref=?",
            (transport, model_ref),
        ).fetchone()
        if row is None:
            return None
        return DeadModel(
            transport=row["transport"],
            model_ref=row["model_ref"],
            reason=row["reason"],
            retry_after=row["retry_after"],
            recorded_at=row["recorded_at"],
            probe_claimed_at=row["probe_claimed_at"],
        )

    def is_dead(self, transport: str, model_ref: str) -> bool:
        """Whether this pair should be skipped right now, without a network call.

        A terminal entry is always dead. A temporary entry is dead only
        while the clock has not yet passed its `retry_after` -- once it has,
        the pair is live again automatically. The row is left in place
        either way; it is harmless history until something overwrites or
        clears it, and re-checking the clock on every read is cheaper than
        a sweep that has to run on a schedule.
        """
        entry = self.get(transport, model_ref)
        if entry is None:
            return False
        if entry.retry_after is None:
            return True
        return self._clock() < entry.retry_after

    # ------------------------------------------------------------- probing

    def claim_probe(self, transport: str, model_ref: str, now: float) -> bool:
        """Try to become the single caller allowed to send the HALF_OPEN trial call.

        Mirrors the circuit-breaker HALF_OPEN state: once `now` has passed a
        TEMPORARY entry's `retry_after`, exactly one caller should make the
        trial request while every other concurrent caller keeps treating
        the model as still dead -- a stampede of simultaneous "is it back
        yet" calls defeats the point of remembering it was dead at all.

        Returns False, never raises, for every case where probing is not
        this caller's job right now: no entry (nothing dead to probe), a
        TERMINAL entry (`retry_after=None` never opens a half-open window,
        per the module docstring), an entry still fully OPEN (`now` has not
        reached `retry_after` yet), or an entry whose HALF_OPEN slot
        another caller already claimed within `PROBE_CLAIM_TIMEOUT_SECONDS`.

        The read (is this pair half-open and unclaimed?) and the write
        (claim it) happen under one `exclusive()` acquisition -- serialising
        them individually would let two callers each read "unclaimed"
        before either writes, both then claim, and the single-claimant
        guarantee this method exists for is gone. Same race `LeaseStore.
        acquire` guards against for path leases.
        """
        with self._conn.exclusive():
            row = self._conn.execute(
                "SELECT retry_after, probe_claimed_at FROM dead_models"
                " WHERE transport=? AND model_ref=?",
                (transport, model_ref),
            ).fetchone()
            if row is None or row["retry_after"] is None:
                return False  # unknown pair, or TERMINAL -- never half-open
            if now < row["retry_after"]:
                return False  # still fully OPEN
            claimed_at = row["probe_claimed_at"]
            if claimed_at is not None and now - claimed_at < PROBE_CLAIM_TIMEOUT_SECONDS:
                return False  # someone else's claim is still live
            with self._conn:
                self._conn.execute(
                    "UPDATE dead_models SET probe_claimed_at=? WHERE transport=? AND model_ref=?",
                    (now, transport, model_ref),
                )
            return True

    def report_probe(
        self,
        transport: str,
        model_ref: str,
        ok: bool,
        now: float,
        *,
        retry_after: float | None = None,
    ) -> None:
        """Resolve a HALF_OPEN trial: `ok=True` closes the breaker, `ok=False` reopens it.

        `ok=True` clears the entry exactly like an explicit `clear()` --
        the model answered, so it goes back to having no row at all, same
        as any model that was never marked dead.

        `ok=False` keeps the entry TEMPORARY (a failed probe never promotes
        it to TERMINAL -- only a real `ModelUnavailableError` earns that,
        per the module's terminal-vs-temporary split) and picks the next
        `retry_after`: the caller's value if given, else an exponential-ish
        backoff on the interval that produced the previous one (see
        `DEFAULT_PROBE_BACKOFF_SECONDS` / `MAX_PROBE_BACKOFF_SECONDS`).
        Reuses the prior `reason` when there was a prior entry, since a
        failed probe on an already-recorded pair isn't a new failure mode.

        A TERMINAL entry is left untouched on `ok=False` -- defensive only,
        since `claim_probe` never grants a claim for one in the first
        place, so a well-behaved caller can't reach this. Guards against a
        misused/forged claim silently downgrading a permanent death into a
        self-healing one.
        """
        with self._conn.exclusive():
            entry = self.get(transport, model_ref)
            if ok:
                self.clear(transport, model_ref)
                return
            if entry is not None and entry.terminal:
                return
            if retry_after is None:
                if entry is not None and entry.retry_after is not None:
                    prior_window = max(
                        entry.retry_after - entry.recorded_at, DEFAULT_PROBE_BACKOFF_SECONDS
                    )
                else:
                    prior_window = DEFAULT_PROBE_BACKOFF_SECONDS
                retry_after = now + min(prior_window * 2, MAX_PROBE_BACKOFF_SECONDS)
            reason = entry.reason if entry is not None else "half-open probe failed"
            self.mark_dead(transport, model_ref, reason=reason, retry_after=retry_after, now=now)


__all__ = [
    "DEAD_MODELS_SCHEMA",
    "DEFAULT_PROBE_BACKOFF_SECONDS",
    "MAX_PROBE_BACKOFF_SECONDS",
    "PROBE_CLAIM_TIMEOUT_SECONDS",
    "DeadModel",
    "DeadModelStore",
]
