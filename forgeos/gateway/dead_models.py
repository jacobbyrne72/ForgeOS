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
    PRIMARY KEY (transport, model_ref)
);
"""


class DeadModel(BaseModel):
    """One (transport, model_ref) pair's recorded death."""

    transport: str
    model_ref: str
    reason: str
    retry_after: float | None = None
    recorded_at: float

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
    ) -> None:
        """Record that `model_ref` stopped working on `transport`.

        `retry_after=None` (the default) is TERMINAL: permanent until
        `clear()`. A finite `retry_after` (epoch seconds, same clock as
        `clock=`) is TEMPORARY and self-heals once that time passes.
        """
        with self._conn:
            self._conn.execute(
                "INSERT INTO dead_models (transport, model_ref, reason, retry_after, recorded_at)"
                " VALUES (?,?,?,?,?)"
                " ON CONFLICT(transport, model_ref) DO UPDATE SET"
                " reason=excluded.reason, retry_after=excluded.retry_after,"
                " recorded_at=excluded.recorded_at",
                (transport, model_ref, reason, retry_after, self._clock()),
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


__all__ = ["DEAD_MODELS_SCHEMA", "DeadModel", "DeadModelStore"]
