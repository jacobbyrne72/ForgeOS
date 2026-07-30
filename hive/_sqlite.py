"""A sqlite connection that is actually safe to share between threads.

Every store in hive (`Ledger`, `EventLog`, `LeaseStore`, `AvoidanceLog`,
`SpanStore`, `ClaimStore`) opens one connection with `check_same_thread=False`
and hands it to whoever asks. That flag only disables Python's *thread-affinity
check* — it does not make the connection safe for concurrent use, and the
docstrings that cited it as concurrency protection were mistaken.

While tasks ran one at a time this was harmless. The moment the Forge started
executing the ready set in a thread pool it became live, and it surfaced exactly
the way this class of bug always does: a suite that passes when a test runs alone
and fails once under load, with a task dying before it was even routed
(`worker_id=''`) and no explanation beyond "worker thread raised".

Two hazards, both real:

- **Shared cursors.** Two threads calling `execute` on one connection interleave
  on the same underlying statement machinery; sqlite3 raises "recursive use of
  cursors not allowed", or one thread's rows arrive on the other's cursor.
- **Torn transactions.** `with conn:` is a transaction. Two threads inside it at
  once means one commits the other's half-finished work.

The fix is one reentrant lock per connection, held across a whole statement *and*
across a whole `with conn:` block. Reads are materialised before the lock is
released, so no cursor ever outlives the lock that protected it.

This serialises database access, which is the correct trade here: the expensive
part of a task is the worker call, which happens outside any of this. Contention
on microsecond-scale sqlite statements costs nothing measurable, and the
alternative — a thread-local connection per store — trades this bug for
`database is locked` under write contention plus a connection leak per worker
thread.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


class Result:
    """Rows already fetched, so nothing depends on a cursor after the lock drops.

    Mirrors the slice of the cursor API hive actually uses (`fetchone`,
    `fetchall`, iteration, `lastrowid`, `rowcount`). Deliberately not a full
    cursor: a partial result that still needed the connection would reintroduce
    precisely the cross-thread hazard this module exists to remove.
    """

    __slots__ = ("_rows", "_pos", "lastrowid", "rowcount")

    def __init__(self, rows: list[Any], lastrowid: int | None, rowcount: int) -> None:
        self._rows = rows
        self._pos = 0
        self.lastrowid = lastrowid
        self.rowcount = rowcount

    def fetchone(self) -> Any:
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchall(self) -> list[Any]:
        rows = self._rows[self._pos:]
        self._pos = len(self._rows)
        return rows

    def __iter__(self):
        return iter(self.fetchall())

    def __len__(self) -> int:
        return len(self._rows)


class GuardedConnection:
    """A `sqlite3.Connection` with one reentrant lock in front of it.

    Reentrant on purpose: `_tx()` opens `with conn:` and then runs several
    `execute` calls inside it. A plain lock would deadlock on the first one.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()

    # -- transaction ------------------------------------------------------
    # The lock spans the entire block, not each statement inside it. Locking
    # per statement would let a second thread commit in the middle of this
    # transaction, which is the failure `with conn:` exists to prevent.

    def __enter__(self) -> GuardedConnection:
        self._lock.acquire()
        try:
            self._conn.__enter__()
        except BaseException:
            self._lock.release()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            return bool(self._conn.__exit__(exc_type, exc, tb))
        finally:
            self._lock.release()

    # -- statements -------------------------------------------------------

    def execute(self, sql: str, params: Any = ()) -> Result:
        with self._lock:
            cur = self._conn.execute(sql, params)
            # `description` is None for statements that return no rows; calling
            # fetchall() on those is wasted work, not an error.
            rows = cur.fetchall() if cur.description is not None else []
            return Result(rows, cur.lastrowid, cur.rowcount)

    def executemany(self, sql: str, seq: Any) -> Result:
        with self._lock:
            cur = self._conn.executemany(sql, seq)
            return Result([], cur.lastrowid, cur.rowcount)

    def executescript(self, script: str) -> Result:
        with self._lock:
            cur = self._conn.executescript(script)
            return Result([], cur.lastrowid, cur.rowcount)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def rollback(self) -> None:
        with self._lock:
            self._conn.rollback()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- passthrough ------------------------------------------------------

    @property
    def row_factory(self):
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value) -> None:
        with self._lock:
            self._conn.row_factory = value

    @property
    def raw(self) -> sqlite3.Connection:
        """The underlying connection, for the rare caller that needs it.

        Using it drops every guarantee above, so it is named to make that
        obvious at the call site rather than hidden behind attribute access.
        """
        return self._conn


def connect(path: str | Path, *, timeout: float = 30.0) -> GuardedConnection:
    """Open a WAL-mode connection guarded for concurrent use.

    `timeout` is the busy timeout: WAL lets readers run while a writer holds the
    write lock, but a second *writer* still waits. Without a timeout that wait
    fails immediately with `database is locked`; with one it blocks briefly and
    succeeds, which is the behaviour a parallel run needs.
    """
    p = str(path)
    if p != ":memory:":
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, check_same_thread=False, timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    return GuardedConnection(conn)


__all__ = ["GuardedConnection", "Result", "connect"]
