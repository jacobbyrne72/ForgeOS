"""Deterministic path-lease store. Stops two workers corrupting the same file.

Workers take out a lease before touching a path pattern. Two exclusive claims
on overlapping paths must never both succeed, and a dead worker's lease must
not block the queue forever once it expires. SQLite is the source of truth,
same pattern as `events.EventLog`: no in-memory lock table to fall out of
sync with reality.
"""

from __future__ import annotations

import fnmatch
import os
import sqlite3
import time

import psutil
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .contracts import new_id
from ._sqlite import connect as _sql_connect


class LeaseType(str, Enum):
    WRITE = "write"  # exclusive: conflicts with any overlapping WRITE or READ
    READ = "read"  # shared: only conflicts with an overlapping WRITE


LEASES_SCHEMA = """
CREATE TABLE IF NOT EXISTS path_leases (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    repo_id TEXT NOT NULL,
    path_pattern TEXT NOT NULL,
    lease_type TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    released_at REAL
);
CREATE INDEX IF NOT EXISTS idx_lease_repo ON path_leases(repo_id, released_at, expires_at);
"""


def _normalize(pattern: str) -> str:
    """Windows backslashes and a leading "./" both mean the same posix path."""
    p = pattern.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


def _segment_matches(a: str, b: str) -> bool:
    if a == b:
        return True
    # A wildcard segment is a pattern, not a literal — check it against the
    # other side both ways since either one (or neither) may hold the glob.
    if any(ch in a for ch in "*?[") or any(ch in b for ch in "*?["):
        return fnmatch.fnmatch(a, b) or fnmatch.fnmatch(b, a)
    return False


def patterns_overlap(a: str, b: str) -> bool:
    """True if some path could satisfy both patterns.

    Compares path segments left to right. A "**" segment matches everything
    below it, so it ends the comparison immediately. Running out of segments
    on one side without a mismatch is also an overlap — a bare directory
    pattern like "src/router" is implicitly a prefix over everything under
    it, the same convention `contracts.Scope.covers` uses.
    """
    a_segs = _normalize(a).split("/")
    b_segs = _normalize(b).split("/")
    if a_segs == b_segs:
        return True
    for sa, sb in zip(a_segs, b_segs):
        if sa == "**" or sb == "**":
            return True
        if not _segment_matches(sa, sb):
            return False
    return True


class Lease(BaseModel):
    id: str = Field(default_factory=lambda: new_id("lease"))
    task_id: str
    repo_id: str
    path_pattern: str
    lease_type: LeaseType
    acquired_at: float = Field(default_factory=time.time)
    expires_at: float
    released_at: float | None = None
    # 0 means "owner unknown" (a pre-migration row): liveness reaping never
    # applies, only the TTL does. Never guess an owner that wasn't recorded.
    owner_pid: int = 0

    def is_expired(self, at: float | None = None) -> bool:
        at = at if at is not None else time.time()
        return at >= self.expires_at


def _conflicting_types(a: LeaseType, b: LeaseType) -> bool:
    """READ+READ is the only combination that never conflicts."""
    return not (a is LeaseType.READ and b is LeaseType.READ)


class LeaseStore:
    """SQLite-backed path leases. Same connection style as `events.EventLog`."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = _sql_connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(LEASES_SCHEMA)
        # Upgrade path for a store persisted before owner liveness existed.
        # A fresh or already-migrated database raises "duplicate column
        # name", the expected harmless case — same pattern as
        # `gateway.dead_models`' probe column and the Ledger's `generation`.
        try:
            self._conn.execute(
                "ALTER TABLE path_leases ADD COLUMN owner_pid INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------- io

    def _row_to_lease(self, r: sqlite3.Row) -> Lease:
        return Lease(
            id=r["id"],
            task_id=r["task_id"],
            repo_id=r["repo_id"],
            path_pattern=r["path_pattern"],
            lease_type=LeaseType(r["lease_type"]),
            acquired_at=r["acquired_at"],
            expires_at=r["expires_at"],
            released_at=r["released_at"],
            owner_pid=r["owner_pid"] if "owner_pid" in r.keys() else 0,
        )

    def _insert(self, lease: Lease) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO path_leases"
                " (id, task_id, repo_id, path_pattern, lease_type, acquired_at,"
                "  expires_at, released_at, owner_pid)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    lease.id,
                    lease.task_id,
                    lease.repo_id,
                    lease.path_pattern,
                    lease.lease_type.value,
                    lease.acquired_at,
                    lease.expires_at,
                    lease.released_at,
                    lease.owner_pid,
                ),
            )

    # -------------------------------------------------------------- queries

    def active(self, repo_id: str | None = None) -> list[Lease]:
        """Leases that are neither released nor past their TTL."""
        at = time.time()
        sql = "SELECT * FROM path_leases WHERE released_at IS NULL AND expires_at > ?"
        args: list = [at]
        if repo_id is not None:
            sql += " AND repo_id = ?"
            args.append(repo_id)
        rows = self._conn.execute(sql, args).fetchall()
        return [self._row_to_lease(r) for r in rows]

    def conflicts(self, repo_id: str, path_pattern: str, lease_type: LeaseType | str) -> list[Lease]:
        """Active leases in `repo_id` that would block `path_pattern`/`lease_type`.

        This is a raw query — it does not know or care whose task is asking,
        so a task's own prior lease still shows up here. `acquire()` is what
        turns that into a no-op instead of a rejection.
        """
        lease_type = LeaseType(lease_type)
        return [
            lease
            for lease in self.active(repo_id=repo_id)
            if _conflicting_types(lease.lease_type, lease_type)
            and patterns_overlap(lease.path_pattern, path_pattern)
        ]

    def holder(self, repo_id: str, path: str) -> Lease | None:
        """The lease that currently governs a concrete file path, if any.

        By the conflict invariant a path can have at most one active WRITE
        lease, or any number of active READ leases, never both — so WRITE
        sorts first and ties break on acquisition order for determinism.
        """
        matches = [lease for lease in self.active(repo_id=repo_id)
                   if patterns_overlap(lease.path_pattern, path)]
        if not matches:
            return None
        matches.sort(key=lambda m: (m.lease_type is not LeaseType.WRITE, m.acquired_at))
        return matches[0]

    # ------------------------------------------------------------- mutation

    def acquire(
        self,
        task_id: str,
        repo_id: str,
        path_pattern: str,
        lease_type: LeaseType | str,
        ttl_seconds: float,
    ) -> Lease | None:
        """Grant a lease, or None if it conflicts with another task's lease.

        `ttl_seconds` may be zero or negative on purpose — that is how tests
        (and callers) create an already-expired lease deterministically,
        without sleeping or mocking the clock.
        """
        lease_type = LeaseType(lease_type)

        # The entire decision runs under one lock. Checking for conflicts and
        # then inserting are two statements, and serialising them individually
        # is not enough: two threads can each read "no conflict" before either
        # writes, and both are then granted the same WRITE lease. Observed with
        # 16 threads racing one path — two tasks granted, which is precisely the
        # corruption path leases exist to prevent. Every guard here is only as
        # strong as the window between reading it and acting on it.
        # `exclusive_write`, not `exclusive`: the latter takes a per-process
        # threading lock, and two processes on one state dir have two of them.
        # `forge.py` and `dashboard/app.py` already open separate LeaseStores
        # against the same leases.db. Measured with `exclusive()`: two
        # connections racing this method past a dead owner were BOTH granted a
        # WRITE lease on the same path, 5 runs out of 5. `BEGIN IMMEDIATE`
        # serialises the whole decision across processes, and its rollback is
        # what stops the reap below leaking an open transaction when this
        # returns None.
        with self._conn.exclusive_write():

            own = [
                lease
                for lease in self.active(repo_id=repo_id)
                if lease.task_id == task_id
                and patterns_overlap(lease.path_pattern, path_pattern)
            ]
            if own:
                return own[0]  # re-acquiring your own overlapping lease is a no-op, not a conflict

            blockers = [lease for lease in self.conflicts(repo_id, path_pattern, lease_type)
                        if lease.task_id != task_id]
            # Owner liveness: a blocker whose recording process is provably
            # dead cannot release its lease and cannot be mid-write either —
            # waiting out its TTL is pure wall-clock loss (observed live: a
            # crash-killed run's leases deferred every later job on the same
            # state dir for the full 600s budget window). Reap it now.
            # Conservative by construction: owner_pid 0 (pre-migration row)
            # and the current pid never reap, and a recycled pid reads as
            # alive — the failure mode is "waits for TTL", never "two live
            # owners". `psutil` is already a hard dependency.
            live_blockers = []
            for lease in blockers:
                if (
                    lease.owner_pid
                    and lease.owner_pid != os.getpid()
                    and not psutil.pid_exists(lease.owner_pid)
                ):
                    self._conn.execute(
                        "UPDATE path_leases SET released_at = ? WHERE id = ? AND released_at IS NULL",
                        (time.time(), lease.id),
                    )
                    continue
                live_blockers.append(lease)
            if live_blockers:
                return None

            at = time.time()
            lease = Lease(
                task_id=task_id,
                repo_id=repo_id,
                path_pattern=path_pattern,
                lease_type=lease_type,
                acquired_at=at,
                expires_at=at + ttl_seconds,
                owner_pid=os.getpid(),
            )
            self._insert(lease)
            return lease

    def release(self, lease_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE path_leases SET released_at = ? WHERE id = ? AND released_at IS NULL",
                (time.time(), lease_id),
            )

    def expire_stale(self, now: float | None = None) -> int:
        """Reap leases past their TTL so a dead worker cannot block forever."""
        at = now if now is not None else time.time()
        expired = 0
        with self._conn:
            cur = self._conn.execute(
                "UPDATE path_leases SET released_at = expires_at"
                " WHERE released_at IS NULL AND expires_at <= ?",
                (at,),
            )
            expired = cur.rowcount
        return expired


__all__ = ["Lease", "LeaseStore", "LeaseType", "LEASES_SCHEMA", "patterns_overlap"]
