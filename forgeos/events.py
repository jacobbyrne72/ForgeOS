"""Event-sourced kernel. The log is the truth; state is a projection of it.

This is what makes "kill the manager, resume without redoing work" a property
rather than a hope. Materialised state is a cache — if it disagrees with the log,
the log wins and the cache is rebuilt.

Append-only on purpose: events are never updated or deleted, so an attempt's
history cannot be quietly rewritten to look better than it was.
"""

from __future__ import annotations

import json
import sqlite3
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .contracts import FailureClass, TaskState, new_id, now
from ._sqlite import connect as _sql_connect


class EventType(str, Enum):
    MISSION_CREATED = "mission_created"
    MISSION_COMPILED = "mission_compiled"
    MISSION_COMPLETED = "mission_completed"
    MISSION_RESUMED = "mission_resumed"
    TASK_CREATED = "task_created"
    TASK_READY = "task_ready"
    WORKER_ASSIGNED = "worker_assigned"
    SESSION_STARTED = "session_started"
    ACTION_RECORDED = "action_recorded"
    CHECKPOINT_CREATED = "checkpoint_created"
    TASK_BLOCKED = "task_blocked"
    FOLLOWUP_ISSUED = "followup_issued"
    WORKER_REASSIGNED = "worker_reassigned"
    VERIFICATION_STARTED = "verification_started"
    TEST_RUN_COMPLETED = "test_run_completed"
    REVIEW_COMPLETED = "review_completed"
    # A worker finished and reported success. NOT the same as accepted: the merge
    # gate has not ruled yet, and it refuses work that has no tests, no evidence,
    # a failing security scan, or no independent review.
    TASK_COMPLETED = "task_completed"
    # The merge gate allowed it. This is the only event that may be counted as
    # "accepted" — the headline cost-per-accepted-task metric divides by it, and
    # counting merely-finished work there makes the number flatter itself exactly
    # as the harness is built to prevent.
    TASK_ACCEPTED = "task_accepted"
    TASK_REJECTED = "task_rejected"
    GOVERNOR_TRIPPED = "governor_tripped"
    LEASE_ACQUIRED = "lease_acquired"
    LEASE_RELEASED = "lease_released"
    # A worker's report was rejected because its generation was not the task's
    # current one -- it was reclaimed (heartbeat expiry) and reassigned, and
    # this report is the orphaned, still-running previous attempt phoning home
    # after the fact. Deliberately absent from `_PROJECTION` below: a fenced
    # report has no effect on task state by definition, and adding it there
    # would be the exact bug this event exists to prevent, expressed in the
    # projection instead of the scheduler.
    REPORT_FENCED = "report_fenced"
    # A lifecycle hook (forgeos/hooks.py) ran for pre_route, pre_execute,
    # post_gate, or job_end. Deliberately absent from `_PROJECTION`: a hook
    # observes and, for the first two events only, may veto BEFORE this event
    # is even reached (the veto itself surfaces as a TASK_REJECTED/failed
    # outcome through the normal path). This event is a record of what the
    # hook said, never a state transition in its own right.
    HOOK_INVOKED = "hook_invoked"


# The only transitions the projection honours. An event that would move a task
# somewhere illegal is recorded but does not corrupt the projection.
_PROJECTION: dict[EventType, TaskState] = {
    EventType.TASK_CREATED: TaskState.PENDING,
    EventType.TASK_READY: TaskState.QUEUED,
    EventType.WORKER_ASSIGNED: TaskState.QUEUED,
    EventType.SESSION_STARTED: TaskState.RUNNING,
    EventType.ACTION_RECORDED: TaskState.RUNNING,
    EventType.CHECKPOINT_CREATED: TaskState.RUNNING,
    EventType.FOLLOWUP_ISSUED: TaskState.RUNNING,
    EventType.WORKER_REASSIGNED: TaskState.QUEUED,
    EventType.TASK_BLOCKED: TaskState.BLOCKED,
    EventType.VERIFICATION_STARTED: TaskState.VERIFYING,
    EventType.TEST_RUN_COMPLETED: TaskState.VERIFYING,
    EventType.REVIEW_COMPLETED: TaskState.VERIFYING,
    # Both project to DONE: the work is finished either way, and resume must not
    # redo it. What differs is whether it was *approved*, which is a question for
    # the metric, not for the state machine.
    EventType.TASK_COMPLETED: TaskState.DONE,
    EventType.TASK_ACCEPTED: TaskState.DONE,
    EventType.TASK_REJECTED: TaskState.FAILED,
    EventType.GOVERNOR_TRIPPED: TaskState.PAUSED,
}

EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    job_id TEXT NOT NULL,
    task_id TEXT,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ev_job ON event_log(job_id, seq);
CREATE INDEX IF NOT EXISTS idx_ev_task ON event_log(task_id, seq);
"""


class Event(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ev"))
    job_id: str
    type: EventType
    task_id: str | None = None
    payload: dict = Field(default_factory=dict)
    created_at: float = Field(default_factory=now)
    seq: int = 0

    @property
    def failure_class(self) -> FailureClass | None:
        fc = self.payload.get("failure_class")
        try:
            return FailureClass(fc) if fc else None
        except ValueError:
            return None


class EventLog:
    """Append-only event store with state projection."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = _sql_connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(EVENTS_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------------------------------------------------------------- append

    def append(
        self,
        job_id: str,
        type: EventType,
        *,
        task_id: str | None = None,
        **payload,
    ) -> Event:
        ev = Event(job_id=job_id, type=type, task_id=task_id, payload=payload)
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO event_log (id, job_id, task_id, type, payload, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (ev.id, ev.job_id, ev.task_id, ev.type.value, json.dumps(ev.payload), ev.created_at),
            )
            ev.seq = int(cur.lastrowid or 0)
        return ev

    # ----------------------------------------------------------------- read

    def _row_to_event(self, r: sqlite3.Row) -> Event:
        return Event(
            id=r["id"],
            job_id=r["job_id"],
            task_id=r["task_id"],
            type=EventType(r["type"]),
            payload=json.loads(r["payload"]),
            created_at=r["created_at"],
            seq=r["seq"],
        )

    def replay(self, job_id: str | None = None, after_seq: int = 0,
               limit: int | None = None) -> list[Event]:
        """Events in `seq` order, oldest first.

        `limit` returns the NEWEST `limit` events, still oldest-first. A feed
        that wants "the last 30 things that happened" would otherwise pull the
        entire history and slice the tail: the dashboard's activity panel did
        exactly that on a 2-second poll, so the work grew with total lifetime
        events forever while the answer stayed 30 rows. Ordering the tail in
        SQL and reversing keeps the cost proportional to what is returned.
        """
        sql = "SELECT * FROM event_log WHERE seq > ?"
        args: list = [after_seq]
        if job_id:
            sql += " AND job_id=?"
            args.append(job_id)
        if limit is not None and limit >= 0:
            rows = self._conn.execute(
                sql + " ORDER BY seq DESC LIMIT ?", [*args, limit]
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = self._conn.execute(sql + " ORDER BY seq", args).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count(self, job_id: str | None = None) -> int:
        """How many events exist, without materialising any of them."""
        sql = "SELECT COUNT(*) AS n FROM event_log"
        args: list = []
        if job_id:
            sql += " WHERE job_id=?"
            args.append(job_id)
        return int(self._conn.execute(sql, args).fetchone()["n"])

    def count_by_type(self, event_type: "EventType", job_id: str | None = None) -> int:
        """Count one event type in SQL.

        `_summary()` counted `TASK_ACCEPTED` by replaying every event ever
        recorded and filtering in Python, on every dashboard poll.
        """
        sql = "SELECT COUNT(*) AS n FROM event_log WHERE type=?"
        args: list = [event_type.value]
        if job_id:
            sql += " AND job_id=?"
            args.append(job_id)
        return int(self._conn.execute(sql, args).fetchone()["n"])

    def for_task(self, task_id: str) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM event_log WHERE task_id=? ORDER BY seq", (task_id,)
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def last_seq(self) -> int:
        r = self._conn.execute("SELECT COALESCE(MAX(seq),0) AS s FROM event_log").fetchone()
        return int(r["s"])

    # ------------------------------------------------------------ projection

    def project_task_states(self, job_id: str) -> dict[str, TaskState]:
        """Rebuild every task's state from the log alone.

        Terminal states are sticky: once a task is accepted or rejected, a late
        straggler event from a killed worker must not resurrect it.
        """
        states: dict[str, TaskState] = {}
        for ev in self.replay(job_id):
            if not ev.task_id:
                continue
            current = states.get(ev.task_id)
            if current in (TaskState.DONE, TaskState.FAILED):
                continue
            mapped = _PROJECTION.get(ev.type)
            if mapped is not None:
                states[ev.task_id] = mapped
        return states

    def resumable_tasks(self, job_id: str) -> list[str]:
        """Tasks that were mid-flight when the process died.

        These are what a restarted kernel picks back up — and crucially, work that
        already reached a terminal state is not redone.
        """
        return [
            tid
            for tid, st in self.project_task_states(job_id).items()
            if st in (TaskState.RUNNING, TaskState.QUEUED, TaskState.VERIFYING, TaskState.BLOCKED)
        ]

    def attempt_count(self, task_id: str) -> int:
        return sum(1 for e in self.for_task(task_id) if e.type is EventType.SESSION_STARTED)

    def failure_classes(self, task_id: str) -> list[FailureClass]:
        out = []
        for e in self.for_task(task_id):
            fc = e.failure_class
            if fc:
                out.append(fc)
        return out


__all__ = ["Event", "EventLog", "EventType", "EVENTS_SCHEMA"]
