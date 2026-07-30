"""Read-only teammate-awareness board.

The problem this solves: path leases already make a path collision *impossible*,
but a worker that gets refused a lease only learns "no" — not who holds the path,
what they are doing, or whether a narrower request would succeed. A refusal with
no story behind it turns a coordination problem into a mystery, and a worker that
cannot see why it is blocked will retry, escalate, or work around the lock — all
of which cost tokens for no work done.

`TeamBoard` answers, cheaply and without mutating anything: who is working right
now, on what, holding which paths; who owns a given path; would my proposed paths
collide, and with whom; which of my candidate paths are actually free; and a
compact digest of all of it, bounded in size, safe to paste into a worker prompt.

Everything here is a query. `TeamBoard` never calls `LeaseStore.acquire`,
`.release`, or `.expire_stale` — mixing "check" with "mutate" would let a status
read silently change who owns a path, which is exactly the class of bug path
leases exist to rule out.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from ..contracts import TERMINAL_STATES, TaskState, TestResults
from ..events import EventLog, EventType
from ..leases import Lease, LeaseStore, LeaseType
from ..ledger import Ledger

# Board digest bounds — this is the token-sensitive part. A board that grows
# with team size would add cost to every prompt it is injected into, so every
# dimension (workers shown, paths per worker, goal text) is capped, and the
# digest says how much it left out rather than pretending the team is smaller
# than it is.
BOARD_MAX_WORKERS = 8
BOARD_MAX_PATHS_PER_WORKER = 3
BOARD_MAX_GOAL_CHARS = 60


class WorkerActivity(BaseModel):
    """A snapshot of one worker's in-flight task, right now."""

    worker_id: str
    task_id: str
    subject: str
    state: TaskState
    owned_paths: list[str] = Field(default_factory=list)
    started_at: float
    last_heartbeat: float
    current_goal: str = ""
    tests: TestResults | None = None
    seconds_active: float = 0.0


class PathCollision(BaseModel):
    """One requested path that is already held by someone else's task."""

    path: str
    holder_worker: str
    holder_task: str
    holder_goal: str = ""


class CollisionReport(BaseModel):
    """Answer to "would my proposed paths collide, and with whom?"."""

    collides: bool
    overlapping: list[PathCollision] = Field(default_factory=list)
    suggestion: str = ""


class TeamBoard:
    """Live, read-only view over a `LeaseStore` + `EventLog` (+ optional `Ledger`).

    `Ledger` is optional: when given, it is used as the canonical source for a
    task's `subject` (it is the durable record). When absent, the board falls
    back to the `subject` carried on the task's `TASK_CREATED` event — the
    event log alone is always enough to answer every question here.
    """

    def __init__(self, leases: LeaseStore, events: EventLog, ledger: Ledger | None = None):
        self._leases = leases
        self._events = events
        self._ledger = ledger

    # ------------------------------------------------------------- activity

    def activities(self, job_id: str, *, now: float | None = None) -> list[WorkerActivity]:
        """Who is working, on what, holding which paths, right now.

        Scoped to tasks that have actually started (a `SESSION_STARTED` event
        exists) and are not yet in a terminal state — an assigned-but-not-yet-
        running task has no heartbeat or goal to report, and a finished task is
        not a teammate to avoid colliding with.
        """
        task_ids = sorted({ev.task_id for ev in self._events.replay(job_id) if ev.task_id})
        out = []
        for task_id in task_ids:
            activity = self._activity_for_task(task_id, now=now)
            if activity is not None:
                out.append(activity)
        return out

    def who_owns(self, path: str, repo_id: str, *, now: float | None = None) -> WorkerActivity | None:
        """Which worker currently governs this concrete file path, if any."""
        lease = self._leases.holder(repo_id, path)
        if lease is None:
            return None
        return self._activity_for_task(lease.task_id, now=now)

    def would_collide(
        self,
        paths: list[str],
        repo_id: str,
        *,
        task_id: str,
        now: float | None = None,
    ) -> CollisionReport:
        """Would taking `paths` (as WRITE) collide with another task's lease?

        A blocking lease held by `task_id` itself does not count — re-checking
        your own scope should never look like a collision with yourself.
        """
        free: list[str] = []
        overlapping: list[PathCollision] = []
        for path in paths:
            blockers = [
                lease
                for lease in self._leases.conflicts(repo_id, path, LeaseType.WRITE)
                if lease.task_id != task_id
            ]
            if not blockers:
                free.append(path)
                continue
            holder_lease = self._pick_holder(blockers)
            holder = self._activity_for_task(holder_lease.task_id, now=now)
            overlapping.append(
                PathCollision(
                    path=path,
                    holder_worker=holder.worker_id if holder else "",
                    holder_task=holder_lease.task_id,
                    holder_goal=holder.current_goal if holder else "",
                )
            )
        return CollisionReport(
            collides=bool(overlapping),
            overlapping=overlapping,
            suggestion=self._collision_suggestion(free, overlapping),
        )

    def idle_paths(self, candidate_paths: list[str], repo_id: str) -> list[str]:
        """Of `candidate_paths`, which are actually free (no active lease blocks them)?

        Lets a blocked worker propose a narrower, non-colliding scope instead of
        waiting on a path it does not actually need.
        """
        return [
            path
            for path in candidate_paths
            if not self._leases.conflicts(repo_id, path, LeaseType.WRITE)
        ]

    def board(self, job_id: str, *, now: float | None = None) -> str:
        """Compact text digest of `activities(job_id)`, safe to inject into a prompt.

        Bounded on every axis that could otherwise grow with team size: number
        of workers shown, paths listed per worker, and goal text length. When
        entries are left out, the digest says how many rather than silently
        truncating.
        """
        activities = sorted(self.activities(job_id, now=now), key=lambda a: a.task_id)
        if not activities:
            return ""
        shown = activities[:BOARD_MAX_WORKERS]
        elided = len(activities) - len(shown)
        lines = [self._board_line(a) for a in shown]
        if elided > 0:
            lines.append(f"...+{elided} more teammate(s) not shown")
        return "\n".join(lines)

    # ------------------------------------------------------------- internals

    def _activity_for_task(self, task_id: str, *, now: float | None = None) -> WorkerActivity | None:
        task_events = self._events.for_task(task_id)
        if not task_events:
            return None

        session_starts = [ev for ev in task_events if ev.type is EventType.SESSION_STARTED]
        if not session_starts:
            return None  # assigned but never actually started — nothing to report yet

        job_id = task_events[-1].job_id
        state = self._events.project_task_states(job_id).get(task_id)
        if state is None or state in TERMINAL_STATES:
            return None  # finished (or somehow unknown) — not a live teammate

        worker_id = ""
        current_goal = ""
        tests: TestResults | None = None
        for ev in task_events:
            worker = ev.payload.get("worker")
            if worker:
                worker_id = worker
            goal = ev.payload.get("goal")
            if goal:
                current_goal = goal
            raw_tests = ev.payload.get("tests")
            if isinstance(raw_tests, dict):
                tests = TestResults(passed=raw_tests.get("passed", 0), failed=raw_tests.get("failed", 0))
        if not worker_id:
            return None  # no WORKER_ASSIGNED/SESSION_STARTED ever recorded a worker

        subject = self._subject_for(task_id, task_events)
        started_at = session_starts[-1].created_at
        last_heartbeat = task_events[-1].created_at
        at = now if now is not None else time.time()
        owned_paths = sorted(
            {lease.path_pattern for lease in self._leases.active() if lease.task_id == task_id}
        )

        return WorkerActivity(
            worker_id=worker_id,
            task_id=task_id,
            subject=subject,
            state=state,
            owned_paths=owned_paths,
            started_at=started_at,
            last_heartbeat=last_heartbeat,
            current_goal=current_goal,
            tests=tests,
            seconds_active=max(0.0, at - started_at),
        )

    def _subject_for(self, task_id: str, task_events: list) -> str:
        if self._ledger is not None:
            row = self._ledger.task(task_id)
            if row is not None and row["subject"]:
                return row["subject"]
        for ev in task_events:
            if ev.type is EventType.TASK_CREATED:
                return ev.payload.get("subject", "")
        return ""

    @staticmethod
    def _pick_holder(blockers: list[Lease]) -> Lease:
        """Deterministic pick when several leases block a path: earliest wins."""
        return sorted(blockers, key=lambda lease: (lease.acquired_at, lease.id))[0]

    @staticmethod
    def _collision_suggestion(free: list[str], overlapping: list[PathCollision]) -> str:
        if not overlapping:
            if not free:
                return "no paths requested"
            verb = "is" if len(free) == 1 else "are"
            return f"{', '.join(free)} {verb} free"
        parts = []
        if free:
            verb = "is" if len(free) == 1 else "are"
            parts.append(f"{', '.join(free)} {verb} free")
        for collision in overlapping:
            who = collision.holder_worker or "an unknown worker"
            goal = f" (goal: {collision.holder_goal})" if collision.holder_goal else ""
            parts.append(f"{collision.path} is held by {who} on {collision.holder_task}{goal}")
        return "; ".join(parts)

    @staticmethod
    def _board_line(activity: WorkerActivity) -> str:
        shown_paths = activity.owned_paths[:BOARD_MAX_PATHS_PER_WORKER]
        extra = len(activity.owned_paths) - len(shown_paths)
        path_str = ",".join(shown_paths) + (f"+{extra}" if extra > 0 else "")

        goal = activity.current_goal[:BOARD_MAX_GOAL_CHARS]
        if len(activity.current_goal) > BOARD_MAX_GOAL_CHARS:
            goal = goal.rstrip() + "…"

        bits = [activity.task_id, activity.worker_id, activity.state.value]
        if path_str:
            bits.append(f"owns=[{path_str}]")
        if goal:
            bits.append(f'goal="{goal}"')
        if activity.tests is not None:
            bits.append(f"tests={activity.tests.passed}p/{activity.tests.failed}f")
        return " ".join(bits)


__all__ = [
    "BOARD_MAX_GOAL_CHARS",
    "BOARD_MAX_PATHS_PER_WORKER",
    "BOARD_MAX_WORKERS",
    "CollisionReport",
    "PathCollision",
    "TeamBoard",
    "WorkerActivity",
]
