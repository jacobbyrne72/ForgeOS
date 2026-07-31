"""The deterministic scheduler. Sits ABOVE the LLM manager, not below it.

This is ordinary code and it consumes zero model tokens. It owns the task queue,
the dependency graph, worker assignment, path leases, heartbeat expiry and retry
policy. The cheap LLM manager is woken only for what genuinely needs judgement:
decomposing an ambiguous task, reading a compact heartbeat, choosing to split or
escalate.

That split is the point. A manager that narrates every scheduling decision spends a
second stream of tokens roughly proportional to all worker output — supervision
ending up more expensive than the work it supervises. Every decision moved into this
file is a permanent saving rather than a one-off discount.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..contracts import Budget, TaskSpec, TaskState, WorkerReport, new_id, now
from ..events import EventLog, EventType
from ..leases import LeaseStore, LeaseType
from ..ledger import Ledger
from ..registry import Registry
from .governor import Action, Decision, Governor

# A worker silent for this long is presumed dead and its work reclaimed.
DEFAULT_HEARTBEAT_TIMEOUT = 300.0
# Leases outlive a single attempt so a brief stall does not hand the path away.
DEFAULT_LEASE_TTL = 1800.0


class Assignment(BaseModel):
    """One task handed to one worker, with the leases that make it safe."""

    id: str = Field(default_factory=lambda: new_id("asn"))
    job_id: str
    task_id: str
    worker_id: str
    lease_ids: list[str] = Field(default_factory=list)
    assigned_at: float = Field(default_factory=now)
    last_heartbeat: float = Field(default_factory=now)
    # The task's generation at the moment this assignment was made. The caller
    # that eventually builds this worker's WorkerReport must stamp it with
    # this value (WorkerReport.generation) -- that round trip is what lets
    # Scheduler.report tell a live worker's result from an orphaned one still
    # reporting under a generation the task has since moved past. The same
    # value fences Scheduler.heartbeat, so an orphan cannot refresh the
    # liveness of whichever attempt holds this task_id now.
    generation: int = 0

    def stale(self, timeout: float = DEFAULT_HEARTBEAT_TIMEOUT, at: float | None = None) -> bool:
        return ((at or now()) - self.last_heartbeat) > timeout


class Scheduler:
    def __init__(
        self,
        ledger: Ledger,
        events: EventLog,
        leases: LeaseStore,
        registry: Registry,
        governor: Governor,
        *,
        max_parallel: int = 4,
        heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT,
        lease_ttl: float = DEFAULT_LEASE_TTL,
    ):
        self.ledger = ledger
        self.events = events
        self.leases = leases
        self.registry = registry
        self.governor = governor
        self.max_parallel = max_parallel
        self.heartbeat_timeout = heartbeat_timeout
        self.lease_ttl = lease_ttl
        self._active: dict[str, Assignment] = {}

    # ------------------------------------------------------------- submit

    def submit(self, job, tasks: list[TaskSpec]) -> str:
        self.ledger.open_job(job)
        self.events.append(job.id, EventType.MISSION_CREATED, objective=job.objective)
        for t in tasks:
            self.ledger.add_task(t)
            self.events.append(job.id, EventType.TASK_CREATED, task_id=t.id, subject=t.subject)
        return job.id

    # -------------------------------------------------------------- queue

    def _task_states(self, job_id: str) -> dict[str, TaskState]:
        return {r["id"]: TaskState(r["state"]) for r in self.ledger.tasks_for_job(job_id)}

    def ready_tasks(self, job_id: str, dependencies: dict[str, list[str]] | None = None) -> list[str]:
        """Tasks whose dependencies are all DONE and which are not finished or running.

        A task whose dependency FAILED is never ready. Running it anyway would spend
        tokens building on something known to be broken.
        """
        dependencies = dependencies or {}
        states = self._task_states(job_id)
        ready: list[str] = []
        for tid, st in states.items():
            if st in (TaskState.DONE, TaskState.FAILED, TaskState.RUNNING, TaskState.PAUSED):
                continue
            deps = dependencies.get(tid, [])
            if any(states.get(d) is not TaskState.DONE for d in deps):
                continue
            ready.append(tid)
        return sorted(ready)

    def blocked_tasks(self, job_id: str, dependencies: dict[str, list[str]]) -> dict[str, str]:
        """Tasks that cannot proceed, and why — surfaced rather than silently skipped."""
        states = self._task_states(job_id)
        out: dict[str, str] = {}
        for tid, deps in dependencies.items():
            if states.get(tid) in (TaskState.DONE, TaskState.FAILED):
                continue
            failed = [d for d in deps if states.get(d) is TaskState.FAILED]
            if failed:
                out[tid] = f"dependency failed: {','.join(sorted(failed))}"
                continue
            pending = [d for d in deps if states.get(d) is not TaskState.DONE]
            if pending:
                out[tid] = f"waiting on: {','.join(sorted(pending))}"
        return out

    # ------------------------------------------------------------- assign

    @property
    def active_count(self) -> int:
        return len(self._active)

    def at_capacity(self) -> bool:
        return self.active_count >= self.max_parallel

    def assign(
        self,
        job_id: str,
        task_id: str,
        *,
        repo_id: str = "default",
        needs_file_edits: bool = True,
        worker_id: str | None = None,
    ) -> Assignment | None:
        """Assign one task: take capacity and path leases for a chosen worker.

        When `worker_id` is supplied, the caller (the Router) has already made the
        selection and this method binds leases and bookkeeping to THAT worker. It
        must not select again: two independent selections merge into one record —
        the router's tier stapled to the scheduler's worker — and every downstream
        stat is then keyed to a (worker, tier) pair that no decision ever produced.
        When `worker_id` is None the scheduler picks for itself, which keeps
        `assign_next` and direct callers working.

        Returns None when at capacity, when no worker can do it, or when another
        task already holds a conflicting write lease. Each of those is a reason to
        wait, never a reason to run the task anyway.
        """
        if self.at_capacity():
            return None

        row = self.ledger.task(task_id)
        if row is None:
            return None

        import json

        required = json.loads(row["capabilities"])
        scope = json.loads(row["scope"])
        paths: list[str] = scope.get("paths") or []

        if worker_id is None:
            stats = self._worker_stats(required)
            candidate = self.registry.pick(
                required, stats=stats, needs_file_edits=needs_file_edits and bool(paths)
            )
            if candidate is None:
                return None
            chosen = candidate.worker.worker_id
            measured, reason = candidate.measured, candidate.reason
        else:
            chosen = worker_id
            measured, reason = False, "worker supplied by caller (router decision)"

        # Take every write lease first. A partial lease set is worse than none —
        # the worker would start editing paths it does not own.
        acquired: list[str] = []
        for p in paths:
            lease = self.leases.acquire(task_id, repo_id, p, LeaseType.WRITE, self.lease_ttl)
            if lease is None:
                for lid in acquired:
                    self.leases.release(lid)
                return None
            acquired.append(lease.id)
            self.events.append(
                job_id, EventType.LEASE_ACQUIRED, task_id=task_id, path=p, worker=chosen
            )

        asn = Assignment(
            job_id=job_id,
            task_id=task_id,
            worker_id=chosen,
            lease_ids=acquired,
            generation=self.ledger.task_generation(task_id),
        )
        self._active[task_id] = asn
        self.ledger.set_task_state(task_id, TaskState.RUNNING)
        self.events.append(
            job_id,
            EventType.WORKER_ASSIGNED,
            task_id=task_id,
            worker=asn.worker_id,
            measured=measured,
            reason=reason,
        )
        self.events.append(job_id, EventType.SESSION_STARTED, task_id=task_id, worker=asn.worker_id)
        return asn

    def assign_next(self, job_id: str, dependencies: dict[str, list[str]] | None = None, **kw):
        for tid in self.ready_tasks(job_id, dependencies):
            asn = self.assign(job_id, tid, **kw)
            if asn is not None:
                return asn
        return None

    def _worker_stats(self, required: list[str]) -> dict[str, dict]:
        """Measured history per worker across the required capabilities.

        Delegates to `Ledger.worker_stats`, which deduplicates by task in SQL.
        Merging per-capability rows here instead would count a task tagged
        ["edit","python","mechanical"] as three attempts and skew the win rate the
        router depends on.
        """
        return {row["worker_id"]: row for row in self.ledger.worker_stats(required)}

    # ------------------------------------------------------------- report

    def heartbeat(
        self,
        task_id: str,
        *,
        generation: int | None = None,
        worker_id: str | None = None,
    ) -> bool:
        """Refresh an assignment's liveness, iff the caller still holds it.

        The same fence as `report`, for the same reason: a reclaimed worker is
        never killed, so it keeps heartbeating against a task_id whose slot a
        DIFFERENT attempt now occupies. Unfenced, those ticks refresh the
        replacement's stamp and a genuinely stuck attempt looks healthy
        indefinitely -- the one thing expiry exists to catch. Returns True
        when the stamp was refreshed; an orphan handed False has learned its
        attempt is over.

        Fenced against the in-memory assignment rather than the ledger, unlike
        `report`: cheaper (heartbeats are frequent, and this keeps them off
        the database) and correct, since `_active[task_id].last_heartbeat` is
        itself the value being guarded. `worker_id` is checked too -- two
        assignments can share a generation if a task is assigned twice with no
        reclaim between, and the counter alone would not separate them.

        An unstamped call is accepted, deliberately UNLIKE `report`, because
        the costs are not symmetric: an unstamped report accepted after a bump
        corrupts the ledger permanently and refusing it costs one retry, while
        an unstamped heartbeat refused after a bump leaves the current healthy
        worker no way to prove liveness -- reclaimed, reassigned, refused,
        forever -- and accepting it costs only the late reclaim this shortens.

        A fenced heartbeat is not logged: a fenced report is discarded work
        worth an operator's attention once, but a fenced heartbeat is a zombie
        on a timer, and one event per tick is the supervision cost this file
        exists to avoid.
        """
        asn = self._active.get(task_id)
        if asn is None:
            return False
        if generation is not None and generation != asn.generation:
            return False
        if worker_id is not None and worker_id != asn.worker_id:
            return False
        asn.last_heartbeat = now()
        return True

    def report(self, report: WorkerReport, *, job_id: str, budget: Budget | None = None) -> Decision:
        """Record a worker's result, then let the governor rule on it.

        Fencing happens first, before anything else in this method runs. A
        report whose generation does not match the task's current one came
        from a worker that outlived its assignment -- reclaimed by
        `expire_heartbeats` and (possibly) handed to someone else, but never
        actually killed, so it eventually calls in anyway. That report is
        recorded as a fenced event for an operator to see and then discarded:
        no state change, no lease release, no spend, no governor call. Acting
        on it even partially would mean the task's truth in the ledger is
        whichever of two attempts happened to finish last, which is exactly
        the corruption this exists to prevent.
        """
        accepted, current_generation = self.ledger.record_report_if_current(report)
        if not accepted:
            self.events.append(
                job_id,
                EventType.REPORT_FENCED,
                task_id=report.task_id,
                worker=report.worker_id,
                report_generation=report.generation,
                current_generation=current_generation,
            )
            return Decision(
                action=Action.CONTINUE,
                reason=(
                    f"fenced stale report: worker={report.worker_id!r} task={report.task_id} "
                    f"report_generation={report.generation!r} current_generation={current_generation}"
                ),
            )

        asn = self._active.get(report.task_id)

        # heartbeat() already carries task_id; the event log takes it as its own
        # column, so drop it from the payload rather than passing it twice.
        payload = {k: v for k, v in report.heartbeat().items() if k != "task_id"}
        self.events.append(
            job_id,
            EventType.CHECKPOINT_CREATED,
            task_id=report.task_id,
            **payload,
        )

        if report.state is TaskState.DONE:
            # COMPLETED, not ACCEPTED. A worker reporting success is a claim, not a
            # verdict — the merge gate still has to see tests, evidence, a clean
            # security scan and an independent review. Emitting ACCEPTED here made
            # every finished task count toward cost-per-accepted-task, including
            # ones the gate then refused, which is the self-flattering metric this
            # project exists to avoid.
            self.events.append(job_id, EventType.TASK_COMPLETED, task_id=report.task_id)
        elif report.state is TaskState.FAILED:
            self.events.append(job_id, EventType.TASK_REJECTED, task_id=report.task_id)
        elif report.state is TaskState.BLOCKED:
            self.events.append(job_id, EventType.TASK_BLOCKED, task_id=report.task_id,
                               blocker=report.blocker)

        # Any state that is not actively RUNNING releases the slot and its leases.
        # Releasing only on DONE/FAILED left a BLOCKED worker — one waiting on a
        # human approval, which the hard rules make common — holding its assignment
        # forever. On a machine whose execution capacity is 1, that pinned slot made
        # every later task fail "no capacity" and killed the whole job.
        if report.state is not TaskState.RUNNING and asn is not None:
            self._release(asn, job_id)

        decision = self.governor.check_task(report.task_id, job_id, budget or Budget())
        if decision.action is Action.TRIP:
            self.ledger.set_task_state(report.task_id, TaskState.PAUSED)
            if asn is not None:
                self._release(asn, job_id)
        return decision

    def _release(self, asn: Assignment, job_id: str) -> None:
        for lid in asn.lease_ids:
            self.leases.release(lid)
            self.events.append(job_id, EventType.LEASE_RELEASED, task_id=asn.task_id, lease=lid)
        self._active.pop(asn.task_id, None)

    # -------------------------------------------------- liveness & recovery

    def expire_heartbeats(self, at: float | None = None) -> list[str]:
        """Reclaim tasks whose worker went silent.

        The leases go back to the pool and the task returns to the queue, so a dead
        worker costs one timeout rather than blocking its paths indefinitely.
        """
        reclaimed: list[str] = []
        for task_id, asn in list(self._active.items()):
            if asn.stale(self.heartbeat_timeout, at):
                # Bump BEFORE anything else: the fence must be up before the
                # lease is even freed, let alone before any replacement worker
                # could be assigned. A late report from `asn`'s worker carries
                # `asn.generation`, which is now stale the instant this runs.
                new_generation = self.ledger.bump_generation(task_id)
                self.events.append(
                    asn.job_id, EventType.WORKER_REASSIGNED, task_id=task_id,
                    reason="heartbeat expired", worker=asn.worker_id,
                    generation=new_generation,
                )
                self._release(asn, asn.job_id)
                self.ledger.set_task_state(task_id, TaskState.QUEUED)
                reclaimed.append(task_id)
        return reclaimed

    def resume(self, job_id: str) -> list[str]:
        """Rebuild the queue from the event log after a crash.

        Work that already reached a terminal state is NOT redone — that is the whole
        return on event sourcing, and re-running an accepted task would pay twice
        for the same result.
        """
        resumable = self.events.resumable_tasks(job_id)
        for tid in resumable:
            self.ledger.set_task_state(tid, TaskState.QUEUED)
        self._active.clear()
        for lease in self.leases.active():
            if lease.task_id in resumable:
                self.leases.release(lease.id)
        return sorted(resumable)


__all__ = [
    "Assignment",
    "DEFAULT_HEARTBEAT_TIMEOUT",
    "DEFAULT_LEASE_TTL",
    "Scheduler",
]
