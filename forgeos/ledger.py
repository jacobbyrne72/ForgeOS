"""SQLite truth for jobs, tasks, spend, escalations and governor trips.

Two rules shape this file:

1. Spend is integer microdollars. A float comparison in a budget check is a cap
   that silently does not hold.
2. Cached input tokens are recorded separately from fresh ones. Cached input is
   roughly a tenth the price, so a system that cannot see its cache-hit rate
   cannot tell an optimisation from a regression. Measured, not claimed.

The router, governor and dashboard all read from here. Nothing else is truth.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from .contracts import (
    Escalation,
    EscalationKind,
    GovernorTrip,
    JobSpec,
    TaskSpec,
    TaskState,
    WorkerReport,
    new_id,
    now,
)
from ._sqlite import GuardedConnection, connect as _sql_connect

# Below this many calls in a period (baseline OR recent), a hit-rate is noise --
# one cached call out of two is a coin flip, not a signal. `cache_health`
# refuses to render a verdict for a group until both periods clear this bar,
# rather than raising a false regression alarm off n=2.
CACHE_HEALTH_MIN_CALLS = 5

# Size, in calls, of the "recent" window compared back against everything
# earlier in that same group's own history. Fixed-size so the comparison means
# the same thing early and late in a long-running job.
CACHE_HEALTH_RECENT_CALLS = 10

# A baseline hit-rate at or below this is indistinguishable from "this
# provider/model never served a cache hit" -- below it there is no working
# state to regress FROM, so the honest label is "no_cache_support", never
# "regressed to zero".
CACHE_HEALTH_FLOOR_PCT = 1.0

# Recent must fall below this fraction of baseline to count as a regression
# rather than ordinary call-to-call noise (an occasional cache miss even while
# caching is, on the whole, still working).
CACHE_HEALTH_REGRESSION_RATIO = 0.5

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    objective TEXT NOT NULL,
    cwd TEXT NOT NULL,
    state TEXT NOT NULL,
    max_usd_micros INTEGER NOT NULL,
    max_seconds INTEGER NOT NULL,
    max_iterations INTEGER NOT NULL,
    created_at REAL NOT NULL,
    closed_at REAL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    parent_id TEXT,
    subject TEXT NOT NULL,
    description TEXT NOT NULL,
    acceptance TEXT NOT NULL,
    scope TEXT NOT NULL,
    capabilities TEXT NOT NULL,
    budget TEXT NOT NULL,
    depends_on TEXT NOT NULL DEFAULT '[]',
    state TEXT NOT NULL,
    -- Bumped every time a task is reclaimed from a worker that went silent
    -- (Scheduler.expire_heartbeats), BEFORE the freed lease/slot can reach a
    -- replacement. A WorkerReport.report() call is only honoured when it
    -- carries this exact value -- see Ledger.record_report_if_current. This is
    -- what fences an orphaned-but-still-running worker out after its task has
    -- moved on: it started thread pool execution, and nothing before it
    -- stopped a reclaimed worker's late report from overwriting a result
    -- nobody was waiting on.
    generation INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_job ON tasks(job_id, state);

CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    worker_id TEXT NOT NULL,
    state TEXT NOT NULL,
    confidence REAL NOT NULL,
    summary TEXT NOT NULL,
    files_touched TEXT NOT NULL,
    commands_run TEXT NOT NULL,
    evidence TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    usd_micros INTEGER NOT NULL,
    seconds REAL NOT NULL,
    verdict TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reports_task ON reports(task_id);

-- kind: 'call' for a fresh model call, 'cache' for a cache-served call.
CREATE TABLE IF NOT EXISTS spend (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    task_id TEXT,
    worker_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    tokens_cached_in INTEGER NOT NULL,
    usd_micros INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spend_job ON spend(job_id);
CREATE INDEX IF NOT EXISTS idx_spend_task ON spend(task_id);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    task_id TEXT,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id, created_at);

CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at REAL NOT NULL,
    resolved_at REAL,
    resolution TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_esc_open ON escalations(job_id, resolved_at);

CREATE TABLE IF NOT EXISTS trips (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    task_id TEXT,
    reason TEXT NOT NULL,
    limit_desc TEXT NOT NULL,
    observed TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def _generation_accepted(report_generation: int | None, current_generation: int) -> bool:
    """Whether a report's stamped generation still authorizes it to take effect.

    `report_generation is None` means the report came from a caller that
    predates generation fencing -- it must not crash, but it also cannot PROVE
    it belongs to the live attempt, so it is trusted only when the task has
    never been reclaimed (`current_generation == 0`, the value every task is
    created with and the only value a report with no opinion on generation
    could possibly have been issued under). The moment a bump has happened,
    an unstamped report is exactly as unverifiable as a stale one and is
    fenced the same way. This is the deliberate choice: existing callers that
    never set `generation` keep working on the common, never-reclaimed path,
    while the one path this feature exists to close -- a reclaimed task's
    orphaned worker reporting late -- is fenced regardless of whether that
    orphan happens to be an old caller or a new one.
    """
    if report_generation is None:
        return current_generation == 0
    return report_generation == current_generation


class Ledger:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = _sql_connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        # Upgrade path for a `tasks` table persisted before generation fencing
        # existed. `CREATE TABLE IF NOT EXISTS` in SCHEMA above never adds a
        # column to an already-existing table, so a pre-existing on-disk ledger
        # needs this explicitly. A fresh (or already-migrated) database raises
        # "duplicate column name", which is the expected, harmless case.
        try:
            self._conn.execute("ALTER TABLE tasks ADD COLUMN generation INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute("ALTER TABLE tasks ADD COLUMN depends_on TEXT NOT NULL DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[GuardedConnection]:
        with self._conn:
            yield self._conn

    # ---------------- jobs ----------------

    def open_job(self, job: JobSpec) -> str:
        with self._tx() as c:
            c.execute(
                "INSERT INTO jobs (id, objective, cwd, state, max_usd_micros, max_seconds,"
                " max_iterations, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    job.id,
                    job.objective,
                    job.cwd,
                    TaskState.RUNNING.value,
                    job.budget.max_usd_micros,
                    job.budget.max_seconds,
                    job.budget.max_iterations,
                    job.created_at,
                ),
            )
        return job.id

    def close_job(self, job_id: str, state: TaskState) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE jobs SET state=?, closed_at=? WHERE id=?",
                (state.value, now(), job_id),
            )

    def job(self, job_id: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def reopen_job(self, job_id: str) -> None:
        """Mark a persisted job active again after a crash-safe resume."""
        with self._tx() as c:
            c.execute(
                "UPDATE jobs SET state=?, closed_at=NULL WHERE id=?",
                (TaskState.RUNNING.value, job_id),
            )

    def active_jobs(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM jobs WHERE closed_at IS NULL ORDER BY created_at DESC"
        ).fetchall()

    # ---------------- tasks ----------------

    def add_task(self, task: TaskSpec, state: TaskState = TaskState.PENDING) -> str:
        with self._tx() as c:
            c.execute(
                "INSERT INTO tasks (id, job_id, parent_id, subject, description, acceptance,"
                " scope, capabilities, budget, depends_on, state, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task.id,
                    task.job_id,
                    task.parent_id,
                    task.subject,
                    task.description,
                    json.dumps(task.acceptance),
                    task.scope.model_dump_json(),
                    json.dumps(task.capabilities),
                    task.budget.model_dump_json(),
                    json.dumps(task.depends_on),
                    state.value,
                    task.created_at,
                    task.created_at,
                ),
            )
        return task.id

    def set_task_state(self, task_id: str, state: TaskState) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE tasks SET state=?, updated_at=? WHERE id=?",
                (state.value, now(), task_id),
            )

    def task(self, task_id: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()

    def tasks_for_job(self, job_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM tasks WHERE job_id=? ORDER BY created_at", (job_id,)
        ).fetchall()

    def unfinished_tasks(self, job_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM tasks WHERE job_id=? AND state NOT IN (?,?) ORDER BY created_at",
            (job_id, TaskState.DONE.value, TaskState.FAILED.value),
        ).fetchall()

    def recent_tasks(self, limit: int = 500) -> list[sqlite3.Row]:
        """The `limit` most recently created tasks across ALL jobs, newest first.

        Read-only, and deliberately not scoped to one job: `economy.preflight`'s
        repeat-work check needs to catch a duplicate submitted from a different
        job, not just a duplicate within the same one. `limit` bounds the cost
        of every caller that scans this (a fingerprint has no stored column to
        index on -- see `economy.preflight.task_fingerprint`'s docstring for
        why) to a fixed number of row decodes regardless of how large the ledger
        has grown, at the honest cost of not finding a match older than the
        `limit`-th most recent task.
        """
        return self._conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    # ---------------- generation fencing ----------------
    # A task's generation counter starts at 0 and only moves forward. It exists
    # to fence out a worker that outlived its assignment: `bump_generation` is
    # called the moment a task is reclaimed, before any replacement worker can
    # start, and `record_report_if_current` refuses to act on a report stamped
    # with anything but the task's current generation.

    def task_generation(self, task_id: str) -> int:
        """The task's current generation, or 0 if the task is unknown.

        0 doubles as "never reclaimed" for `_generation_accepted`'s backwards
        compatibility rule, and as a harmless default for a missing task rather
        than raising -- callers that care whether the task exists already have
        `task()` for that.
        """
        row = self.task(task_id)
        return int(row["generation"]) if row is not None else 0

    def bump_generation(self, task_id: str) -> int:
        """Advance a task's generation by one and return the new value.

        Called by the scheduler at the moment a task is reclaimed from a
        worker that went silent -- BEFORE its lease is freed or its state
        returns to QUEUED, so the fence is up before a replacement worker
        could possibly exist. `generation = generation + 1` is one atomic
        UPDATE statement (SQLite evaluates it under the single write lock
        `_tx()` takes), so no separate read-modify-write guard is needed here
        the way `record_report_if_current` needs `exclusive()` below.
        """
        with self._tx() as c:
            c.execute(
                "UPDATE tasks SET generation = generation + 1, updated_at = ? WHERE id = ?",
                (now(), task_id),
            )
            row = c.execute("SELECT generation FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return int(row["generation"]) if row is not None else 0

    def record_report_if_current(self, report: WorkerReport) -> tuple[bool, int]:
        """Record a report iff its generation still matches the task's current one.

        Returns `(accepted, current_generation_at_check_time)`. A caller that
        gets `accepted=False` back MUST NOT treat this as a normal report: the
        task's state was not touched and nothing was written to `reports` --
        the scheduler is responsible for logging the rejection as its own
        event so an operator can still see that a stale worker phoned home.

        The read (current generation) and the conditional write (record or
        not) happen under one `exclusive()` acquisition on purpose: this is a
        check-then-act sequence, and checking-then-acting as two separately
        locked steps would let a `bump_generation` land in the gap between
        them -- a report that was current at the moment it was checked could
        still be recorded a moment after it stopped being current. The same
        hazard `leases.LeaseStore.acquire` had to close for lease grants.
        """
        with self._conn.exclusive():
            current = self.task_generation(report.task_id)
            if not _generation_accepted(report.generation, current):
                return False, current
            self.record_report(report)
            return True, current

    # ---------------- reports ----------------

    def record_report(self, report: WorkerReport) -> str:
        rid = new_id("rep")
        with self._tx() as c:
            c.execute(
                "INSERT INTO reports (id, task_id, worker_id, state, confidence, summary,"
                " files_touched, commands_run, evidence, tokens_in, tokens_out, usd_micros,"
                " seconds, verdict, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rid,
                    report.task_id,
                    report.worker_id,
                    report.state.value,
                    report.confidence,
                    report.summary,
                    json.dumps(report.files_touched),
                    json.dumps(report.commands_run),
                    report.evidence,
                    report.tokens_in,
                    report.tokens_out,
                    report.usd_micros,
                    report.seconds,
                    report.verdict.value if report.verdict else None,
                    report.created_at,
                ),
            )
            c.execute(
                "UPDATE tasks SET state=?, updated_at=? WHERE id=?",
                (report.state.value, now(), report.task_id),
            )
        return rid

    def reports_for_task(self, task_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM reports WHERE task_id=? ORDER BY created_at", (task_id,)
        ).fetchall()

    # ---------------- spend ----------------

    def record_spend(
        self,
        job_id: str,
        worker_id: str,
        model: str,
        usd_micros: int,
        *,
        task_id: str | None = None,
        kind: str = "call",
        tokens_in: int = 0,
        tokens_out: int = 0,
        tokens_cached_in: int = 0,
    ) -> str:
        """Record a model call. Never use a result whose spend was not recorded."""
        sid = new_id("spd")
        with self._tx() as c:
            c.execute(
                "INSERT INTO spend (id, job_id, task_id, worker_id, kind, model, tokens_in,"
                " tokens_out, tokens_cached_in, usd_micros, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    job_id,
                    task_id,
                    worker_id,
                    kind,
                    model,
                    tokens_in,
                    tokens_out,
                    tokens_cached_in,
                    usd_micros,
                    now(),
                ),
            )
        return sid

    def job_spend_micros(self, job_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(usd_micros),0) AS t FROM spend WHERE job_id=?", (job_id,)
        ).fetchone()
        return int(row["t"])

    def task_spend_micros(self, task_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(usd_micros),0) AS t FROM spend WHERE task_id=?", (task_id,)
        ).fetchone()
        return int(row["t"])

    def task_count(self, job_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE job_id=?", (job_id,)
        ).fetchone()
        return int(row["n"])

    def cache_stats(self, job_id: str | None = None) -> dict[str, int | float]:
        """Fresh vs cached input tokens. The cache-hit rate is the headline lever."""
        sql = (
            "SELECT COALESCE(SUM(tokens_in),0) AS fresh,"
            " COALESCE(SUM(tokens_cached_in),0) AS cached,"
            " COALESCE(SUM(tokens_out),0) AS out FROM spend"
        )
        args: tuple = ()
        if job_id:
            sql += " WHERE job_id=?"
            args = (job_id,)
        r = self._conn.execute(sql, args).fetchone()
        fresh, cached = int(r["fresh"]), int(r["cached"])
        total = fresh + cached
        return {
            "fresh_in": fresh,
            "cached_in": cached,
            "tokens_out": int(r["out"]),
            "cache_hit_pct": round(100.0 * cached / total, 1) if total else 0.0,
        }

    def spend_rows(self, job_id: str) -> list[sqlite3.Row]:
        """Per-call spend for a job: model, token shape, and the micros actually billed.

        The row-level granularity `job_spend_micros`/`cache_stats` sum away — a job
        that used two different models needs each call priced against ITS OWN model
        card, not the job's aggregate token counts against one card. Callers that
        need to decompose `usd_micros` by cost component (fresh input / cached
        input / output) start here rather than re-deriving this query.
        """
        sql = (
            "SELECT model, tokens_in, tokens_out, tokens_cached_in, usd_micros"
            " FROM spend WHERE job_id=? ORDER BY created_at, rowid"
        )
        return self._conn.execute(sql, (job_id,)).fetchall()

    def cache_health(self, job_id: str | None = None, *, group_by: str = "worker") -> dict:
        """Is prompt caching still working, or did it silently stop?

        `cache_stats` reports today's hit-rate but has no memory, so a real
        regression and "this model never cached in the first place" look
        identical: a low percentage. This tells them apart by comparing each
        group's RECENT calls against that SAME group's own earlier baseline --
        never a fixed cutoff, which would misfire the moment it met a
        provider/model that never had caching to lose.

        `group_by` is "worker" (default) or "model": a worker can rotate
        between models mid-job, and grouping by worker alone would dilute a
        regression tied to one specific model, so the caller picks the lens.

        Returns ``{"status": overall, "group_by": ..., "groups": [...]}``.
        Each group's ``status`` is exactly one of:

        - ``"ok"``                 recent hit rate holds up against its own baseline
        - ``"regressed"``          baseline showed real caching; recent fell hard below it
        - ``"no_cache_support"``   baseline itself is ~0 -- nothing to regress FROM
        - ``"insufficient_data"``  fewer than `CACHE_HEALTH_MIN_CALLS` in a period

        ``overall`` is ``"regressed"`` if ANY group regressed (the one
        actionable case), otherwise the least-informative honest label rather
        than a false ``"ok"``.
        """
        if group_by not in ("worker", "model"):
            raise ValueError(f"group_by must be 'worker' or 'model', got {group_by!r}")
        column = "worker_id" if group_by == "worker" else "model"

        # kind='call' excludes the tier-prior "estimate" rows an unmetered
        # subscription worker charges (forgeos/forge.py `_run_task`) -- those never
        # carry real provider usage numbers, so mixing them in would water down
        # a genuine gateway cache signal with structurally-unmeasurable zeros.
        sql = f"SELECT {column} AS grp, tokens_in, tokens_cached_in FROM spend WHERE kind='call'"
        args: tuple = ()
        if job_id:
            sql += " AND job_id=?"
            args = (job_id,)
        sql += " ORDER BY created_at, rowid"  # rowid tiebreaks calls sharing one timestamp

        by_group: dict[str, list[sqlite3.Row]] = {}
        for row in self._conn.execute(sql, args).fetchall():
            by_group.setdefault(row["grp"], []).append(row)

        def hit_pct(period: list[sqlite3.Row]) -> float | None:
            if not period:
                return None
            fresh = sum(r["tokens_in"] for r in period)
            cached = sum(r["tokens_cached_in"] for r in period)
            total = fresh + cached
            return round(100.0 * cached / total, 1) if total else 0.0

        groups = []
        for key in sorted(by_group):
            calls = by_group[key]
            n = len(calls)
            recent_n = min(CACHE_HEALTH_RECENT_CALLS, n)
            recent = calls[n - recent_n:]
            baseline = calls[: n - recent_n]

            baseline_hit_pct = hit_pct(baseline)
            recent_hit_pct = hit_pct(recent)

            if len(baseline) < CACHE_HEALTH_MIN_CALLS or len(recent) < CACHE_HEALTH_MIN_CALLS:
                status = "insufficient_data"
            elif (baseline_hit_pct or 0.0) <= CACHE_HEALTH_FLOOR_PCT:
                status = "no_cache_support"
            elif (recent_hit_pct or 0.0) < (baseline_hit_pct or 0.0) * CACHE_HEALTH_REGRESSION_RATIO:
                status = "regressed"
            else:
                status = "ok"

            groups.append({
                "group": key,
                "status": status,
                "baseline_calls": len(baseline),
                "recent_calls": len(recent),
                "baseline_hit_pct": baseline_hit_pct,
                "recent_hit_pct": recent_hit_pct,
            })

        if any(g["status"] == "regressed" for g in groups):
            overall = "regressed"
        elif not groups or all(g["status"] == "insufficient_data" for g in groups):
            overall = "insufficient_data"
        elif all(g["status"] in ("no_cache_support", "insufficient_data") for g in groups):
            overall = "no_cache_support"
        else:
            overall = "ok"

        return {"status": overall, "group_by": group_by, "groups": groups}

    # ---------------- events / escalations / trips ----------------

    def record_event(self, job_id: str, kind: str, detail: str = "", task_id: str | None = None) -> str:
        eid = new_id("evt")
        with self._tx() as c:
            c.execute(
                "INSERT INTO events (id, job_id, task_id, kind, detail, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (eid, job_id, task_id, kind, detail, now()),
            )
        return eid

    def events_for_job(self, job_id: str, limit: int = 200) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM events WHERE job_id=? ORDER BY created_at DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()

    def open_escalation(self, esc: Escalation) -> str:
        with self._tx() as c:
            c.execute(
                "INSERT INTO escalations (id, job_id, task_id, worker_id, kind, detail,"
                " created_at, resolved_at, resolution) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    esc.id,
                    esc.job_id,
                    esc.task_id,
                    esc.worker_id,
                    esc.kind.value,
                    esc.detail,
                    esc.created_at,
                    esc.resolved_at,
                    esc.resolution,
                ),
            )
        return esc.id

    def resolve_escalation(self, esc_id: str, resolution: str) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE escalations SET resolved_at=?, resolution=? WHERE id=?",
                (now(), resolution, esc_id),
            )

    def open_escalations(self, job_id: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM escalations WHERE resolved_at IS NULL"
        args: tuple = ()
        if job_id:
            sql += " AND job_id=?"
            args = (job_id,)
        return self._conn.execute(sql + " ORDER BY created_at", args).fetchall()

    def record_trip(self, trip: GovernorTrip) -> str:
        tid = new_id("trip")
        with self._tx() as c:
            c.execute(
                "INSERT INTO trips (id, job_id, task_id, reason, limit_desc, observed, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (tid, trip.job_id, trip.task_id, trip.reason, trip.limit, trip.observed, trip.created_at),
            )
        return tid

    def trips_for_job(self, job_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM trips WHERE job_id=? ORDER BY created_at", (job_id,)
        ).fetchall()

    # ---------------- aggregates the router and governor need ----------------

    def capability_stats(self, capability: str, min_attempts: int = 1) -> list[dict]:
        """Measured track record per worker for one capability.

        This is what makes routing data-driven instead of assumed. Workers are
        credited by what they actually shipped, not by what they claim.

        An attempt is one (worker, task) pair scored on its FINAL report — not one
        row per report. Counting every report would penalise a worker for checking
        in: a worker that reports BLOCKED to ask a question and then succeeds would
        score 50%, and one that heartbeats five times before shipping would score
        20%. Routing would then learn to prefer workers that stay silent, which is
        the opposite of what the escalation design depends on.
        """
        rows = self._conn.execute(
            """
            WITH final AS (
                SELECT r.worker_id AS worker_id,
                       r.task_id   AS task_id,
                       r.state     AS state,
                       r.verdict   AS verdict,
                       ROW_NUMBER() OVER (
                           PARTITION BY r.worker_id, r.task_id
                           ORDER BY r.created_at DESC, r.rowid DESC
                       ) AS rn
                FROM reports r
                JOIN tasks t ON t.id = r.task_id
                WHERE t.capabilities LIKE ?
            ),
            spend AS (
                SELECT r.worker_id AS worker_id,
                       r.task_id   AS task_id,
                       SUM(r.usd_micros) AS usd_micros,
                       SUM(r.seconds)    AS seconds
                FROM reports r
                JOIN tasks t ON t.id = r.task_id
                WHERE t.capabilities LIKE ?
                GROUP BY r.worker_id, r.task_id
            )
            SELECT f.worker_id AS worker_id,
                   COUNT(*) AS attempts,
                   SUM(CASE WHEN f.state='done' AND COALESCE(f.verdict,'pass')<>'fail'
                            THEN 1 ELSE 0 END) AS wins,
                   AVG(s.usd_micros) AS avg_usd_micros,
                   AVG(s.seconds) AS avg_seconds
            FROM final f
            JOIN spend s ON s.worker_id = f.worker_id AND s.task_id = f.task_id
            WHERE f.rn = 1
            GROUP BY f.worker_id
            HAVING attempts >= ?
            """,
            (f'%"{capability}"%', f'%"{capability}"%', min_attempts),
        ).fetchall()
        return [
            {
                "worker_id": r["worker_id"],
                "attempts": int(r["attempts"]),
                "wins": int(r["wins"]),
                "win_rate": (int(r["wins"]) / int(r["attempts"])) if r["attempts"] else 0.0,
                "avg_usd_micros": int(r["avg_usd_micros"] or 0),
                "avg_seconds": float(r["avg_seconds"] or 0.0),
            }
            for r in rows
        ]

    def worker_stats(self, capabilities: list[str], min_attempts: int = 1) -> list[dict]:
        """Track record across SEVERAL capabilities, deduplicated by task.

        Use this instead of summing `capability_stats` over a worker's capability
        list. A task tagged ["edit","python","mechanical"] matches three times, so
        naive merging counts one attempt as three and skews both the win rate and
        the average spend. Deduplication has to happen in SQL, on (worker, task),
        because the caller cannot see which tasks overlapped.
        """
        if not capabilities:
            return []

        like_clause = " OR ".join("t.capabilities LIKE ?" for _ in capabilities)
        params = [f'%"{c}"%' for c in capabilities]

        rows = self._conn.execute(
            f"""
            WITH matched AS (
                SELECT r.worker_id, r.task_id, r.state, r.verdict, r.usd_micros,
                       r.seconds, r.created_at, r.rowid AS rid
                FROM reports r
                JOIN tasks t ON t.id = r.task_id
                WHERE {like_clause}
            ),
            final AS (
                SELECT worker_id, task_id, state, verdict,
                       ROW_NUMBER() OVER (
                           PARTITION BY worker_id, task_id
                           ORDER BY created_at DESC, rid DESC
                       ) AS rn
                FROM matched
            ),
            spend AS (
                SELECT worker_id, task_id,
                       SUM(usd_micros) AS usd_micros,
                       SUM(seconds) AS seconds
                FROM matched
                GROUP BY worker_id, task_id
            )
            SELECT f.worker_id AS worker_id,
                   COUNT(*) AS attempts,
                   SUM(CASE WHEN f.state='done' AND COALESCE(f.verdict,'pass')<>'fail'
                            THEN 1 ELSE 0 END) AS wins,
                   AVG(s.usd_micros) AS avg_usd_micros,
                   AVG(s.seconds) AS avg_seconds
            FROM final f
            JOIN spend s ON s.worker_id = f.worker_id AND s.task_id = f.task_id
            WHERE f.rn = 1
            GROUP BY f.worker_id
            HAVING attempts >= ?
            """,
            (*params, min_attempts),
        ).fetchall()
        return [
            {
                "worker_id": r["worker_id"],
                "attempts": int(r["attempts"]),
                "wins": int(r["wins"]),
                "win_rate": (int(r["wins"]) / int(r["attempts"])) if r["attempts"] else 0.0,
                "avg_usd_micros": int(r["avg_usd_micros"] or 0),
                "avg_seconds": float(r["avg_seconds"] or 0.0),
            }
            for r in rows
        ]

    def file_touch_counts(self, task_id: str) -> dict[str, int]:
        """How often each file was touched. Loop-detector input."""
        counts: dict[str, int] = {}
        for row in self.reports_for_task(task_id):
            for f in json.loads(row["files_touched"]):
                counts[f] = counts.get(f, 0) + 1
        return counts

    def evidence_fingerprints(self, task_id: str) -> list[str]:
        """Ordered evidence strings, so 'churning with no result delta' is detectable."""
        return [row["evidence"] for row in self.reports_for_task(task_id)]


def open_ledger(path: str | Path) -> Ledger:
    return Ledger(path)


__all__ = ["Ledger", "open_ledger", "SCHEMA"]


def _selfcheck() -> None:  # pragma: no cover - manual smoke
    with closing(Ledger(":memory:")) as led:
        job = JobSpec(objective="smoke", cwd=".")
        led.open_job(job)
        t = TaskSpec(job_id=job.id, subject="s", description="d", capabilities=["python"])
        led.add_task(t)
        led.record_spend(job.id, "w1", "free/model", 1234, task_id=t.id, tokens_in=10, tokens_cached_in=90)
        led.open_escalation(Escalation(job_id=job.id, task_id=t.id, kind=EscalationKind.STUCK))
        print("spend:", led.job_spend_micros(job.id), "cache:", led.cache_stats(job.id))


if __name__ == "__main__":  # pragma: no cover
    _selfcheck()
