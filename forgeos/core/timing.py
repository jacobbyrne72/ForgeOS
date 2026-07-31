"""Latency attribution — where every second of a mission actually goes.

Without this, effort chases the wrong bar. A team can spend weeks shaving
prompt sizes while container startup or a slow test suite is consuming most
of every mission's wall clock. `SpanStore` records where time is spent;
`attribution()` rolls spans up into the single number that matters — the
bottleneck — while refusing to let any of it disappear silently:

- Time inside the job's wall-clock window that no span covers is reported as
  `unmeasured`, not folded invisibly into whichever phase happened to be
  open. An attribution that quietly sums to 100% of wall clock is lying
  about how much of the mission was ever instrumented.
- Parallel workers legitimately produce overlapping spans, so summed
  `measured` time can exceed `wall_clock`. That is surfaced as
  `parallelism_ratio`, not clamped away — clamping it would hide the one
  number that says "N workers were busy at once" instead of "something is
  wrong with the clock".
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

from ..contracts import new_id, now
from .._sqlite import connect as _sql_connect


class Phase(str, Enum):
    MODEL_REASONING = "model_reasoning"
    TOOL_CALL = "tool_call"
    TEST_RUN = "test_run"
    ENV_STARTUP = "env_startup"
    CONTEXT_RETRIEVAL = "context_retrieval"
    QUEUE_WAIT = "queue_wait"
    MANAGER = "manager"
    MERGE_WAIT = "merge_wait"
    HUMAN_APPROVAL = "human_approval"
    IDLE = "idle"


SPANS_SCHEMA = """
CREATE TABLE IF NOT EXISTS span (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    task_id TEXT,
    phase TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_span_job ON span(job_id, started_at);
CREATE INDEX IF NOT EXISTS idx_span_task ON span(task_id, started_at);
"""


class Span(BaseModel):
    id: str = Field(default_factory=lambda: new_id("span"))
    job_id: str
    task_id: str | None = None
    phase: Phase
    started_at: float = Field(default_factory=now)
    ended_at: float | None = None
    detail: str = ""

    @property
    def open(self) -> bool:
        return self.ended_at is None

    @property
    def duration(self) -> float:
        """Elapsed seconds — real if closed, elapsed-so-far if still open.

        Live wall-clock read for a still-open span (uses the real clock, not a
        caller-supplied reference time) — this is a convenience for ad hoc
        inspection. `SpanStore.attribution()` does not use this property; it
        snapshots one reference time and applies it to every open span so a
        multi-span rollup is internally consistent.
        """
        end = self.ended_at if self.ended_at is not None else now()
        return max(0.0, end - self.started_at)


class PhaseTotal(BaseModel):
    phase: Phase
    seconds: float
    pct_of_measured: float


class TaskTotal(BaseModel):
    task_id: str
    seconds: float


class Attribution(BaseModel):
    job_id: str
    wall_clock: float
    measured: float
    unmeasured: float
    parallelism_ratio: float
    bottleneck: Phase | None
    by_phase: list[PhaseTotal] = Field(default_factory=list)


class SpanStore:
    """Append-mostly span store. Same shape as `EventLog`: SQLite, WAL, Row factory.

    Spans are not events — a span is closed by updating its `ended_at` in
    place rather than appending a closing event, because a phase's duration
    is the thing being measured and a two-event join for every reader would
    just be `end - start` with extra steps. The log-of-truth property that
    matters for `EventLog` (never rewrite history) does not apply here: a
    span's `ended_at` genuinely is unknown until the phase ends.
    """

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = _sql_connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SPANS_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------------------------------------------------------------- write

    def start(
        self,
        job_id: str,
        phase: Phase,
        *,
        task_id: str | None = None,
        detail: str = "",
        started_at: float | None = None,
    ) -> Span:
        span = Span(
            job_id=job_id,
            task_id=task_id,
            phase=phase,
            detail=detail,
            started_at=started_at if started_at is not None else now(),
        )
        with self._conn:
            self._conn.execute(
                "INSERT INTO span (id, job_id, task_id, phase, started_at, ended_at, detail)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    span.id,
                    span.job_id,
                    span.task_id,
                    span.phase.value,
                    span.started_at,
                    span.ended_at,
                    span.detail,
                ),
            )
        return span

    def end(self, span_id: str, at: float | None = None) -> Span:
        end_at = at if at is not None else now()
        with self._conn:
            self._conn.execute("UPDATE span SET ended_at=? WHERE id=?", (end_at, span_id))
        return self._get(span_id)

    @contextmanager
    def measure(
        self,
        job_id: str,
        phase: Phase,
        *,
        task_id: str | None = None,
        detail: str = "",
        started_at: float | None = None,
    ) -> Iterator[Span]:
        """Open a span for the body of the `with` block, close it no matter what.

        An exception in the body must not silently lose the time it consumed —
        the `finally` closes the span before the exception propagates.
        """
        span = self.start(job_id, phase, task_id=task_id, detail=detail, started_at=started_at)
        try:
            yield span
        finally:
            self.end(span.id)

    # ----------------------------------------------------------------- read

    def _row_to_span(self, r: sqlite3.Row) -> Span:
        return Span(
            id=r["id"],
            job_id=r["job_id"],
            task_id=r["task_id"],
            phase=Phase(r["phase"]),
            started_at=r["started_at"],
            ended_at=r["ended_at"],
            detail=r["detail"],
        )

    def _get(self, span_id: str) -> Span:
        row = self._conn.execute("SELECT * FROM span WHERE id=?", (span_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such span: {span_id}")
        return self._row_to_span(row)

    def open_spans(self, job_id: str | None = None) -> list[Span]:
        sql = "SELECT * FROM span WHERE ended_at IS NULL"
        args: list = []
        if job_id is not None:
            sql += " AND job_id=?"
            args.append(job_id)
        rows = self._conn.execute(sql + " ORDER BY started_at", args).fetchall()
        return [self._row_to_span(r) for r in rows]

    def spans_for_job(self, job_id: str) -> list[Span]:
        rows = self._conn.execute(
            "SELECT * FROM span WHERE job_id=? ORDER BY started_at", (job_id,)
        ).fetchall()
        return [self._row_to_span(r) for r in rows]

    # ------------------------------------------------------------ rollups

    def attribution(self, job_id: str) -> Attribution:
        """Roll every span for `job_id` up into a wall-clock-vs-measured picture.

        `wall_clock` is the span of time covered by the spans themselves
        (earliest `started_at` to latest end) — there is no separate job
        start/end marker to anchor to, so the recorded spans are the only
        honest source for it.

        A single reference "now" is snapshotted once and used for every still-
        open span, so a rollup across many open spans is internally
        consistent (unlike calling the real clock once per span).
        """
        spans = self.spans_for_job(job_id)
        if not spans:
            return Attribution(
                job_id=job_id,
                wall_clock=0.0,
                measured=0.0,
                unmeasured=0.0,
                parallelism_ratio=0.0,
                bottleneck=None,
                by_phase=[],
            )

        reference_now = now()
        ends = [s.ended_at if s.ended_at is not None else reference_now for s in spans]
        wall_clock = max(0.0, max(ends) - min(s.started_at for s in spans))

        durations = [max(0.0, end - s.started_at) for s, end in zip(spans, ends, strict=True)]
        measured = sum(durations)

        phase_totals: dict[Phase, float] = {}
        for span, dur in zip(spans, durations, strict=True):
            phase_totals[span.phase] = phase_totals.get(span.phase, 0.0) + dur

        by_phase = [
            PhaseTotal(
                phase=phase,
                seconds=total,
                pct_of_measured=(total / measured * 100.0) if measured > 0 else 0.0,
            )
            for phase, total in phase_totals.items()
        ]
        by_phase.sort(key=lambda pt: pt.seconds, reverse=True)
        bottleneck = by_phase[0].phase if by_phase else None

        # Overlap can make measured > wall_clock (legitimate parallelism), so
        # the gap the other way — wall_clock time nothing covers — is floored
        # at zero rather than reported as a nonsensical negative "unmeasured".
        unmeasured = max(0.0, wall_clock - measured)
        # The overlap direction is the interesting one and is NOT clamped:
        # a ratio > 1 is the signal that workers ran concurrently.
        parallelism_ratio = (measured / wall_clock) if wall_clock > 0 else 0.0

        return Attribution(
            job_id=job_id,
            wall_clock=wall_clock,
            measured=measured,
            unmeasured=unmeasured,
            parallelism_ratio=parallelism_ratio,
            bottleneck=bottleneck,
            by_phase=by_phase,
        )

    def slowest_tasks(self, job_id: str, n: int = 5) -> list[TaskTotal]:
        """Tasks ranked by total span duration, worst first.

        Spans with no `task_id` (job-level phases such as queueing or manager
        overhead) are not attributable to a task and are excluded here — they
        still show up in `attribution()`'s per-phase totals.
        """
        spans = self.spans_for_job(job_id)
        reference_now = now()
        totals: dict[str, float] = {}
        for s in spans:
            if s.task_id is None:
                continue
            end = s.ended_at if s.ended_at is not None else reference_now
            totals[s.task_id] = totals.get(s.task_id, 0.0) + max(0.0, end - s.started_at)
        ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        return [TaskTotal(task_id=tid, seconds=sec) for tid, sec in ranked[:n]]


__all__ = [
    "Attribution",
    "Phase",
    "PhaseTotal",
    "Span",
    "SPANS_SCHEMA",
    "SpanStore",
    "TaskTotal",
]
