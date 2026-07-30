"""Avoided spend, measured honestly. The counterfactual ledger.

"Tokens saved" is a counterfactual: the spend that did not happen cannot be
observed, so its baseline has to be measured some other way — a tiktoken count
of the full file that was NOT read, the ledger row of the identical call that
was NOT repeated. Every row therefore carries `baseline_source`, a plain
statement of HOW its baseline was measured. A blank source is rejected at the
edge: a guessed baseline is not evidence, it is marketing.

Append-only, same shape as the event log: a saving that can be quietly edited
afterwards is a saving nobody should believe.
"""

from __future__ import annotations

import sqlite3
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from ..contracts import new_id, now


class AvoidanceMethod(str, Enum):
    """How a call was avoided outright — not how it was made cheaper."""

    CACHE_HIT = "cache_hit"          # answer served from a cache instead of a model
    PACK = "pack"                    # bounded evidence pack sent instead of a full read
    DEDUP = "dedup"                  # identical finished/in-flight work reused
    VERIFY_GATE = "verify_gate"      # skipped work already proven green
    DETERMINISTIC = "deterministic"  # plain code replaced a model call


AVOIDANCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS avoidance (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    task_id TEXT,
    method TEXT NOT NULL,
    baseline_tokens INTEGER NOT NULL CHECK (baseline_tokens >= 0),
    actual_tokens INTEGER NOT NULL CHECK (actual_tokens >= 0),
    baseline_source TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_avoid_job ON avoidance(job_id);
CREATE INDEX IF NOT EXISTS idx_avoid_method ON avoidance(method);
"""


class Avoidance(BaseModel):
    """One avoided (or reduced) call and the measured baseline it is judged against."""

    id: str = Field(default_factory=lambda: new_id("avd"))
    job_id: str
    task_id: str | None = None
    method: AvoidanceMethod
    baseline_tokens: int = Field(ge=0)
    actual_tokens: int = Field(ge=0)
    baseline_source: str
    created_at: float = Field(default_factory=now)

    @field_validator("baseline_source")
    @classmethod
    def _measured_not_guessed(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "baseline_source must say how the baseline was measured; a guessed baseline is not evidence"
            )
        return v.strip()

    @property
    def saved_tokens(self) -> int:
        """Never negative: an avoidance that cost more than its baseline saved nothing."""
        return max(0, self.baseline_tokens - self.actual_tokens)


def _pct(saved: int, baseline: int) -> float:
    return round(100.0 * saved / baseline, 2) if baseline else 0.0


class AvoidanceLog:
    """Append-only SQLite store of avoided spend."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(AVOIDANCE_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------------------------------------------------------------- record

    def record(
        self,
        job_id: str,
        method: AvoidanceMethod,
        *,
        baseline_tokens: int,
        actual_tokens: int,
        baseline_source: str,
        task_id: str | None = None,
    ) -> str:
        """Validate and persist one avoidance. Returns the row id."""
        row = Avoidance(
            job_id=job_id,
            task_id=task_id,
            method=method,
            baseline_tokens=baseline_tokens,
            actual_tokens=actual_tokens,
            baseline_source=baseline_source,
        )
        with self._conn:
            self._conn.execute(
                "INSERT INTO avoidance (id, job_id, task_id, method, baseline_tokens,"
                " actual_tokens, baseline_source, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    row.id,
                    row.job_id,
                    row.task_id,
                    row.method.value,
                    row.baseline_tokens,
                    row.actual_tokens,
                    row.baseline_source,
                    row.created_at,
                ),
            )
        return row.id

    # ----------------------------------------------------------------- read

    def _row(self, r: sqlite3.Row) -> Avoidance:
        return Avoidance(
            id=r["id"],
            job_id=r["job_id"],
            task_id=r["task_id"],
            method=AvoidanceMethod(r["method"]),
            baseline_tokens=r["baseline_tokens"],
            actual_tokens=r["actual_tokens"],
            baseline_source=r["baseline_source"],
            created_at=r["created_at"],
        )

    def for_job(self, job_id: str) -> list[Avoidance]:
        rows = self._conn.execute(
            "SELECT * FROM avoidance WHERE job_id=? ORDER BY created_at", (job_id,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def totals(self, job_id: str | None = None) -> dict:
        """Aggregate saved tokens, overall and per method.

        Savings are clamped per row BEFORE summing: one method overspending its
        baseline must not silently cancel a genuine saving elsewhere. An empty
        store returns zeros — no data is a zero, never a crash.
        """
        sql = (
            "SELECT method, COUNT(*) AS n,"
            " COALESCE(SUM(baseline_tokens),0) AS baseline,"
            " COALESCE(SUM(actual_tokens),0) AS actual,"
            " COALESCE(SUM(MAX(0, baseline_tokens - actual_tokens)),0) AS saved"
            " FROM avoidance"
        )
        args: list = []
        if job_id:
            sql += " WHERE job_id=?"
            args.append(job_id)
        rows = self._conn.execute(sql + " GROUP BY method", args).fetchall()

        by_method = {
            r["method"]: {
                "count": r["n"],
                "baseline_tokens": r["baseline"],
                "actual_tokens": r["actual"],
                "saved_tokens": r["saved"],
                "saved_pct": _pct(r["saved"], r["baseline"]),
            }
            for r in rows
        }
        baseline = sum(m["baseline_tokens"] for m in by_method.values())
        actual = sum(m["actual_tokens"] for m in by_method.values())
        saved = sum(m["saved_tokens"] for m in by_method.values())
        return {
            "baseline_tokens": baseline,
            "actual_tokens": actual,
            "saved_tokens": saved,
            "saved_pct": _pct(saved, baseline),
            "by_method": by_method,
        }


__all__ = ["AVOIDANCE_SCHEMA", "Avoidance", "AvoidanceLog", "AvoidanceMethod"]
