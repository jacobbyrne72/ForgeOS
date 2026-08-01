"""Seed a realistic job into the forgeos stores so the dashboard renders populated.

Uses only the real modules and real code paths — no fixture JSON. If the dashboard
looks right on this data, it looks right on production data, because it is the same
data path. Idempotent enough to re-run.
"""

from __future__ import annotations

import os
from pathlib import Path

from forgeos.contracts import (
    Budget,
    Escalation,
    EscalationKind,
    JobSpec,
    Scope,
    TaskSpec,
    TaskState,
    TestResults,
    Verdict,
    WorkerReport,
    to_micros,
)
from forgeos.core.governor import Governor
from forgeos.core.scheduler import Scheduler
from forgeos.economy.avoidance import AvoidanceLog, AvoidanceMethod
from forgeos.events import EventLog
from forgeos.leases import LeaseStore
from forgeos.ledger import Ledger
from forgeos.registry import default_registry

HOME = Path(os.path.expanduser("~/.forgeos"))


def main() -> int:
    HOME.mkdir(parents=True, exist_ok=True)
    led = Ledger(HOME / "ledger.db")
    ev = EventLog(HOME / "events.db")
    ls = LeaseStore(HOME / "leases.db")
    av = AvoidanceLog(HOME / "avoidance.db")

    reg = default_registry()
    gov = Governor(led, ev)
    sch = Scheduler(led, ev, ls, reg, gov, max_parallel=3)

    job = JobSpec(
        objective="Add provider failover without changing the public API",
        cwd=str(Path.cwd()),
        budget=Budget(max_usd=8.0, max_seconds=7200, max_iterations=40),
    )

    tasks = [
        TaskSpec(job_id=job.id, subject="Map retry call paths", description="Locate parse_retry_after and callers.",
                 capabilities=["search", "locate"], scope=Scope(paths=["forgeos/gateway/"]),
                 acceptance=["callers listed with file:line"]),
        TaskSpec(job_id=job.id, subject="Normalise Retry-After parsing",
                 description="Support integer seconds and HTTP-date; malformed falls back to backoff.",
                 capabilities=["edit", "python", "mechanical"], scope=Scope(paths=["forgeos/gateway/health.py"]),
                 acceptance=["pytest tests/test_gateway.py -q passes"], budget=Budget(max_usd=2.0)),
        TaskSpec(job_id=job.id, subject="Regression test for malformed values",
                 description="Add one test per malformed shape.", capabilities=["test", "python"],
                 scope=Scope(paths=["tests/test_gateway.py"]),
                 acceptance=["new test fails before fix, passes after"], budget=Budget(max_usd=1.5)),
        TaskSpec(job_id=job.id, subject="Independent review of the diff",
                 description="Review diff + evidence only. No implementer transcript.",
                 capabilities=["review"], acceptance=["no unresolved risk"], budget=Budget(max_usd=1.0)),
    ]

    sch.submit(job, tasks)
    scout, impl, test, review = tasks

    # --- scout finishes cheaply on a free local worker -------------------
    a = sch.assign(job.id, scout.id, needs_file_edits=False)
    led.record_spend(job.id, a.worker_id if a else "omc.explore", "auto:free", 0,
                     task_id=scout.id, tokens_in=1800, tokens_cached_in=6200, tokens_out=340)
    sch.report(WorkerReport(task_id=scout.id, worker_id=a.worker_id if a else "omc.explore",
                            state=TaskState.DONE, verdict=Verdict.PASS, confidence=0.9,
                            goal="Map retry call paths",
                            files_touched=[], commands_run=["rg -n parse_retry_after"],
                            evidence="3 callers found", tokens_in=1800, tokens_out=340,
                            usd_micros=0, seconds=11.0), job_id=job.id)
    av.record(job_id=job.id, task_id=scout.id, method=AvoidanceMethod.PACK,
              baseline_tokens=48120, actual_tokens=6400,
              baseline_source="tiktoken count of full forgeos/gateway/ tree")

    # --- implementer: one failed attempt, then green ---------------------
    b = sch.assign(job.id, impl.id)
    wid = b.worker_id if b else "omc.executor"
    led.record_spend(job.id, wid, "claude/sonnet", to_micros(0.42), task_id=impl.id,
                     tokens_in=9200, tokens_cached_in=21400, tokens_out=2600)
    sch.report(WorkerReport(task_id=impl.id, worker_id=wid, state=TaskState.BLOCKED,
                            confidence=0.55, goal="Normalise Retry-After parsing",
                            files_touched=["forgeos/gateway/health.py"],
                            commands_run=["pytest tests/test_gateway.py -q"],
                            evidence="1 failed, 12 passed",
                            tests=TestResults(passed=12, failed=1),
                            blocker="HTTP-date rounds down by one second",
                            next_action="inspect clock rounding", needs_manager=True,
                            escalations=[EscalationKind.LOW_CONFIDENCE],
                            tokens_in=9200, tokens_out=2600, usd_micros=to_micros(0.42),
                            seconds=196.0), job_id=job.id, budget=impl.budget)
    esc = Escalation(job_id=job.id, task_id=impl.id, worker_id=wid,
                     kind=EscalationKind.LOW_CONFIDENCE,
                     detail="HTTP-date rounding off by one; requested normalisation rule")
    led.open_escalation(esc)

    sch.assign(job.id, impl.id)
    led.record_spend(job.id, wid, "claude/sonnet", to_micros(0.31), task_id=impl.id,
                     tokens_in=3100, tokens_cached_in=28800, tokens_out=1500)
    sch.report(WorkerReport(task_id=impl.id, worker_id=wid, state=TaskState.DONE,
                            verdict=Verdict.PASS, confidence=0.92,
                            goal="Normalise Retry-After parsing",
                            files_touched=["forgeos/gateway/health.py"],
                            commands_run=["pytest tests/test_gateway.py -q"],
                            evidence="13 passed", tests=TestResults(passed=13, failed=0),
                            tokens_in=3100, tokens_out=1500, usd_micros=to_micros(0.31),
                            seconds=142.0), job_id=job.id, budget=impl.budget)
    led.resolve_escalation(esc.id, "normalisation rule confirmed: floor to whole seconds")
    av.record(job_id=job.id, task_id=impl.id, method=AvoidanceMethod.CACHE_HIT,
              baseline_tokens=28800, actual_tokens=3100,
              baseline_source="measured cached-prefix tokens reported by provider")

    # --- test task on the free local worker ------------------------------
    c = sch.assign(job.id, test.id)
    twid = c.worker_id if c else "omc.test-engineer"
    led.record_spend(job.id, twid, "jcode-0.58", 0, task_id=test.id,
                     tokens_in=900, tokens_cached_in=4100, tokens_out=620)
    sch.report(WorkerReport(task_id=test.id, worker_id=twid, state=TaskState.DONE,
                            verdict=Verdict.PASS, confidence=0.88,
                            goal="Regression test for malformed values",
                            files_touched=["tests/test_gateway.py"],
                            commands_run=["pytest tests/test_gateway.py -q"],
                            evidence="16 passed", tests=TestResults(passed=16, failed=0),
                            tokens_in=900, tokens_out=620, usd_micros=0, seconds=38.0),
                job_id=job.id)
    av.record(job_id=job.id, task_id=test.id, method=AvoidanceMethod.DETERMINISTIC,
              baseline_tokens=12000, actual_tokens=0,
              baseline_source="tiktoken count of full pytest log that a model would have read")

    # --- reviewer: different family from the implementer -----------------
    d = sch.assign(job.id, review.id, needs_file_edits=False)
    rwid = d.worker_id if d else "omc.code-reviewer"
    led.record_spend(job.id, rwid, "auto:free", 0, task_id=review.id,
                     tokens_in=2400, tokens_cached_in=5100, tokens_out=480)
    sch.report(WorkerReport(task_id=review.id, worker_id=rwid, state=TaskState.DONE,
                            verdict=Verdict.PASS, confidence=0.85, goal="Independent review",
                            commands_run=["git diff --stat"], evidence="no unresolved risk",
                            tokens_in=2400, tokens_out=480, usd_micros=0, seconds=64.0),
                job_id=job.id)

    led.close_job(job.id, TaskState.DONE)

    print(f"job        : {job.id}")
    print(f"spend      : ${led.job_spend_micros(job.id)/1e6:.4f}")
    print(f"cache      : {led.cache_stats(job.id)}")
    print(f"avoidance  : {av.totals(job.id)}")
    print(f"escalations: {len(led.open_escalations(job.id))} open")
    print(f"db dir     : {HOME}")
    for s in (led, ev, ls, av):
        try:
            s.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
