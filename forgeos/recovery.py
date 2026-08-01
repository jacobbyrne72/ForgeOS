"""Deterministic, provider-free next-action guidance for ForgeOS operators.

The dashboard already knows the facts needed to recover a crashed run: the
persisted task states, the operator halt flag, and queue heartbeat health.  This
module turns those facts into a small versioned report.  It recommends commands
but never executes them; resuming a job can spend money and therefore remains a
human decision.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping


RECOVERY_SCHEMA = "forgeos.recovery.v1"


def _command(*parts: str) -> tuple[list[str], str]:
    argv = [str(part) for part in parts]
    return argv, subprocess.list2cmdline(argv)


def _resume_action(job: Mapping[str, Any], *, state_dir: Path, unfinished: int) -> dict[str, Any]:
    job_id = str(job.get("id", ""))
    argv, command = _command(
        "python", "-m", "forgeos", "resume", job_id,
        "--state-dir", str(state_dir),
    )
    halted = bool(job.get("halted"))
    return {
        "kind": "resume_halted_job" if halted else "resume_job",
        "severity": "blocked" if halted else "attention",
        "job_id": job_id,
        "objective": str(job.get("objective", "")),
        "unfinished_tasks": unfinished,
        "reason": (
            "The job is halted; inspect the halt reason before resuming."
            if halted
            else "The persisted ledger contains unfinished tasks."
        ),
        "argv": argv,
        "command": command,
        "requires_operator_confirmation": True,
    }


def build_recovery(snapshot: Mapping[str, Any], *, state_dir: str | Path) -> dict[str, Any]:
    """Build a stable next-action report from one dashboard snapshot.

    The input is deliberately a mapping rather than a live Ledger.  Recovery
    guidance must describe the exact observation an operator exported, not a
    second set of queries taken at a different time.
    """
    state_path = Path(state_dir)
    jobs_view = snapshot.get("jobs") or {}
    jobs = jobs_view.get("jobs", []) if isinstance(jobs_view, Mapping) else []
    actions: list[dict[str, Any]] = []
    unfinished_jobs = 0
    unfinished_tasks = 0
    halted_jobs = 0

    for job in jobs if isinstance(jobs, list) else []:
        if not isinstance(job, Mapping):
            continue
        counts = job.get("task_counts_by_state") or {}
        if not isinstance(counts, Mapping):
            counts = {}
        done = int(counts.get("done", 0) or 0)
        failed = int(counts.get("failed", 0) or 0)
        total = int(job.get("task_count", 0) or 0)
        unfinished = max(0, total - done - failed)
        if job.get("halted"):
            halted_jobs += 1
        if unfinished:
            unfinished_jobs += 1
            unfinished_tasks += unfinished
            actions.append(_resume_action(job, state_dir=state_path, unfinished=unfinished))

    queue = snapshot.get("queue") or {}
    stale_queue = bool(
        isinstance(queue, Mapping)
        and queue.get("heartbeat_valid")
        and queue.get("stale")
    )
    if stale_queue:
        queue_dir = str(queue.get("queue_dir", state_path / "queue"))
        argv, command = _command(
            "python", "-m", "forgeos", "queue-status",
            "--queue", queue_dir, "--json",
        )
        actions.append({
            "kind": "inspect_stale_queue",
            "severity": "blocked" if not queue.get("owner_active") else "attention",
            "reason": (
                "The queue owner is active but its heartbeat is stale."
                if queue.get("owner_active")
                else "The queue heartbeat is stale and no owner holds the queue lock."
            ),
            "queue_dir": queue_dir,
            "argv": argv,
            "command": command,
            "requires_operator_confirmation": False,
        })

    severity_order = {"blocked": 0, "attention": 1, "info": 2}
    actions.sort(key=lambda action: (severity_order.get(str(action.get("severity")), 9), str(action.get("kind"))))
    status = "attention" if actions else "clear"
    return {
        "schema": RECOVERY_SCHEMA,
        "captured_at": snapshot.get("captured_at"),
        "state_dir": str(state_path),
        "status": status,
        "action_count": len(actions),
        "summary": {
            "unfinished_jobs": unfinished_jobs,
            "unfinished_tasks": unfinished_tasks,
            "halted_jobs": halted_jobs,
            "stale_queue": stale_queue,
        },
        "actions": actions,
        "next_action": actions[0] if actions else None,
        "note": (
            "Recommendations are provider-free and never execute automatically."
            if actions
            else "No persisted recovery action is currently indicated."
        ),
    }


__all__ = ["RECOVERY_SCHEMA", "build_recovery"]
