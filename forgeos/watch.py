"""Two unrelated 24/7 monitoring loops share this module by name coincidence,
not by design -- `watch()` (below) already existed and is load-bearing
(`forgeos/cli.py`'s `cmd_watch`, `forgeos/__init__.py`'s re-export), so
`watch_queue()` was added alongside it under a different name rather than
overwriting either.

`watch()`: continuous cost monitoring with anomaly detection.

Reads the ledger in real time and flags when spend rate
deviates beyond expected bounds. Detects:
- Cost spikes (>3x rolling average)
- Silent burn (steady spend with zero task completion)
- Budget leaks (tasks that never complete)

`watch_queue()`: unattended job-queue daemon -- forgeos runs 24/7 without a
human watching. Poll a queue directory for job-spec JSON files, run each one
through a single long-lived `Forge` instance (`Forge.run(executor=None)`'s
own routed default: whichever backend the router picks, never a hand-rolled
executor), then file the spec -- and a receipt -- into `done/` or `failed/`.
This is the process-registry thesis applied to jobs instead of daemons:
supervision is a tiny heartbeat file an external monitor can poll, never a
transcript nobody reads until something has already gone wrong.

Three deliberate positions for `watch_queue()`:

1. **A malformed spec never crashes the loop.** It is refused straight to
   `failed/` with a `.reason.txt` sidecar explaining why, and the daemon moves
   on to the next file. The whole point of an unattended queue is that one bad
   file must not stop every job behind it.
2. **Budget is never invented.** `budget_usd` must be present in the spec.
   Falling back to `Budget`'s own pydantic default would silently let a typo
   in one queued file run at whatever cap forgeos happens to default to today
   -- the one thing a cost-governed harness must never do quietly.
3. **The halt flag is the same file the dashboard already writes.**
   `Forge._halted(job_id)` already reads `<state_dir>/halts.json` per job;
   watch mode has no single job id (every queued file becomes its own fresh
   job with a fresh, random id), so it checks the same file under a reserved
   sentinel key instead of inventing a second halt mechanism.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .contracts import Budget, TaskSpec

# Reserved job id under which an operator halts the WHOLE `watch_queue`
# daemon (as opposed to one job) in the shared halts.json -- see
# `_watch_halted`.
WATCH_HALT_KEY = "__watch__"


def watch(ledger, job_id: str | None = None, interval_seconds: int = 30, max_alerts: int = 10) -> list[dict]:
    """Monitor spend rate and return anomalies."""
    alerts: list[dict] = []
    history: list[tuple[float, float]] = []  # (timestamp, spend_usd)
    threshold_multiplier = 3.0

    start = time.time()
    last_check = start

    while len(alerts) < max_alerts:
        time.sleep(interval_seconds)
        now = time.time()
        spent = _get_spend_since(ledger, last_check, job_id=job_id)
        history.append((now, spent))

        if len(history) >= 3:
            avg = sum(h[1] for h in history[-6:]) / min(len(history), 6)
            if spent > avg * threshold_multiplier and spent > 0:
                alerts.append(
                    {
                        "type": "cost_spike",
                        "timestamp": datetime.utcnow().isoformat(),
                        "spend_this_interval": spent,
                        "rolling_average": avg,
                        "multiplier": round(spent / avg, 2),
                    }
                )
        last_check = now

    return alerts


def _get_spend_since(ledger, since_ts: float, job_id: str | None = None) -> float:
    # Use ledger methods to get spend since timestamp
    if job_id:
        rows = ledger._conn.execute(
            "SELECT SUM(usd_micros) FROM spend WHERE job_id=? AND created_at > ?",
            (job_id, since_ts),
        ).fetchone()
    else:
        rows = ledger._conn.execute(
            "SELECT SUM(usd_micros) FROM spend WHERE created_at > ?",
            (since_ts,),
        ).fetchone()
    return int(rows[0] or 0) / 1_000_000


# --------------------------------------------------------- watch_queue daemon


class WatchStats(BaseModel):
    """What one `watch_queue()` call did. Small enough to print or assert on directly."""

    jobs_done: int = 0
    jobs_failed: int = 0
    halted: bool = False


class _JobSpecError(Exception):
    """A queue file that cannot become a job. Carries the operator-facing reason."""


def _parse_job_spec(raw: object) -> tuple[str, str, Budget, list[TaskSpec]]:
    """Validate a queue file's parsed JSON against the real pydantic contracts.

    Raises `_JobSpecError` with a human-readable reason on any problem --
    never lets a `KeyError`/`TypeError` surface as an unexplained crash.
    """
    if not isinstance(raw, dict):
        raise _JobSpecError("job spec must be a JSON object")

    objective = raw.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        raise _JobSpecError("missing or empty 'objective'")

    cwd = raw.get("cwd", ".")
    if not isinstance(cwd, str):
        raise _JobSpecError("'cwd' must be a string")

    if "budget_usd" not in raw:
        raise _JobSpecError(
            "missing 'budget_usd' -- refusing rather than running with a made-up cap"
        )
    budget_kwargs: dict = {}
    try:
        budget_kwargs["max_usd"] = float(raw["budget_usd"])
        if "max_seconds" in raw:
            budget_kwargs["max_seconds"] = int(raw["max_seconds"])
        if "max_iterations" in raw:
            budget_kwargs["max_iterations"] = int(raw["max_iterations"])
        budget = Budget(**budget_kwargs)
    except (TypeError, ValueError, ValidationError) as exc:
        raise _JobSpecError(f"invalid budget: {exc}") from exc

    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list):
        raise _JobSpecError("'tasks' must be a list")

    tasks: list[TaskSpec] = []
    for i, t in enumerate(raw_tasks):
        if not isinstance(t, dict):
            raise _JobSpecError(f"tasks[{i}] must be an object")
        try:
            # job_id is overwritten the moment the job starts
            # (`Forge.run`: `for t in tasks: t.job_id = job.id`), so any value
            # here is a placeholder -- but TaskSpec requires the field.
            tasks.append(TaskSpec(**{**t, "job_id": ""}))
        except ValidationError as exc:
            raise _JobSpecError(f"tasks[{i}] invalid: {exc}") from exc

    return objective, cwd, budget, tasks


def _refuse(spec_path: Path, failed_dir: Path, reason: str) -> None:
    dest = failed_dir / spec_path.name
    shutil.move(str(spec_path), str(dest))
    (failed_dir / f"{spec_path.stem}.reason.txt").write_text(reason, encoding="utf-8")


def _write_receipt(target_dir: Path, stem: str, result) -> None:
    (target_dir / f"{stem}.receipt.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )


def _process_one(spec_path: Path, forge, done_dir: Path, failed_dir: Path, stats: WatchStats) -> None:
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _refuse(spec_path, failed_dir, f"invalid JSON: {exc}")
        stats.jobs_failed += 1
        return

    try:
        objective, cwd, budget, tasks = _parse_job_spec(raw)
    except _JobSpecError as exc:
        _refuse(spec_path, failed_dir, str(exc))
        stats.jobs_failed += 1
        return

    try:
        result = forge.run(objective, tasks, cwd=cwd, budget=budget)
    except Exception as exc:  # noqa: BLE001 - one bad job must not kill the daemon
        _refuse(spec_path, failed_dir, f"forge.run raised {exc.__class__.__name__}: {exc}")
        stats.jobs_failed += 1
        return

    # A failed job still gets a receipt -- that is the culture this harness
    # runs on (AGENTS.md rule 8: report failure faithfully).
    target_dir = done_dir if result.all_accepted else failed_dir
    _write_receipt(target_dir, spec_path.stem, result)
    shutil.move(str(spec_path), str(target_dir / spec_path.name))
    if result.all_accepted:
        stats.jobs_done += 1
    else:
        stats.jobs_failed += 1


def _watch_halted(state_dir: Path) -> bool:
    """Whether the operator asked this daemon to stop, read from the same
    `halts.json` `Forge._halted` reads and the dashboard's `HaltStore`
    writes. `watch_queue` has no single job id to key off of, so it checks a
    reserved sentinel (`WATCH_HALT_KEY`) in that same shared file. Same
    defensive parsing as `Forge._halted`: missing or corrupt means not halted.
    """
    flag = Path(state_dir) / "halts.json"
    if not flag.exists():
        return False
    try:
        data = json.loads(flag.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        return False
    entry = data.get(WATCH_HALT_KEY)
    if isinstance(entry, dict):
        return bool(entry.get("halted", True))
    return bool(entry)


def _write_heartbeat(queue_dir: Path, *, state: str, current_job: str | None, stats: WatchStats) -> None:
    payload = {
        "ts": time.time(),
        "state": state,
        "current_job": current_job,
        "jobs_done": stats.jobs_done,
        "jobs_failed": stats.jobs_failed,
    }
    (queue_dir / "heartbeat.json").write_text(json.dumps(payload), encoding="utf-8")


def watch_queue(
    queue_dir: str | Path,
    *,
    state_dir: str | Path | None = None,
    once: bool = False,
    poll_interval: float = 5.0,
    forge_factory: Callable[[], object] | None = None,
) -> WatchStats:
    """Poll `queue_dir/incoming/*.json`, run each through a long-lived `Forge`,
    file the result into `done/`/`failed/`. See the module docstring for the
    three rules this loop enforces.

    `forge_factory` is injectable so tests exercise the real file-moving,
    receipt-writing, heartbeat, and halt logic below against a fake `Forge`
    instead of a real one -- a real `Forge()` opens five sqlite stores under
    `state_dir` as a side effect of construction, which a test must not do.
    """
    from .forge import DEFAULT_HOME, Forge

    queue_dir = Path(queue_dir)
    resolved_state_dir = Path(state_dir) if state_dir else DEFAULT_HOME
    incoming = queue_dir / "incoming"
    done_dir = queue_dir / "done"
    failed_dir = queue_dir / "failed"
    for d in (incoming, done_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    forge_factory = forge_factory or (lambda: Forge(home=resolved_state_dir))
    forge = forge_factory()
    stats = WatchStats()

    try:
        while True:
            _write_heartbeat(queue_dir, state="idle", current_job=None, stats=stats)
            if _watch_halted(resolved_state_dir):
                stats.halted = True
                return stats

            pending = sorted(incoming.glob("*.json"))
            for spec_path in pending:
                if _watch_halted(resolved_state_dir):
                    stats.halted = True
                    _write_heartbeat(queue_dir, state="idle", current_job=None, stats=stats)
                    return stats
                _write_heartbeat(queue_dir, state="running", current_job=spec_path.name, stats=stats)
                _process_one(spec_path, forge, done_dir, failed_dir, stats)

            _write_heartbeat(queue_dir, state="idle", current_job=None, stats=stats)

            if once:
                return stats
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        stats.halted = True
        return stats
    finally:
        forge.close()


__all__ = ["WATCH_HALT_KEY", "WatchStats", "watch", "watch_queue"]
