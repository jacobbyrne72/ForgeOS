"""Operator dashboard for the forgeos token-efficiency harness.

Read-only reporting over four stores forgeos already writes to (`Ledger`,
`EventLog`, `AvoidanceLog`, `LeaseStore`) plus one write action: a manual
per-job halt flag. The halt is a flag, never a delete and never a budget
edit -- widening a budget to make a job finish is the one hard rule this
dashboard must never help anyone break (see AGENTS.md hard rule 1).

The headline number is `cost_per_accepted_task`, not raw spend and not cost
per call. A router that halves per-call cost while doubling retries makes
things worse; per-call metrics would score that as a win. "Accepted" is
`EventType.TASK_ACCEPTED` from the event log -- a task the manager actually
signed off on, not one that merely finished.

Subscription "cap burn" and API "dollars" are different currencies and are
never summed into one total (a dollar figure for subscription-routed work
would be fiction). Forge persists provider-reported quota telemetry beside the
ledger; this dashboard reads that snapshot but never probes a provider itself.

Binds 127.0.0.1 only -- see `forgeos/dashboard/__init__.py`.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse

from .._sqlite import connect as guarded_connect
from ..contracts import from_micros
from ..economy.avoidance import AvoidanceLog
from ..events import EventLog, EventType
from ..ledger import Ledger
from ..leases import LeaseStore
from ..core.quota import QUOTA_SNAPSHOT_SCHEMA, QuotaSource, QuotaTracker
from ..diagnostics import record_degradation
from ..forgebench_table import load_receipt, row_for_receipt
from ..leaderboard import build_leaderboard
from ..recovery import build_recovery
from ..registry import MIN_ATTEMPTS_TO_TRUST, default_registry

HOST = "127.0.0.1"
PORT = 8899

STATIC_DIR = Path(__file__).parent / "static"

# A same-origin request from the dashboard itself sends one of these. A native
# client (curl, a test) sends no Origin at all and is allowed; a browser on another
# site sends its own origin and is refused.
ALLOWED_WS_ORIGINS = frozenset({
    f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}",
})

LEDGER_DB = "ledger.db"
EVENTS_DB = "events.db"
AVOIDANCE_DB = "avoidance.db"
LEASES_DB = "leases.db"
HALTS_FILE = "halts.json"
QUOTA_FILE = "quota.json"
DASHBOARD_SNAPSHOT_SCHEMA = "forgeos.dashboard_snapshot.v1"

WS_POLL_SECONDS = 2.0


def default_state_dir() -> Path:
    """Where the dashboard looks for forgeos's data when no path is given.

    `FORGEOS_STATE_DIR` wins if set (matches the env-var-only-references-a-name
    convention `forgeos/settings.py` already uses); otherwise `.forgeos` under the
    current working directory, so each repo forgeos is pointed at gets its own.
    """
    return Path(os.environ.get("FORGEOS_STATE_DIR", str(Path.cwd() / ".forgeos")))


# --------------------------------------------------------------------- halts


class HaltStore:
    """Operator kill-flag store, owned entirely by the dashboard.

    A halt is not spend, not a task result, and not a row the governor's own
    tables model -- so it does not belong in `ledger.py`, and this repo's
    scope for this change is additive only (never edit an existing module).
    Plain JSON on purpose: "does not delete anything" is trivially true when
    there is nothing here to cascade-delete.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def _read(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
            return json.loads(raw) if raw else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def is_halted(self, job_id: str) -> bool:
        return bool(self._read().get(job_id, {}).get("halted", False))

    def halt(self, job_id: str, reason: str = "") -> dict[str, Any]:
        data = self._read()
        entry = {"halted": True, "at": time.time(), "reason": reason}
        data[job_id] = entry
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return entry


# ------------------------------------------------------------- row helpers


def _row(r: sqlite3.Row) -> dict[str, Any]:
    return dict(r)


def _job_dict(job: sqlite3.Row, *, ledger: Ledger, halts: HaltStore, task_count: int | None = None,
              counts_by_state: dict[str, int] | None = None) -> dict[str, Any]:
    closed_at = job["closed_at"]
    return {
        "id": job["id"],
        "objective": job["objective"],
        "cwd": job["cwd"],
        "state": job["state"],
        "created_at": job["created_at"],
        "closed_at": closed_at,
        "elapsed_seconds": (closed_at if closed_at is not None else time.time()) - job["created_at"],
        "spend_usd": from_micros(ledger.job_spend_micros(job["id"])),
        "budget": {
            "max_usd": from_micros(job["max_usd_micros"]),
            "max_seconds": job["max_seconds"],
            "max_iterations": job["max_iterations"],
        },
        "task_count": task_count,
        "task_counts_by_state": counts_by_state or {},
        "halted": halts.is_halted(job["id"]),
    }


def _task_dict(ledger: Ledger, t: sqlite3.Row) -> dict[str, Any]:
    reports = ledger.reports_for_task(t["id"])
    last = reports[-1] if reports else None
    return {
        "id": t["id"],
        "job_id": t["job_id"],
        "parent_id": t["parent_id"],
        "subject": t["subject"],
        "description": t["description"],
        "acceptance": json.loads(t["acceptance"]),
        "capabilities": json.loads(t["capabilities"]),
        "budget": json.loads(t["budget"]),
        "state": t["state"],
        "created_at": t["created_at"],
        "updated_at": t["updated_at"],
        "spend_usd": from_micros(ledger.task_spend_micros(t["id"])),
        "attempts": len(reports),
        "last_report": (
            {
                "worker_id": last["worker_id"],
                "state": last["state"],
                "confidence": last["confidence"],
                "verdict": last["verdict"],
                "spend_usd": from_micros(last["usd_micros"]),
                "seconds": last["seconds"],
                "created_at": last["created_at"],
            }
            if last is not None
            else None
        ),
    }


def _with_depth(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Indentation level for a simple parent/child DAG view.

    `tasks` only ever carries `parent_id` (the schema has no `depends_on`
    column), so a tree over `parent_id` is what is actually persisted --
    guarded against cycles defensively, since a cycle here would mean a
    broken invariant elsewhere, not something this endpoint should hang on.
    """
    by_id = {t["id"]: t for t in tasks}

    def depth(tid: str, seen: frozenset[str]) -> int:
        node = by_id.get(tid)
        if node is None or node["parent_id"] is None or tid in seen or len(seen) > 100:
            return 0
        return 1 + depth(node["parent_id"], seen | {tid})

    for t in tasks:
        t["depth"] = depth(t["id"], frozenset())
    return tasks


def _event_feed(ledger: Ledger, event_log: EventLog, job_id: str, limit: int = 300) -> list[dict[str, Any]]:
    """Merge both event stores this codebase has.

    `ledger.py` keeps its own free-form `events` table (`record_event` /
    `events_for_job`) -- the halt endpoint below writes to it, so it is the
    only place an operator-triggered halt shows up. `events.py`'s `EventLog`
    is the structured, typed event-sourced kernel (`TASK_CREATED`,
    `TASK_ACCEPTED`, ...) that `project_task_states` and the headline metric
    are built on. An operator's "event feed" is honestly both, normalised to
    one shape and interleaved by time.
    """
    combined: list[dict[str, Any]] = []
    for row in ledger.events_for_job(job_id, limit=limit):
        combined.append(
            {
                "source": "ledger",
                "kind": row["kind"],
                "detail": row["detail"],
                "task_id": row["task_id"],
                "created_at": row["created_at"],
            }
        )
    for ev in event_log.replay(job_id):
        combined.append(
            {
                "source": "kernel",
                "kind": ev.type.value,
                "detail": json.dumps(ev.payload) if ev.payload else "",
                "task_id": ev.task_id,
                "created_at": ev.created_at,
            }
        )
    combined.sort(key=lambda e: e["created_at"], reverse=True)
    return combined[:limit]


def _worker_lanes(ledger: Ledger, tasks: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """Which worker touched which tasks in this job, and at what cost."""
    lanes: dict[str, dict[str, Any]] = {}
    for t in tasks:
        for r in ledger.reports_for_task(t["id"]):
            lane = lanes.setdefault(
                r["worker_id"],
                {"worker_id": r["worker_id"], "task_ids": [], "reports": 0, "spend_usd": 0.0},
            )
            if t["id"] not in lane["task_ids"]:
                lane["task_ids"].append(t["id"])
            lane["reports"] += 1
            lane["spend_usd"] += from_micros(r["usd_micros"])
    return sorted(lanes.values(), key=lambda lane: -lane["spend_usd"])



class ChatProposal(BaseModel):
    """Body of POST /api/chat/propose. Module-level on purpose: with
    `from __future__ import annotations` in force, FastAPI resolves handler
    annotations against module globals, and a class local to `create_app` is
    invisible there -- the parameter silently became a query param."""

    objective: str
    max_usd: float = 1.0


class ChatApproval(BaseModel):
    digest: str
    dry_run: bool = True


# ------------------------------------------------------------------- app


def create_app(
    state_dir: str | Path,
    queue_dir: str | Path | None = None,
    leaderboard_dir: str | Path | None = None,
) -> FastAPI:
    """Build one dashboard instance over the stores rooted at `state_dir`.

    A factory rather than a module-level singleton so tests can point each
    instance at its own `tmp_path` -- a fresh, empty set of sqlite files with
    no shared state between tests.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    queue_dir = Path(
        queue_dir
        if queue_dir is not None
        else os.environ.get("FORGEOS_QUEUE_DIR", str(state_dir / "queue"))
    )
    leaderboard_dir = Path(
        leaderboard_dir
        if leaderboard_dir is not None
        else os.environ.get("FORGEOS_LEADERBOARD_DIR", str(state_dir / "receipts"))
    )

    ledger = Ledger(state_dir / LEDGER_DB)
    event_log = EventLog(state_dir / EVENTS_DB)
    avoidance_log = AvoidanceLog(state_dir / AVOIDANCE_DB)
    lease_store = LeaseStore(state_dir / LEASES_DB)
    halts = HaltStore(state_dir / HALTS_FILE)
    quota_path = state_dir / QUOTA_FILE

    # `Ledger` exposes `active_jobs()` (open only) and `job(id)` (one row) but
    # no "every job, open or closed" aggregate -- adding one means editing
    # ledger.py, which is out of scope for this change. This is a second
    # read-only connection to the same file Ledger already owns, used only
    # for report queries; it never writes, so it does not touch AGENTS.md
    # hard rule 2 ("never bypass the ledger"), which is about recording
    # spend outside `record_spend`, not about additional read-only listing.
    # Guarded, not raw: FastAPI runs sync endpoints on a threadpool, so several
    # requests hit this connection at once. `check_same_thread=False` alone only
    # silences the thread-affinity check — it does not make concurrent use safe,
    # and two simultaneous dashboard polls are enough to interleave cursors.
    reports_conn = guarded_connect(ledger.path)
    reports_conn.row_factory = sqlite3.Row

    def _all_jobs(limit: int = 500) -> list[sqlite3.Row]:
        return reports_conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    def _quota_view() -> dict[str, Any]:
        """Read the last local quota snapshot without making a provider call."""
        try:
            tracker = QuotaTracker.load(quota_path)
        except ValueError as exc:
            return {
                "schema": QUOTA_SNAPSHOT_SCHEMA,
                "available": False,
                "error": str(exc),
                "states": [],
                "parked_jobs": 0,
                "cap_burn": {
                    "measured": False,
                    "value_pct": None,
                    "currency": "subscription_cap",
                    "note": "quota snapshot is invalid; no cap figure inferred",
                },
            }

        states = []
        for state in tracker.states():
            remaining = state.pct_remaining
            states.append(
                {
                    "provider": state.provider,
                    "model": state.model,
                    "window": state.window.value,
                    "pct_remaining": remaining,
                    "pct_burn": round(100.0 - remaining, 3) if remaining is not None else None,
                    "resets_at": state.resets_at,
                    "seconds_until_reset": state.seconds_until_reset(),
                    "exhausted": state.exhausted,
                    "available": state.available(),
                    "source": state.source.value,
                    "observed_at": state.observed_at,
                    "detail": state.detail,
                }
            )
        measured = [
            row for row in states
            if row["source"] == QuotaSource.REPORTED.value and row["pct_burn"] is not None
        ]
        if len(measured) == 1:
            cap_burn = {
                "measured": True,
                "value_pct": measured[0]["pct_burn"],
                "currency": "subscription_cap",
                "note": "one provider window reported by the provider",
            }
        elif measured:
            cap_burn = {
                "measured": False,
                "value_pct": None,
                "currency": "subscription_cap",
                "note": "multiple provider windows present; inspect /api/quota rather than averaging them",
            }
        else:
            cap_burn = {
                "measured": False,
                "value_pct": None,
                "currency": "subscription_cap",
                "note": "no provider-reported percentage in the local quota snapshot",
            }
        return {
            "schema": QUOTA_SNAPSHOT_SCHEMA,
            "available": quota_path.exists(),
            "error": None,
            "states": states,
            "parked_jobs": len(tracker.parked()),
            "cap_burn": cap_burn,
        }

    def _summary() -> dict[str, Any]:
        all_jobs = _all_jobs()
        # Measured and modelled are reported SEPARATELY. A tier prior for a
        # flat-rate seat (kind='estimate', no tokens) used to be summed into
        # the same "$ spend" figure as provider-billed calls: measured on a
        # real job, $0.0008 of billed calls sat under one $0.06 tier prior and
        # the page read "$0.06 spend". Cost-per-accepted-task is built from the
        # MEASURED half only -- dividing a modelled number by a real task count
        # produces a per-task cost nobody was invoiced.
        splits = [ledger.job_spend_split(j["id"]) for j in all_jobs]
        measured_micros = sum(m for m, _ in splits)
        modelled_micros = sum(x for _, x in splits)
        spend_usd_total = from_micros(measured_micros)
        modelled_usd_total = from_micros(modelled_micros)
        cache = ledger.cache_stats(None)
        avoided = avoidance_log.totals(None)
        accepted = event_log.count_by_type(EventType.TASK_ACCEPTED)
        cost_per_accepted_task = (spend_usd_total / accepted) if accepted else None
        return {
            "spend_usd": spend_usd_total,
            "modelled_usd": modelled_usd_total,
            "cap_burn": _quota_view()["cap_burn"],
            "cache_hit_pct": cache["cache_hit_pct"],
            "avoided_tokens": avoided["saved_tokens"],
            "active_jobs": len(ledger.active_jobs()),
            "open_escalations": len(ledger.open_escalations(None)),
            "accepted_tasks": accepted,
            "cost_per_accepted_task": cost_per_accepted_task,
        }

    def _jobs_list() -> dict[str, Any]:
        out = []
        for job in _all_jobs():
            tasks = ledger.tasks_for_job(job["id"])
            counts: dict[str, int] = {}
            for t in tasks:
                counts[t["state"]] = counts.get(t["state"], 0) + 1
            out.append(
                _job_dict(job, ledger=ledger, halts=halts, task_count=len(tasks), counts_by_state=counts)
            )
        return {"jobs": out}

    def _workers_list() -> dict[str, Any]:
        registry = default_registry()
        capabilities = sorted({c for w in registry.all() for c in w.capabilities})

        merged: dict[str, dict[str, Any]] = {}
        for w in registry.all():
            merged[w.worker_id] = {
                "worker_id": w.worker_id,
                "adapter": w.adapter.value,
                "model": w.model,
                "agent_type": w.agent_type,
                "tier": w.tier.value,
                "capabilities": sorted(w.capabilities),
                "prior_win_rate": w.prior_win_rate,
                "attempts": 0,
                "wins": 0,
                "avg_usd_micros": 0,
                "avg_seconds": 0.0,
            }

        # One query across every capability, deduplicated by task in SQL. Summing
        # per-capability rows here counted a task tagged ["edit","python",
        # "mechanical"] as three attempts, which showed a worker at 50% when it had
        # in fact gone 1-for-1 — and the router reads these same numbers.
        for row in ledger.worker_stats(sorted(capabilities), min_attempts=1):
            wid = row["worker_id"]
            m = merged.setdefault(
                wid,
                {
                    "worker_id": wid,
                    "adapter": None,
                    "model": "",
                    "agent_type": "",
                    "tier": None,
                    "capabilities": [],
                    "prior_win_rate": None,
                    "attempts": 0,
                    "wins": 0,
                    "avg_usd_micros": 0,
                    "avg_seconds": 0.0,
                },
            )
            m["attempts"] = row["attempts"]
            m["wins"] = row["wins"]
            m["avg_usd_micros"] = row["avg_usd_micros"]
            m["avg_seconds"] = row["avg_seconds"]

        workers = []
        for m in merged.values():
            m["measured"] = m["attempts"] >= MIN_ATTEMPTS_TO_TRUST
            m["win_rate"] = (m["wins"] / m["attempts"]) if m["attempts"] else None
            m["avg_spend_usd"] = from_micros(m["avg_usd_micros"])
            workers.append(m)
        workers.sort(key=lambda w: (-w["attempts"], w["worker_id"]))
        return {"workers": workers}

    def _economy(job_id: str | None) -> dict[str, Any]:
        totals = avoidance_log.totals(job_id)
        cache = ledger.cache_stats(job_id)
        cache_health = ledger.cache_health(job_id)
        return {**totals, "cache": cache, "cache_health": cache_health}

    def _activity_feed(limit: int = 60) -> dict[str, Any]:
        """Most recent operator-visible events across every job, newest first.

        This is the cross-job counterpart to `_event_feed` above: same two
        stores, same merge-and-normalise shape, just without a job filter.
        `event_log.replay()` with no `job_id` is not a new query shape --
        `_summary()` already replays every event across every job to count
        `TASK_ACCEPTED`, so this reads nothing the dashboard doesn't already read.
        """
        combined: list[dict[str, Any]] = []
        # Bounded. This runs on a 2-second websocket poll; replaying the whole
        # event history to return `limit` rows made the work grow with total
        # lifetime events forever while the answer stayed the same size.
        for ev in event_log.replay(limit=limit):
            combined.append(
                {
                    "source": "kernel",
                    "job_id": ev.job_id,
                    "task_id": ev.task_id,
                    "kind": ev.type.value,
                    "detail": json.dumps(ev.payload) if ev.payload else "",
                    "created_at": ev.created_at,
                }
            )
        for job in _all_jobs():
            for row in ledger.events_for_job(job["id"], limit=limit):
                combined.append(
                    {
                        "source": "ledger",
                        "job_id": job["id"],
                        "task_id": row["task_id"],
                        "kind": row["kind"],
                        "detail": row["detail"],
                        "created_at": row["created_at"],
                    }
                )
        combined.sort(key=lambda e: e["created_at"], reverse=True)
        return {"events": combined[:limit]}

    def _leaderboard_view() -> dict[str, Any]:
        """Read the configured local ForgeBench receipts, never a provider.

        The dashboard must remain useful before the first benchmark exists, so
        a missing directory is an honest empty board rather than an exception.
        A malformed artifact is reported in-band and cannot take down the
        operator page or make a partial board look complete.
        """
        paths = sorted(p for p in leaderboard_dir.glob("*.json") if p.is_file())
        receipts = []
        errors = []
        for path in paths:
            try:
                receipt = load_receipt(path)
                # Validate at the file boundary. One half-written or unrelated
                # JSON artifact must not erase otherwise useful local rankings.
                row_for_receipt(path, receipt)
                receipts.append((path, receipt))
            except ValueError as exc:
                errors.append(str(exc))
        try:
            board = build_leaderboard(receipts)
        except ValueError as exc:
            errors.append(str(exc))
            board = build_leaderboard([])
        return {
            **board,
            "available": bool(receipts),
            "partial": bool(receipts and errors),
            "receipt_dir": str(leaderboard_dir),
            "errors": errors,
        }

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:  # pragma: no cover - exercised by process exit, not tests
            ledger.close()
            event_log.close()
            avoidance_log.close()
            lease_store.close()
            reports_conn.close()

    app = FastAPI(title="forgeos operator dashboard", lifespan=lifespan)

    # Binding to 127.0.0.1 does NOT make this private to the operator: the browser
    # is on the loopback interface too. Any page the operator visits while forgeos runs
    # can open ws://127.0.0.1:8899/ws and receive job objectives and absolute repo
    # paths, because a WebSocket handshake is exempt from same-origin policy and the
    # Origin header is the only thing distinguishing the dashboard from any other
    # site. TrustedHost additionally blocks DNS rebinding against the REST routes.
    # Starlette compares the hostname with the port stripped, so no port patterns.
    # "testserver" is TestClient's default Host; it is not a resolvable public name,
    # so a browser cannot be induced to send it to loopback and allowing it opens no
    # rebinding path.
    app.add_middleware(TrustedHostMiddleware,
                       allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    # ------------------------------------------------------------- pages

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # ----------------------------------------------------------- read api

    @app.get("/api/summary")
    def get_summary() -> dict[str, Any]:
        return _summary()

    @app.get("/api/quota")
    def get_quota() -> dict[str, Any]:
        return _quota_view()

    @app.get("/api/queue/status")
    def get_queue_status(stale_after_seconds: float = 30.0) -> dict[str, Any]:
        """Read queue liveness without starting Forge or contacting a provider."""
        if not math.isfinite(stale_after_seconds) or stale_after_seconds <= 0:
            raise HTTPException(status_code=422, detail="stale_after_seconds must be positive and finite")
        from ..watch import queue_status

        return queue_status(
            queue_dir, stale_after_seconds=stale_after_seconds
        ).model_dump(mode="json")

    @app.get("/api/jobs")
    def get_jobs() -> dict[str, Any]:
        return _jobs_list()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = ledger.job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")

        task_rows = ledger.tasks_for_job(job_id)
        tasks = _with_depth([_task_dict(ledger, t) for t in task_rows])
        lanes = _worker_lanes(ledger, task_rows)
        events = _event_feed(ledger, event_log, job_id)
        trips = [_row(t) for t in ledger.trips_for_job(job_id)]
        escalations = [_row(e) for e in ledger.open_escalations(job_id)]

        task_ids = {t["id"] for t in task_rows}
        active_leases = [
            lease.model_dump(mode="json") for lease in lease_store.active() if lease.task_id in task_ids
        ]

        base = _job_dict(job, ledger=ledger, halts=halts, task_count=len(task_rows))
        return {
            **base,
            "tasks": tasks,
            "worker_lanes": lanes,
            "events": events,
            "trips": trips,
            "open_escalations": escalations,
            "active_leases": active_leases,
        }

    @app.get("/api/economy")
    def get_economy(job_id: str | None = None) -> dict[str, Any]:
        return _economy(job_id)

    @app.get("/api/activity")
    def get_activity(limit: int = 60) -> dict[str, Any]:
        return _activity_feed(limit)

    @app.get("/api/leaderboard")
    def get_leaderboard() -> dict[str, Any]:
        return _leaderboard_view()

    @app.get("/api/workers")
    def get_workers() -> dict[str, Any]:
        return _workers_list()

    @app.get("/api/providers")
    def get_providers() -> dict[str, Any]:
        """Which providers are connected, and which can actually be reached.

        Two different questions, deliberately answered separately. `status` is what
        settings believes (enabled, installed, key present); `transport` is whether
        a call would have anywhere to go. They came apart in practice — a provider
        read "ready" while no transport existed for it, so an API key could be
        configured, look correct, and be silently unreachable.

        Never returns a key value. `env_key` is the variable NAME, which is what an
        operator needs in order to fix a missing key, and is not itself a secret.
        """
        from ..gateway.client import default_transports
        from ..settings import Settings

        settings = Settings.load()
        transports = default_transports(settings)
        # A transport with an empty `serves` is a fan-out gateway: it can answer
        # for any provider, so every provider counts as reachable through it.
        universal = any(not getattr(t, "serves", set()) and t.name != "litellm"
                        for t in transports)
        reachable = {t.name for t in transports}

        rows = []
        for p in sorted(settings.providers.values(), key=lambda x: (x.kind.value, x.name)):
            rows.append({
                "name": p.name,
                "kind": p.kind.value,
                "auth": p.auth.value,
                "enabled": p.enabled,
                "status": p.status(),
                "usable": p.usable,
                "env_key": p.env_key,        # the NAME only — never the value
                "base_url": p.base_url,
                "capabilities": sorted(p.capabilities),
                "has_transport": p.name in reachable or universal,
            })

        return {
            "providers": rows,
            "transports": [
                {"name": t.name, "serves": sorted(getattr(t, "serves", set())) or ["any"]}
                for t in transports
            ],
            "usable_count": sum(1 for r in rows if r["usable"]),
            "total": len(rows),
        }

    @app.get("/api/snapshot")
    def get_snapshot(stale_after_seconds: float = 30.0) -> dict[str, Any]:
        """Export one coherent, local-only dashboard observation.

        The browser normally polls several endpoints independently. Automation
        and incident capture need one versioned payload instead, so consumers
        can archive the operator view without joining responses from different
        moments. Every component below is read-only and provider-free.
        """
        if not math.isfinite(stale_after_seconds) or stale_after_seconds <= 0:
            raise HTTPException(status_code=422, detail="stale_after_seconds must be positive and finite")
        from ..watch import queue_status

        payload = {
            "schema": DASHBOARD_SNAPSHOT_SCHEMA,
            "captured_at": time.time(),
            "summary": _summary(),
            "quota": _quota_view(),
            "queue": queue_status(
                queue_dir, stale_after_seconds=stale_after_seconds
            ).model_dump(mode="json"),
            "jobs": _jobs_list(),
            "economy": _economy(None),
            "workers": _workers_list(),
            "providers": get_providers(),
            "leaderboard": _leaderboard_view(),
            "activity": _activity_feed(),
        }
        # Keep recovery guidance inside the same captured observation.  The
        # guidance is derived from these exact job/queue facts, not a second
        # set of reads that could disagree with the archived snapshot.
        payload["recovery"] = build_recovery(payload, state_dir=state_dir)
        return payload

    @app.get("/api/recovery")
    def get_recovery(stale_after_seconds: float = 30.0) -> dict[str, Any]:
        """Return provider-free next actions for unfinished local work."""
        return get_snapshot(stale_after_seconds=stale_after_seconds)["recovery"]

    # ---------------------------------------------------------- write api

    @app.post("/api/jobs/{job_id}/halt")
    def halt_job(job_id: str, reason: str = "operator requested halt via dashboard") -> dict[str, Any]:
        job = ledger.job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        entry = halts.halt(job_id, reason=reason)
        # Audit trail only -- `record_event`'s `kind` is a free-form string
        # column on the ledger's own `events` table, not the fixed `EventType`
        # enum in events.py, so this does not require touching either schema.
        ledger.record_event(job_id, "operator_halt_requested", detail=reason)
        return {"job_id": job_id, "halted": True, "at": entry["at"], "reason": entry["reason"]}

    # ------------------------------------------------------- chat / command

    # One manager conversation per dashboard process. The session object is the
    # same `ManagerSession` the CLI uses -- blueprint in, digest-gated approval,
    # job cards out -- so the browser gets the identical spend gate, not a
    # parallel weaker one.
    #
    # THE DIGEST IS ALSO THE CSRF DEFENCE. Binding to 127.0.0.1 does not make
    # these endpoints private: any web page the operator visits can POST to
    # loopback. But a cross-origin page cannot READ a response without CORS
    # headers (which are never emitted), so it can call /chat/propose blind and
    # still never learn the digest -- and /chat/run refuses without the digest
    # of the exact blueprint that was shown. Nothing spends on the strength of
    # a blind POST.
    from ..manager_chat import ApprovalError, ManagerSession, draft_blueprint

    chat_state: dict[str, Any] = {"session": None}
    chat_lock = threading.RLock()

    @app.post("/api/chat/propose")
    def chat_propose(body: ChatProposal) -> dict[str, Any]:
        objective = body.objective.strip()
        if not objective:
            raise HTTPException(status_code=422, detail="objective is empty")
        if not math.isfinite(body.max_usd) or body.max_usd <= 0 or body.max_usd > 50:
            raise HTTPException(status_code=422, detail="max_usd must be in (0, 50]")
        session = ManagerSession(objective=objective)
        blueprint = draft_blueprint(
            objective,
            plan=["compile the objective into a task graph",
                  "run it under the budget cap",
                  "verify through the merge gate"],
            criteria=[("C1", "the compiled job completes and is accepted",
                       ["ledger:job_accepted"])],
            max_usd=body.max_usd,
            write_scope=["**"],
        )
        session.propose(blueprint)
        with chat_lock:
            chat_state["session"] = session
        return {
            "rendered": blueprint.render(),
            "digest": blueprint.contract.digest()[:16],
            "phase": session.phase.value,
        }

    @app.post("/api/chat/run")
    def chat_run(body: ChatApproval) -> dict[str, Any]:
        with chat_lock:
            session = chat_state.get("session")
            if session is None:
                raise HTTPException(status_code=409, detail="nothing proposed yet")
            try:
                session.approve(body.digest)
            except ApprovalError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc

            objective = session.objective
            max_usd = session.blueprint.contract.budget.cash_limit
            # A real approval is a one-shot spend authorization. Keep the
            # session for free previews, but consume it before starting the
            # background job so a replay cannot launch a second paid run.
            if not body.dry_run:
                chat_state["session"] = None
        if body.dry_run:
            # Free preview through the same compiler the real run uses.
            from ..compiler import compile_mission

            mission = compile_mission(objective, cwd=str(Path.cwd()))
            return {
                "ran": False,
                "dry_run": True,
                "tasks": [{"id": t.id, "subject": t.subject,
                           "scope": list(t.scope.paths)} for t in mission.tasks],
            }
        # The real run is synchronous and can take minutes; hand it to a thread
        # so the event loop (and the websocket feed) stays live. The budget cap
        # comes from the APPROVED contract, never from the request body -- a
        # second request cannot widen what was approved.
        from ..__main__ import _run_team

        def _work() -> None:
            try:
                _run_team(objective, cwd=str(Path.cwd()), budget_usd=max_usd,
                          state_dir=str(state_dir), dry_run=False)
            except Exception as exc:
                record_degradation(
                    "dashboard_chat", "background run failed", exc,
                    consequence="the approved chat job did not complete; inspect the activity feed and diagnostics",
                )

        threading.Thread(target=_work, daemon=True).start()
        return {"ran": True, "dry_run": False, "budget_usd": max_usd,
                "note": "running; watch the activity feed and receipts"}

    # ------------------------------------------------------------------ ws

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        # Reject before accepting. The payload below carries job objectives and
        # absolute repo paths; a cross-origin page opening this socket would read
        # them, and no same-origin policy stops it from trying.
        origin = websocket.headers.get("origin")
        # True same-origin, not a hardcoded allowlist: the Origin's host:port
        # must equal the Host this handshake was addressed to. The old
        # `ALLOWED_WS_ORIGINS` froze PORT=8899 into the check, so serving the
        # dashboard on ANY other port rejected its own page's socket and the
        # feed sat on "reconnecting..." forever. Comparing against the
        # request's own Host is port-agnostic and still refuses every
        # cross-site page (their Origin is their site, never this Host) --
        # including a page served from a DIFFERENT loopback port, which the
        # fixed list also refused and which must stay refused: another local
        # server's page reading job objectives is the exact leak the check
        # exists to block. A native client sends no Origin and is allowed.
        if origin is not None:
            from urllib.parse import urlsplit

            host = websocket.headers.get("host", "")
            if urlsplit(origin).netloc != host:
                await websocket.close(code=1008)
                return
        await websocket.accept()
        try:
            while True:
                await websocket.send_json({"type": "summary", "data": _summary()})
                await websocket.send_json({"type": "jobs", "data": _jobs_list()})
                await websocket.send_json({"type": "activity", "data": _activity_feed(30)})
                await asyncio.sleep(WS_POLL_SECONDS)
        except WebSocketDisconnect:
            return

    return app


def main() -> None:  # pragma: no cover - manual launch path, no server started in tests
    """Entry point for `python -m forgeos.dashboard.app`.

    Deliberately does not construct the app at import time -- constructing
    `Ledger`/`EventLog`/etc. has a filesystem side effect (creating `.forgeos/`),
    and importing this module for its constants (`HOST`, `PORT`) or for
    `create_app` must never touch disk on its own.
    """
    import uvicorn

    app = create_app(default_state_dir())
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
