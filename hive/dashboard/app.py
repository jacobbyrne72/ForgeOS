"""Operator dashboard for the hive token-efficiency harness.

Read-only reporting over four stores hive already writes to (`Ledger`,
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
would be fiction). Today hive has no subscription-quota data source wired
into the ledger, so `cap_burn` is reported honestly as unmeasured rather than
invented from spend data.

Binds 127.0.0.1 only -- see `hive/dashboard/__init__.py`.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse

from .._sqlite import connect as guarded_connect
from ..contracts import from_micros
from ..economy.avoidance import AvoidanceLog
from ..events import EventLog, EventType
from ..ledger import Ledger
from ..leases import LeaseStore
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

WS_POLL_SECONDS = 2.0


def default_state_dir() -> Path:
    """Where the dashboard looks for hive's data when no path is given.

    `HIVE_STATE_DIR` wins if set (matches the env-var-only-references-a-name
    convention `hive/settings.py` already uses); otherwise `.hive` under the
    current working directory, so each repo hive is pointed at gets its own.
    """
    return Path(os.environ.get("HIVE_STATE_DIR", str(Path.cwd() / ".hive")))


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


# ------------------------------------------------------------------- app


def create_app(state_dir: str | Path) -> FastAPI:
    """Build one dashboard instance over the stores rooted at `state_dir`.

    A factory rather than a module-level singleton so tests can point each
    instance at its own `tmp_path` -- a fresh, empty set of sqlite files with
    no shared state between tests.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(state_dir / LEDGER_DB)
    event_log = EventLog(state_dir / EVENTS_DB)
    avoidance_log = AvoidanceLog(state_dir / AVOIDANCE_DB)
    lease_store = LeaseStore(state_dir / LEASES_DB)
    halts = HaltStore(state_dir / HALTS_FILE)

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

    def _summary() -> dict[str, Any]:
        all_jobs = _all_jobs()
        spend_micros_total = sum(ledger.job_spend_micros(j["id"]) for j in all_jobs)
        spend_usd_total = from_micros(spend_micros_total)
        cache = ledger.cache_stats(None)
        avoided = avoidance_log.totals(None)
        accepted = sum(1 for ev in event_log.replay() if ev.type is EventType.TASK_ACCEPTED)
        cost_per_accepted_task = (spend_usd_total / accepted) if accepted else None
        return {
            "spend_usd": spend_usd_total,
            "cap_burn": {
                "measured": False,
                "value_pct": None,
                "currency": "subscription_cap",
                "note": (
                    "no subscription-quota source is wired into the ledger yet; "
                    "intentionally not derived from spend_usd -- API dollars and "
                    "subscription cap usage are different currencies and must "
                    "never be summed"
                ),
            },
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

    app = FastAPI(title="hive operator dashboard", lifespan=lifespan)

    # Binding to 127.0.0.1 does NOT make this private to the operator: the browser
    # is on the loopback interface too. Any page the operator visits while hive runs
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

    # ------------------------------------------------------------------ ws

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        # Reject before accepting. The payload below carries job objectives and
        # absolute repo paths; a cross-origin page opening this socket would read
        # them, and no same-origin policy stops it from trying.
        origin = websocket.headers.get("origin")
        if origin is not None and origin not in ALLOWED_WS_ORIGINS:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                await websocket.send_json({"type": "summary", "data": _summary()})
                await websocket.send_json({"type": "jobs", "data": _jobs_list()})
                await asyncio.sleep(WS_POLL_SECONDS)
        except WebSocketDisconnect:
            return

    return app


def main() -> None:  # pragma: no cover - manual launch path, no server started in tests
    """Entry point for `python -m hive.dashboard.app`.

    Deliberately does not construct the app at import time -- constructing
    `Ledger`/`EventLog`/etc. has a filesystem side effect (creating `.hive/`),
    and importing this module for its constants (`HOST`, `PORT`) or for
    `create_app` must never touch disk on its own.
    """
    import uvicorn

    app = create_app(default_state_dir())
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
