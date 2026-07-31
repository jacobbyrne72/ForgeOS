"""`python -m forgeos` — a small, read-only diagnostics entry point.

Two questions an operator asks before and after spending anything:

- **`doctor`**: what could actually run right now? Wraps `Forge.doctor()`
  (machine class, resource pools, provider readiness, capacity-market
  report) with `adapters.factory.runnable_workers()` — which constructs the
  REAL adapter for every registry profile and calls its `.health()`, the
  same ground truth the router itself acts on, not a second and weaker
  approximation of it — plus the price catalog's own staleness accounting
  (`ModelCard.age_days`, `Catalog.stale()`). Every check here is local:
  `shutil.which`, a file on PATH, a local subprocess talking to a CLI or
  daemon already on this machine (e.g. `ollama list`), a JSON cache on
  disk. Nothing here calls out to a remote model provider — that
  reachability probe against real, billed endpoints is `tools/live_check.py`'s
  job, not this one's. `doctor` also prints a **Degradations** section: subsystem
  failures caught and swallowed elsewhere in forgeos this process (see
  `forgeos.diagnostics`) that changed behaviour invisibly -- e.g. a smaller
  fleet than configured, or pool sizes computed from defaults because
  hardware detection failed. This is the one place an operator can see that
  "no error" did not mean "nothing went wrong".

- **`receipts`**: what did it cost? A read-only summary of the ledger
  already on disk — spend by job, spend by worker, cost per accepted task —
  using the same `cost_per_accepted` definition `ForgeResult` reports
  (spend / count of tasks that reached `TaskState.DONE`).

Deliberately a second, narrower entry point next to `forgeos/cli.py`
(invoked as `python -m forgeos.cli ...`), not a replacement for it: this
file reuses `Forge.doctor()` / `runnable_workers()` / `Catalog.stale()`
rather than duplicating their logic, per AGENTS.md's "extend, don't
parallel-build".

Every import below is deferred into the command function that needs it,
matching `cli.py`'s existing style — it keeps `python -m forgeos --help`
fast and makes each command trivial to monkeypatch in isolation from tests.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Callable
from pathlib import Path


def _resolve_state_dir(explicit: str | None) -> Path:
    """Where forgeos's ledger lives: `--state-dir` if given, else the same
    `DEFAULT_HOME` (`Path.home() / ".forgeos"`) that a bare `Forge()` writes
    to — so `receipts` reports on the state a normal run actually produced,
    not a different convention this file invents on its own.
    """
    from forgeos.forge import DEFAULT_HOME

    return Path(explicit) if explicit else DEFAULT_HOME


# ------------------------------------------------------------------- doctor


def _print_worker_runnability(workers) -> None:
    """What would actually execute if the router picked it — not what the
    registry merely lists. `runnable_workers` builds the real adapter for
    each profile and calls its `.health()`, so this is the router's own
    ground truth, reused rather than re-derived from `Settings` alone.
    """
    from forgeos.adapters.factory import runnable_workers

    results = runnable_workers(workers)
    ok = sum(1 for v in results.values() if v.startswith("ok"))
    print(f"Registry: {len(workers)} worker(s)")
    for worker_id in sorted(results):
        print(f"  {worker_id:<24} {results[worker_id]}")
    print(f"Runnable now: {ok}/{len(workers)}")


def _print_catalog_staleness(catalog) -> None:
    cards = catalog.all()
    if not cards:
        print("Price catalog: empty (no local cache found)")
        return
    stale = catalog.stale()
    ages = [c.age_days for c in cards if c.age_days is not None]
    print(f"Price catalog: {len(cards)} model(s), {len(stale)} stale or unstamped (>30d)")
    if ages:
        print(f"  oldest stamped price: {max(ages):.1f} day(s) old")
    else:
        print("  staleness: unknown (no stamped price data found)")


def _print_degradations() -> None:
    """What silently went wrong elsewhere this process -- see `forgeos.diagnostics`.

    This is the whole point of the module it reads from: without this section,
    a smaller-than-configured fleet or a hardware profile built from defaults
    would be invisible to the operator. An empty list is reported as plainly as
    a non-empty one -- "none recorded" is a claim, not silence.
    """
    from forgeos.diagnostics import degradations

    degs = degradations()
    print("Degradations")
    if not degs:
        print("  none recorded this process")
        return
    for d in degs:
        suffix = f" (x{d.count})" if d.count > 1 else ""
        print(f"  [{d.subsystem}] {d.consequence}{suffix}")


def cmd_doctor(args: argparse.Namespace) -> int:
    from forgeos import Forge
    from forgeos.catalog import default_catalog

    try:
        forge = Forge()
    except OSError as exc:
        print(f"Cannot use forgeos's home directory: {exc}")
        print("Fix: check permissions on that path, or set HOME to a writable directory.")
        return 1

    try:
        print(forge.doctor())
        print()
        _print_worker_runnability(forge.registry.all())
        print()
        _print_catalog_staleness(default_catalog())
        print()
        _print_degradations()
    finally:
        forge.close()
    return 0


# ------------------------------------------------------------------ receipts


def cmd_receipts(args: argparse.Namespace) -> int:
    from forgeos.contracts import TaskState

    state_dir = _resolve_state_dir(args.state_dir)
    ledger_path = state_dir / "ledger.db"
    if not ledger_path.exists():
        print(f"No ledger at {ledger_path}.")
        print("Fix: pass --state-dir to a directory forgeos has run a job in, or run one first.")
        return 1

    from forgeos.ledger import Ledger

    try:
        ledger = Ledger(ledger_path)
    except (OSError, sqlite3.Error) as exc:
        print(f"Cannot open ledger at {ledger_path}: {exc}")
        print("Fix: check the file is a valid forgeos ledger and this process can read it.")
        return 1

    try:
        # `Ledger` exposes `active_jobs()` (open jobs only) but no accessor for
        # closed ones, and receipts exists specifically to review finished
        # work. `_conn` is the same private-access pattern `cli.py`'s
        # `cmd_adapt` already uses for a query the public Ledger API doesn't
        # cover; adding one is out of this change's scope (forgeos/ledger.py
        # is not in scope for this task).
        jobs = ledger._conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        if not jobs:
            print("Ledger has no jobs recorded.")
            return 0

        total_spend_micros = 0
        total_accepted = 0
        print(f"Jobs: {len(jobs)}")
        print("-" * 72)
        for job in jobs:
            spend = ledger.job_spend_micros(job["id"])
            tasks = ledger.tasks_for_job(job["id"])
            done = sum(1 for t in tasks if t["state"] == TaskState.DONE.value)
            failed = sum(1 for t in tasks if t["state"] == TaskState.FAILED.value)
            per_accepted = f"${spend / done / 1e6:.4f}" if done else "n/a"
            status = job["state"] + (" (closed)" if job["closed_at"] is not None else "")
            print(f"  [{job['id'][:16]}] {job['objective'][:44]}")
            print(
                f"    state={status}  tasks={len(tasks)} done={done} failed={failed}"
                f"  spend=${spend / 1e6:.4f}  $/accepted={per_accepted}"
            )
            total_spend_micros += spend
            total_accepted += done

        by_worker = ledger._conn.execute(
            "SELECT worker_id, COUNT(*) AS calls, COALESCE(SUM(usd_micros),0) AS micros"
            " FROM spend GROUP BY worker_id ORDER BY micros DESC"
        ).fetchall()
        if by_worker:
            print()
            print("Spend by worker:")
            for row in by_worker:
                print(f"  {row['worker_id']:<28} {row['calls']:>4} call(s)  ${row['micros'] / 1e6:.4f}")

        print()
        overall = f"${total_spend_micros / total_accepted / 1e6:.4f}" if total_accepted else "n/a"
        print(
            f"Total: ${total_spend_micros / 1e6:.4f} across {len(jobs)} job(s),"
            f" {total_accepted} accepted task(s), {overall}/accepted"
        )
        return 0
    finally:
        ledger.close()


# ---------------------------------------------------------------------- watch


def cmd_watch(args: argparse.Namespace) -> int:
    from forgeos.watch import watch_queue

    stats = watch_queue(
        args.queue,
        state_dir=args.state_dir,
        once=args.once,
        poll_interval=args.poll_interval,
    )
    print(
        f"watch: {stats.jobs_done} done, {stats.jobs_failed} failed"
        + (" (halted by operator)" if stats.halted else "")
    )
    return 0


# ------------------------------------------------------------------------ team


def _print_mission_graph(mission) -> None:
    """The plan before money moves: every task's subject, scope, and
    dependency edges, plus any `dependency_conflicts` the compiler could not
    resolve on its own (see `compiler.Mission.dependency_conflicts` -- a
    non-empty list there is a human's call, never auto-resolved).
    """
    print(f"Mission: {mission.objective}")
    print(f"Tasks: {len(mission.tasks)}")
    for t in mission.tasks:
        print(f"  [{t.id[:12]}] {t.subject}")
        print(f"    scope: {' '.join(t.scope.paths) or '(none)'}")
        if t.depends_on:
            print(f"    depends_on: {', '.join(d[:12] for d in t.depends_on)}")
    if mission.dependency_conflicts:
        print()
        print("Dependency conflicts (unresolved -- human call):")
        for conflict in mission.dependency_conflicts:
            print(f"  - {conflict}")


# Substring `core.verify.py` appends to `MergeGate.evaluate`'s reasons when
# `Forge._pick_reviewer` found no second capable worker (see forge.py's
# `_STRUCTURAL_REFUSALS` -- the same substring it matches on). Lives in
# `TaskOutcome.merge_reasons`, not the short `reason` field.
_NO_REVIEW_REFUSAL = "no independent review"


def _print_outcomes(outcomes) -> None:
    for outcome in outcomes:
        status = "accepted" if outcome.accepted else "rejected"
        tier = outcome.tier if outcome.tier is not None else "-"
        print(
            f"  [{outcome.task_id[:12]}] {status:<8} {outcome.subject}"
            f"  worker={outcome.worker_id or '-'} tier={tier}"
            f" attempts={outcome.attempts} cost=${outcome.usd_micros / 1e6:.4f}"
        )
        if outcome.reason:
            print(f"    reason: {outcome.reason}")
        if any(_NO_REVIEW_REFUSAL in r.lower() for r in outcome.merge_reasons):
            print(
                "    review needs a second capable worker -- the registry has "
                "only one that can take this task"
            )


def _run_team(
    objective: str,
    *,
    cwd: str = ".",
    budget_usd: float | None,
    state_dir: str | None,
    dry_run: bool,
    forge_factory: Callable[[], object] | None = None,
) -> int:
    """Compile `objective` into a task graph, print it, then (unless
    `dry_run`) run it through a `Forge` and report what happened.

    `forge_factory` is injectable for the same reason `watch.watch_queue`'s
    is: a real `Forge()` opens five sqlite stores under `state_dir` as a
    side effect of construction, which a test must not depend on.
    """
    from forgeos.compiler import CompilerError, compile_mission

    try:
        mission = compile_mission(objective, cwd=cwd)
    except CompilerError as exc:
        print(f"Cannot compile objective: {exc}")
        return 1

    _print_mission_graph(mission)

    if dry_run:
        return 0

    if budget_usd is None:
        print()
        print("Refusing to run without an explicit budget.")
        print("Fix: pass --budget-usd <dollars> -- forgeos never invents a spending cap.")
        return 1

    from pydantic import ValidationError

    from forgeos.contracts import Budget

    try:
        budget = Budget(max_usd=budget_usd)
    except ValidationError as exc:
        print(f"Invalid --budget-usd: {exc}")
        return 1

    if forge_factory is None:
        resolved_state_dir = _resolve_state_dir(state_dir)
        from forgeos.forge import Forge

        forge_factory = lambda: Forge(home=resolved_state_dir)  # noqa: E731

    try:
        forge = forge_factory()
    except OSError as exc:
        print(f"Cannot use forgeos's home directory: {exc}")
        print("Fix: check permissions on that path, or set HOME to a writable directory.")
        return 1

    # A reviewer worth of independent review, built the same way `Forge.run`
    # builds its own default executor: routed by worker_id, so whichever
    # worker the merge gate's `_pick_reviewer` names actually gets called.
    # Without this, `_run_team` never passed a reviewer at all and every task
    # was refused at the merge gate for "no independent review" -- a fleet
    # with a genuinely single capable worker still gets that same honest
    # refusal (see `Forge._pick_reviewer`'s docstring), but now it is because
    # no second worker exists, not because nothing was ever asked to review.
    default_executor = getattr(forge, "default_executor", None)
    if callable(default_executor):
        reviewer = default_executor(cwd=cwd)
    else:
        # Keep lightweight Forge-shaped test doubles compatible with the CLI
        # seam; real Forge instances take the ledger-owned Gateway path above.
        from forgeos.adapters.routed import routed_executor

        reviewer = routed_executor(forge.registry, forge.ledger, cwd=cwd)

    try:
        result = forge.run(
            mission.objective,
            mission.tasks,
            cwd=cwd,
            budget=budget,
            dependencies=mission.dependencies,
            reviewer=reviewer,
        )
    finally:
        forge.close()

    print()
    _print_outcomes(result.outcomes)

    print()
    per_accepted = (
        f"${result.cost_per_accepted:.4f}" if result.cost_per_accepted is not None else "n/a"
    )
    print(
        f"Result: accepted={result.accepted} rejected={result.rejected}"
        f"  spend=${result.spend_usd:.4f}  $/accepted={per_accepted}"
    )
    if result.halted_reason:
        print(f"Halted: {result.halted_reason}")

    merge_warnings = [w for o in result.outcomes for w in o.merge_warnings]
    if merge_warnings:
        print("Merge warnings:")
        for warning in merge_warnings:
            print(f"  - {warning}")

    return 0 if (result.accepted > 0 and result.rejected == 0) else 2


def cmd_team(args: argparse.Namespace) -> int:
    return _run_team(
        args.objective,
        cwd=args.cwd or ".",
        budget_usd=args.budget_usd,
        state_dir=args.state_dir,
        dry_run=args.dry_run,
    )


# ------------------------------------------------------------------- serve-mcp


def cmd_serve_mcp(args: argparse.Namespace) -> int:
    from forgeos.mcp_server import serve

    serve(state_dir=args.state_dir, queue_dir=args.queue)
    return 0


# ---------------------------------------------------------------------- memory


def cmd_memory(args: argparse.Namespace) -> int:
    """Mine the ledger/event log for recurring patterns and file each as an
    unverified claim candidate (see `forgeos.knowledge.automemory`). Never
    promotes anything -- see `forgeos.knowledge.claims.ClaimStore.promote`
    for the separate, evidence-gated step that does.
    """
    if not args.mine:
        print("Nothing to do -- pass --mine to scan the ledger/events for lesson candidates.")
        return 1

    state_dir = _resolve_state_dir(args.state_dir)
    ledger_path = state_dir / "ledger.db"
    events_path = state_dir / "events.db"
    if not ledger_path.exists() or not events_path.exists():
        print(f"No ledger/events under {state_dir}.")
        print("Fix: pass --state-dir to a directory forgeos has run a job in, or run one first.")
        return 1

    from forgeos.events import EventLog
    from forgeos.knowledge.automemory import file_candidates, mine_lessons
    from forgeos.knowledge.claims import ClaimStore
    from forgeos.ledger import Ledger

    ledger = Ledger(ledger_path)
    events = EventLog(events_path)
    claims_store = ClaimStore(state_dir / "claims.db")
    try:
        candidates = mine_lessons(ledger, events)
        if not candidates:
            print("No repeated patterns found -- nothing to file.")
            return 0

        print(f"Mined {len(candidates)} lesson candidate(s):")
        for c in candidates:
            print(f"  [{c.kind.value}] {c.subject} (x{c.occurrences})")
            print(f"    {c.claim_text}")

        filed = file_candidates(claims_store, candidates)
        print()
        print(
            f"Filed {len(filed)} claim(s) into {state_dir / 'claims.db'} -- all UNVERIFIED, "
            "pending corroboration/promotion. Nothing here is an instruction yet."
        )
        return 0
    finally:
        ledger.close()
        events.close()
        claims_store.close()


# ----------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m forgeos",
        description=(
            "ForgeOS diagnostics — read-only, no network. Never probes a remote "
            "provider; use tools/live_check.py to test a real, billed endpoint."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Runnable workers, provider reachability, price staleness — no network")

    p_receipts = sub.add_parser(
        "receipts", help="Ledger summary: spend by job/worker, cost per accepted task"
    )
    p_receipts.add_argument(
        "--state-dir", help="Where the ledger lives (default: forgeos's DEFAULT_HOME, ~/.forgeos)"
    )

    p_watch = sub.add_parser(
        "watch", help="Unattended job-queue daemon: poll --queue, run each job, write receipts"
    )
    p_watch.add_argument("--queue", required=True, help="Directory with incoming/done/failed queue subdirs")
    p_watch.add_argument(
        "--state-dir", help="Where the ledger/halt-flag live (default: forgeos's DEFAULT_HOME, ~/.forgeos)"
    )
    p_watch.add_argument("--once", action="store_true", help="Process the current backlog once and exit")
    p_watch.add_argument(
        "--poll-interval", type=float, default=5.0,
        help="Seconds between polls when not --once (default: 5)",
    )

    p_team = sub.add_parser(
        "team", help="Compile an objective into a task graph and run it end-to-end"
    )
    p_team.add_argument("objective", help="Natural-language objective to decompose and run")
    p_team.add_argument("--cwd", help="Working directory to decompose/run against (default: .)")
    p_team.add_argument(
        "--budget-usd", type=float, dest="budget_usd",
        help="Hard USD cap for the whole job (required unless --dry-run)",
    )
    p_team.add_argument(
        "--state-dir", help="Where the ledger lives (default: forgeos's DEFAULT_HOME, ~/.forgeos)"
    )
    p_team.add_argument(
        "--dry-run", action="store_true",
        help="Print the compiled task graph and exit without spending anything",
    )

    p_mcp = sub.add_parser(
        "serve-mcp",
        help="MCP server over stdio -- expose doctor/receipts/plan/submit-job as tools for any MCP-speaking agent",
    )
    p_mcp.add_argument(
        "--state-dir", help="Where the ledger lives for forgeos_receipts (default: forgeos's DEFAULT_HOME, ~/.forgeos)"
    )
    p_mcp.add_argument(
        "--queue", help="Queue directory for forgeos_submit_job (omit to leave that tool unavailable)"
    )

    p_memory = sub.add_parser(
        "memory", help="Mine ledger/events for recurring patterns -> unverified claim candidates"
    )
    p_memory.add_argument(
        "--mine", action="store_true", help="Scan and file lesson candidates (the only mode today)"
    )
    p_memory.add_argument(
        "--state-dir", help="Where the ledger/events/claims live (default: forgeos's DEFAULT_HOME, ~/.forgeos)"
    )

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    dispatch = {
        "doctor": cmd_doctor,
        "receipts": cmd_receipts,
        "watch": cmd_watch,
        "team": cmd_team,
        "serve-mcp": cmd_serve_mcp,
        "memory": cmd_memory,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
