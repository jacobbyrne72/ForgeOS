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
  job, not this one's.

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

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    dispatch = {
        "doctor": cmd_doctor,
        "receipts": cmd_receipts,
        "watch": cmd_watch,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
