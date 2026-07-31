"""forgeos CLI — dispatch subcommands."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from forgeos.compiler import compile_mission
from forgeos.circuit_breaker import CircuitBreaker
from forgeos.prompt_cache import PromptCache


def cmd_run(args) -> int:
    from forgeos import Forge

    Forge()
    mission = compile_mission(args.objective, cwd=args.cwd or ".")
    print(f"Mission compiled: {len(mission.tasks)} tasks")
    for t in mission.tasks:
        print(f"  [{t.id[:12]}] {t.subject}")
    print(f"Spending: ${args.budget:.2f}" if hasattr(args, "budget") else "")
    print("Use forge resume <job-id> to track progress.")
    return 0


def cmd_resume(args) -> int:
    from forgeos import open_ledger

    home = Path(args.state_dir or Path.cwd() / ".forgeos")
    ledger = open_ledger(home / "ledger.db")
    job_row = ledger.job(args.job_id)
    if job_row is None:
        print(f"ERROR: Job '{args.job_id}' not found")
        return 1
    print(f"Job {args.job_id}: state={job_row['state']}")
    return 0


def cmd_report(args) -> int:
    from forgeos import open_ledger

    home = Path(args.state_dir or Path.cwd() / ".forgeos")
    ledger = open_ledger(home / "ledger.db")
    job_row = ledger.job(args.job_id)
    if job_row is None:
        print(f"ERROR: Job '{args.job_id}' not found")
        return 1
    total_spend = ledger.job_spend_micros(args.job_id)
    total_tasks = ledger.task_count(args.job_id)
    print(f"Job: {job_row['objective']}")
    print(f"State: {job_row['state']}")
    print(f"Total cost: ${total_spend / 1e6:.4f}")
    print(f"Tasks: {total_tasks}")
    if args.breakdown and total_tasks > 0:
        print()
        print("Per-task breakdown:")
        print("-" * 60)
        for task in ledger.tasks_for_job(args.job_id):
            task_spend = ledger.task_spend_micros(task.id)
            print(f"  [{task.id[:12]}] {task.subject[:50]}")
            print(f"    cost: ${task_spend / 1e6:.4f}  state: {task.state.value}")
    return 0


def cmd_doctor(args) -> int:
    from forgeos.settings import Settings

    settings = Settings.load()
    usable = settings.usable_providers()
    total = len(settings.providers)
    score = round(len(usable) / total * 100, 1) if total else 0
    print(f"Provider readiness: {len(usable)}/{total} usable ({score}%)")
    for p in sorted(settings.providers.values(), key=lambda x: x.name):
        status = "ready" if p.usable else f"unavailable ({p.status()})"
        print(f"  {p.name:<12} {p.kind.value:<10} {status}")
    cache = PromptCache(home=Path.home() / ".forgeos")
    cs = cache.stats()
    print(f"\nPrompt cache: {cs['entries']} entries ({cs['utilization_pct']}% full)")
    cache.close()
    return 0


def cmd_init(args) -> int:
    cwd = Path(args.cwd or Path.cwd())
    forgeos_dir = cwd / ".forgeos"
    forgeos_dir.mkdir(exist_ok=True)
    live_dir = cwd / ".forgeos-live"
    live_dir.mkdir(exist_ok=True)
    claude = cwd / "CLAUDE.md"
    if not claude.exists():
        claude.write_text(f"# CLAUDE.md — {cwd.name}\n\nCodebase overview.\n", encoding="utf-8")
    settings_path = forgeos_dir / "settings.json"
    if not settings_path.exists():
        from forgeos.settings import default_settings

        settings_path.write_text(
            default_settings().model_dump_json(indent=2),
            encoding="utf-8",
        )
    agents = cwd / "AGENTS.md"
    if not agents.exists():
        agents.write_text(f"# AGENTS.md\n\nProject: {cwd.name}\n", encoding="utf-8")
    print("Init complete! Run 'forge run <objective>' to start.")
    return 0


def cmd_compile(args) -> int:
    mission = compile_mission(args.objective, cwd=args.cwd or ".")
    print(f"Mission: {args.objective[:100]}")
    print(f"Tasks: {len(mission.tasks)}")
    for t in mission.tasks:
        print(f"  [{t.id[:12]}] {t.subject}")
        print(f"    scope: {' '.join(t.scope.paths[:3])}")
        print(f"    caps: {', '.join(t.capabilities)}")
    return 0


def cmd_breaker(args) -> int:
    breaker = CircuitBreaker()
    stats = breaker.stats()
    if not stats:
        print("No worker history yet.")
        return 0
    for wid, s in stats.items():
        icon = {"closed": "[OK]", "open": "[TRIP]", "half_open": "[TEST]"}.get(s["state"], "[?]")
        print(f"{icon} {wid}: {s['state']} ({s['total_calls']} calls, {s['failure_rate']:.0%} fail)")
    return 0


def cmd_cache(args) -> int:
    cache = PromptCache()
    if args.cache_action == "clear":
        cache.clear()
        print("Cache cleared.")
    elif args.cache_action == "stats":
        print(json.dumps(cache.stats(), indent=2))
    elif args.cache_action == "prune":
        cache._purge_expired()
        print("Pruned expired entries.")
    cache.close()
    return 0


def cmd_compress(args):
    from forgeos.context_compress import compress_context

    files = [(f, open(f).read()) for f in args.files] if args.files else []
    result = compress_context(args.objective, files)
    for path, filtered in result:
        print(f"=== {path} ===")
        print(filtered[:500])
    print(f"Compressed {len(files)} -> {len(result)} relevant file(s)")
    return 0


def cmd_adapt(args):
    from forgeos.adapt import AdapterProfiler

    profiler = AdapterProfiler()
    # Load any existing profiles from ledger
    try:
        from forgeos import open_ledger
        from pathlib import Path

        ledger = open_ledger(Path.home() / ".forgeos" / "ledger.db")
        rows = ledger._conn.execute(
            "SELECT task_id, worker_id, usd_micros, seconds FROM reports WHERE task_id IS NOT NULL"
        ).fetchall()
        for row in rows:
            profiler.record_task(
                row.get("worker_id", "unknown"),
                row.get("usd_micros", 0),
                row.get("seconds", 0),
                True,
            )
    except Exception:
        pass
    decision = profiler.best_adapter(
        required_capabilities=set(args.capabilities.split(",")) if args.capabilities else set(),
        budget_usd_micros=args.budget,
    )
    if decision:
        print(
            json.dumps(
                {
                    "adapter": decision.adapter_name,
                    "reason": decision.reason,
                    "estimated_cost_usd": decision.estimated_cost_usd,
                    "estimated_seconds": decision.estimated_seconds,
                    "confidence": decision.confidence,
                },
                indent=2,
            )
        )
    else:
        print("No suitable adapter found")
    return 0


def cmd_bench(args):
    from forgeos.bench import run_benchmark_cli

    print(run_benchmark_cli(args.objective, iterations=args.iterations or 3))
    return 0


def cmd_watch(args):
    from forgeos.watch import watch

    print(f"Watching for cost anomalies (interval={args.interval}s)...")
    alerts = watch(None, interval_seconds=args.interval, max_alerts=args.max_alerts or 5)
    for a in alerts:
        print(json.dumps(a, indent=2))
    return 0


def cmd_fleet(args) -> int:
    """Show every provider, what it costs, and what's cheapest RIGHT NOW.

    The screenshot command. One glance tells you: what do I have, what's
    alive, and what order should I burn through them to make my
    subscriptions last longest.

    Deliberately does NOT probe the filesystem (no shutil.which, no
    subprocess). That's what `forge doctor` is for. Fleet reads config
    and prints instantly.
    """
    from forgeos.settings import Settings, ProviderKind, AuthMode

    settings = Settings.load()

    print("forgeos fleet — your AI capacity, ranked by cost")
    print("=" * 62)

    # Group by cost tier — the ordering IS the product
    free, subscription, metered = [], [], []
    for p in sorted(settings.providers.values(), key=lambda x: x.name):
        if not p.enabled:
            continue
        entry = {"name": p.name, "kind": p.kind.value, "auth": p.auth.value}

        if p.kind == ProviderKind.LOCAL:
            free.append(entry)
        elif p.auth == AuthMode.SUBSCRIPTION:
            subscription.append(entry)
        else:
            metered.append(entry)

    def print_group(title, entries, hint):
        if not entries:
            return
        print(f"\n  {title}")
        print(f"  {hint}")
        print(f"  {'-' * 56}")
        for e in entries:
            if e["auth"] == "subscription":
                cost_label = "$0 marginal (you already paid)"
            elif e["kind"] == "local":
                cost_label = "$0 (runs on your hardware)"
            elif e["kind"] == "gateway":
                cost_label = "pay-per-token (multi-provider)"
            else:
                cost_label = "pay-per-token"
            print(f"  • {e['name']:<14} {cost_label}")

    print_group(
        "FREE / LOCAL — burn these first, they cost nothing",
        free,
        "Ollama, local models. Unlimited. No quota. No meter.",
    )
    print_group(
        "SUBSCRIPTION — you already paid, use every drop",
        subscription,
        "Claude, Codex, Copilot. Flat-rate seat. Every task = $0 extra.",
    )
    print_group(
        "METERED — last resort, every token costs real money",
        metered,
        "DeepSeek, OpenRouter. Only when subscriptions are exhausted.",
    )

    # The money shot
    print(f"\n{'=' * 62}")
    print("  YOUR ROUTING LADDER (cheapest first)")
    print(f"  {'-' * 56}")
    rung = 1
    for label, group in [("free/local", free), ("subscription", subscription), ("metered", metered)]:
        if group:
            names = ", ".join(e["name"] for e in group)
            print(f"  {rung}. {label:<14} → {names}")
            rung += 1
    if rung == 1:
        print("  (nothing configured — run 'forge init' then add providers)")

    if subscription:
        names = ", ".join(e["name"] for e in subscription)
        print(f"\n  → forgeos routes through {names} BEFORE touching metered API.")
        print("    Every task your subscription handles = $0 extra cost.")
        print("    Same subscription. 5x more tasks. That's the product.")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="forge", description="ForgeOS — cost-governed AI coding")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="What can this machine do right now")
    sub.add_parser("fleet", help="What you have, what's alive, what's cheapest TODAY")
    p_init = sub.add_parser("init", help="Scan repo and generate files")
    p_init.add_argument("--cwd", default=None)
    p_run = sub.add_parser("run", help="Compile and run objective")
    p_run.add_argument("objective")
    p_run.add_argument("--cwd", default=".")
    p_resume = sub.add_parser("resume", help="Resume a crashed job")
    p_resume.add_argument("job_id")
    p_resume.add_argument("--state-dir", default=None)
    p_report = sub.add_parser("report", help="Cost breakdown for a job")
    p_report.add_argument("job_id")
    p_report.add_argument("--state-dir", default=None)
    p_compile = sub.add_parser("compile", help="Dry-run compile")
    p_compile.add_argument("objective")
    p_compile.add_argument("--cwd", default=".")
    p_cache = sub.add_parser("cache", help="Manage prompt cache")
    p_cache.add_argument("cache_action", choices=["clear", "stats", "prune"])
    sub.add_parser("breaker", help="Circuit breaker state")
    p_compress = sub.add_parser("compress", help="AST-based context compression")
    p_compress.add_argument("objective")
    p_compress.add_argument("--files", nargs="*", default=[])
    p_adapt = sub.add_parser("adapt", help="Adaptive adapter selection")
    p_adapt.add_argument("--capabilities", default="")
    p_adapt.add_argument("--budget", type=int, default=None)
    p_bench = sub.add_parser("bench", help="Reproducible cost benchmark")
    p_bench.add_argument("objective")
    p_bench.add_argument("--iterations", type=int, default=3)
    p_watch = sub.add_parser("watch", help="Continuous cost monitoring")
    p_watch.add_argument("--interval", type=int, default=30)
    p_watch.add_argument("--max-alerts", type=int, default=5)
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    dispatch = {
        "run": cmd_run,
        "resume": cmd_resume,
        "report": cmd_report,
        "adapt": cmd_adapt,
        "compress": cmd_compress,
        "bench": cmd_bench,
        "watch": cmd_watch,
        "doctor": cmd_doctor,
        "init": cmd_init,
        "compile": cmd_compile,
        "cache": cmd_cache,
        "breaker": cmd_breaker,
        "fleet": cmd_fleet,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
