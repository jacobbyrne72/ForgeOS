"""forgeos CLI — dispatch subcommands."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from forgeos.compiler import compile_mission
from forgeos.circuit_breaker import CircuitBreaker
from forgeos.prompt_cache import PromptCache

def cmd_run(args) -> int:
    from forgeos import Forge
    forge = Forge()
    mission = compile_mission(args.objective, cwd=args.cwd or ".")
    print(f"Mission compiled: {len(mission.tasks)} tasks")
    for t in mission.tasks:
        print(f"  [{t.id[:12]}] {t.subject}")
    print(f"Spending: ${args.budget:.2f}" if hasattr(args, 'budget') else "")
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
    print(f"Job: {job_row['objective']}")
    print(f"State: {job_row['state']}")
    print(f"Budget: ${ledger.job_spend_micros(args.job_id) / 1e6:.4f}")
    print(f"Tasks: {ledger.task_count(args.job_id)}")
    return 0

def cmd_doctor(args) -> int:
    from forgeos.settings import Settings, default_settings
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

def main() -> int:
    parser = argparse.ArgumentParser(prog="forge", description="ForgeOS CLI")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="Readiness score")
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
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    dispatch = {
        "run": cmd_run, "resume": cmd_resume, "report": cmd_report,
        "compress": cmd_compress, "bench": cmd_bench, "watch": cmd_watch,
        "doctor": cmd_doctor, "init": cmd_init, "compile": cmd_compile,
        "cache": cmd_cache, "breaker": cmd_breaker,
    }
    return dispatch[args.command](args)

if __name__ == "__main__":
    sys.exit(main())
