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
    """Run an objective through the same guarded team path as the main CLI.

    The old handler only compiled a mission and printed an *estimated* saving,
    then returned success without creating a job or invoking a worker. That was
    especially dangerous for a command named ``run``: a green exit code looked
    like completed work. Reuse ``python -m forgeos``'s tested runner so budget,
    review, ledger, and refusal semantics stay single-sourced.
    """
    from forgeos.__main__ import _run_team

    return _run_team(
        args.objective,
        cwd=args.cwd or ".",
        budget_usd=args.budget,
        state_dir=args.state_dir,
        dry_run=args.dry_run,
    )


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
            task_id = task["id"]
            task_spend = ledger.task_spend_micros(task_id)
            print(f"  [{task_id[:12]}] {task['subject'][:50]}")
            print(f"    cost: ${task_spend / 1e6:.4f}  state: {task['state']}")
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
    cache = PromptCache()
    try:
        cs = cache.stats()
        print(f"\nPrompt cache: {cs['entries']} entries ({cs['utilization_pct']}% full)")
    finally:
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
        claude.write_text(f"# CLAUDE.md - {cwd.name}\n\nCodebase overview.\n", encoding="utf-8")
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
    try:
        if args.cache_action == "clear":
            cache.clear()
            print("Cache cleared.")
        elif args.cache_action == "stats":
            print(json.dumps(cache.stats(), indent=2))
        elif args.cache_action == "prune":
            cache._purge()
            print("Pruned expired entries.")
    finally:
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


def cmd_forgebench(args) -> int:
    from forgeos.forgebench import main as forgebench_main

    argv = ["--budget-usd", str(args.budget_usd)]
    if args.model:
        argv += ["--model", args.model]
    if args.dry_run:
        argv.append("--dry-run")
    if args.skip_baseline:
        argv.append("--skip-baseline")
    if args.ledger_path:
        argv += ["--ledger-path", args.ledger_path]
    if getattr(args, "json_out", ""):
        argv += ["--json-out", args.json_out]
    return forgebench_main(argv)


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

    print("forgeos fleet - your AI capacity, ranked by cost")
    print("=" * 62)

    # Group by cost tier - the ordering IS the product
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
            print(f"  - {e['name']:<14} {cost_label}")

    print_group(
        "FREE / LOCAL - burn these first, they cost nothing",
        free,
        "Ollama, local models. Unlimited. No quota. No meter.",
    )
    print_group(
        "SUBSCRIPTION - you already paid, use every drop",
        subscription,
        "Claude, Codex, Copilot. Flat-rate seat. Every task = $0 extra.",
    )
    print_group(
        "METERED - last resort, every token costs real money",
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
            print(f"  {rung}. {label:<14} -> {names}")
            rung += 1
    if rung == 1:
        print("  (nothing configured - run 'forge init' then add providers)")

    if subscription:
        names = ", ".join(e["name"] for e in subscription)
        print(f"\n  -> forgeos routes through {names} BEFORE touching metered API.")
        print("    Every task your subscription handles = $0 extra cost.")
        print("    Same subscription. 5x more tasks. That's the product.")
    print()
    return 0


def cmd_replace(args):
    from forgeos.cost_replacer import CostReplacer
    r = CostReplacer()
    print("=== Cost Replacer ===")
    print("Scanning for expensive patterns...")
    print()
    demo_texts = [
        "Use GPT-4 to summarize this text",
        "Translate this to Spanish",
        "Extract key entities from this text",
        "Review this code for errors",
    ]
    for t in demo_texts:
        mod, reps = r.replace(t)
        if reps:
            for rep in reps:
                print(f"  {rep['type']}: ${rep['savings_usd']:.3f} saved, {rep['saves_tokens']} tokens")
        else:
            print(f"  no replacement for: {t[:40]}...")
    print()
    print("Total savings:", r.report())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forge", description="ForgeOS - cost-governed AI coding")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="What can this machine do right now")
    sub.add_parser("fleet", help="What you have, what's alive, what's cheapest TODAY")
    p_init = sub.add_parser("init", help="Scan repo and generate files")
    p_init.add_argument("--cwd", default=None)
    p_run = sub.add_parser("run", help="Compile and run objective")
    p_run.add_argument("objective")
    p_run.add_argument("--cwd", default=".")
    p_run.add_argument(
        "--budget-usd", "--budget", type=float, default=None, dest="budget",
        help="Hard USD cap for the whole job (required unless --dry-run)",
    )
    p_run.add_argument("--state-dir", default=None)
    p_run.add_argument("--dry-run", action="store_true")
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
    p_models = sub.add_parser("models", help="Select cheapest capable model")
    sub.add_parser("profile", help="Model performance profiles")
    p_opt = sub.add_parser("optimize", help="Show cost optimization plan for a task type")
    p_auto = sub.add_parser("auto", help="Automatic cost optimization with savings report")
    p_auto.add_argument("--task-type", default="code_gen", dest="task_type")
    p_auto.add_argument("--daily-tasks", type=int, default=100, dest="daily_tasks")
    p_batch = sub.add_parser("batch", help="Batch cost projection across multiple tasks")
    p_batch.add_argument("--daily-tasks", type=int, default=100, dest="daily_tasks")
    p_opt.add_argument("task_type", nargs="?", default="code_gen", choices=["simple_edit","code_gen","security_scan","review","planning","debug","refactor"])
    p_opt.add_argument("--daily-tasks", type=int, default=100, dest="daily_tasks")
    p_models.add_argument("--capabilities", default="")
    p_models.add_argument("--complexity", default="simple", choices=["simple","medium","complex"])
    p_models.add_argument("--max-cost", type=float, default=1.0, dest="max_cost")
    p_compress.add_argument("objective")
    p_compress.add_argument("--files", nargs="*", default=[])
    p_output_compress = sub.add_parser("output-compress", help="Compress generated output to a token budget")
    p_output_compress.add_argument("--max-tokens", type=int, default=100, dest="max_tokens")
    p_adapt = sub.add_parser("adapt", help="Adaptive adapter selection")
    p_adapt.add_argument("--capabilities", default="")
    p_adapt.add_argument("--budget", type=int, default=None)
    p_bench = sub.add_parser("bench", help="Reproducible cost benchmark")
    p_bench.add_argument("objective")
    p_bench.add_argument("--iterations", type=int, default=3)
    p_forgebench = sub.add_parser(
        "forgebench",
        help="Release 0.1 exit gate: paired baseline vs ForgeOS on a pinned task suite",
    )
    p_forgebench.add_argument("--model", default="")
    p_forgebench.add_argument("--budget-usd", type=float, default=0.50, dest="budget_usd")
    p_forgebench.add_argument("--dry-run", action="store_true", dest="dry_run")
    p_forgebench.add_argument("--skip-baseline", action="store_true", dest="skip_baseline")
    p_forgebench.add_argument("--ledger-path", default=":memory:", dest="ledger_path")
    p_forgebench.add_argument("--json-out", default="", dest="json_out")
    p_watch = sub.add_parser("watch", help="Continuous cost monitoring")
    p_watch.add_argument("--interval", type=int, default=30)
    p_watch.add_argument("--max-alerts", type=int, default=5)
    p_budget = sub.add_parser("budget", help="Token budget enforcer")
    p_budget.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    p_budget.add_argument("--warn-at", type=float, default=0.8, dest="warn_at")
    sub.add_parser("replace", help="Replace expensive calls with cheap equivalents")
    sub.add_parser("adaptbatch", help="Auto-select cheapest batch strategy for any workload")
    sub.add_parser("smartbatch", help="ML-inspired batch cost predictor using history")
    p_audit = sub.add_parser("audit", help="Scan project for AI cost waste")
    sub.add_parser("route", help="Route tasks to cheapest execution path")
    p_prompt = sub.add_parser("prompt", help="Optimize a prompt for token cost")
    p_prompt.add_argument("--max-tokens", type=int, default=100, dest="max_tokens")
    p_local = sub.add_parser("local", help="Execute a safe local code demo")
    p_local.add_argument("--code", default="")
    p_retry = sub.add_parser("retry", help="Show cost-aware retry policy")
    p_retry.add_argument("--max-retries", type=int, default=3, dest="max_retries")
    p_schedule = sub.add_parser("schedule", help="Schedule tasks into cost-aware batches")
    p_schedule.add_argument("--batch-size", type=int, default=5, dest="batch_size")
    p_forecast = sub.add_parser("forecast", help="Forecast future savings")
    p_forecast.add_argument("--forecast-days", type=int, default=30, dest="forecast_days")
    p_guard = sub.add_parser("guard", help="Check cost guard decisions")
    p_guard.add_argument("--guard-budget", type=float, default=1.0, dest="guard_budget")
    sub.add_parser("track", help="Show recorded cost savings")
    sub.add_parser("efficiency", help="Show token efficiency metrics")
    sub.add_parser("dashboard", help="Show the ForgeOS cost dashboard")
    p_purge = sub.add_parser("purge", help="Purge expensive prompt-cache entries")
    p_shrink = sub.add_parser("shrink", help="Shrink prompts to save tokens")
    p_rank = sub.add_parser("model-rank", help="Rank models by cost-effectiveness")
    p_trunc = sub.add_parser("truncate", help="Truncate responses to save tokens")
    p_proj = sub.add_parser("proj", help="Project batch costs before execution")
    p_proj.add_argument("--tasks", type=int, default=2, dest="num_tasks")
    p_trunc.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    p_rank.add_argument("--complexity", type=str, default="simple", dest="complexity")
    p_rank.add_argument("--max-cost", type=float, default=0.01, dest="max_cost")
    p_shrink.add_argument("--tokens", type=int, default=2048, dest="max_tokens")
    p_purge.add_argument("--cost-limit", type=float, default=0.01, dest="cost_limit")
    p_format = sub.add_parser("format", help="Show local formatting capability")
    p_format.add_argument("--formatter-tool", default="ruff", dest="formatter_tool")
    p_audit.add_argument("--dir", default=".", dest="audit_dir")
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    dispatch = {
        "run": cmd_run,
        "resume": cmd_resume,
        "report": cmd_report,
        "adapt": cmd_adapt,
        "compress": cmd_compress,
        "models": cmd_models,
        "profile": cmd_profile,
        "optimize": cmd_optimize,
        "auto": cmd_auto,
        "batch": cmd_batch,
        "bench": cmd_bench,
        "forgebench": cmd_forgebench,
        "watch": cmd_watch,
        "budget": cmd_budget,
        "replace": cmd_replace,
        "adaptbatch": cmd_adbatch,
        "smartbatch": cmd_smartbatch,
        "audit": cmd_audit,
        "prompt": cmd_prompt_opt,
        "route": cmd_route,
        "track": cmd_track,
        "local": cmd_local,
        "retry": cmd_retry,
        "schedule": cmd_schedule,
        "guard": cmd_guard,
        "forecast": cmd_forecast,
        "efficiency": cmd_efficiency,
        "dashboard": cmd_dashboard,
        "output-compress": cmd_output_compress,
        "format": cmd_format,
        "purge": cmd_purge,
        "shrink": cmd_shrink,
        "model-rank": cmd_model_rank,
        "truncate": cmd_truncate,
        "proj": cmd_project,
        "doctor": cmd_doctor,
        "init": cmd_init,
        "compile": cmd_compile,
        "cache": cmd_cache,
        "breaker": cmd_breaker,
        "fleet": cmd_fleet,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 2
    return handler(args)


def cmd_batch(args):
    from forgeos.batch_optimize import BatchOptimizer
    b = BatchOptimizer()
    # Default demo tasks
    tasks = [
        {"type": "code_gen", "name": "implement feature"},
        {"type": "review", "name": "review PR"},
        {"type": "simple_edit", "name": "format code"},
        {"type": "debug", "name": "fix null pointer"},
        {"type": "security_scan", "name": "scan diff"},
    ]
    s = b.run_batch(tasks[:args.daily_tasks // 5])
    print(b.print_report(s))
    return 0



def cmd_budget(args):
    from forgeos.token_budget import TokenBudget
    print("=== Token Budget Enforcer ===")
    print("Max tokens:", args.max_tokens)
    print("Warn at:", args.warn_at)
    print()
    # Demo: check a short prompt
    budget = TokenBudget(max_tokens=args.max_tokens, warn_ratio=args.warn_at)
    short_prompt = "Write a function that adds two numbers."
    result = budget.check(short_prompt)
    print("Short prompt check:", result["action"])
    print("  Est tokens:", result["estimated_tokens"])
    print("  Remaining:", result["budget_remaining"])
    print()
    # Demo: check a large prompt
    big_prompt = "x" * (args.max_tokens + 2000)
    result = budget.check(big_prompt)
    print("Oversized prompt check:", result["action"])
    print("  Est tokens:", result["estimated_tokens"])
    print("  Remaining:", result["budget_remaining"])
    print()
    print("Budget prevents overspending on single requests.")
    return 0



def cmd_smartbatch(args):
    from forgeos.smart_batch import SmartBatchPredictor
    p = SmartBatchPredictor()
    # Seed with history demo
    for _ in range(5):
        p.record("code_gen", 0.018, 200)
    for _ in range(3):
        p.record("review", 0.015, 600)
    # Predict for a mixed workload
    result = p.predict_batch([
        {"type": "code_gen"}, {"type": "code_gen"},
        {"type": "review"}, {"type": "debug"},
    ])
    print("=== Smart Batch Predictor ===")
    print("Total tasks:", result["total_tasks"])
    print()
    for pred in result["predictions"]:
        print("  " + pred["task_type"] + ": " + str(pred["count"]) + "x, $" + str(round(pred["per_task_usd"], 4)) + "/task (" + pred["source"] + ", " + str(pred["history_samples"]) + " history samples)")
    print()
    print("Total estimated:", "$" + str(round(result["total_estimated_usd"], 4)))
    print("Monthly projection:", "$" + str(round(result["monthly_projection"], 2)))
    print("Yearly projection:", "$" + str(round(result["yearly_projection"], 2)))
    return 0







def cmd_track(args):
    from forgeos.cost_tracker import CostTracker
    t = CostTracker()
    print("=== Cost Tracker ===")
    total = t.total_saved()
    print("Total savings: $" + str(total["total_usd"]))
    print("Total events:", total["total_events"])
    print("Total tokens saved:", total["total_tokens"])
    print()
    by_type = t.by_task_type()
    if by_type:
        print("By task type:")
        for bt in by_type:
            print("  " + bt["task_type"] + ": $" + str(bt["total_usd"]) + " (" + str(bt["count"]) + " events)")
    print()
    proj = t.monthly_projection()
    print("Monthly projection: $" + str(proj["monthly_usd"]))
    print("Yearly projection: $" + str(proj["yearly_usd"]))
    print()
    recent = t.recent(5)
    if recent:
        print("Recent events:")
        for r in recent:
            print("  " + r["task_type"] + ": $" + str(r["saved_usd"]) + " (" + r["strategy"] + ")")
    return 0

def cmd_route(args):
    from forgeos.cost_router import CostRouter
    router = CostRouter()
    print("=== Cost Router ===")
    demo_tasks = [
        {"type": "format"}, {"type": "format"},
        {"type": "lint"}, {"type": "summarize"},
        {"type": "code_gen"}, {"type": "debug"},
        {"type": "security_scan"}, {"type": "grep"},
    ]
    result = router.route_many(demo_tasks)
    print("Tasks:", result["total_tasks"])
    print("Total cost: $" + str(result["total_cost_usd"]))
    print()
    for route, data in result["routes"].items():
        print("  " + route + ": " + str(data["count"]) + " tasks, $" + str(round(data["cost"], 4)))
    print()
    print("Savings vs full model for all: $" + str(result["savings_vs_full_model"]))
    return 0



def cmd_prompt_opt(args):
    from forgeos.prompt_optimizer import optimize_prompt
    # Demo prompt with excessive tokens
    demo = "You are an AI assistant." + chr(10) + chr(10) + "Previous turns:" + chr(10)
    for i in range(15):
        demo += "- User: Task " + str(i) + chr(10) + "- Assistant: Response " + str(i) + chr(10)
    demo += chr(10) + "Now write something useful."
    optimized, stats = optimize_prompt(demo, max_tokens=args.max_tokens)
    print("=== Prompt Optimizer ===")
    print("Original tokens:", stats["original_tokens"])
    print("Optimized tokens:", stats["optimized_tokens"])
    print("Tokens saved:", stats["tokens_saved"])
    print("Savings:", stats["savings_pct"], "%")
    print("Under budget:", stats["under_budget"])
    print()
    reduction_pct = round((1 - stats["optimized_tokens"] / max(1, stats["original_tokens"])) * 100, 1)
    print("Token reduction:", reduction_pct, "%")
    return 0

def cmd_local(args):
    from forgeos.local_exec import LocalExecutor
    e = LocalExecutor()
    code = args.code
    if not code:
        code = "x = 2 + 2" + chr(10) + "print(x)"
    result = e.execute(code)
    print("=== Local Executor ===")
    print("Executed locally:", result["executed_locally"])
    print("Cost: $" + str(result["cost"]))
    print("Reason:", result["reason"])
    if result["output"]:
        print("Output:", result["output"].strip())
    if result["error"]:
        print("Error:", result["error"])
    print()
    print("Local calls:", e.local_calls)
    print("API calls:", e.api_calls)
    return 0


def cmd_retry(args):
    from forgeos.cost_retry import CostRetry
    retry = CostRetry(max_retries=args.max_retries)
    print("=== Cost-Aware Retry ===")
    print("Max retries:", retry.max_retries)
    print("Base budget:", retry.base_cost_budget)
    print("Retry cost per call: $0.03")
    print()
    print("How it works:")
    print("  - Exponential backoff: wait doubles each retry")
    print("  - Budget check: stops if next retry costs more than remaining budget")
    print("  - Waste detection: skips retry on non-recoverable errors")
    print()
    print("Savings report:", retry.savings_report())
    return 0


def cmd_schedule(args):
    from forgeos.cost_scheduler import CostScheduler
    s = CostScheduler(batch_size=args.batch_size)
    # Demo workload
    for i in range(5):
        s.add_task("code_gen", {"name": f"feature {i}"})
    for i in range(3):
        s.add_task("review", {"name": f"PR #{i}"})
    for i in range(2):
        s.add_task("format", {"name": f"format {i}"})
    plan = s.schedule()
    print("=== Cost Scheduler ===")
    print("Total tasks:", plan["total_tasks"])
    print("Total batches:", plan["total_batches"])
    print("Total cost: $" + str(plan["total_cost"]))
    print()
    for b in plan["batches"]:
        print("  " + b["task_type"] + " batch " + str(b["batch_index"]) + ": " + str(b["batch_size"]) + " tasks, $" + str(b["cost"]))
    print()
    print("Savings vs all-API ($0.03 each): $" + str(round(plan["total_tasks"] * 0.03 - plan["total_cost"], 4)))
    return 0


def cmd_dashboard(args):
    from forgeos.cost_audit import CostAuditor
    from forgeos.cost_guard import CostGuard
    from forgeos.cost_tracker import CostTracker
    print("=" * 56)
    print("  FORGEOS COST DASHBOARD")
    print("=" * 56)
    print()

    # 1. Tracker summary
    tracker = CostTracker()
    total = tracker.total_saved()
    print("  SAVINGS TRACKER")
    print("    Total events:", total["total_events"])
    print("    Total saved: $" + str(total["total_usd"]))
    print("    Tokens saved:", total["total_tokens"])
    proj = tracker.monthly_projection()
    print("    Monthly projection: $" + str(proj["monthly_usd"]))
    print("    Yearly projection: $" + str(proj["yearly_usd"]))
    print()

    # 2. By task type
    by_type = tracker.by_task_type()
    if by_type:
        print("  BY TASK TYPE")
        for bt in by_type:
            print("    " + bt["task_type"] + ": $" + str(bt["total_usd"]) + " (" + str(bt["count"]) + " events)")
        print()

    # 3. Cost auditor
    auditor = CostAuditor()
    audit = auditor.audit_directory()
    print("  COST AUDIT")
    print("    Files scanned:", audit["files_audited"])
    print("    Issues found:", audit["total_issues"])
    print("    Potential savings: $" + str(audit["total_potential_savings_usd"]))
    print()

    # 4. Guard status
    guard = CostGuard(budget_usd=10.0)
    print("  BUDGET GUARD")
    print("    Budget: $" + str(guard.budget_usd))
    print("    Breaches:", guard.breaches)
    print()

    # 5. Efficiency
    print("  EFFICIENCY")
    print("    Track this command for ongoing metrics")
    print()

    # 6. Router
    print("  ROUTING RULES")
    rules = {
        "format": "local",
        "lint": "local",
        "summarize": "cheap_model",
        "code_gen": "full_model",
        "debug": "full_model",
        "review": "full_model",
        "security_scan": "full_model",
    }
    for task, route in rules.items():
        print("    " + task + " -> " + route)
    print()
    print("=" * 56)
    print("  v" + __import__("forgeos").__version__)
    print("=" * 56)
    return 0


def cmd_output_compress(args):
    from forgeos.output_compressor import compress_output
    # Create a large demo output
    demo = "\n".join(["model generated line " + str(i) + ": code here" for i in range(200)])
    result, info = compress_output(demo, target_tokens=args.max_tokens)
    print("=== Output Compressor ===")
    print("Original tokens:", info["original_tokens"])
    print("Compressed tokens:", info["compressed_tokens"])
    print("Tokens saved:", info["tokens_saved"])
    print("Savings:", str(info["savings_pct"]) + "%")
    print()
    print("Preview:", result[:300])
    return 0


def cmd_project(args):
    from forgeos.batch_projection import BatchProjection
    bp = BatchProjection()
    tasks = [{"type": "code_gen", "name": "task_1"}, {"type": "review", "name": "task_2"}]
    for i in range(2, args.num_tasks):
        tasks.append({"type": "code_gen", "name": "task_" + str(i+1)})
    bp.print_projection(tasks)
    return 0


def cmd_truncate(args):
    from forgeos.output_compressor import truncate_response
    demo = "This is a long response that goes on and on and takes up many tokens. " * 100
    result, info = truncate_response(demo, max_tokens=args.max_tokens)
    print("=== Response Truncator ===")
    print("Original tokens:", info["original_tokens"])
    print("Final tokens:", info["final_tokens"])
    print("Tokens saved:", info["tokens_saved"])
    print("Savings:", str(info["savings_pct"]) + "%")
    return 0


def cmd_model_rank(args):
    from forgeos.model_ranker import ModelRanker
    ranker = ModelRanker()
    print("=== Model Ranker ===")
    print()
    print("Top 5 models by quality-per-dollar:")
    for m in ranker.rank()[:5]:
        print(f"  {m['model']} ({m['provider']}): ${m['cost_per_1k_tokens']}/1k tokens, quality={m['quality_score']}, value={m['quality_per_dollar']}")
    print()
    best = ranker.recommend(max_cost=args.max_cost)
    if best:
        print("Recommendation:", best["recommendation"])
        print("Reason:", best["reason"])
        print("Est. cost/task: ${:.6f}".format(best["estimated_cost_per_task"]))
    else:
        print("No model found within budget")
    return 0


def cmd_shrink(args):
    from forgeos.prompt_optimizer import shrink_prompt
    demo = "Explain the concept of recursion in detail.  " * 10
    result, info = shrink_prompt(demo, target_tokens=args.max_tokens)
    print("=== Prompt Shrinker ===")
    print("Original tokens:", info["original_tokens"])
    print("Shrunk tokens:", info["shrunk_tokens"])
    print("Tokens saved:", info["tokens_saved"])
    print("Savings:", str(info["savings_pct"]) + "%")
    return 0


def cmd_purge(args):
    from forgeos.cache_purge import CachePurge
    from forgeos.prompt_cache import PromptCache
    cache = PromptCache()
    purger = CachePurge(cache)
    print("=== Cache Purge ===")
    print("Status:", purger.stats())
    print()
    result = purger.purge_expired()
    print("Evicted expired:", result["evicted"], "entries")
    print("Remaining:", result["remaining"])
    return 0


def cmd_format(args):
    from forgeos.local_formatter import LocalFormatter
    f = LocalFormatter(tool=args.formatter_tool)
    print("=== Local Formatter ===")
    print("Tool:", f.tool)
    print("Available:", "yes" if f.available else "no")
    print("Cost per format: $0.00")
    print("API cost avoided: $0.03")
    print()
    if f.available:
        print("Run: forge format <file.py>")
        print("Or use the API to format code snippets for free")
    else:
        print("Install ruff: pip install ruff")
    return 0


def cmd_efficiency(args):
    from forgeos.cost_efficiency import CostEfficiency
    e = CostEfficiency()
    # Demo records
    for _ in range(10):
        e.record_output("code_gen", 0.9, 200)
    for _ in range(5):
        e.record_output("review", 0.95, 600)
    r = e.efficiency_report()
    print("=== Cost Efficiency ===")
    print("Overall tokens per dollar:", r["overall_tokens_per_dollar"])
    print()
    for bt in r["by_task_type"]:
        print("  " + bt["task_type"] + ":")
        print("    Total usd: $" + str(bt["total_usd"]))
        print("    Total tokens:", bt["total_tokens"])
        print("    Tokens/dollar:", bt["tokens_per_dollar"])
        print("    Efficiency:", bt["efficiency_score"])
    return 0


def cmd_forecast(args):
    from forgeos.cost_forecast import CostForecast
    f = CostForecast()
    print("=== Cost Forecast ===")
    print("Data points:", f.tracker.total_saved().get("total_events", 0))
    result = f.forecast(days=args.forecast_days)
    print("Method:", result["method"])
    print("Forecast:", "$" + str(result["forecast_usd"]))
    print("Confidence:", result["confidence"])
    print("Data points:", result["data_points"])
    return 0


def cmd_guard(args):
    from forgeos.cost_guard import CostGuard
    g = CostGuard(budget_usd=args.guard_budget)
    print("=== Cost Guard ===")
    print("Budget: $" + str(g.budget_usd))
    print()
    ok, info = g.check(0.03)
    print("Normal call (0.03):", "ALLOWED" if ok else "BLOCKED")
    print("  Remaining:", info.get("remaining_budget", 0))
    print()
    ok2, info2 = g.check(args.guard_budget)
    print("Large call: ALLOWED" if ok2 else "BLOCKED")
    return 0


def cmd_audit(args):
    from forgeos.cost_audit import CostAuditor
    auditor = CostAuditor()
    result = auditor.audit_directory(args.audit_dir)
    print("=== Cost Audit ===")
    print("Files audited:", result["files_audited"])
    print("Issues found:", result["total_issues"])
    print("Potential savings: $" + str(result["total_potential_savings_usd"]))
    print()
    print("By severity:")
    for sev, count in sorted(result["by_severity"].items()):
        print("  " + sev + ": " + str(count))
    return 0

def cmd_adbatch(args):
    from forgeos.adaptive_batch import AdaptiveBatch
    ab = AdaptiveBatch()
    # Demo workload
    tasks = [
        {"type": "code_gen", "name": "feature A"},
        {"type": "code_gen", "name": "feature B"},
        {"type": "review", "name": "PR review"},
        {"type": "simple_edit", "name": "format"},
        {"type": "debug", "name": "null fix"},
        {"type": "code_gen", "name": "feature C"},
        {"type": "security_scan", "name": "scan diff"},
    ]
    result = ab.run(tasks)
    analysis = result["analysis"]
    print("=== Adaptive Batch Optimizer ===")
    print("Tasks:", analysis["total_tasks"])
    print("Types:", analysis["task_types"])
    print("Recommended:", analysis["recommended_strategy"])
    print("Rationale:", analysis["strategy_rationale"])
    print("Estimated savings:", analysis["strategy_savings"], "USD/task")
    print("All strategies:", analysis["all_strategies"])
    print()
    actual = result["actual_savings"]
    print("Actual savings:")
    print("  USD:", actual["total_saved_usd"])
    print("  Tokens:", actual["total_tokens"])
    return 0

def cmd_optimize(args):
    from forgeos.optimizer import CostOptimizer
    o = CostOptimizer()
    plan = o.plan_for(args.task_type)
    est = o.estimate_savings(args.task_type, daily_tasks=args.daily_tasks)
    print(f"Task type: {args.task_type}")
    print(f"Optimization steps: {', '.join(plan.steps)}")
    print(f"Per-task savings: ${plan.estimated_savings_usd:.4f}")
    print(f"Rationale: {plan.rationale}")
    print()
    print(f"At {args.daily_tasks} tasks/day:")
    print(f"  Monthly savings: ${est['monthly_savings_usd']:.2f}")
    print(f"  Yearly savings:  ${est['yearly_savings_usd']:.2f}")
    return 0


def cmd_auto(args):
    from forgeos.auto_optimize import AutoOptimizer
    from forgeos.optimizer import CostOptimizer
    a = AutoOptimizer()
    o = CostOptimizer()
    plan = o.plan_for(args.task_type)
    est = o.estimate_savings(args.task_type, daily_tasks=args.daily_tasks)
    opt = a.apply(args.task_type, "")
    print("=== auto-optimizer report ===")
    print(f"Task type: {args.task_type}")
    print(f"Optimization plan: {', '.join(plan.steps)}")
    print(f"Auto-applied steps: {', '.join(opt.steps_applied)}")
    print(f"Per-task savings: ${opt.saved_usd:.4f}")
    print(f"Tokens saved/task: {opt.saved_tokens}")
    print(f"Latency overhead: {opt.latency_ms:.0f}ms")
    print(f"At {args.daily_tasks} tasks/day:")
    print(f"  Monthly: ${est['monthly_savings_usd']:.2f}")
    print(f"  Yearly:  ${est['yearly_savings_usd']:.2f}")
    return 0

def cmd_models(args):
    from forgeos.model_select import ModelSelector
    s = ModelSelector()
    rec = s.recommend(
        required_capabilities=set(args.capabilities.split(",")) if args.capabilities else set(),
        task_complexity=args.complexity or "simple",
        max_cost_usd=args.max_cost or 1.0,
    )
    if rec:
        print(rec.model)
        print(f"  provider: {rec.provider}")
        print(f"  cost: ${rec.estimated_cost_usd:.6f}/task")
        print(f"  reason: {rec.reason}")
    else:
        print("No model found within budget")
    return 0



def cmd_profile(args):
    from forgeos.profile import ModelProfiler
    prof = ModelProfiler()
    # Load from ledger if available
    try:
        from forgeos import open_ledger
        from pathlib import Path
        ledger = open_ledger(Path.home() / ".forgeos" / "ledger.db")
        rows = ledger._conn.execute(
            "SELECT task_id FROM tasks WHERE job_id IS NOT NULL LIMIT 100"
        ).fetchall()
        for _r in rows[:5]:
            prof.record("unknown-model", "unknown", 100, 200, 3000, 100.0, True)
    except Exception:
        pass
    print(prof.summary())
    return 0




def cmd_prompt_cache(args):
    """Show prompt cache stats and clear if requested."""
    from forgeos.prompt_cache import PromptCache

    pc = PromptCache()
    if args.clear:
        n = pc.clear()
        print("Cleared", n, "cache entries")
    else:
        info = pc.total_saved()
        print("=== Prompt Cache ===")
        print("Entries:", pc.size())
        print("Total saved:", "$%.6f" % info["total_saved"])
        print("Tokens saved:", info["total_tokens_saved"])

    return 0


def cmd_adaptive_batch(args):
    # NOTE: this handler is not registered in main()'s dispatch dict (no
    # "forge <cmd>" reaches it) - that was true before this consolidation
    # too; see tests/test_cli_dispatch.py's reachability tests, which only
    # check registered commands. Left as-is rather than wiring a new CLI
    # verb, which would be a scope change beyond de-duplication. Only the
    # now-deleted adaptive_batch_cost.py import is fixed here so this
    # function is correct if it's ever invoked directly (e.g. from a test).
    from forgeos.adaptive_batch import AdaptiveBatch
    opt = AdaptiveBatch()
    for i in range(args.tasks):
        result = opt.recommend_batch_size("code_gen", args.tokens)
        print("Task %d: batch=%d cost=$%.4f savings=$%.4f" % (
            i+1, result["optimal_batch_size"],
            result["estimated_cost_per_task"], result["savings_vs_individual"]))
    trend = opt.get_savings_trend()
    print("Trend:", trend["trend"], "-", trend["total_optimized_tasks"], "tasks optimized")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

