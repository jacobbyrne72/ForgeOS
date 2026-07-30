"""Run a real Forge job end to end against a real provider.

The whole chain, nothing faked:

    settings -> transport -> GatewayWorkerAdapter -> adapter_executor
      -> Forge (route, lease, execute, reduce, verify, merge gate) -> ledger

Writes to `.forgeos/` in the current directory so the dashboard has something real
to show. Spends real money — cents at most with the default budget, but real.

    python tools/live_job.py --env ~/.hermes/.env
    python tools/live_job.py --env ~/.hermes/.env --model deepseek/deepseek-chat

Key values are never printed or stored; only variable NAMES are reported.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forgeos.adapters.executor import adapter_executor  # noqa: E402
from forgeos.adapters.factory import build_adapter  # noqa: E402
from forgeos.catalog import default_catalog  # noqa: E402
from forgeos.contracts import Budget, Scope, TaskSpec  # noqa: E402
from forgeos.forge import Forge  # noqa: E402
from forgeos.gateway.client import Gateway, default_transports  # noqa: E402
from forgeos.registry import Adapter, CostTier, Registry, WorkerProfile  # noqa: E402
from forgeos.settings import ProviderKind, Settings  # noqa: E402
from live_check import load_env  # noqa: E402


def pick_model(catalog, settings, explicit: str) -> str:
    if explicit:
        return explicit
    for p in sorted(settings.providers.values(), key=lambda x: x.name):
        if p.kind is not ProviderKind.API or not p.usable:
            continue
        cards = [c for c in catalog.all() if c.provider == p.name]
        if cards:
            return min(cards, key=lambda c: c.input_cost_per_1m + c.output_cost_per_1m).ref
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="~/.hermes/.env")
    ap.add_argument("--model", default="")
    ap.add_argument("--state-dir", default=".forgeos")
    ap.add_argument("--budget-usd", type=float, default=0.10)
    args = ap.parse_args()

    settings = Settings.load()
    wanted = tuple(p.env_key for p in settings.providers.values() if p.env_key)
    names = load_env(Path(os.path.expanduser(args.env)), wanted)
    print(f"keys loaded (names only): {', '.join(names) or '(none)'}")
    settings = Settings.load()

    catalog = default_catalog()
    model_ref = pick_model(catalog, settings, args.model)
    if not model_ref:
        print("no usable API provider — set a key first")
        return 1
    card = catalog.get(model_ref)
    if card is None:
        print(f"{model_ref} is not in the catalog")
        return 1
    print(f"model: {model_ref}  (${card.input_cost_per_1m}/1M in, ${card.output_cost_per_1m}/1M out)")

    state_dir = Path(args.state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    # A worker whose capabilities are honest about what an HTTP completion can do:
    # it produces text. It cannot edit files, so `can_edit_files` stays False and
    # the task below asks for analysis rather than an edit.
    worker = WorkerProfile(
        worker_id=f"gateway.{card.provider}",
        adapter=Adapter.GATEWAY,
        model=model_ref,
        tier=CostTier.CHEAP,
        capabilities={"summarize", "review", "plan", "classify", "triage"},
        can_edit_files=False,
        prior_win_rate=0.7,
        est_seconds=20.0,
    )
    reviewer = WorkerProfile(
        worker_id=f"gateway.{card.provider}.reviewer",
        adapter=Adapter.GATEWAY,
        model=model_ref,
        tier=CostTier.CHEAP,
        capabilities={"review", "verify"},
        can_edit_files=False,
        prior_win_rate=0.7,
        est_seconds=20.0,
    )
    forge = Forge(home=state_dir, registry=Registry([worker, reviewer]))
    gateway = Gateway(
        catalog=catalog,
        ledger=forge.ledger,
        settings=settings,
        transports=default_transports(settings),
    )

    def remaining() -> int:
        return int(args.budget_usd * 1_000_000)

    executors = {}
    for profile in (worker, reviewer):
        adapter, why = build_adapter(
            profile, gateway=gateway, job_id="live", remaining_micros=remaining
        )
        if adapter is None:
            print(f"could not build {profile.worker_id}: {why}")
            return 1
        print(f"{profile.worker_id}: {why}")
        executors[profile.worker_id] = adapter_executor(
            adapter, cwd=str(state_dir), model_profile=model_ref, timeout_seconds=120
        )

    execute = executors[worker.worker_id]
    # A real second call from a genuinely different worker id. The merge gate
    # rejects a reviewer whose id is derived from the implementer's, so this
    # cannot be satisfied by relabelling the same worker.
    review = executors[reviewer.worker_id]

    task = TaskSpec(
        job_id="",
        subject="Summarise the cost rule",
        description=(
            "In one sentence: why is cost per accepted task a better optimisation "
            "target than cost per model call? Answer in under 40 words."
        ),
        capabilities=["summarize"],
        scope=Scope(paths=[]),
        acceptance=["a one sentence answer is produced"],
        budget=Budget(max_usd=args.budget_usd, max_seconds=120),
    )

    print("\nrunning...\n")
    result = forge.run(
        "prove the gateway path end to end",
        tasks=[task],
        executor=execute,
        reviewer=review,
        cwd=str(state_dir),
        budget=Budget(max_usd=args.budget_usd),
    )

    print(f"accepted={result.accepted} rejected={result.rejected}")
    for o in result.outcomes:
        print(f"  worker={o.worker_id} tier={o.tier} reason={o.reason}")
        if o.merge_reasons:
            print(f"  merge_reasons={o.merge_reasons}")
        print(f"  spend=${o.usd_micros / 1_000_000:.6f}")
    print(f"\njob spend  ${result.spend_usd:.6f}")
    print(f"cache hit  {result.cache_hit_pct:.1f}%")
    print(f"state dir  {state_dir}")
    print("\nstart the dashboard against this run:")
    print(f"  FORGEOS_STATE_DIR={state_dir} python -m forgeos.dashboard.app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
