"""Prove the gateway path end to end against real providers.

Run this after adding an API key to confirm the whole chain works: settings sees
the provider, a transport is built for it, the preflight prices the call, the
call goes out, the response is priced, and the spend lands in the ledger.

    python tools/live_check.py --env ~/.hermes/.env
    python tools/live_check.py --provider deepseek --dry-run

**It spends real money.** Tiny amounts — the probe prompt is a few dozen tokens —
but it is a real call to a real billed endpoint, so it is opt-in and prints the
estimate before spending anything. `--dry-run` stops after the estimate.

Key values are never printed, logged, or written anywhere. `--env` loads a
dotenv-style file into this process only; the script reports the *name* of each
variable it picked up and nothing else.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forgeos.catalog import default_catalog  # noqa: E402
from forgeos.contracts import Budget, JobSpec  # noqa: E402
from forgeos.economy.preflight import CallRefused, estimate_call  # noqa: E402
from forgeos.gateway.client import Gateway, GatewayRequest, default_transports  # noqa: E402
from forgeos.gateway.dead_models import DeadModelStore  # noqa: E402
from forgeos.ledger import open_ledger  # noqa: E402
from forgeos.settings import ProviderKind, Settings  # noqa: E402

DEAD_MODELS_PATH = Path(os.path.expanduser("~/.forgeos/dead_models.db"))

PROBE = "Reply with exactly one word: ok"


def load_env(path: Path, wanted: tuple[str, ...]) -> list[str]:
    """Load only the named variables. Returns the NAMES loaded, never values.

    Loading the whole file would pull unrelated credentials (exchange keys, bot
    tokens) into a process that has no business holding them.
    """
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        if name not in wanted:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="", help="dotenv file to read provider keys from")
    ap.add_argument("--provider", default="", help="only probe this provider")
    ap.add_argument("--model", default="", help="explicit model ref, e.g. deepseek/deepseek-chat")
    ap.add_argument("--dry-run", action="store_true", help="stop after the cost estimate")
    ap.add_argument("--budget-usd", type=float, default=0.05, help="hard ceiling for this probe")
    ap.add_argument("--tries", type=int, default=6,
                    help="candidate models to try per provider before giving up")
    ap.add_argument("--max-output", type=int, default=64,
                    help="output token cap; reasoning models need headroom to emit any text at all")
    args = ap.parse_args()

    settings = Settings.load()
    wanted = tuple(
        p.env_key for p in settings.providers.values() if p.env_key
    )
    if args.env:
        names = load_env(Path(os.path.expanduser(args.env)), wanted)
        print(f"loaded {len(names)} key(s) from {args.env}: {', '.join(names) or '(none)'}")
        print("  (names only — no value is ever printed)\n")

    # Reload so `usable` is recomputed now that the env is populated.
    settings = Settings.load()

    print(f"{'provider':<12} {'kind':<10} {'status'}")
    print("-" * 46)
    for p in sorted(settings.providers.values(), key=lambda x: (x.kind.value, x.name)):
        if p.kind in (ProviderKind.API, ProviderKind.GATEWAY):
            print(f"{p.name:<12} {p.kind.value:<10} {p.status()}")

    transports = default_transports(settings)
    print("\ntransports built:")
    for t in transports:
        serves = getattr(t, "serves", set())
        print(f"  {t.name:<12} serves={sorted(serves) if serves else 'any'}")

    catalog = default_catalog()
    # Per provider, an ordered candidate list rather than a single pick. Catalogued
    # "free" slugs rot — a provider retires a free tier and the model id 404s while
    # every catalogue still lists it at $0. Falling through to the next candidate is
    # what a free-tier pool has to do anyway, so the probe does it too and reports
    # which slugs are actually alive.
    plans: list[tuple[str, list[str]]] = []
    if args.model:
        card = catalog.get(args.model)
        plans = [(card.provider if card else "?", [args.model])]
    else:
        for p in sorted(settings.providers.values(), key=lambda x: x.name):
            if p.kind is not ProviderKind.API or not p.usable:
                continue
            if args.provider and p.name != args.provider:
                continue
            cards = [c for c in catalog.all() if c.provider == p.name]
            if not cards:
                continue
            # Cheapest first — a probe should never reach for a premium model.
            cards.sort(key=lambda c: (c.input_cost_per_1m + c.output_cost_per_1m))
            plans.append((p.name, [c.ref for c in cards[: args.tries]]))

    if not plans:
        print("\nno usable API provider with a catalogued model — nothing to probe")
        return 1

    ledger = open_ledger(":memory:")
    job_id = ledger.open_job(
        JobSpec(
            objective="live provider check",
            cwd=str(Path.cwd()),
            budget=Budget(max_usd=args.budget_usd),
        )
    )
    # Persistent across runs (not ":memory:" like the ledger above): this is
    # the actual fix for re-probing the same retired free-tier slug on every
    # invocation -- a model_ref this script already learned is 404 on a given
    # transport stays skipped, network call and all, until DEAD_MODELS_PATH is
    # cleared or DeadModelStore.clear() is called for that pair explicitly.
    dead_models = DeadModelStore(DEAD_MODELS_PATH)
    gateway = Gateway(catalog=catalog, ledger=ledger, settings=settings, transports=transports,
                       dead_models=dead_models)

    failures = 0
    attempted = 0
    for provider, refs in plans:
        print(f"\n=== {provider} ===")
        succeeded = False
        for ref in refs:
            card = catalog.get(ref)
            if card is None:
                print(f"  {ref}: NOT IN CATALOG")
                continue

            request = GatewayRequest(
                model_ref=ref, prompt_tail=PROBE, max_output_tokens=args.max_output
            )
            est = estimate_call(request.prompt, request.max_output_tokens, card)
            print(f"  {ref}")
            print(f"    estimate  ${est.usd_micros / 1_000_000:.6f} "
                  f"({est.tokens_in} in / {est.tokens_out} out cap, exact_tokens={est.exact})")
            if args.dry_run:
                succeeded = True
                break

            attempted += 1
            try:
                resp = gateway.complete(
                    request,
                    job_id=job_id,
                    task_id=None,
                    worker_id=f"live-check.{card.provider}",
                    remaining_micros=(
                        ledger.job(job_id)["max_usd_micros"] - ledger.job_spend_micros(job_id)
                    ),
                )
            except CallRefused as e:
                print(f"    REFUSED   {e}")
                break  # a refusal is the budget talking; the next model will not help
            except Exception as e:
                print(f"    dead      {e}"[:400])
                continue

            print(f"    reply     {resp.text.strip()[:60]!r}")
            print(f"    actual    ${resp.usd_micros / 1_000_000:.6f} "
                  f"({resp.tokens_in} in + {resp.tokens_cached_in} cached / {resp.tokens_out} out)")
            print(f"    measured  exact_usage={resp.exact_usage} cache_hit={resp.cache_hit}")
            print(f"    served by {resp.provider_used} as {resp.model_used}")
            succeeded = True
            break

        if not succeeded:
            print(f"  NO LIVE MODEL for {provider} in {len(refs)} candidate(s)")
            failures += 1

    spent = ledger.job_spend_micros(job_id)
    print(f"\nledger recorded ${spent / 1_000_000:.6f} across {attempted} call(s)")
    print("this figure came from the ledger, not from the responses above — "
          "if it were zero, the governor would have nothing to enforce against")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
