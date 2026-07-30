"""A/B the harness against the naive way of doing the same job.

Same question, same model, same provider, same pricing code. The only difference
is what happens *around* the call:

    baseline   dump every candidate file into the prompt, ask, take the answer
    forgeos       rank and pack a bounded capsule, byte-stable prefix, capped output

Run it:

    python tools/ab_bench.py --env ~/.hermes/.env
    python tools/ab_bench.py --model openrouter/openrouter/free --repeat 2

Both arms are priced by `Gateway.complete` and recorded in the same ledger, so
the cost comparison is the system measuring itself rather than a claim about
itself. **Both answers are printed in full.** A cheaper wrong answer is not a
saving, and a benchmark that hides the outputs is asking you to take the
interesting half on trust.

Honest about what this does and does not show:

- It measures the *context* lever (send less), which is the one that dominates
  in practice. It does not exercise routing, retries, or the merge gate.
- One question is one data point. `--repeat` runs the pair more than once, which
  also surfaces the prefix-cache effect on the second and later runs.
- The baseline is deliberately the *reasonable* naive approach — the relevant
  files, not the whole repository. Padding it would manufacture a bigger number.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forgeos.catalog import default_catalog  # noqa: E402
from forgeos.contracts import Budget, JobSpec  # noqa: E402
from forgeos.economy.capsule import CapsuleBuilder, RefKind, make_ref  # noqa: E402
from forgeos.economy.preflight import count_tokens  # noqa: E402
from forgeos.gateway.client import Gateway, GatewayRequest, default_transports  # noqa: E402
from forgeos.ledger import open_ledger  # noqa: E402
from forgeos.settings import ProviderKind, Settings  # noqa: E402
from live_check import load_env  # noqa: E402

QUESTION = (
    "In this codebase, which function records spend to the ledger, and what "
    "specifically prevents the same model call being charged twice? "
    "Name the function and the guard. Answer in under 80 words."
)

# The files a person would reach for. Not the whole repo — an inflated baseline
# would manufacture a saving rather than measure one.
CANDIDATES = [
    "forgeos/forge.py",
    "forgeos/ledger.py",
    "forgeos/gateway/client.py",
    "forgeos/adapters/gateway_worker.py",
    "forgeos/core/governor.py",
]

CAPSULE_BUDGET_TOKENS = 1_500
WINDOW = 14  # lines of context kept around each hit


@dataclass
class Arm:
    name: str
    tokens_in: int = 0
    tokens_cached: int = 0
    tokens_out: int = 0
    usd_micros: int = 0
    seconds: float = 0.0
    answers: list[str] = field(default_factory=list)
    cache_hits: int = 0
    calls: int = 0
    rounds: list[dict] = field(default_factory=list)

    def add(self, resp, seconds: float) -> None:
        self.rounds.append({
            "in": resp.tokens_in, "cached": resp.tokens_cached_in,
            "out": resp.tokens_out, "usd": resp.usd_micros,
            "sec": seconds, "hit": resp.cache_hit,
        })
        self.tokens_in += resp.tokens_in
        self.tokens_cached += resp.tokens_cached_in
        self.tokens_out += resp.tokens_out
        self.usd_micros += resp.usd_micros
        self.seconds += seconds
        self.cache_hits += 1 if resp.cache_hit else 0
        self.calls += 1
        self.answers.append(resp.text.strip())


def build_baseline_prompt(root: Path) -> str:
    """Everything relevant, concatenated. What you do when you have no retrieval."""
    parts = []
    for rel in CANDIDATES:
        p = root / rel
        if p.exists():
            parts.append(f"===== {rel} =====\n{p.read_text(encoding='utf-8', errors='replace')}")
    return "\n\n".join(parts) + f"\n\n{QUESTION}"


def _score(block: str, terms: list[str]) -> int:
    low = block.lower()
    return sum(low.count(t) for t in terms)


def build_capsule_prompt(root: Path) -> tuple[str, dict]:
    """Rank, window, and pack to a hard budget.

    Deterministic and model-free: term hits over the same candidate files, kept
    to a window around each match, best-scoring first until the budget is spent.
    No model is consulted to decide what to send — that would spend tokens to
    save tokens.
    """
    terms = ["record_spend", "spend", "double", "charge", "ledger",
             "already_recorded", "usd_micros", "twice"]

    blocks: list[tuple[int, str, str]] = []
    for rel in CANDIDATES:
        p = root / rel
        if not p.exists():
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        hits = [i for i, ln in enumerate(lines)
                if any(t in ln.lower() for t in ("record_spend", "already_recorded",
                                                 "usd_micros", "charged"))]
        # Merge overlapping windows so one region is not sent three times.
        spans: list[tuple[int, int]] = []
        for i in hits:
            lo, hi = max(0, i - WINDOW // 2), min(len(lines), i + WINDOW // 2)
            if spans and lo <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
            else:
                spans.append((lo, hi))
        for lo, hi in spans:
            body = "\n".join(lines[lo:hi])
            blocks.append((_score(body, terms), f"{rel}:{lo + 1}-{hi}", body))

    blocks.sort(key=lambda b: -b[0])

    # A ContextRef is a reference, not a payload — it carries a hash and a token
    # price so the same slice is never counted twice. The bodies stay here, keyed
    # by ref, and only the ones the builder accepted get sent.
    bodies: dict[str, str] = {}
    builder = CapsuleBuilder(budget=CAPSULE_BUDGET_TOKENS)
    for score, ref, body in blocks:
        path, span = ref.rsplit(":", 1)
        uri = f"file_slice://{path}#L{span}"
        bodies[uri] = body
        builder.add(make_ref(RefKind.FILE_SLICE, uri, body, f"{score} term hits"))

    capsule = builder.finish(objective=QUESTION, acceptance=["names the function and the guard"],
                            write_scope=[])
    packed = "\n\n".join(f"===== {r.ref} =====\n{bodies[r.ref]}" for r in builder.items)
    stats = {
        "blocks_found": len(blocks),
        "blocks_sent": len(builder.items),
        "blocks_dropped": len(builder.excluded),
        "capsule_tokens": builder.total_tokens,
        "read_scope": len(capsule.read_scope),
    }
    return packed + f"\n\n{QUESTION}", stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="~/.hermes/.env")
    ap.add_argument("--model", default="", help="e.g. deepseek/deepseek-chat or openrouter/openrouter/free")
    ap.add_argument("--repeat", type=int, default=2, help="run the pair N times (shows cache effect)")
    ap.add_argument("--budget-usd", type=float, default=0.50)
    ap.add_argument("--max-output", type=int, default=200)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent

    settings = Settings.load()
    wanted = tuple(p.env_key for p in settings.providers.values() if p.env_key)
    names = load_env(Path(os.path.expanduser(args.env)), wanted)
    print(f"keys loaded (names only): {', '.join(names) or '(none)'}")
    settings = Settings.load()

    catalog = default_catalog()
    model_ref = args.model
    if not model_ref:
        for p in sorted(settings.providers.values(), key=lambda x: x.name):
            if p.kind is ProviderKind.API and p.usable:
                cards = [c for c in catalog.all() if c.provider == p.name]
                if cards:
                    model_ref = min(cards, key=lambda c: c.input_cost_per_1m
                                    + c.output_cost_per_1m).ref
                    break
    card = catalog.get(model_ref) if model_ref else None
    if card is None:
        print(f"no usable model (asked for {model_ref!r})")
        return 1

    baseline_prompt = build_baseline_prompt(root)
    capsule_prompt, cap_stats = build_capsule_prompt(root)

    b_tok = count_tokens(baseline_prompt, card.model_id).tokens
    h_tok = count_tokens(capsule_prompt, card.model_id).tokens

    print(f"\nmodel      {model_ref}  (${card.input_cost_per_1m}/1M in, "
          f"${card.output_cost_per_1m}/1M out)")
    print(f"question   {QUESTION[:70]}...")
    print(f"\nprompt size")
    print(f"  baseline  {b_tok:>7,} tokens  ({len(CANDIDATES)} whole files)")
    print(f"  forgeos      {h_tok:>7,} tokens  ({cap_stats['blocks_sent']} of "
          f"{cap_stats['blocks_found']} ranked blocks, {cap_stats['blocks_dropped']} over budget)")
    if b_tok:
        print(f"  reduction {100 * (1 - h_tok / b_tok):>6.1f}%")

    ledger = open_ledger(":memory:")
    job_id = ledger.open_job(JobSpec(objective="a/b bench", cwd=str(root),
                                     budget=Budget(max_usd=args.budget_usd)))
    gateway = Gateway(catalog=catalog, ledger=ledger, settings=settings,
                      transports=default_transports(settings))

    baseline, forgeos = Arm("baseline"), Arm("forgeos")

    def call(arm: Arm, prompt: str, prefix: str = "") -> bool:
        req = GatewayRequest(model_ref=model_ref, prompt_prefix=prefix, prompt_tail=prompt,
                             max_output_tokens=args.max_output)
        t0 = time.monotonic()
        try:
            resp = gateway.complete(
                req, job_id=job_id, task_id=None, worker_id=f"bench.{arm.name}",
                remaining_micros=(args.budget_usd * 1_000_000) - ledger.job_spend_micros(job_id),
            )
        except Exception as e:
            print(f"  {arm.name}: FAILED {str(e)[:160]}")
            return False
        arm.add(resp, time.monotonic() - t0)
        return True

    # forgeos keeps a byte-stable prefix so a provider can serve a cache hit on the
    # runs after the first. The baseline has no such discipline, which is the
    # point: caching is not free, it is a property you have to preserve.
    FORGEOS_PREFIX = "You are answering a question about a Python codebase.\n"

    print(f"\nrunning {args.repeat} round(s)...")
    for i in range(args.repeat):
        ok_b = call(baseline, baseline_prompt)
        ok_h = call(forgeos, capsule_prompt, prefix=FORGEOS_PREFIX)
        print(f"  round {i + 1}: baseline={'ok' if ok_b else 'failed'} "
              f"forgeos={'ok' if ok_h else 'failed'}")

    print("\n" + "=" * 74)
    print(f"{'':10} {'calls':>5} {'tok in':>9} {'cached':>8} {'tok out':>8} "
          f"{'USD':>11} {'sec':>7}")
    for arm in (baseline, forgeos):
        print(f"{arm.name:10} {arm.calls:>5} {arm.tokens_in:>9,} {arm.tokens_cached:>8,} "
              f"{arm.tokens_out:>8,} {arm.usd_micros / 1e6:>11.6f} {arm.seconds:>7.1f}")

    # Per round, because the two regimes tell different stories and reporting
    # only the flattering one would be the sort of benchmark this project exists
    # to distrust. Round 1 is cold: the baseline pays full freight for every
    # token. Later rounds hit the provider's prompt cache, which makes a big
    # repeated prompt much cheaper — so the gap narrows sharply.
    #
    # Which number matters depends on the workload. Identical prompts repeated
    # all day look like the warm row; real tasks vary their prompt and land
    # nearer the cold one.
    print("\nper round (cold first, then provider-cached)")
    print(f"{'round':>5} {'arm':10} {'fresh in':>9} {'cached':>8} {'out':>6} "
          f"{'USD':>10} {'hit':>4}")
    for i in range(max(len(baseline.rounds), len(forgeos.rounds))):
        for arm in (baseline, forgeos):
            if i < len(arm.rounds):
                r = arm.rounds[i]
                print(f"{i + 1:>5} {arm.name:10} {r['in']:>9,} {r['cached']:>8,} "
                      f"{r['out']:>6,} {r['usd'] / 1e6:>10.6f} {'yes' if r['hit'] else 'no':>4}")
        if baseline.rounds and forgeos.rounds and i < min(len(baseline.rounds), len(forgeos.rounds)):
            b, h = baseline.rounds[i]["usd"], forgeos.rounds[i]["usd"]
            if h:
                print(f"{'':5} {'ratio':10} {b / h:>9.1f}x cheaper")

    if baseline.usd_micros and forgeos.calls:
        saved = 1 - (forgeos.usd_micros / baseline.usd_micros)
        print(f"\ncost      forgeos is {saved * 100:.1f}% cheaper overall "
              f"({baseline.usd_micros / 1e6:.6f} -> {forgeos.usd_micros / 1e6:.6f})")
        if baseline.usd_micros > forgeos.usd_micros:
            print(f"          {baseline.usd_micros / max(forgeos.usd_micros, 1):.1f}x")
    if baseline.seconds and forgeos.seconds:
        print(f"latency   {baseline.seconds:.1f}s -> {forgeos.seconds:.1f}s "
              f"({(1 - forgeos.seconds / baseline.seconds) * 100:+.1f}%)")
    print(f"cache     baseline {baseline.cache_hits}/{baseline.calls} hits, "
          f"forgeos {forgeos.cache_hits}/{forgeos.calls} hits")
    print(f"ledger    ${ledger.job_spend_micros(job_id) / 1e6:.6f} recorded across both arms")

    print("\n" + "=" * 74)
    print("ANSWERS — judge these yourself; a cheaper wrong answer is not a saving.\n")
    for arm in (baseline, forgeos):
        print(f"--- {arm.name} ---")
        print(_console_safe(arm.answers[0] if arm.answers else "(no answer)"))
        print()
    return 0


def _console_safe(text: str) -> str:
    """Models emit characters a legacy console encoding cannot represent.

    A narrow no-break space in an answer crashed this tool on a cp1252 stdout
    *after* both calls had been paid for — losing the entire result to a
    formatting detail. Re-encoding through the console's own codec keeps the
    answer readable and the run non-fatal.
    """
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
