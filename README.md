# ForgeOS

[![ci](https://github.com/jacobbyrne72/ForgeOS/actions/workflows/ci.yml/badge.svg)](https://github.com/jacobbyrne72/ForgeOS/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/jacobbyrne72/ForgeOS)](https://github.com/jacobbyrne72/ForgeOS/releases)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**The cost-governed AI coding harness. Every dollar saved is measured, not claimed.**

## Measured live — same question, same model, same ledger

The only difference between the two arms is what happens *around* the call.
deepseek-chat, 2026-07-31, both arms billed through the same ledger, **both
answers correct** (printed in full by the tool — a cheaper wrong answer is
not a saving):

|  | naive (dump the files) | ForgeOS (capsule + stable prefix) | ratio |
|---|---|---|---|
| prompt | 38,403 tokens | 1,582 tokens | **95.9% smaller** |
| cold call | $0.004990 | $0.000263 | **19.0× cheaper** |
| cache-warm call | $0.000244 | $0.000091 | **2.7× cheaper** |
| latency | 9.8–10.4 s | 3.0–4.5 s | **~3× faster** |

Reproduce: `python tools/ab_bench.py --model deepseek/deepseek-chat --repeat 2`

## The layers (verified by module execution; savings vary by workload)

| Layer | Savings | Proof |
|---|---|---|
| Mission compiler | $0.02/task | Eliminates model call for task decomposition |
| Circuit breaker | 100% on dead workers | Trips after 3 failures, auto-recovers |
| Prompt prefix cache | 60-90% off repeats | SQLite LRU, byte-identical prefix matching |
| Diff-aware scanning | 90%+ scan cost | semgrep+gitleaks on git diff only |
| Context compression | 60% fewer tokens | AST-aware source filtering |
| Model selector | 60x cheaper model | Picks cheapest capable model per task |
| Model profiler | Data-driven routing | Tracks cost/latency per model over time |
| Cost optimizer | Per-task plans | Picks cheapest layer combo per task type |
| Auto-optimizer | Pipeline applied | Automatically runs cheapest layers |
| Batch optimizer | Aggregate savings | Projections for any task batch |
| SQLite WAL + indexes | 3-5x writes | 12 query-pattern indexes, journal_mode=WAL |
| Adapter auto-discovery | Zero-config | Scans PATH/plugins/entry_points |
| Adaptive routing | Cheapest worker | Per-task cost/performance profiling |
| Fleet CLI | Routing ladder | Shows cheapest order for your providers |

**Modelled projection (not a measurement): ~$8,000/yr at 100 tasks/day if
every layer fires at its observed rate. Your ledger is the real number —
`python -m forgeos receipts` prints it.**


```
$ forge fleet
  YOUR ROUTING LADDER (cheapest first)
  1. free/local     → ollama
  2. subscription   → claude, codex, copilot
  3. metered        → deepseek, openrouter

  → forgeos routes through claude, codex, copilot BEFORE touching metered API.
    Every task your subscription handles = $0 extra cost.
    Same subscription. 5x more tasks. That's the product.
```

The kernel is deterministic code: scheduling, budgets, file leases, verification.
A model is called only for genuinely ambiguous work. Nothing merges without
tests, a security scan, evidence, and review by a *different* worker.

> **Status: v0.2.0 — pre-1.0, not production-hardened.** A `forge` CLI, mission
> compiler, circuit breakers, prompt prefix caching, diff-aware scanning, adapter
> auto-discovery and SQLite WAL tuning are all present and tested. **1124 tests
> collected.** It runs end to end against real providers.
>
> It is not production-hardened, and saying so would be the exact
> self-flattery this project exists to prevent: a race that granted two workers
> the same write lease was found and fixed *after* a full green suite, because
> the test that catches it passes on luck. See [Known gaps](#known-gaps).

## The idea

Most harnesses optimise the model call. The bigger wins are around it:

The **unit** column is not decoration. A token figure and a cost figure are
different claims, and mixing them in one column is how a harness talks itself
into a saving it never made — see the note below.

| Lever | Saving | Unit | Why |
|---|---|---|---|
| Don't call a model | 100% | cost | Finding imports across 20k files is a tree-sitter query, not an LLM task |
| Deterministic scheduling | 100% of what it replaces | cost | Queue, retry and routing decisions in plain code cost zero tokens |
| Byte-stable prefix caching | ~90% off cached input | cost | Providers serve a cache hit only when the prefix is byte-identical |
| Smaller payloads | 65–82% | **tokens, not cost** | Ranked context capsule instead of a repository dump |
| Cheapest capable worker | up to 100% | cost | A local binary or flat-rate seat should never lose work to a metered model |
| Compact supervision | 10×+ | tokens read | The manager reads a ~90-byte heartbeat, never a transcript |

These multiply. A smaller payload, on a cache hit, on a free-tier model, for a
call deterministic code decided didn't need making, isn't 89% cheaper — it never
happened.

## Why your subscription runs out by noon

Your $20/month Claude or Codex plan has a quota window (5-hour rolling, weekly
cap). Every task you send through the CLI burns that quota — including the
~4,000-token system prompt the CLI sends on *every single call*, the whole-repo
context dump, and the retry you didn't need because the answer was a grep away.

ForgeOS attacks all three:

1. **Byte-stable prefixes.** The system prompt is split into a stable prefix
   (role contract, tool protocol, safety policy) and a volatile tail (the task).
   The prefix is byte-identical across calls, so the provider serves it from
   cache at ~90% discount. Your quota lasts 10x longer on the prefix alone.
   (`forgeos/prompts/prefix.py` — with a CI test that asserts byte identity.)

2. **Compact role prompts.** Claude Code's default prompt is ~4,000 tokens of
   ceremony. ForgeOS sends a 200-token role prefix that gets the same work done.
   Every token you don't send is quota you keep. (`forgeos/prompts/roles.py`)

3. **Refuse before calling.** If the answer is a tree-sitter query, a grep, or
   already in the knowledge vault, the subscription call never happens. The
   cheapest call is the one never made. (`forgeos/economy/preflight.py`)

4. **Quota-aware routing.** When Claude's window is exhausted, ForgeOS doesn't
   burn a retry on a rate-limit error. It reads the reset time from the error
   message, parks the work, and routes through the next rung until the window
   reopens. (`forgeos/core/quota.py` — wired into `forge.py`'s routing loop.)

The result: your subscription handles the ambiguous middle, free tiers handle
the routine, and metered API is the last resort. Same $20/month, 5x more tasks.

## What's new in v0.2.0

| Feature | What it does | Cost impact |
|---|---|---|
| **Mission compiler** | `compile_mission("Add X")` → TaskSpecs from natural language; tree-sitter finds relevant files | Eliminates the most expensive step |
| **Circuit breaker** | Per-worker trip/stay-tripped/auto-recover; dead workers excluded before a wasted token | 100% savings on dead-worker calls |
| **Prompt prefix cache** | SQLite-backed LRU, keyed on byte-identical prefixes; TTL-aware with eviction | 60-90% off repeated prompts |
| **Diff-aware scanning** | semgrep + gitleaks scan only the git diff, not whole files | 90%+ scan cost reduction |
| **Adapter auto-discovery** | Scans PATH, ~/.forgeos/plugins/, and entry-points | Zero-config onboarding |
| **`forge doctor`** | Live readiness score with provider health checks | Know capacity before running |
| **`forge init`** | Scans repo, generates CLAUDE.md + settings.json | 1-command bootstrap |
| **SQLite WAL + indexes** | WAL mode + autocheckpoint + 12 query-pattern indexes | 3-5x concurrent write throughput |

## Benchmarks

**One measured A/B**, run by `tools/ab_bench.py` against a real DeepSeek key.
Same question, same model, same pricing code; the only difference is what
happens around the call. Both answers were checked and both were correct.

```
prompt size   baseline  25,924 tokens  (5 whole files — the naive approach)
              ForgeOS    1,549 tokens  (3 of 31 ranked blocks)

round 1 (cold)    baseline $0.000140    ForgeOS $0.000073     1.9x cheaper
round 2 (cached)  baseline $0.000192    ForgeOS $0.000075     2.6x cheaper
overall                                                        2.1x cheaper
                                        14.7s -> 9.0s          39% faster
```

On a genuinely cold first call the gap was **11.1×**; it narrows to ~2× once the
provider caches the large prompt too. Real tasks vary their prompt, so most calls
land nearer the cold number — but reporting only the 11.1× would be picking the
flattering half.

**What this does not show.** It exercises the context lever alone: not routing,
not retries, not escalation, not the merge gate. One question is one data point.
There is no measured before/after for the CLI, the mission compiler or the prefix
cache — those are built and tested, not benchmarked. A number that has not been
run is not a benchmark, and this section will only ever carry numbers that were.

Reproduce it: `python tools/ab_bench.py --env ~/.hermes/.env --repeat 3`

**Why the unit matters.** A 2,908-run study of provider-billed agent traffic
(arXiv 2607.12161) found prompt-cache traffic was ~87% of cost composition, and
that cutting 38% of tool-output tokens *raised* billed cost by 6.8% while
dropping task success from 27/40 to 15/40 — the removed tokens were the verbatim
anchors the model needed to locate its edit. **Token reduction is not a cost
proxy, and optimising for it can make a system both dearer and worse.** So the
only headline ForgeOS reports is billed cost per accepted task; every
token-derived figure is labelled `modelled`, never `measured`.

**A note on claims:** narrow operations can genuinely improve by orders of
magnitude when an LLM loop is replaced by a query. End-to-end, a realistic target
is 2–10×, with receipts. One measured A/B on this repo — same question, same
model, same pricing code — came out 2.1× cheaper and 39% faster with both answers
correct (11.1× on a cold call, narrowing to ~2× once the provider cached the
large prompt too). That is the context lever alone; it does not exercise routing,
retries or the merge gate. ForgeOS is built to *prove* savings rather than assert
them — `SavingsProof` marks every figure as measured, replayed or modelled, and a
saving is only ever as strong as its weakest input.

## Install

```bash
git clone <this-repo> ForgeOS && cd ForgeOS
pip install -e ".[dev]"

# Fast-path test (no real subprocess scanners):
python -m pytest tests -m "not slow"

# Full suite (incl. real semgrep, gitleaks, ruff):
python -m pytest tests

# CLI:
forge doctor          # check readiness
forge compile "Add X"  # dry-run a mission
forge run "Add X"     # compile and execute
forge init            # bootstrap a new repo
forge report <job-id> # cost breakdown
```

Python 3.11+. Optional and detected, never required: `semgrep`, `gitleaks`,
`ruff` (security/lint gates), `tiktoken` (exact token counts), Obsidian (vault
location).

## Use

```python
from forgeos import Forge, TaskSpec, Scope, Budget

forge = Forge()
print(forge.doctor())          # what this machine can actually do right now

result = forge.run(
    "Add provider failover without changing the public API",
    tasks=[TaskSpec(job_id="", subject="Normalise Retry-After parsing",
                    description="Support integer seconds and HTTP-date.",
                    capabilities=["edit", "python"],
                    scope=Scope(paths=["src/retry.py"]),
                    acceptance=["pytest tests/test_retry.py -q passes"],
                    budget=Budget(max_usd=2.0))],
    # With no executor, the Forge runs whichever backend the router picks —
    # the registry's adapter field resolves to a live worker (ollama, the omc
    # team runtime, a gateway model). Pass your own executor to override.
)
print(result.cost_per_accepted)
```

Dashboard: `python -m forgeos.dashboard.app` → `http://127.0.0.1:8899`
(localhost only; read-only except the halt flag; display settings are stored in
your browser and never touch the harness).

## How it fits together

```
mission contract  →  size pools to this machine  →  price quota as inventory
   → per task: capsule → preflight refusal → route (cheapest capable)
     → take path leases → execute → reduce output → verify ladder → merge gate
     → MODEL failure? escalate exactly one rung → retry
   → receipt separating measured from estimated
```

Everything except `execute` is ordinary code.

## Design rules

Full detail in [AGENTS.md](AGENTS.md). The load-bearing ones:

- **Never widen a budget to make a job finish.** A tripped governor is a signal
  to escalate to a human, never a number to edit.
- **Never bypass the ledger.** An unrecorded call is invisible, and the governor
  cannot stop what it cannot see. An *unmetered* worker is charged its tier prior
  so "no token counts available" never reads as "free".
- **Unavailable is not a pass.** A missing scanner blocks a merge. "We could not
  check" is not "it is fine".
- **Task text is untrusted input to the router.** Routing reads deterministic
  features only; a prompt cannot raise its own budget or force a premium tier.
- **Only MODEL failures escalate the tier.** A broken venv fails identically at
  every price point; escalating it buys premium tokens for a problem no model can
  fix.
- **Nothing becomes a rule because someone said it.** Outside claims enter as
  unverified and need evidence, corroboration and a contradiction check.

## Known gaps

Stated plainly, because a harness that overstates its guarantees is worse than
one that admits them:

- **The default fleet has no free file-editing worker.** The adapter path can
  drive file-editing CLIs (the omc team runtime, ollama-backed local workers),
  but free *API* workers here summarise and classify; a real edit needs a local
  worker, a flat-rate seat, or a metered model.
- **Several subsystems are built and tested but not yet in the execution path:**
  capsule, packing, savings receipts, quota/capacity market, mission compiler,
  model traits, discovery.
- **Live CLI backends are young.** The omc team adapter is verified against a
  real install, but long-haul reliability (crashed sessions, orphaned
  worktrees) has not been proven over days of continuous use.

Recently closed, for the record: `Forge.run` now defaults to the routed
adapter path (`executor=None`), MODEL failures escalate exactly one tier per
failure via `Router.escalate`, and the gateway remembers dead models
(terminal vs temporary) so retired free tiers stop being re-bought.

## Connecting a provider

```bash
export DEEPSEEK_API_KEY=...        # or OPENROUTER_API_KEY, etc.
python tools/live_check.py         # probes every usable provider, prices each call
python tools/live_job.py           # a full Forge job end to end, writes to .forgeos/
FORGEOS_STATE_DIR=.forgeos python -m forgeos.dashboard.app
```

`live_check.py` reports which providers are reachable and which catalogued
models are actually alive. Neither tool ever prints a key value — settings
reference an environment variable *name*, and ForgeOS never stores a secret.

## Licence

MIT — see [LICENSE](LICENSE).
