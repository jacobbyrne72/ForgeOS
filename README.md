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
| cold call | $0.004990 | $0.000263 | 19.0× cheaper |
| cache-warm call | $0.000244 | $0.000091 | **2.7× cheaper** |
| latency | 9.8–10.4 s | 3.0–4.5 s | **~3× faster** |

Reproduce live: `python tools/ab_bench.py --live --model deepseek/deepseek-chat --repeat 2`
Preview without spending: `python tools/ab_bench.py --model deepseek/deepseek-chat --json-out artifacts/ab-bench.json`

**Read the warm row, not the cold one.** 19× is the first, uncached call on
one question. Once the provider caches the large naive prompt too, the gap
narrows to 2.7×. Real tasks vary their prompt so most calls land between —
the honest single number is ~2×, and quoting 19× alone would be picking the
flattering half of our own measurement.

## Team mode — N agents, zero interruptions

Several agents can work one project at once without interrupting each
other, because each interference class is removed by construction, not by
prompt etiquette. A parallel-safety pre-flight sequences tasks that would
silently conflict before they're ever run together, path leases stop
same-file writes in-process, and an awareness board lets each agent see
what its teammates currently hold. Per-task worktrees add filesystem-level
isolation on top of leases — opt-in, still landing. Nothing merges without
a security scan, a test-tampering check, and independent review by a
genuinely different worker, and per-worker budgets keep any single agent
from spending the team's money unaccounted.

```
python -m forgeos team "<objective>" --dry-run
```

That's a free preview — it compiles and prints the task graph without
spending anything. Full mechanism-by-mechanism breakdown (merged vs. still
landing) in [docs/TEAM.md](docs/TEAM.md); a runnable walkthrough in
[examples/README.md](examples/README.md).

## The benchmark gate

`forge forgebench` runs a pinned 6-task suite through two arms and applies the
blueprint's savings classes. deepseek-chat, both arms on the same ledger, same
prompt settings, class A paired:

```
           attempted accepted    tok in  tok out         USD
baseline           6        5    73,056      212    0.006249
forgeos            6        5    15,047      202    0.001109

RELEASE 0.1 EXIT GATE: PASS
```

Acceptance is **equal**, so the cost comparison is not void and the harness
prints a figure. Two consecutive runs:

| | run 1 | run 2 |
|---|---|---|
| acceptance | 5/6 vs 5/6 | 5/6 vs 5/6 |
| cash cost | −78.8% | −82.3% |
| input tokens | −79.3% | −79.4% |
| wall time | −51.6% | −52.1% |

The provenance on those figures reads `replayed`, not `measured`, because the
baseline is a separate paired execution rather than a direct observation of the
same run. That distinction is enforced in code
([`savings.py`](forgeos/economy/savings.py)), not left to the writer.

Caveats, stated because they are the ones that matter:

- **Six tasks is small**, and two runs is two runs. The input-token figure is
  stable to within 0.1pp; cash cost moved 3.5pp between them. Treat the
  direction as the result.
- **One model, one repo.** Measured with deepseek-chat against this codebase.
  Nothing here establishes what it does on yours.
- **`ledger-dedup-guard` fails in both arms** — its answer spans two files. Both
  arms fail it equally, so it is not tilting the comparison.

Two findings from getting here are worth more than the percentage:

1. Three of six tasks had been returning **empty or truncated text while billing
   a full quota of output tokens**, and the suite scored each as the model
   getting the question wrong. The cause was `reasoning_effort` defaulting to
   `medium`: the model spent its entire output budget on chain-of-thought before
   writing anything. Turning it off on both arms cut output tokens ~66% and
   raised acceptance from 3/6 to 5/6. The harness now prints `empty!` and
   `cut off!` distinctly from `fail`, because they are its own failures.
2. The retriever was previously ranked against a hardcoded term list that
   **contained the answers to the six benchmark questions**. That is tuning on
   the test set, and any saving it measured would say nothing about a repo it
   had not seen. Ranking now derives from each task's own objective.

Reproduce without spending: `forge forgebench --dry-run --json-out artifacts/forgebench.json`.
The receipt contains all six pinned task contracts, per-task acceptance slots,
proof hashes, and the modelled suite estimate; `--live` is not implicit.
Aggregate receipts without making provider calls:
`forge forgebench-table artifacts/forgebench-*.json --json-out artifacts/forgebench-table.json`.
The repository tool `python tools/aggregate_bench.py ...` remains an equivalent
compatibility entry point.
The table keeps dry-run, failed, aborted, and voided receipts visible, and only
computes savings from measured live Class-A runs with matching acceptance.

## The layers — built and tested, NOT individually benchmarked

Each of these exists and has tests. What none of them has is a measured
before/after in isolation, so this table deliberately carries no savings column:
a number that has not been run is not a saving, and the only end-to-end
measurement is the gate above.

| Layer | What it does |
|---|---|
| Mission compiler | Turns an objective into TaskSpecs without a model call |
| Circuit breaker | Trips a dead worker after repeated failure, auto-recovers |
| Prompt prefix cache | SQLite LRU keyed on byte-identical prefixes |
| Anthropic cache breakpoints | Emits `cache_control` so Claude actually caches |
| Diff-aware scanning | semgrep + gitleaks over the git diff, not whole files |
| Context capsule | Ranked, budgeted context with graduated trimming |
| Deterministic lowerer | Asks whether the task needs a model at all |
| Model selector / profiler | Routes on measured cost and latency per model |
| Free-tier pool | Resolves every free model in the live price catalog, skipping known-dead |
| Generation fencing | A reclaimed task's zombie worker cannot write results |
| Path leases | Two workers can never hold the same write path |
| Merge gate | Tests + security + evidence + a genuinely different reviewer |

Your ledger is the real number — `forge receipts` (or `python -m forgeos receipts`)
prints it.


```
$ forge fleet
  YOUR ROUTING LADDER (cheapest first)
  1. free/local     → ollama
  2. subscription   → claude, codex, copilot
  3. metered        → deepseek, openrouter

  → forgeos routes through claude, codex, copilot BEFORE touching metered API.
    Every task your subscription handles = $0 extra cost.
    Every task a flat-rate seat absorbs is a task you do not meter.
```

The kernel is deterministic code: scheduling, budgets, file leases, verification.
A model is called only for genuinely ambiguous work. Nothing merges without
tests, a security scan, evidence, and review by a *different* worker.

> **Status: v0.6.10 — pre-1.0, not production-hardened.** A `forge` CLI, mission
> compiler, circuit breakers, prompt prefix caching, diff-aware scanning, adapter
> auto-discovery and SQLite WAL tuning are all present and tested. **1863 tests
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
   cache at the provider's cached-input rate (~90% off that portion). How
   much longer your quota lasts is unmeasured — it depends on your prefix
   share, and we have not run that experiment.
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
the routine, and metered API is the last resort. The multiplier depends on
your workload; ForgeBench is how you find yours rather than trusting ours.

## What's new in v0.6.10

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

Two benchmarks, both paired, both run against a real DeepSeek key with the same
model and the same pricing code on both arms:

| | what it measures | scope | result |
|---|---|---|---|
| [`tools/ab_bench.py`](tools/ab_bench.py) | the context lever alone | **one** question | table at the top of this README |
| [`forge forgebench`](forgeos/forgebench.py) | the Release 0.1 exit gate | **six** pinned tasks | [above](#the-benchmark-gate) |

An earlier edition of this section quoted a *different* ab_bench run (25,924
baseline tokens, $0.000140 cold) alongside the top table's 38,403 tokens /
$0.004990, both described as "one measured A/B". Two contradictory numbers for
one claim is worse than no number, so the stale run is gone rather than
reconciled after the fact — the reproduce command below is the authority.

**What these do not show.** Together they exercise the context lever and the
paired-costing machinery. They do not measure routing, retries, escalation, the
merge gate, the mission compiler or the prefix cache — those are built and
tested, not benchmarked. Both run one model against one repo. A number that has
not been run is not a benchmark, and this section will only ever carry numbers
that were.

Reproduce it live: `python tools/ab_bench.py --live --env ~/.hermes/.env --repeat 3`

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
git clone https://github.com/jacobbyrne72/ForgeOS.git && cd ForgeOS
pip install -e ".[dev]"

# Fast-path test (no real subprocess scanners):
python -m pytest tests -m "not slow"

# Full suite (incl. real semgrep, gitleaks, ruff):
python -m pytest tests
```

**Everything below spends nothing.** Run the whole block; none of it can bill
you.

```bash
forge doctor --probe        # which providers actually work (lists models, never completes)
forge init                  # scan this repo, write CLAUDE.md + local settings
forge compile "Add X"       # see the task graph a model call would have produced
forge run "Add X" --dry-run # same, through the full runner
forge forgebench --dry-run  # price the benchmark suite without running it
forge receipts              # what you have actually spent, from the ledger
```

**These spend real money.** Separated deliberately: the two `run` forms differ
by one flag, and a first-timer pasting a mixed block would find that out from
their bill.

```bash
forge run "Add X" --budget-usd 0.50   # hard cap; refuses rather than exceeding it
forge forgebench --budget-usd 0.25    # paired live benchmark, both arms
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
  capsule, packing, savings receipts, mission compiler, model traits, discovery.
- **Quota telemetry is durable but still operator-fed.** Provider-reported quota
  facts persist in `.forgeos/quota.json` and appear at `/api/quota`; ForgeOS does
  not probe or invent subscription usage, so live source adapters remain future
  work. Inspect the same local snapshot from the terminal with
  `forge quota --json`.
- **Live CLI backends are young.** The omc team adapter is verified against a
  real install, but long-haul reliability (crashed sessions, orphaned
  worktrees) has not been proven over days of continuous use.

Recently closed, for the record: `Forge.run` now defaults to the routed
adapter path (`executor=None`), MODEL failures escalate exactly one tier per
failure via `Router.escalate`, and the gateway remembers dead models
(terminal vs temporary) so retired free tiers stop being re-bought. The default
Forge also lazily wires a ledger-owned Gateway: `auto:free` resolves against
usable catalogued providers and falls through deterministic free candidates
without requiring callers to hand-build a gateway or executor.

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
