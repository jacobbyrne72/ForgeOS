# hive

A cost-governed harness for AI coding agents. A deterministic kernel owns
scheduling, budgets, file leases and verification; a cheap model wakes only when
something genuinely ambiguous happens; nothing merges without tests, a security
scan, evidence, and a review by a *different* worker.

The optimisation target is **cost per accepted task** — not tokens, not calls.
A router that halves per-call cost while doubling retries has made things worse,
and per-call metrics score that as a win.

> **Status: pre-1.0.** It runs end to end against real providers — a task has
> been routed, executed, reviewed by a second worker, judged by the merge gate
> and billed to the ledger, using a live DeepSeek and OpenRouter key. 1000+
> tests pass. It is not yet production-ready: see [Known gaps](#known-gaps).

## The idea

Most harnesses optimise the model call. The bigger wins are around it:

| Lever | Saving | Why |
|---|---|---|
| Don't call a model | 100% | Finding imports across 20k files is a tree-sitter query, not an LLM task |
| Deterministic scheduling | 100% of what it replaces | Queue, retry and routing decisions in plain code cost zero tokens |
| Byte-stable prefix caching | ~90% off cached input | Providers serve a cache hit only when the prefix is byte-identical |
| Smaller payloads | 65–82% | Ranked context capsule instead of a repository dump |
| Cheapest capable worker | up to 100% | A local Rust binary should never lose work to a metered model |
| Compact supervision | 10×+ | The manager reads a ~90-byte heartbeat, never a transcript |

These multiply. A smaller payload, on a cache hit, on a free-tier model, for a
call deterministic code decided didn't need making, isn't 89% cheaper — it never
happened.

**A note on claims:** narrow operations can genuinely improve by orders of
magnitude when an LLM loop is replaced by a query. End-to-end, a realistic target
is 2–10×, with receipts. hive is built to *prove* savings rather than assert
them — `SavingsProof` marks every figure as measured, replayed or modelled, and
a saving is only ever as strong as its weakest input.

## Install

```bash
git clone <this-repo> hive && cd hive
pip install -e ".[dev]"
python -m pytest tests -m "not slow"     # fast path
python -m pytest tests                   # everything, incl. real scanners
```

Python 3.11+. Optional and detected, never required: `semgrep`, `gitleaks`,
`ruff` (security/lint gates), `tiktoken` (exact token counts), Obsidian (vault
location).

## Use

```python
from hive import Forge, TaskSpec, Scope, Budget

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
    executor=my_executor,       # you supply this — see Known gaps
    reviewer=my_reviewer,       # must be a different worker
)
print(result.cost_per_accepted)
```

Dashboard: `python -m hive.dashboard.app` → `http://127.0.0.1:8899` (localhost
only; read-only except the halt flag).

## How it fits together

```
mission contract  →  size pools to this machine  →  price quota as inventory
   → per task: capsule → preflight refusal → route (cheapest capable)
     → take path leases → execute → reduce output → verify ladder → merge gate
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

- **No worker can edit files yet.** The gateway worker returns text; it has no
  filesystem. So the merge gate correctly refuses everything it produces (no
  tests, no evidence, no commands run). Driving a real coding CLI — which *can*
  edit — needs the ACP or omc-team adapter wired through `build_adapter`.
  **This is the next thing to build.**
- **The default fleet has no free file-editing worker.** Free workers here
  summarise and classify. Every real edit currently costs money.
- **Several subsystems are built and tested but not yet in the execution path:**
  capsule, packing, savings receipts, quota/capacity market, mission compiler,
  model traits, discovery.
- **Router escalation is not wired.** `Router.escalate` exists; a MODEL failure
  currently retries at the same tier.
- **Catalogued free tiers rot.** Provider catalogues list models at $0 that the
  vendor has since retired; a 404 is handled and falls through to the next
  candidate, but the catalogue itself is not self-healing.

## Connecting a provider

```bash
export DEEPSEEK_API_KEY=...        # or OPENROUTER_API_KEY, etc.
python tools/live_check.py         # probes every usable provider, prices each call
python tools/live_job.py           # a full Forge job end to end, writes to .hive/
HIVE_STATE_DIR=.hive python -m hive.dashboard.app
```

`live_check.py` reports which providers are reachable and which catalogued
models are actually alive. Neither tool ever prints a key value — settings
reference an environment variable *name*, and hive never stores a secret.

## Licence

MIT — see [LICENSE](LICENSE).
