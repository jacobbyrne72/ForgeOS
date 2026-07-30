# ForgeOS — Roadmap and Positioning

*The operating system for cheap, fast, verified AI coding.*

This document is the working brainstorm for taking ForgeOS from a sound kernel
to the most-watched agent-infrastructure repo of the year. It is split into:
what we already have that nobody else ships, what to build next, what to mine
from existing open source, and how to launch. Claims here follow the house
rule: **measured, replayed or modelled — never asserted.**

---

## 1. Why this can win

Every developer on X is complaining about the same thing: subscription and API
spend evaporating. Codex caps, Claude limits, OpenRouter bills. The market gap
is not "another agent framework" — it is **cost governance as a first-class
kernel**, the way an OS owns memory and CPU so processes cannot ruin the
machine for each other.

ForgeOS's differentiators, already built and tested:

| Differentiator | Status | Nobody else ships |
|---|---|---|
| Cost per **accepted** task as the only optimisation target | ✅ kernel-wide | Frameworks report per-call cost, which rewards doubled retries |
| Integer-microdollar ledger, no float drift, no unrecorded call | ✅ | Most harnesses have no ledger at all |
| `SavingsProof` receipts: measured vs replayed vs modelled | ✅ | Everyone else asserts "90% cheaper" |
| Deterministic pre-model economy: capsule, preflight refusal, test selection, output reduction | ✅ modules | The cheapest call is the one never made |
| Budget as contract: a tripped governor escalates to a human, never auto-widens | ✅ | Agent loops that "just retry" empty accounts |
| Second-worker review + merge gate (tests, scanners, evidence) | ✅ | Self-review is a rubber stamp |
| Router reads deterministic features only — prompt injection cannot buy premium tier | ✅ | Published attacks do exactly this to LLM routers |
| Dead-model memory (terminal vs temporary) so free-tier rot self-heals | ✅ | Catalogues everywhere list retired $0 models |
| Machine-sized pools (reasoning vs execution) from live pressure | ✅ | Fixed parallelism starves workstations or thrashes laptops |

## 2. The gap between "kernel" and "product" (build order)

1. **Routed executor glue** — `build_adapter` + `adapter_executor` exist and are
   tested; `Forge.run` still requires a hand-rolled executor. One module closes
   it: registry profile → adapter → bridged executor, `Forge.run(executor=None)`
   defaults to it. *This turns ForgeOS from a library into a tool.*
2. **Router escalation wiring** — `Router.escalate` exists; MODEL failures
   currently retry at the same tier. Wire the failure class into re-routing.
3. **`forgeos` CLI** — `forgeos run "objective"`, `forgeos doctor`,
   `forgeos receipts`, `forgeos dash`. The README demo must be one command.
4. **Live free-fleet defaults** — ship a curated fleet: local Ollama for
   summarise/classify, free OpenRouter tiers for drafting, metered models only
   at the top of the ladder, dead-model store keeping the list honest.
5. **The benchmark that sells it** — one reproducible script that runs the same
   task list through (a) a naive single-model loop and (b) ForgeOS, and prints
   both bills with receipts. Post the table, not adjectives. (`tools/ab_bench.py`
   is the seed.)

## 3. Novel tools worth building (none exist in the wild today)

Ranked by (novelty × demand ÷ effort):

1. **Subscription orchestration** — treat Claude Code / Codex / Gemini CLI
   *subscription seats* as capacity-priced workers alongside API keys. Quota
   inventory (already in `core/quota.py`) prices "you have N Sonnet messages
   left this week" so the router spends flat-rate seats before metered tokens.
   Nobody arbitrages subscriptions vs API — everyone just complains about caps.
2. **Byte-stable prefix compiler** — a prompt assembler whose contract is byte
   identity of the prefix across calls (provider cache hits ≈ 90% off input).
   Assert with a CI test: same fleet, same day → identical prefix bytes.
   (`prompts/prefix.py` is the seed; make it a public, framework-agnostic tool.)
3. **Cost-per-accepted-task leaderboard** — a public, reproducible harness where
   models/fleets compete on `$ per merged, reviewed, scanner-clean task`, not
   on benchmark accuracy. This is the SWE-bench nobody has run: the one with a
   bill attached. Massive social-proof flywheel.
4. **The refusal engine as a product** — preflight that *refuses* calls
   (answer already in vault, task is a tree-sitter query, capsule over budget)
   with a receipt for every refusal. "Your agent spent $0.00 because it read
   the codebase index" is a screenshot people share.
5. **Fleet doctor** — `forgeos doctor` already answers "what can this machine
   run right now"; extend to live-probe free tiers and print the cheapest
   viable ladder for the day. Free-tier rot is universal pain; a self-healing
   answer is shareable output.

## 4. Mine, don't rewrite — the vendored upstreams

Already cloned under `vendor/` (mined for patterns, never committed):

| Repo | What we take |
|---|---|
| `OmniRoute` | Terminal-vs-temporary provider status (already ported to `gateway/dead_models.py`); free-tier ranking |
| `RouteLLM` | Router calibration ideas: win-rate thresholds per tier |
| `GPTCache` | Semantic-cache lookups as a preflight refusal source |
| `LLMLingua` | Prompt compression as a capsule post-pass (10–20× on prose) |
| `aider` | Repo-map ranking, edit-block formats that survive cheap models |
| `mini-swe-agent` | Minimal agent loop as a reference executor |
| `serena` | Symbol-level navigation for the capsule builder |
| `repomix` / `rendergit` | Repo packing baselines to beat in benchmarks |
| `acp-python-sdk` / `agent-client-protocol` / `claude-agent-acp` / `codex-acp` | ACP adapters: drive Claude Code / Codex / any ACP CLI as workers |
| `claude-squad` / `conductor` / `cli-agent-orchestrator` / `humanlayer` | Multi-CLI orchestration + human-approval patterns |
| `llm-council` | Multi-model review panels (second-worker review generalised) |
| `evalite` | Lightweight eval harness patterns for the leaderboard |
| `spec-kit` | Spec-driven task decomposition for the manager |
| `software-agent-sdk` | Agent SDK surface to mirror for embedders |
| `sandcastle` | Sandboxed execution patterns |
| `minbpe` | Cheap local token counting fallback |
| `system_prompts_leaks` / `agent-rules-books` / `pocock-skills` | Prompt/skill corpus for role prefixes |
| `autoresearch` | Literature-sweep automation for the knowledge vault |

Worth cloning next (direct-use candidates):

- `BerriAI/litellm` — already an optional dependency; pin and wrap, don't fork.
- `tree-sitter` grammars — the "don't call a model" lever for code queries.
- `ast-grep/ast-grep` — structural search binary for capsule + avoidance.
- `openai/codex` + `anthropics/claude-code` (docs/CLIs) — worker seats to drive.
- `sst/opencode`, `google-gemini/gemini-cli` — more free/flat-rate seats.
- `microsoft/LLMLingua-2` — faster compression model for the capsule.
- `qdrant/fastembed` or `lancedb` — embedded vault search when Obsidian absent.

## 5. Launch plan (repo-of-the-year mechanics)

What actually moved the needle for breakout infra repos: one undeniable demo,
one number, zero setup friction.

- **The hero demo**: a 60-second asciinema — `pip install forgeos && forgeos run
  "fix issue #42"` — ending with a printed receipt: what ran where, what was
  refused, what it cost, what the naive loop would have cost.
- **The number**: the A/B bill table from `tools/ab_bench.py`, reproducible by
  anyone with the same keys. Honest range (2–10× end-to-end), receipts linked.
- **README above the fold**: the bill table, then the install line, then the
  architecture diagram. Design rules and philosophy below the fold.
- **Weekly "free fleet report"**: automated post listing which free tiers are
  alive/dead (dead-model store output). Recurring, useful, citable — the
  growth loop runs itself.
- **Integrations as PRs to other repos**: a ForgeOS executor for aider, an
  OMC adapter (done), an ACP adapter — every upstream integration is
  distribution.
- **What we never do**: inflate claims. The repo's credibility *is* the
  receipts culture. One caught exaggeration kills the premise.

## 6. Non-goals

- Not a chat UI, not a model, not a hosting business.
- No domain logic (see AGENTS.md — domain-specific defaults are bugs).
- No growth hack that spends user money to look fast: speed comes from
  refusal, caching, and free-first routing — with the bill as proof.
