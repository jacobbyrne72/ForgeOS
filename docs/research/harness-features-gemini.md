# Mined: google-gemini/gemini-cli

Read-only feature mining pass over `vendor/gemini-cli` (docs + `packages/core`,
`packages/cli` structure). No edits made outside this file. ForgeOS context:
`docs/ROADMAP.md`, `docs/TEAM.md`.

## Inventory

| Feature | Implementation (gemini-cli) | ForgeOS | Target module |
|---|---|---|---|
| Checkpointing + `/restore` | Shadow git repo `~/.gemini/history/<hash>` + JSON checkpoint per write; `docs/cli/checkpointing.md` | Missing | new `forgeos/checkpoint.py` |
| Rewind (`/rewind`, chat+code rollback across compression) | `docs/cli/rewind.md` | Missing (no chat-turn abstraction to rewind) | n/a — architecture mismatch |
| Memory hierarchy (`GEMINI.md`, `/memory show\|reload`, JIT per-dir scan, `@import`) | `docs/cli/gemini-md.md`; core memory discovery service | Partial — `AGENTS.md` is a static root file only; `forgeos/knowledge/memory.py` is a different concept (session-fact DB, not instructional context) | `forgeos/knowledge/memory.py` |
| Auto Memory (background transcript mining → reviewable `.patch`/`SKILL.md` candidates, human-approved) | `docs/cli/auto-memory.md` | Missing | `forgeos/knowledge/memory.py` |
| MCP client (discovery, trust bypass, per-tool confirm, resources, OAuth, `gemini mcp` CLI) | `packages/core/src/tools/mcp-client.ts`, `mcp-tool.ts`; `docs/tools/mcp-server.md` | Missing native client — `mcp` hits in `adapters/discovery.py`/`cli_team.py` only describe *other* CLIs' MCP config, not a ForgeOS tool registry | n/a — ForgeOS drives worker CLIs rather than hosting its own tool registry |
| Sandboxing (Seatbelt/Docker/Podman/Windows Native/gVisor/LXC; tool-level vs process; sandbox expansion dialog) | `docs/cli/sandbox.md` | Missing — isolation today is the read-size gate (`forgeos/policy.py`) + worktrees, not OS-level execution sandboxing | new `forgeos/sandbox.py` |
| OpenTelemetry surfaces (logs/metrics/traces catalog, GCP/Jaeger export, User-Agent surface tagging) | `docs/cli/telemetry.md` | Partial — `ledger.py`/`forge receipts`/`dashboard/app.py` give cost telemetry, no OTel spans or structured event taxonomy | `forgeos/ledger.py`, `forgeos/dashboard/app.py` |
| Extension system (bundle prompts/MCP/commands/themes/hooks/subagents/skills; gallery install) | `docs/extensions/index.md` | Missing — likely a non-goal (ROADMAP §6: no domain logic, single coherent kernel) | n/a |
| ReAct loop hooks + tool confirmation UX (`BeforeTool`/`AfterTool`, `ask_user`, approval modes, TOML policy engine) | `docs/hooks/index.md`, `docs/reference/policy-engine.md` | Partial — `forgeos/policy.py` is a narrow read-size gate; `MergeGate`/reviewer is merge-time only, no interactive per-call confirmation (ForgeOS is headless-first by design) | `forgeos/policy.py` |
| Context compression | Core "Chat history compression" (`docs/core/index.md`) | **Have** — `context_compress.py`, `compact_multi_turn.py`, `output_compressor.py`, byte-stable prefix compiler (ROADMAP §2) | n/a — already ahead (CI-testable byte identity vs. best-effort lossless) |
| IDE integration (VS Code companion, ACP registry, native diffing) | `docs/ide-integration/index.md` | **Have** via ACP — `forgeos/adapters/acp.py` (task #14 done) | n/a |
| Token caching / cost signal (`/stats` cached tokens) | `docs/cli/token-caching.md` | **Have** — `prompt_cache.py`, `dedup_cache.py`, byte-stable prefix compiler targets ~90% cache-hit discount explicitly | n/a — already ahead |
| Model routing / steering (fallback chain, local Gemma router, Plan-mode Pro→Flash phase switch) | `docs/cli/model-routing.md`, `docs/cli/plan-mode.md` | **Have**, differentiated — `core/router.py`, `model_ranker.py`, `cost_router.py`; AGENTS.md rule 9 explicitly routes on deterministic features only (gemini-cli's model classifier is exactly the injection surface this avoids) | n/a |
| Skills system (Agent Skills standard, discovery tiers, `activate_skill`, progressive disclosure) | `docs/cli/skills.md` | Missing (relies on the surrounding session's skills, not ForgeOS-native) | n/a — likely non-goal |
| Git worktrees (`--worktree`, one per session, manual resume) | `docs/cli/git-worktrees.md` | **Have, more advanced** — `forgeos/worktrees.py`: deterministic per-*task* worktree, `git merge-tree` pre-check, real `--no-ff` merge, plus batch-bisection merge (`batchmerge.py`) gemini-cli has no equivalent of | n/a |
| Policy engine tiers (Default/Extension/Workspace/User/Admin priority formula, `argsPattern`/`commandPrefix` regex, per-subagent rules) | `docs/reference/policy-engine.md` | Partial — same gap as ReAct/confirmation row above | `forgeos/policy.py` |
| System prompt override (`GEMINI_SYSTEM_MD` full replace + `${AgentSkills}`/`${AvailableTools}` substitution) | `docs/cli/system-prompt.md` | Missing | `forgeos/instructions.py` |
| Subagents (`.md`+YAML frontmatter, tool wildcards, isolated context, per-subagent policy, recursion protection) | `docs/core/subagents.md` | Partial — `manager.py`/`compiler.py` decompose to worker tasks (coarser, flat — no nested-agent recursion risk to protect against) | n/a-ish; consider if user-authored specialist workers become a goal |

## Notable / novel patterns

- **Auto Memory** (background LLM pass over idle transcripts → draft `.patch`/`SKILL.md`, human-reviewed inbox, never auto-applied) has no equivalent anywhere in ForgeOS or the other mined repos. Directly reusable idea for `forgeos/knowledge/memory.py`: mine `EventLog`/worker reports for recurring corrections instead of relying on workers to self-record.
- **Sandbox expansion** (proactive or reactive modal asking for exactly the extra permission a failing command needs, scoped to that run) is a cleaner UX than a static allow/deny gate — relevant if `forgeos/policy.py` ever governs write/shell tools, not just reads.
- **Plan-mode phase-based model routing** (Pro while read-only planning, auto-switch to Flash once a plan is approved and execution starts) is a routing signal ForgeOS doesn't use (routes on task features/cost, not on read-only-vs-write phase) — arguably a cheap additional signal for `core/router.py`.
- **Policy engine's tiered priority formula** (`tier_base + priority/1000`, Admin > User > Workspace > Extension > Default) is a clean pattern if `forgeos/policy.py` ever grows beyond the single read-size gate into a general rule engine.

## Top 5 ports, ranked

1. **Auto Memory** → `forgeos/knowledge/memory.py`. Novel, no prior art in the other mined repos, and slots directly into the existing `Ledger`/`EventLog`/`recall()` machinery — highest novelty × lowest effort.
2. **Sandbox expansion dialog** → new `forgeos/sandbox.py`. ForgeOS has no OS-level execution isolation today; this is the one gap with real security stakes (trading-adjacent hard rules already forbid live orders — a sandbox is the mechanical backstop).
3. **`GEMINI_SYSTEM_MD`-style prompt override + variable substitution** → `forgeos/instructions.py`. Small, self-contained, gives operators an escape hatch without forking the harness.
4. **Policy engine tiering** → generalize `forgeos/policy.py` from a single read-size gate into the Default/Workspace/User/Admin priority model, reusing the existing "mechanical > instructional" enforcement philosophy already proven there.
5. **OTel-shaped event taxonomy** → `forgeos/ledger.py`/`forgeos/dashboard/app.py`. ForgeOS already has the receipts; standardizing the event names/attributes (their `gemini_cli.tool_call`-style catalog) makes ForgeOS pluggable into existing observability backends for free.
