# Harness features mined from openai/codex

Source: `vendor/codex` (shallow clone, `codex-rs/` Rust workspace, ~90 crates).
Root `docs/*.md` are mostly stub redirects to `developers.openai.com/codex/*` — the
real answers are only in source, so every row below cites `codex-rs` file:line,
not the vendored docs. ForgeOS status is read against `docs/ROADMAP.md` and
`docs/TEAM.md` plus a live grep of `forgeos/` (not vendor). Read-only survey,
nothing built or changed.

## Inventory

| Feature | Codex mechanism (receipt) | ForgeOS status | Target |
|---|---|---|---|
| Session/rollout persistence + resume | JSONL per session + SQLite mirror index; reverse-scan reconstruction resolves surviving compaction checkpoint before replaying forward. `rollout/src/recorder.rs:85,982,1047`; `core/src/session/rollout_reconstruction.rs:113`; `protocol/src/protocol.rs:3195` `RolloutItem`; `app-server-protocol/src/protocol/v2/thread.rs:325` `ThreadResumeParams` | **PARTIAL** — `forgeos/adapters/acp.py:18` passes through the *worker's own* `session/resume`; no ForgeOS-owned durable log. Task #13 (in progress) is exactly this gap. | `forgeos/core/rollout.py` (new), wired into `forge.py` attempt loop |
| Sandboxing (seatbelt/landlock/Windows) + approval policy | `sandboxing/src/seatbelt.rs:352,623` (sbpl profiles via `include_str!`); `linux-sandbox/src/landlock.rs`; `windows-sandbox-rs/src/acl.rs`; `protocol/src/protocol.rs:910` `AskForApproval`, `:997` `SandboxPolicy`; `core/src/safety.rs:32` `assess_patch_safety`; `execpolicy/` Starlark rule DSL | **MISSING, by design** — ForgeOS delegates execution to worker CLIs (Claude Code/Codex/etc.) that own their own sandbox; the orchestration layer never runs untrusted code directly | N/A unless `local_exec.py` grows into direct execution |
| Config system (config.toml, profiles, managed layers) | `config/src/config_toml.rs:150` `ConfigToml`; `config/src/profile_toml.rs:24` `ConfigProfile`; `config/src/config_layer_source.rs:6` precedence incl. `Mdm`/`EnterpriseManaged`; `config/src/config_requirements.rs:162` `allow_managed_hooks_only` | **PARTIAL** — `forgeos/settings.py` covers provider roles + "never store a secret" / "disabled is a gate," no named profile overlays, no managed/enterprise precedence tier | `forgeos/settings.py` (see Port #5) |
| MCP client surface | `codex-mcp/src/connection_manager.rs:1`, `runtime.rs:73` `McpRuntime`, `binding.rs:30` `McpBinding` | **MISSING** — ForgeOS speaks ACP (`adapters/acp.py`) to drive workers, not MCP | `forgeos/adapters/mcp_client.py` (new) |
| MCP server surface | `mcp-server/src/lib.rs:1` "Prototype MCP server" | **MISSING** — nothing lets another agent call ForgeOS as a tool | `forgeos/mcp_server.py` (see Port #2) |
| Code-mode (execute code, not tool-call chains) | `code-mode-protocol/src/description.rs:12` V8-isolate exec tool; `core/src/tools/mod.rs:68` `ToolMode::CodeMode`; `v8-poc/src/lib.rs` | **N/A** — ForgeOS routes to worker CLIs, doesn't run its own model tool-loop | low priority |
| exec-server (remote/cross-OS exec) | `exec-server/README.md:1` JSON-RPC subprocess server, Noise-encrypted relay for remote mode; `exec-server-protocol/src/protocol.rs:84+` full exec+FS+HTTP RPC | **MISSING** — `local_exec.py` is a thin local wrapper, no cross-machine exec | future, once multi-machine fleets are a stated goal (not yet) |
| apply-patch tool (custom diff format) | `apply-patch/src/parser.rs:37` `*** Begin/End Patch` markers, `:130` `parse_patch` | **MISSING, by design** — file edits are the worker CLI's job | N/A |
| shell-escalation (execve intercept, Run/Escalate/Deny) | `shell-escalation/README.md:1` `codex-execve-wrapper` | **MISSING** — same delegation reason as above | N/A |
| Prompt caching strategy | Provider-side prefix cache keyed by session id, trusted as-is. `core/src/client.rs:482` `prompt_cache_key` | **HAVE — stronger** — `forgeos/prompts/prefix.py` engineers explicit byte-identity (stable prefix / volatile tail split, `fingerprint()` drift check, CI-assertable), ROADMAP.md differentiator #2. Codex trusts the provider; ForgeOS verifies. | — |
| Context compaction | `core/src/compact.rs`; `analytics/src/facts.rs:274` `CompactionTrigger`/`Reason`; config-driven `model_auto_compact_token_limit`; resumable chain via `protocol.rs:3229` `CompactedItem` | **PARTIAL** — `forgeos/compact_multi_turn.py` (naive last-N + summarize) and `forgeos/context_compress.py` (keyword-window filter, 60–90% cut) exist, no config-driven trigger/telemetry, no resumable chain | `compact_multi_turn.py` — add trigger config + ledger event |
| Hooks / notify | `protocol/src/protocol.rs:1492` `HookEventName` (PreToolUse..Stop); `hooks/src/engine/discovery.rs:57,63,80-83` managed-only gate | **MISSING** — `forgeos/policy.py:76` only comments on *Claude Code's* PreToolUse convention; no native lifecycle hooks on ForgeOS's own wave/gate/merge events | `forgeos/core/hooks.py` (see Port #1) |
| Skills / slash-commands | `core-skills/src/loader.rs:139` `SKILL.md` convention; `tui/src/slash_command.rs:12` separate hardcoded enum | **N/A** — ForgeOS is a headless CLI/library, no TUI, no skill-discovery convention of its own | low priority |
| Multi-agent local coordination | `agent-graph-store/src/store.rs:17` SQLite thread-spawn lineage graph; `collaboration-mode-templates/` Plan/Execute/Pair system-prompt overlays | **HAVE — more rigorous for the local/git case** — TEAM.md mechanisms 1–9: parallel-safety pre-flight, `leases.py`, `awareness.py` (TeamBoard), `worktrees.py`, `batchmerge.py`, `verify.py` (MergeGate), `ledger.py`. No formal queryable lineage graph (Mission parent/child lives ad hoc in `compiler.py`) | formalize lineage in `ledger.py` (not top-5, cheap follow-on) |
| Cloud/remote agent tasks | `cloud-tasks-client/src/api.rs:136` `CloudBackend` trait against a hosted ChatGPT backend (list/create/apply/preflight/sibling-attempts) | **MISSING** — everything runs on the local machine's `Forge.run` | not planned — ForgeOS's differentiator is local cost governance, not hosted execution |
| agent-identity + external-agent-migration | `agent-identity/src/lib.rs:97` Ed25519+JWT per-runtime id; `external-agent-migration/src/migration_source.rs:49` `Cla`/`Cur` importers for `.claude/`/`.cursor/` | **MISSING** — no per-agent crypto identity (not needed locally); no importer for existing Claude Code/Cursor config | migration pattern → onboarding importer (see Port #3) |

## Top 5 ports (novelty × demand ÷ effort)

1. **Hooks / notify system.** Port `HookEventName`'s shape onto ForgeOS's existing
   extension points — `Forge.run`'s wave boundaries, `MergeGate.evaluate`,
   `Scheduler.assign` — as `PreTask`/`PostGate`/`PreMerge`/`PostMerge`/`BudgetTripped`
   events, each a registered-callable list. Mirror `allow_managed_hooks_only`
   (`config/src/config_requirements.rs:162`) so a managed layer can freeze
   user hooks for team deployments. Cheapest way to let third parties (Slack
   notify, custom scanners) attach without touching core, and it's the exact
   "integrations are distribution" lever ROADMAP.md §5 asks for.
   Receipts: `protocol/src/protocol.rs:1492-1504`; `hooks/src/engine/discovery.rs:57,63,80-83`.

2. **MCP server surface for ForgeOS itself.** Wrap `forge run`/`doctor`/`receipts`/`dash`
   as MCP tools so any MCP-aware agent can delegate a task to ForgeOS and get a
   `SavingsProof` receipt back. Codex's own `mcp-server` crate is explicitly a
   "prototype" — proof this doesn't need to be heavyweight. Turns ForgeOS from
   "a CLI you run" into "a cost-governed worker any harness can call."
   Receipts: `mcp-server/src/lib.rs:1`.

3. **External-agent-migration importer.** Mirror the `Cla`/`Cur` pattern —
   read `.claude/` or `.cursor/`, migrate hooks/MCP config/AGENTS files — as
   `forge init --import claude-code|cursor`, seeding `forgeos/settings.py` +
   `instructions.py`. Serves the "zero setup friction" launch mechanic in
   ROADMAP.md §5; the #1 objection to a new CLI is an already-tuned AGENTS.md.
   Receipts: `external-agent-migration/src/migration_source.rs:49-54`;
   `source/cla.rs:20`; `source/cur.rs:20-22`.

4. **Rollout log + replay-based resume.** Don't port the full SQLite-mirror +
   reverse-scan machinery wholesale — codex resumes a *chat session*, ForgeOS
   resumes a *task attempt*, different problems. Port the shape: an
   append-only JSONL log per attempt (`forgeos/core/rollout.py`) with turn
   boundaries + the adapter's own checkpoint token; resume = replay-to-last-
   committed-turn, then re-attach. This is task #13, already in progress —
   codex's design is the reference architecture, not an import target.
   Receipts: `rollout/src/recorder.rs:85,982,1047`;
   `core/src/session/rollout_reconstruction.rs:113-122` (the "resolve the
   surviving checkpoint first, replay forward second" idea is worth stealing
   even in miniature).

5. **Config layering with a managed/locked tier.** `settings.py` already
   encodes "never store a secret" and "disabled is a gate" as hard properties;
   codex's precedence (`session < project < user < enterprise-managed < mdm`)
   is the natural next layer once ForgeOS has more than one operator — a
   `requirements.toml`-equivalent a company drops in to freeze budget ceilings
   or provider allowlists a local config can't override. Lower urgency than
   1–4 today (ForgeOS is single-operator), cheap to design right while
   `settings.py` is still small.
   Receipts: `config/src/config_layer_source.rs:6-25`;
   `config/src/config_requirements.rs:162,889,979`; `config/src/state.rs:69,85`.
