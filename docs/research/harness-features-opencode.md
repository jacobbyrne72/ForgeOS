# Harness features mined from sst/opencode

Read-only mining pass over `vendor/opencode` (TypeScript, Effect-based DI,
Bun/Node). ForgeOS is a Python cost-governance kernel with no chat UI, no
TUI, and no IDE surface (`docs/ROADMAP.md` non-goals) — so opencode's value
here is almost entirely in its *harness* patterns (provider routing,
permissions, tool registry, context compaction, plugin surface), not its
product shell. Method: directory map, then targeted reads of the smallest
file that defines each concept.

## Inventory

| Feature | Implementation | ForgeOS have/partial/missing | Target module |
|---|---|---|---|
| Client/server split | `packages/server` (Effect `HttpApiApp`, handlers per resource) talks to `packages/client` (codegen'd from `packages/protocol` groups) over a typed HTTP API; `packages/opencode/src/server/server.ts` boots the Node HTTP listener with mDNS discovery | Missing — ForgeOS has no long-lived server/client split, only in-process `Forge.run` | n/a (out of scope per non-goals) |
| Share/session model | `packages/opencode/src/share/session.ts` — `SessionShare.share/unshare` wraps `Session.Service`, publishes to `ShareNext`, config gate (`share: "disabled"/"auto"`) | Missing | n/a |
| Provider abstraction (75+ models) | `packages/opencode/src/provider/provider.ts` — `BUNDLED_PROVIDERS` map of lazy `import()`s per vendor SDK (Bedrock, Anthropic, Azure, Google, OpenRouter, xAI, Mistral, …), `ModelsDev` catalog, `ProviderTransform`, SSE-timeout wrapping (`wrapSSE`) | Partial — `forgeos/core/router.py` + `forgeos/catalog.py` do tier routing but without opencode's per-SDK lazy-import/timeout wrapper pattern | `forgeos/gateway` (no `gateway/` dir yet — router+catalog play this role) |
| Provider auth | `packages/opencode/src/provider/auth.ts`, `packages/opencode/src/account/*` — per-provider OAuth/API-key account records | Have (adapters own their own auth) | `forgeos/adapters/*` |
| Model health / dead-model tracking | `packages/opencode/src/provider/model-status.ts` (`ModelStatus` service) | Have, independently — ForgeOS's "dead-model memory" claim (ROADMAP.md diff table) | `forgeos/catalog.py` |
| LSP integration | `packages/opencode/src/lsp/{lsp,client,server,launch,diagnostic,language}.ts` — spawns real language servers, exposes `Symbol`/`DocumentSymbol`/`Range`/diagnostics; surfaced to the agent via `tool/lsp.ts` (flag-gated `experimentalLspTool`) | Missing | n/a — candidate for a future "code intelligence" preflight source (cheaper than a model call for "what calls X") |
| TUI architecture | `packages/tui/src` — SolidJS + `@opentui/solid` reactive renderer (`app.tsx`), context-provider tree (theme/route/sync/permission/kv/…), dialog stack, plugin `feature-plugins/` slots (home/sidebar/system) | Missing (non-goal) | n/a |
| Permission model | `packages/opencode/src/permission/{index,evaluate,arity}.ts` — `evaluate(permission, pattern, ...rulesets)` does last-match wildcard rule lookup (`Wildcard.match`) over layered rulesets, action ∈ {allow, deny, ask}; unresolved asks queue as `Deferred` promises answered via `reply()`; `acp/permission.ts` adapts the same model to ACP's `requestPermission` | Partial — ForgeOS's leases/governor gate *scheduling*, not *tool-call* risk; no wildcard allow/deny/ask ruleset for individual tool invocations | new: `forgeos/permission.py` |
| Plugin/extension surface | `packages/opencode/src/plugin/{loader,index,shared}.ts` — resolves npm or local plugin specs, staged pipeline (install → entry → compatibility → load) with per-stage error reporting and retry; `packages/plugin/src/index.ts` types the `PluginInput` (client, project, directory, shell, `experimental_workspace.register`) and `ToolDefinition`/`WorkspaceAdapter` extension points; first-party plugins ship for Azure/Cloudflare/DigitalOcean/Modal/OpenAI/Snowflake/xAI | Missing — no plugin/tool-extension loader in ForgeOS | new: `forgeos/plugins/` |
| Agent-mode design (build vs plan) | `packages/opencode/src/tool/plan.ts` (`plan_exit` tool asks the user via `Question.Service`, then flips the session's active agent to `"build"` by injecting a synthetic user message with `agent: "build"`); modes/agents are markdown files loaded by `config/agent.ts` (`load`/`loadMode`, `{agent,agents}/**/*.md`, `{mode,modes}/*.md`) with YAML-frontmatter-defined permission rulesets | Missing — no plan/build mode split; ForgeOS tasks don't have a user-gated "switch mode" step | n/a (ForgeOS is non-interactive by design) |
| Tool registry | `packages/opencode/src/tool/registry.ts` — `ToolRegistry.Service` composes builtin tools + config-dir custom tools (`{tool,tools}/*.{js,ts}` glob) + plugin tools + MCP tools into one list; per-call filtering by model id (e.g. `apply_patch` only for non-`oss` `gpt-*`) and by agent (`task` tool description lists only agents the *caller's* permission ruleset doesn't deny); JSON-Schema is regenerated from Zod at the boundary so plugin tools built with Zod interop with the LLM's schema | Partial — ForgeOS doesn't have a declarative tool catalog with per-model/per-agent visibility filtering | `forgeos/tools.py` (new) |
| Context management | `packages/opencode/src/session/compaction.ts` — turn-boundary detection, `PRUNE_MINIMUM`/`PRUNE_PROTECT` thresholds, `preserveRecentBudget` (config override or 25% of usable context, clamped 2k–8k tokens), protected tool ids (`skill`), summary carried as a synthetic `compaction` part rather than deleting history | Partial — `forgeos/context_compress.py`/`compact_multi_turn.py` exist but not the turn-aware protect-recent-percentage algorithm | `forgeos/context_compress.py` |
| MCP client | `packages/opencode/src/mcp/index.ts` — stdio/SSE/StreamableHTTP transports, OAuth device flow (`oauth-provider.ts`/`oauth-callback.ts`), tool-list-changed notifications, catalog sanitization (`McpCatalog.sanitize`) | Missing | n/a |
| **Novel: ACP server surface** | `packages/opencode/src/acp/*` — opencode itself implements the *agent side* of Agent Client Protocol (`@agentclientprotocol/sdk`): `initialize`/`newSession`/`loadSession`/`prompt`/permission bridging, so any ACP client (Zed, etc.) can drive opencode as a worker | ForgeOS already has the *client* side (`forgeos/adapters/acp.py` drives ACP workers); opencode's `src/acp/service.ts` is a reference for the reverse direction if ForgeOS ever exposes itself as an ACP-drivable worker | `forgeos/adapters/acp.py` (reference only) |
| **Novel: workspace adapter plugin API** | `packages/plugin/src/index.ts` (`WorkspaceAdapter` type: `configure/create/remove/target`) + `control-plane/workspace.ts` — a plugin registers a named adapter (e.g. a cloud sandbox) and the rest of the system addresses it through one `WorkspaceTarget` (`local` dir vs `remote` URL+headers), uniformly | Partial — ForgeOS's `forgeos/worktrees.py` only does local git worktrees; no registered-adapter abstraction for "local vs remote execution target" | `forgeos/worktrees.py` |
| **Novel: skill discovery/pull** | `packages/opencode/src/skill/discovery.ts` — pulls a remote `index.json` of `{name, files[], version}`, downloads missing `SKILL.md`-bearing skills into a local cache with bounded concurrency | Missing | n/a |

## Top 5 ports (ranked, novelty × demand ÷ effort for ForgeOS specifically)

1. **Permission model** (`permission/evaluate.ts`) — the wildcard last-match
   ruleset with an explicit `ask` state and a `Deferred`-backed pending queue
   is a clean, ~40-line pattern for "should this tool call escalate to a
   human" that ForgeOS's leases/governor don't currently express at the
   tool-call granularity. Lowest effort, directly extends the "budget as
   contract... escalates to a human" principle already in `docs/ROADMAP.md`.
2. **Workspace adapter plugin API** (`plugin/src/index.ts` `WorkspaceAdapter`
   + `control-plane/workspace.ts`) — generalizes `forgeos/worktrees.py` from
   "always a local git worktree" to "local or remote target, chosen by a
   registered adapter." Matches the roadmap's `sandcastle` mining goal.
3. **Context compaction algorithm** (`session/compaction.ts`) —
   turn-boundary-aware pruning with a percentage-of-usable-context protected
   tail is more principled than ForgeOS's current `context_compress.py`;
   directly feeds the "byte-stable prefix compiler" roadmap item (protect the
   tail, never mutate the cached prefix).
4. **Tool registry composition + per-model/per-agent filtering**
   (`tool/registry.ts`) — the pattern of building one tool list from
   builtin+plugin+MCP sources and filtering it per model id and per caller's
   permission ruleset is a ready-made shape for ForgeOS's capsule/preflight
   tool-availability gating.
5. **Provider lazy-import + SSE-timeout wrapper** (`provider/provider.ts`
   `BUNDLED_PROVIDERS`, `wrapSSE`, `timeoutController`) — small, mechanical,
   and solves a real gap (ForgeOS's router has no per-provider SSE
   read-timeout wrapper today).

## Summary

opencode is a full chat-product harness (client/server, TUI, share links,
LSP, MCP) that ForgeOS's non-goals explicitly rule out porting wholesale.
The mineable value is five specific subsystems — permission ruleset
evaluation, a registered workspace-adapter abstraction, turn-aware context
compaction, tool-registry composition/filtering, and provider lazy-import
plumbing — all narrow, dependency-light TypeScript files with direct Python
analogs already stubbed in ForgeOS (`permission` missing, `worktrees.py`,
`context_compress.py`, no `tools.py`, `core/router.py`). Two things are
worth a second look but aren't ports: opencode's own ACP server
implementation (`src/acp/`) as a reference for exposing ForgeOS as an
ACP-drivable worker, and its skill-discovery-by-remote-index pattern, which
has no current ForgeOS equivalent or obvious need.
