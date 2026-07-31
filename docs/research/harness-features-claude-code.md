# Harness features — Claude Code (mined from public surface, not source)

**Source note:** `anthropics/claude-code` ships no application source — this repo is
CHANGELOG.md (465KB, ~4,900 lines back to early releases), docs pointers, `.claude/`
commands, `.devcontainer/`, `examples/`, and `plugins/`. Everything below is inferred
from that public surface plus two secondary sources: `vendor/system_prompts_leaks/Anthropic/Claude Code/`
(binary-extracted agent/skill/reminder prompts, captured at CLI v2.1.211 — a third party's
reverse-engineering, not Anthropic-published) and `vendor/claude-agent-acp/` (Anthropic's
official ACP adapter package, source-available). No claim here reflects the actual
TypeScript implementation.

## Feature inventory

| Feature | Documented behavior | ForgeOS | Target module |
|---|---|---|---|
| Hooks (Pre/PostToolUse, SessionStart/End, UserPromptSubmit, Stop/SubagentStop) | 9 lifecycle events; JSON stdin, `permissionDecision` (`allow`/`ask`/`deny`/`defer`), `additionalContext` injection, `updatedInput`/`updatedToolOutput` rewrite, agent-scoped hooks in frontmatter | Partial — `hooks/ripper_gate.py` is one hard-deny PreToolUse-style gate for Claude Code itself; no general lifecycle hook bus for ForgeOS's own task pipeline | `forgeos/core/` hook dispatcher |
| Skills (`SKILL.md`, bundled + plugin + project) | Folder-with-`SKILL.md` or flat `.md`; `disableBundledSkills`, `/reload-skills`, `context: fork`, `alwaysLoad`-style deferral | Missing — no skills directory or discovery in `forgeos/` | none yet |
| Subagents + teams (Agent tool, `claude agents`, implicit teams) | Depth-limited nesting (default 3, cap 200/session), background-by-default, `SendMessage`/`TaskCreate`/`TaskList` mailbox, one implicit team per session (`TeamCreate`/`TeamDelete` removed) | Partial — `core/awareness.py` (TeamBoard), `core/scheduler.py` wave dispatch, and `worktrees.py` give an analogous but non-Task-tool coordination model | `forgeos/core/manager.py`, `awareness.py` |
| MCP client + server | stdio/HTTP/SSE servers, OAuth, pagination, `alwaysLoad`, per-server `disallowedTools` | Missing — adapters (`acp.py`, `ollama.py`, `local_command.py`, `gateway_worker.py`) are CLI/process adapters, not MCP | `forgeos/adapters/` (new mcp.py) |
| Plan mode | Read-only-until-approved mode; static Bash analyzer + auto classifier; `/plan`, "Refine with Ultraplan" web handoff | Missing | none |
| Permission modes + allowlists | `default`→`manual` rename, `acceptEdits`/`bypassPermissions`/`plan`; `Tool(param:value)` rule syntax (e.g. `Agent(model:opus)`) | Partial — `governor.py`/`may_start` is a pre-admission gate but not a mode/allowlist grammar | `forgeos/core/governor.py` |
| Session resume/continue (`--resume`, `--continue`, `/rewind`) | Transcript-based resume, background-task reattachment, `/rewind` file-state undo (79× transcript pruning) | Have (recent) — `quota.py`/`quota_ingest.py` persist provider facts across restarts, and the adapter checkpoint/resume wiring into the Forge attempt loop landed this session; not independently verified by this miner | `forgeos/adapter/` checkpoint layer |
| Memory (`CLAUDE.md` hierarchy) | Project/user/enterprise precedence, `@path/to/file.md` imports (since early releases), conditional `.claude/rules/*.md` via `paths:` frontmatter, `/doctor` trim suggestions | Have (repo has `AGENTS.md`, per-dir conventions) but no import/precedence engine of its own | n/a — ForgeOS is a CLI orchestrator, not itself an agent memory consumer |
| Background tasks | Subagents/shells run backgrounded by default, notify on completion, survive `/clear`/resume, per-session caps | Partial — `Forge.run` wave loop runs tasks in a thread pool concurrently but has no detached/notify-on-completion UX | `forgeos/forge.py` |
| Sandboxing | Bubblewrap (Linux)/seatbelt-style network+filesystem isolation, `sandbox.network.strictAllowlist`, `sandbox.credentials`, devcontainer firewall (`init-firewall.sh`) | Partial — `worktrees.py` isolates the *filesystem effect* of a task, not a general command/network sandbox | `forgeos/worktrees.py` (extend), new sandbox module |
| Checkpointing | Bounded checkpoint disk usage via pruned file-history backups; ACP-side no dedicated checkpoint API found | Have (recent) — landed this session per team task tracker; not independently verified by this miner | `forgeos/adapter/` |
| Output styles | Plugin-shareable system-prompt presets (`keep-coding-instructions` frontmatter); deprecated then un-deprecated | Missing | n/a |
| Statusline | `/statusline`, JSON payload (`context_window`, `rate_limits`, `added_dirs`), multi-line support | Different form — `forge dash` web dashboard covers similar ground, no terminal statusline | n/a |

## Notable surface not in the standard inventory

- **Managed-settings / MDM fleet policy** (`examples/mdm/`, `examples/settings/`): three-tier
  lax/strict/bash-sandbox templates, deployed via Jamf/Intune/GPO, sit above user config in
  precedence and can't be overridden locally — a governance pattern ForgeOS's budget-governor
  concept doesn't yet have an enterprise-policy equivalent for.
- **`security-guidance` plugin** (`plugins/security-guidance/README.md`): three-layer review —
  instant regex patterns on Edit/Write, an LLM diff review on Stop, and an agentic
  Read/Grep/Glob-driven commit reviewer that traces cross-file data flow (IDOR, SSRF). Directly
  analogous to ForgeOS's "merge gate + independent review" (`TEAM.md` §7) but triggered off
  hooks rather than a task-graph merge check.
- **`hookify` plugin**: declarative hooks authored as `.claude/hookify.*.local.md` with YAML
  frontmatter (event/pattern/action: warn|block) instead of editing `hooks.json` — a
  lower-friction hook-authoring UX worth copying if ForgeOS grows a general hook bus.
  Note: this is a community-style convenience layer bundled in the official plugins directory,
  not a claim about the core hooks engine.
- **`ralph-wiggum` plugin**: the Stop hook re-feeds the same prompt until a completion string
  appears or `--max-iterations` is hit — a minimal, hook-native version of an autonomous retry
  loop; comparable in spirit to ForgeOS's wave scheduler but session-local and prompt-driven
  rather than task-graph-driven.
- **`claude-agent-acp` nested subagent transcripts**: ACP 1.2 has no standard subagent
  relationship, so the adapter forwards subagent text/thinking/tool-calls tagged with
  `_meta.claudeCode.parentToolUseId`/`subagent: true`, gated behind a client capability flag —
  relevant if ForgeOS's ACP adapter (`adapters/acp.py`) needs to surface nested-agent activity
  to a UI.
- **Devcontainer firewall** (`.devcontainer/init-firewall.sh`, `NET_ADMIN`/`NET_RAW` caps):
  reference implementation of network egress control at the container level, complementary to
  (not a replacement for) an in-process sandbox.

## Top 5 ports, ranked

1. **Declarative security-guidance-style merge gate** — port the 3-layer pattern (pattern
   regex → LLM diff review → agentic cross-file reviewer) onto ForgeOS's existing merge gate
   (`TEAM.md` §7), since ForgeOS already has the review-lane primitive; this mainly adds the
   pattern-rule and diff-review layers ahead of the expensive agentic pass.
2. **Checkpoint/resume disk-pruning strategy** — the adapter checkpoint/resume wiring itself
   landed this session (team tracker), but Claude Code's specific technique — bounded
   checkpoint disk via pruning *superseded* backups rather than capping count/age, and
   resume that surfaces orphaned background work as a single summary — is still a usable
   refinement reference.
3. **Hook bus with a `deny`/`ask`/`defer` decision model** — generalizes `ripper_gate.py` from
   one hard-coded gate into a lifecycle event bus (PreToolUse-equivalent at minimum), giving
   ForgeOS's own task pipeline the same "mechanical enforcement beats instruction" property the
   ripper gate already proves for Claude Code itself.
4. **MDM-style managed policy tier** — a settings precedence layer that sits above user/project
   config and can't be overridden, useful once ForgeOS is deployed across a team rather than a
   single operator.
5. **Sandboxed Bash execution** — network/filesystem allow-deny lists per the `sandbox.*`
   settings family; ForgeOS's worktree isolation stops cross-task file collisions but not what
   a single task's shell commands can reach.

## Receipts

- Changelog line numbers cited are from `vendor/claude-code/CHANGELOG.md` (single file, no
  per-version anchors before ~line 3000); e.g. hooks lifecycle: L4818 (UserPromptSubmit),
  L4882 (Stop/SubagentStop split), L4764 (SessionStart), L4669 (SessionEnd), L3980 (agent
  frontmatter hooks), L2568 (`defer` decision), L1605 (`continueOnBlock`), L1029
  (Stop `additionalContext`). Memory import: L5080 (`@path/to/file.md`). Subagent caps/teams:
  L92–L842 range (concurrency caps, depth limits, implicit-team removal of `TeamCreate`).
- Doc paths: `vendor/claude-code/plugins/{hookify,security-guidance,ralph-wiggum}/README.md`,
  `vendor/claude-code/examples/{mdm,settings}/README.md`, `vendor/claude-code/.devcontainer/`.
- Secondary (binary-extracted, unofficial): `vendor/system_prompts_leaks/Anthropic/Claude Code/{agents/teammate.md,injected-reminders/teammate.md,bundled-skills/{README.md,memory-types.md},slash-commands/README.md}`.
- ACP: `vendor/claude-agent-acp/README.md` (nested subagent transcript forwarding).
