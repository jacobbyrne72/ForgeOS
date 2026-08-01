# Blueprint vs. built — an honest audit

Two architecture blueprints define the target: a **Model-Native Execution
Compiler** (decide whether to ask, which model, what context, how much
reasoning, what authority, when to stop) and a **Corpus Compiler** (turn
gigabytes into evidence locally, without paying an API).

This file records which parts exist, which are partial, and which are absent.
It is written the same way every other claim here is: verified against the
source tree, not against intention. `HAVE` means the module exists and has
tests. `PARTIAL` means the mechanism exists but not in the shape the
blueprint specifies. `MISSING` means it is not there.

## Engine 1 — Model-Native Execution Compiler

| Blueprint component | State | Where |
|---|---|---|
| MissionContract | PARTIAL | `compiler.py` compiles objectives into TaskSpecs + a dependency graph. There is no user-approval gate, no immutable numbered revision, no `DRAFT → PLAN_PROPOSED → USER_ACCEPTED → MISSION_LOCKED` transition. |
| Deterministic task compiler → DAG | HAVE | `compiler.py`, with `parallel_safety` sequencing pairs that would silently conflict. |
| Economic workflow router | HAVE | `core/router.py` (cheapest tier clearing a risk threshold), `core/market.py` (capacity priced as inventory), `core/quota.py`. |
| Reasoning-effort budgeting | HAVE | `core/effort.py`, on the real call path. |
| **Prompt IR** | **BUILDING** | `prompts/ir.py` — provider-neutral semantics stated once. |
| **Model-native renderers** | **BUILDING** | `prompts/renderers.py` — anthropic XML, openai lean, google query-last, deepseek stable-prefix. |
| Model Execution Profiles | PARTIAL | `models/traits.py` + `registry.py` carry capability and measured win-rate. No per-snapshot profile with acceptance rate, tokens-per-accepted, drift rate, or prompt-policy fields. |
| Context Compiler | HAVE | `economy/capsule.py` — ranked, budgeted, graduated trimming. |
| Byte-stable prefix caching | HAVE | `prompts/prefix.py`, with a CI test asserting byte identity. |
| Tool ABI Compiler | MISSING | Tool schemas are not compiled per-model or trimmed to the active set. |
| SkillJIT | HAVE | `instructions.py` — progressive disclosure, metadata before bodies. |
| Drift Firewall | PARTIAL | `core/governor.py` detects loops and caps attempts. There is no progress-efficiency or drift-tax metric, and no graduated response ladder. |
| Stop-Loss Controller | HAVE | Budget contracts, admission control before spend, governor trips. |
| Failure Attribution | HAVE | `FailureClass`; only MODEL failures escalate a tier. |
| ProofGraph | PARTIAL | Acceptance criteria + merge gate enforce evidence per task. There is no criterion-level proof graph across a mission. |
| Independent verifier | HAVE | Merge gate requires a different worker, and prefers a different provider family. |
| Harness Calibrator | MISSING | Nothing learns prompt/effort/tool policy from measured outcomes. The single largest remaining gap. |
| SavingsProof | HAVE | `economy/savings.py` — measured / replayed / modelled, enforced in code. |
| JobCards with enforced permissions | PARTIAL | `Scope` + capabilities + path leases enforce write authority. There is no typed JobCard with `forbidden`, `stop_conditions`, and per-job token budget. |
| Event-driven manager | PARTIAL | The scheduler is deterministic and the manager is not in the normal loop. Worker events are not a typed protocol (`PROGRESS`, `BLOCKED`, `SCOPE_VIOLATION`, …). |

## Engine 2 — Corpus Compiler

Largely **absent**, and worth stating plainly rather than implying otherwise.

| Blueprint stage | State |
|---|---|
| Content hashing + incremental reingestion | MISSING |
| Format-aware parsing (Docling / Tree-sitter / DuckDB) | MISSING |
| Semantic units instead of fixed chunks | PARTIAL — the capsule ranks code blocks, but there is no document corpus |
| Exact + near dedup (MinHash/SimHash) | MISSING |
| Parquet / content-addressed object store | PARTIAL — reducer content-addresses tool output; no corpus store |
| Lexical index (Tantivy) | MISSING — `rg` is shelled out to, not indexed |
| Code index (Zoekt / Serena) | MISSING as a dependency |
| Optional local embeddings | MISSING |
| Deterministic query planner | PARTIAL — `economy/lowerer.py` asks whether a task needs a model at all |
| Evidence bundle with excluded-candidate counts | PARTIAL — the capsule produces bounded evidence; it does not report what it excluded |

## What this means for priority

The blueprint's own rule — *never pay a model for what deterministic
software does better* — is already the spine of this codebase. The economics
(ledger, budgets, refusals, receipts), the safety (leases, worktrees, merge
gate, independent review), and the routing (tier, effort, quota, affinity)
are built and tested.

What is missing clusters into three:

1. **Model-native rendering** — one semantics, many provider dialects. In
   progress; the largest per-token win available, because DeepSeek's cached
   input is ~50–120× cheaper than uncached and Anthropic/OpenAI/Google each
   reward a different prompt shape.
2. **Calibration** — nothing here learns from its own receipts yet. The
   ledger records every outcome; no loop reads it to promote a better prompt,
   effort level, or tool set. `knowledge/automemory.py` mines lessons but
   files them as unverified claims, not as policy.
3. **Corpus** — the gigabyte-to-evidence pipeline is a separate system that
   does not exist here. It should stay separate: it is a different lifecycle
   (ingest once, query often) from a coding harness (plan, execute, verify).

Nothing in this file should be read as a promise. It is a map of where the
code is, so the next work is chosen on evidence rather than on the most
recently-read document.
