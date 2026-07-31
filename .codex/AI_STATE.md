# AI State

## Current goal
- Continue turning ForgeOS from a verified kernel into a usable, cost-governed agent tool without overstating live-provider evidence.

## Important project facts
- ForgeOS is already cloned at `C:\Users\byrne\Downloads\ForgeOS`.
- `forgeos.cli.cmd_run` previously returned success after only compiling/projecting; it now delegates to the guarded `_run_team` path.
- `python -m forgeos.cli` is now a working module entry point; its guard must remain after all handler definitions.
- `forge fleet` previously crashed on Windows cp1252 because of Unicode arrows; fleet output is now ASCII-only and tested.
- `forge audit --dir .` previously crashed decoding UTF-8 source with the Windows locale; `CostAuditor` now decodes UTF-8 with replacement semantics.
- `forge cache stats` previously crashed because `PromptCache.stats()` was missing; `forge doctor` used an unsupported `home=` argument and `cache prune` called a missing method. All cache CLI paths now use the real API and close connections.
- `forge bench` still called removed `PromptCache.get/put` methods; it now uses `lookup/store` and closes the cache after each full-layer iteration.
- `forge audit --dir .` now prunes generated/vendored trees (`vendor`, `.venv`, `node_modules`, caches, etc.); the real repo audit dropped from 3503 files/2108 findings to 193/38 and still exits 0.
- `Forge.default_executor()` now lazily creates a ledger-owned `Gateway` from the default catalog/settings with persistent `.forgeos/dead_models.db`; `Forge.run(executor=None)` and the CLI reviewer use that shared routed path.
- `Gateway.resolve_model_refs("auto:free")` resolves usable catalogued free models, filters dead transport/model pairs, and `GatewayWorkerAdapter` falls through deterministically when a free slug is retired.
- The free-pool resolver now preserves explicitly selected concrete `:free` slugs instead of silently replacing them.
- `tools/ab_bench.py` is now safe by default: it prices both arms without opening a ledger or touching a transport, emits a Class-D JSON receipt with `--json-out`, and requires explicit `--live` for provider calls.
- `forgeos/forgebench.py` already owns the pinned six-task correctness-gated suite; it now serializes dry-run/live reports as `forgeos.forgebench.v1` JSON via `--json-out`, and `forgeos.cli` forwards that flag.
- `tools/aggregate_bench.py` now builds a model-free `forgeos.forgebench_table.v1` JSON/Markdown table; it preserves every receipt and gates savings to measured live Class-A runs with matching acceptance.
- `forgeos.forgebench_table` is now the reusable implementation behind both `forge forgebench-table`/`python -m forgeos.cli forgebench-table` and the repository compatibility script; the installed console also exposes the canonical read-only `forge receipts` view.
- `QuotaTracker` now persists versioned telemetry snapshots at `.forgeos/quota.json`, preserves per-provider/model parked work across restart, and the dashboard reads them at `/api/quota` without probing providers; subscription cap burn remains unmeasured unless a provider-reported percentage exists.
- The 713-entry catalog is a manifest, not a set of local source clones; broad cloning is intentionally avoided.
- Reference clones are isolated at `C:\Users\byrne\Downloads\ForgeOS-upstreams-2026-07-31`.
- Earlier concurrent routed/reducer changes were committed separately; this task did not modify their files.

## Last changed files
- `forgeos/forgebench.py` — task-derived IDF capsule packing and bounded adaptive budgets.
- `tests/test_forgebench.py` — regression coverage for budget, term extraction, and deterministic packing.
- `tests/test_local_command.py` — aligned event expectations with canonical tool output events.
- `forgeos/cli.py` — real run path, explicit budget/dry-run flags, argv injection, module entry point.
- `tests/test_cli_dispatch.py` — verifies run delegation and budget forwarding.
- `README.md` — documents the hard-cap and dry-run run commands.
- `tests/test_cli_dispatch.py` — verifies fleet output survives cp1252 consoles.
- `forgeos/cost_audit.py` — locale-independent source decoding.
- `tests/test_cost_audit.py` — UTF-8 and malformed-byte regression coverage.
- `tests/test_cost_audit.py` — also verifies generated/vendor pruning.
- `forgeos/prompt_cache.py` — operator stats and explicit close API.
- `tests/test_prompt_cache.py` — cache stats and CLI regression coverage.
- `forgeos/bench.py` — uses the current prompt-cache contract.
- `tests/test_bench.py` — locks the benchmark/cache integration.
- `forgeos/gateway/client.py`, `forgeos/gateway/free_pool.py` — default gateway/free-pool resolution and dead-model filtering.
- `forgeos/adapters/gateway_worker.py`, `forgeos/forge.py`, `forgeos/__main__.py` — shared default gateway and deterministic free fallback.
- `tests/test_free_pool.py`, `tests/test_gateway_worker.py`, `tests/test_routed_executor.py` — regression coverage.
- `README.md`, `docs/ROADMAP.md`, `docs/TEAM.md` — close stale roadmap/team claims.
- `tools/ab_bench.py`, `tests/test_ab_bench.py` — opt-in live A/B benchmark plus no-call receipt coverage.
- `forgeos/forgebench.py`, `forgeos/cli.py`, `tests/test_forgebench.py`, `tests/test_cli_dispatch.py` — pinned-suite JSON receipts and CLI forwarding.
- `tools/aggregate_bench.py`, `tests/test_aggregate_bench.py` — correctness-gated receipt aggregation and public table rendering.
- `forgeos/forgebench_table.py`, `forgeos/cli.py`, `tests/test_cli_dispatch.py` — package-level aggregation and installed-CLI parity for receipts/table commands.
- `forgeos/core/quota.py`, `forgeos/forge.py`, `forgeos/dashboard/app.py`, `tests/test_quota.py`, `tests/test_forge.py`, `tests/test_dashboard.py` — durable quota telemetry, per-model banked-reset handling, and read-only dashboard exposure.
- `forgeos/core/quota_ingest.py`, `forgeos/core/router.py`, `forgeos/registry.py`, `tests/test_quota_ingest.py`, `tests/test_router.py` — offline header/report normalization and effective-cost arbitration for mapped subscription capacity.
- `forgeos/cli.py`, `forgeos/core/quota_ingest.py`, `tests/test_cli_dispatch.py` — safe local `quota ingest` / `ingest` commands for header JSON and copied CLI reports; fixed duration-suffix parsing.
- `forgeos/core/effort.py`, `forgeos/cli.py`, `forgeos/prompts/prefix.py`, `tests/test_effort.py` — task-difficulty effort routing and a real bounded `forge init` repository scan (committed in `4c9390f`).

## Commands run
- `rtk proxy python -m pytest tests -q -m "not slow"`
- `rtk proxy python -m pytest tests/test_forgebench.py tests/test_local_command.py -q`
- `ruff check forgeos tests`; `python -m compileall -q forgeos tests`
- `python -m forgeos.forgebench --dry-run --model deepseek/deepseek-chat --budget-usd 50`
- `python -m forgeos.cli run "add a deterministic retry guard" --dry-run`
- `python -m forgeos.cli forgebench --dry-run --model deepseek/deepseek-chat --budget-usd 50`
- `python -m forgeos.cli fleet`
- `python -m forgeos.cli audit --dir .` (3503 files audited, 2108 findings, exit 0)
- `python -m forgeos.cli audit --dir .` (193 files audited, 38 findings, exit 0 after pruning)
- `python -m forgeos.cli cache stats` and `python -m forgeos.cli doctor`
- `python -m forgeos.cli bench "measure a local retry helper" --iterations 1`
- `python -m forgeos.cli doctor`, `python -m forgeos.cli fleet`, and `python -m forgeos.cli run "summarize the retry policy" --dry-run`
- local default-gateway catalog resolution smoke check (no provider call)
- `python tools/ab_bench.py --model openrouter/openrouter/free --repeat 2 --json-out <temp receipt>`
- checkpointed full suite via `sweep.py` and `pytest --collect-only`: 1729 tests collected, sweep rc 0
- `python -m forgeos.cli forgebench --dry-run --model openrouter/openrouter/free --budget-usd 50 --json-out <temp receipt>`
- `python tools/aggregate_bench.py <temp receipt> --json-out <temp table>` (one dry-run retained, zero eligible savings)
- `python -m forgeos.cli forgebench-table <temp receipt> --json-out <temp table>` (same zero-claim result)
- `python -m forgeos.cli receipts --state-dir <missing temp dir>` (read-only failure, directory not created)
- `rtk proxy python -m pytest tests -q -m "not slow"` after the quota work (1782 passed, 1 failure in concurrent ForgeBench packing work, 17 deselected, 1 existing FastAPI/httpx deprecation warning)
- `python -m pytest tests/test_quota.py -q` (52 passed)
- `python -m pytest tests/test_forge.py tests/test_dashboard.py -q` (53 passed, 1 existing FastAPI/httpx deprecation warning)
- offline dashboard dogfood with a persisted `Weekly: 75% remaining` report (`/api/quota` and summary both showed 25% burn)
- direct `QuotaIngestor` assertions for Anthropic utilization/reset, generic rate limits, vendor exhaustion, and CLI reports (passed); `git diff --check` (passed)
- `python -m pytest tests/test_cli_dispatch.py -q` (147 passed)
- `python -m pytest tests/test_quota_ingest.py -q` (4 passed)
- `python -m pytest tests/test_router.py tests/test_quota.py -q` (80 passed)
- `ruff check forgeos/cli.py forgeos/core/quota_ingest.py tests/test_cli_dispatch.py`; `python -m py_compile ...` (passed)
- `python -m pytest tests/test_effort.py -q` (28 passed)

## Test status
- Passing: full checkpointed suite (`forgeos-final-suite-clean`, rc 0, 188.4s); 26 diagnostics/dashboard-chat tests; 28 effort tests; 147 CLI dispatch tests; 4 quota-ingest tests; 80 router/quota tests; 106 focused quota/Forge/dashboard tests; 164 focused benchmark/CLI/aggregator tests; Ruff; compileall; CLI dogfood and no-call benchmark smoke checks.
- Failing: full non-slow sweep currently has one unrelated concurrent failure in `tests/test_forgebench_packing.py::test_definition_weighting_is_what_makes_that_true` while `forgeos/forgebench.py` is concurrently modified.
- Not run: live provider calls or live execution (intentionally gated); the full suite was run without provider calls.

## Known blockers
- No blocker for the source upgrade. Full catalog clone coverage remains intentionally unperformed because it is 713 repositories.
- Unrelated reducer wrapper-summary fixes were committed concurrently; they were not touched by this benchmark work.
- A concurrent routed-execution change in `forgeos/adapters/routed.py`, `forgeos/forge.py`, and `tests/test_routed_executor.py` was committed separately; it was not touched here. Its focused route tests passed (15 passed).
- Concurrent ForgeBench packing work is dirty in `forgeos/forgebench.py` and `tests/test_forgebench_packing.py`; `forgeos/economy/savings.py` is also dirty. Do not stage or revert those files.
- Additional concurrent adapter, dashboard, economy, gateway, ledger, registry, settings, hooks, MCP, and test changes are currently dirty; do not stage or revert files outside owned hunks.

## Next best steps
- Keep provider calls opt-in; do not run `tools/ab_bench.py --live`, `forge forgebench` live, or a real `forge run` without explicit operator-approved provider/budget calls.
- The `forge` executable is not installed in this shell (`Get-Command forge` returned unavailable); the source-equivalent `python -m forgeos.cli` path is verified without installing anything.
- Vendor-neutral quota ingestion and subscription-vs-API arbitration are committed: `QuotaIngestor`, `market_resource` mappings, effective-cost routing, `Forge.ingest_quota`, `forge quota --json`, and `forge quota ingest`.
- Keep provider calls opt-in; stage only verified owned hunks if touching the concurrently dirty files.
- Current audit of the control-plane/probe work: `tests/test_automemory.py` passed 13; all 11 hook cases passed across the full run/isolation rerun; all 12 MCP stdio cases passed via checkpointed `sweep.py`; `tests/test_probe.py` passed 16; adapter bridge/factory/routed tests passed 65; the dirty effort-propagation test set passed 31; Ruff and py_compile passed for the new modules. The final clean full-suite sweep now passed (rc 0).
- Gateway affinity/provider-signal work is now committed in `e50bab0`: job-scoped warm-seat pinning, TTL/quota/health release, provider signal parsing, and ForgeBench/GatewayWorker affinity wiring. Focused cache-affinity/provider-signal/gateway tests passed 53; Ruff and py_compile passed.
- SQLite/lease hardening is committed in `c2e6dde`; its cross-process `BEGIN IMMEDIATE` path and refused-acquire rollback were verified with `python -m pytest tests/test_sqlite_concurrency.py tests/test_leases.py -q --timeout=300` (52 passed), plus Ruff, py_compile, and `git diff --check`.
- `forgebench_report.json` remains dirty generated output; preserve it and do not stage it with unrelated work.
- Final-suite verification history: an earlier run hit a transient `gateway_worker.py` decorator race during concurrent edits and a second attempt hit sweep's default 120-second timeout. The corrected checkpointed run used `sweep.py --timeout 600` and completed with rc 0 in 188.4s; its capture was capped before the final pytest count, but the process exit is authoritative.
- Post-suite dogfood: `python -m forgeos.cli doctor --json` returned rc 0 with 13/14 declared providers usable; `python -m forgeos doctor` returned rc 0 with 9/10 registry workers runnable and `Degradations: none recorded this process`. Gateway auth remains intentionally unavailable without `FORGEOS_GATEWAY_URL`; no live provider call was made.
- Follow-up observability hardening is committed in `cd62936`: `CostRouter` now uses the circuit-breaker's locked public state API and records inspection degradation; corrupt probe caches and malformed MCP configs are visible through the same bounded recorder. Focused cost-router/discovery/diagnostics tests passed 42, and a fresh checkpointed full suite (`forgeos-final-suite-after-observability`, sweep timeout 600) passed rc 0 in 136s.
- Settings fallback is now visible too (`edea03b`): corrupt settings record a degradation before defaults are used. Focused cost-router/discovery/diagnostics tests passed 43, and the fresh checkpointed full suite (`forgeos-final-suite-after-settings`, sweep timeout 600) passed rc 0 in 123s.
- Queue safety hardening is committed in `4730fc4`: a corrupt `halts.json` no longer disappears silently; `watch_queue` records that it continued and that the operator halt may not have been honored. Watch/diagnostics tests passed 30, and the fresh checkpointed full suite (`forgeos-final-suite-after-watch`, sweep timeout 600) passed rc 0 in 121s.
- Knowledge-scout omissions are now visible in `4abb34d`: registry fetch failures, malformed responses, and individual candidate parse failures record bounded diagnostics while preserving the non-fatal search contract. Scout/diagnostics tests passed 54, and the fresh checkpointed full suite (`forgeos-final-suite-after-scout`, sweep timeout 600) passed rc 0 in 204s.
- New concurrent work is dirty in dashboard/chat, diagnostics, auto-discovery, resources, CLI, and Forge integration files; preserve those files and do not stage or revert them without ownership evidence.
