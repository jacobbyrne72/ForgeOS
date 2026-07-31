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

## Test status
- Passing: 163 focused benchmark/CLI/aggregator tests; 1731-test full suite target (checkpointed sweep rc 0); Ruff; compileall; CLI dogfood and no-call benchmark smoke checks.
- Failing: none observed.
- Not run: none.

## Known blockers
- No blocker for the source upgrade. Full catalog clone coverage remains intentionally unperformed because it is 713 repositories.
- Unrelated reducer wrapper-summary fixes were committed concurrently; they were not touched by this benchmark work.
- A concurrent routed-execution change in `forgeos/adapters/routed.py`, `forgeos/forge.py`, and `tests/test_routed_executor.py` was committed separately; it was not touched here. Its focused route tests passed (15 passed).

## Next best steps
- Keep provider calls opt-in; do not run `tools/ab_bench.py --live`, `forge forgebench` live, or a real `forge run` without explicit operator-approved provider/budget calls.
- Next product candidate: collect explicitly approved measured live receipts and publish a real bill table; never infer live savings from dry-run/modelled data.
