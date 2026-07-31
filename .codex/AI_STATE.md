# AI State

## Current goal
- Make ForgeOS's advertised front door execute real, budget-governed work with honest dry-runs and receipts.

## Important project facts
- ForgeOS is already cloned at `C:\Users\byrne\Downloads\ForgeOS`.
- `forgeos.cli.cmd_run` previously returned success after only compiling/projecting; it now delegates to the guarded `_run_team` path.
- `python -m forgeos.cli` is now a working module entry point; its guard must remain after all handler definitions.
- `forge fleet` previously crashed on Windows cp1252 because of Unicode arrows; fleet output is now ASCII-only and tested.
- `forge audit --dir .` previously crashed decoding UTF-8 source with the Windows locale; `CostAuditor` now decodes UTF-8 with replacement semantics.
- `forge cache stats` previously crashed because `PromptCache.stats()` was missing; `forge doctor` used an unsupported `home=` argument and `cache prune` called a missing method. All cache CLI paths now use the real API and close connections.
- `forge bench` still called removed `PromptCache.get/put` methods; it now uses `lookup/store` and closes the cache after each full-layer iteration.
- `forge audit --dir .` now prunes generated/vendored trees (`vendor`, `.venv`, `node_modules`, caches, etc.); the real repo audit dropped from 3503 files/2108 findings to 193/38 and still exits 0.
- The 713-entry catalog is a manifest, not a set of local source clones; broad cloning is intentionally avoided.
- Reference clones are isolated at `C:\Users\byrne\Downloads\ForgeOS-upstreams-2026-07-31`.
- Existing dirty files before this task: `forgeos/forge.py`, `docs/research/verification-economy.md`, `tests/test_merge_retry.py`.

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

## Test status
- Passing: 4 focused audit/benchmark tests; 1723 full tests (including slow); Ruff; compileall; live CLI dogfood.
- Failing: none observed.
- Not run: none.

## Known blockers
- No blocker for the source upgrade. Full catalog clone coverage remains intentionally unperformed because it is 713 repositories.

## Next best steps
- Commit `5106f02` contains the verified front-door hardening; next work can target a new product capability rather than re-deriving this audit.
