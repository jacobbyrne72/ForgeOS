# ForgeOS Upgrade Audit

Date: 2026-07-31

## Scope

- Target repository: `C:\Users\byrne\Downloads\ForgeOS`
- Source catalog: `C:\Users\byrne\Downloads\forgeos_github_mega_catalog_2026-07-30`
- Catalog entries: 713 unique GitHub origins
- Cloned repositories: 714 repository directories
- Exact catalog origins cloned: 694
- Verified replacement origins: 12
- Free space at audit time: approximately 14.64 GB

Clone roots:

- `C:\Users\byrne\Downloads\ForgeOS-upstreams-2026-07-31`
- `C:\Users\byrne\Downloads\ForgeOS-catalog-p0-2026-07-31`
- `C:\Users\byrne\Downloads\ForgeOS-catalog-p1-partial-2026-07-31`
- `C:\Users\byrne\Downloads\ForgeOS-catalog-p2-partial-2026-07-31`
- `C:\Users\byrne\Downloads\ForgeOS-catalog-other-partial-2026-07-31`
- `C:\Users\byrne\Downloads\ForgeOS-catalog-replacements-2026-07-31`

## ForgeOS changes

- Added a native local-command worker adapter with registry and factory support.
- Added ACP session resume and checkpoint support.
- Fixed fresh SQLite initialization ordering.
- Corrected security scanning to scan requested paths and skip nonexistent task paths.
- Added model profiling and automatic optimization CLI support.
- Repaired CLI dispatch, unknown-command status handling, parser registration, and public exports.
- Added focused tests for local commands, ACP capabilities, adaptation/context compression, and CLI behavior.

## Issue search

Fixed defects found during review:

- CLI `main()` parsed commands but did not dispatch them.
- Unknown parsed commands could fail with an opaque lookup error instead of status `2`.
- Fresh Forge instances could reference `spend` before Ledger tables existed.
- Gitleaks scanned the repository root instead of each requested source path.
- Security checks could deadlock task execution when all reported paths were absent.
- Multiple unused exports, imports, locals, and non-strict `zip()` calls violated the project lint policy.

Current evidence:

- `ruff check forgeos tests`: passes.
- Full test suite: 1259 passed, 1 dependency deprecation warning.
- No production `eval`, `exec`, or `shell=True` found.

The stricter supplemental security rules still flag intentional resilience handlers, runtime assertions, controlled SQL placeholder construction, and bounded subprocess argv. These were reviewed and not changed without a demonstrated behavioral defect.

## Unresolved catalog origins

These catalog URLs were unavailable or stale and were not replaced with unrelated projects:

- `JetBrains/awesome-ai-devtools`
- `lunary-ai/lunary`
- `memary-ai/memary`
- `NirDiamant/AgentsTowardsProduction`
- `NirDiamant/AI_Agents_Book`
- `OWASP/AI-Exchange`
- `xlang-ai/awesome-language-agents`

Verified replacements are recorded separately in the replacements clone root, including `allenai/OLMo-Eval` for the stale OLMoE evaluation URL, `qodo-ai/qodo-cover` for the stale Cover-Agent URL, `ast-grep/ast-grep`, `Azure/counterfit`, `ai-dynamo/nixl`, `MotiaDev/motia`, `microsoft/semanticworkbench`, `sourcegraph/sourcegraph-public-snapshot`, `sourcegraph/cody-public-snapshot`, `wandb/wandb`, `Not-Diamond/notdiamond-python`, and `microsoft/AI-Red-Teaming-Playground-Labs`.
