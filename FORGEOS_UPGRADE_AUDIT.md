# ForgeOS Upgrade Audit

Date: 2026-07-31

## Scope

- Target repository: `C:\Users\byrne\Downloads\ForgeOS`
- Source catalog: `C:\Users\byrne\Downloads\forgeos_github_mega_catalog_2026-07-30`
- Catalog entries: 713 unique GitHub origins
- Cloned repositories: 721 repository directories
- Exact catalog origins cloned: 694
- Verified replacement origins: 15
- Free space at latest audit time: approximately 14.43 GB

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
- Concurrent CLI additions left handlers unreachable, duplicated dispatch keys, and omitted parser arguments; all registered CLI commands now have reachable subparsers.
- The later `format` and `purge` handlers were also missing parser registrations; both now have explicit arguments and reachable dispatch paths.
- An empty mission could leave the CLI optimization plan uninitialized.
- A second `cmd_compress` definition shadowed context compression and required an unregistered argument; it is now a separate `output-compress` command.
- The output-compressor demo contained a malformed newline literal.
- `output_compressor.py` contained additional malformed newline literals that prevented repository-wide parsing; those literals are now valid.
- Benchmarks accepted `iterations=0` and could read uninitialized results; invalid iteration counts now fail explicitly.
- The CLI-team factory imported a nonexistent `CliTeamAdapter` symbol instead of the implemented `OMCTeamAdapter`, making that backend appear unavailable; the factory now builds the implemented adapter.
- Normalized event sequence IDs and watch spend history types to match SQLite/runtime values.
- Fixed Python 3.11-incompatible nested f-strings in model ranking and CLI output.
- Repaired malformed response-truncation newline literals and made the truncator return valid metadata on every path.
- Made missing ledger task/job rows explicit failures and made routed execution safe when no ledger is configured.
- Hardened diff hunk parsing against optional regex groups and normalized security finding counts.

Current evidence:

- `ruff check forgeos tests`: passes.
- Full current test suite: 1301 passed, 1 dependency deprecation warning.
- Focused Forge/lease/CLI regression suite: 202 passed.
- Dependency audit: `uv pip compile pyproject.toml --all-extras` followed by `pip-audit`; no known vulnerabilities found.
- No production `eval`, `exec`, or `shell=True` found.
- Pyright currently reports 7 inferred-type errors, all at dynamic SQLite row access or third-party tree-sitter/gateway protocol boundaries; runtime behavior is covered by the passing suite.

The stricter supplemental security rules still flag intentional resilience handlers, runtime assertions, controlled SQL placeholder construction, and bounded subprocess argv. These were reviewed and not changed without a demonstrated behavioral defect.

## Unresolved catalog origins

These catalog URLs were unavailable or stale and were not replaced with unrelated projects:

- `JetBrains/awesome-ai-devtools`
- `lunary-ai/lunary`
- `NirDiamant/AI_Agents_Book`
- `xlang-ai/awesome-language-agents`

On the latest live check, all four remaining original URLs returned no Git reference from `git ls-remote`: `JetBrains/awesome-ai-devtools`, `lunary-ai/lunary`, `NirDiamant/AI_Agents_Book`, and `xlang-ai/awesome-language-agents`.

Verified replacements are recorded separately in the replacements clone root, including `allenai/OLMo-Eval` for the stale OLMoE evaluation URL, `qodo-ai/qodo-cover` for the stale Cover-Agent URL, `ast-grep/ast-grep`, `Azure/counterfit`, `ai-dynamo/nixl`, `kingjulio8238/Memary`, `MotiaDev/motia`, `microsoft/semanticworkbench`, `NirDiamant/agents-towards-production`, `OWASP/www-project-ai-security-and-privacy-guide`, `sourcegraph/sourcegraph-public-snapshot`, `sourcegraph/cody-public-snapshot`, `wandb/wandb`, `Not-Diamond/notdiamond-python`, and `microsoft/AI-Red-Teaming-Playground-Labs`.

Candidate successors cloned for further provenance review: `NirDiamant/GenAI_Agents` for `NirDiamant/AI_Agents_Book`, `jamesmurdza/awesome-ai-devtools` for `JetBrains/awesome-ai-devtools`, `ysymyth/awesome-language-agents` for `xlang-ai/awesome-language-agents`, and `lunary-ai/lunary-py` for `lunary-ai/lunary`. They are not counted as exact or verified replacements because no redirect or maintainer statement was found.
