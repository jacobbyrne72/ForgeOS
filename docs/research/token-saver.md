# token-saver (ppgranger/token-saver) — mined for ForgeOS

Read in full: `vendor/token-saver` (v2.0, 36 command processors, 853 tests, Apache-2.0).
It solves one narrow problem well — deterministic, zero-latency, no-LLM compression of
*CLI tool output* before it reaches the model — via a Claude Code PreToolUse hook that
rewrites `git diff` etc. into `python3 wrap.py '<cmd>'`, and an Antigravity AfterTool
hook that edits already-captured output. No vector search, no summarizing model, no
caching layer — same "deterministic code beats a model call" bet ForgeOS's own
`lowerer.py` makes, applied to one specific surface.

**Measured vs. asserted, up front**: the README's headline table (60-99% by command
type) and every per-processor doc page are generated from `audit_compression.py` /
`examples/demo.py` — synthetic fixtures the project authored itself, printed via a
manual script, not assertions in the 853-test suite and not sampled from a real usage
corpus. The one *actually* measured number is `src/tracker.py`'s local SQLite ledger
(`savings`/`sessions` tables, WAL mode, real before/after byte counts per real command)
— but it has no `baseline_source`-style provenance discipline: `stats.py` prints
"67.3% saved" with no distinction from `avoidance.py`'s enforced MEASURED/MODELLED
split, and token counts are always a flat `chars/4` estimate, never a real tokenizer
count (`config.py:49` `chars_per_token`). Where ForgeOS already has a `Provenance`
enum and `savings_pct()`'s weaker-of-two-inputs rule (`economy/savings.py:167-222`),
token-saver has one unqualified percentage. Don't import its claimed ratios as fact.

| Technique | Implementation | Claimed saving | Provenance | ForgeOS | Target |
|---|---|---|---|---|---|
| PreToolUse command rewrite → wrapper → compressed replay | `scripts/hook_pretool.py:409-456`, `scripts/wrap.py` | — | mechanism, not a number | **missing** | hooks (#16) |
| Priority-chain processor dispatch, first-match-wins + generic 2nd pass | `src/engine.py:56-141` | — | mechanism | partial (`reducer.py` has 2 processors: pytest + generic; no dispatch chain) | `economy/reducer.py` |
| Quote-aware chain splitting (`&&`/`;`), per-segment routing via marker injection | `src/chain_utils.py`, `scripts/wrap.py:58-115` | — | mechanism | missing | hooks (#16) |
| Command-safety classifier: reject `\|\|`, `$()`, backticks, heredocs, `sudo`, redirects, streaming (`tail -f`, `docker stats`) | `scripts/hook_pretool.py:104-269` | — | mechanism, 174 tests | missing | hooks (#16) |
| "Processor mismatch" telemetry — log when a specialized path under-compresses so weak processors surface empirically | `src/engine.py:119-134`, `tracker.py:223-270` | — | measured (local) | missing (no equivalent self-audit loop for reducer processors) | `economy/reducer.py` |
| Structured pytest reduction: FAILURES parsed, traceback truncated to N lines | `src/processors/test_output.py:82-224`, `_truncate_traceback:67-80` | 95% (500 tests, 2 fail) | asserted (demo script) | **ForgeOS ahead** — `reduce_pytest` groups by root-cause with `count`/`also`, never drops a FAILED id (`economy/reducer.py:291-381`) | n/a |
| Generic fallback: ANSI strip, repeated-line collapse, numeric-normalized similar-line collapse, head/tail truncate | `src/processors/generic.py` | — | mechanism | partial — ForgeOS's `reduce_generic` reserves budget for mid-output error/warning lines before truncating (`economy/reducer.py:397-474`); token-saver's `_truncate_middle` (`generic.py:214-231`) does not scan the middle for signal at all | `economy/reducer.py` (borrow ANSI-strip + repeat-collapse, skip the blind truncate) |
| Structured lock-file compression (npm/yarn/poetry/Cargo/go.sum → name@version only) | `src/processors/file_content.py:370-515` | up to 99% (npm install) | asserted | missing | new reducer variant |
| Structured JSON/YAML/TOML/XML depth-and-list-truncated compression | `src/processors/utils.py:20-64` (`compress_json_value`), `file_content.py:519-605` | — | mechanism | missing | new reducer variant — **see anti-port below** |
| Secret redaction: unambiguous substrings + letter-boundary-guarded ambiguous tokens (`KEY`/`AUTH`/`PASS`) avoid `MONKEY`/`AUTHOR` false-hits; allowlist override | `src/processors/env.py:60-80`, `file_content.py:303-336` | — | mechanism, tested | not found in reviewed modules (`capsule.py`, `context_compress.py`, `prompts/prefix.py`) — check `policy.py`/`security_diff.py` before porting to avoid duplicating | new: content redaction pass |
| Source-code/sensitive-config passthrough allowlist (never touch `.py`/`.env`/`.ini` etc.) | `src/processors/file_content.py:24-88, 178-184` | — | design principle | **ForgeOS ahead** — same principle, but general (`capsule.py`'s whole-line-only `fit()`, citing arXiv 2607.12161) vs. token-saver's fixed extension list | n/a |
| Fail-open on every failure path (bad hook JSON, unknown tool, subprocess crash, corrupt SQLite) | `scripts/hook_pretool.py:410-416`, `core.py:66-85`, `tracker.py:74-92` | — | design principle | aligned (`preflight.py`'s "no still costs nothing"); apply same discipline to hooks (#16) | hooks (#16) |
| Cascading config: defaults → global JSON → project `.token-saver.json` (walks up like `.gitignore`) → env vars, with type-coercion guards | `src/config.py:69-211` | — | mechanism | likely duplicate of existing ForgeOS config — low priority, verify before porting | n/a |

## Top 5 ports (ranked)

1. **PreToolUse-style transparent Bash interception** (`hook_pretool.py` + `wrap.py`). This is the actual delivery mechanism token-saver has and ForgeOS's `reducer.py` doesn't: a hook that rewrites the tool call itself so *every* Bash-tool output gets reduced before the model ever sees it, not just pytest runs a caller explicitly pipes through `reduce_pytest`. Directly on task #16 (deterministic hook lifecycle). Port the classifier (`is_compressible`), not the marker-injection rewrite — see anti-port below.
2. **Processor-mismatch telemetry** (`engine.py:119-134`). Cheap, valuable, and missing: log every time a reducer path ran but didn't hit its compression floor, so weak spots in `economy/reducer.py` surface from real usage instead of manual audit.
3. **Generic-fallback hardening for `reduce_generic`**: ANSI-code stripping and consecutive-repeated-line collapse (`generic.py:58-136`) are safe, structure-blind wins `reduce_generic` doesn't yet do, and they don't touch the middle-budget-reservation logic ForgeOS already does better.
4. **Lock-file-shaped structured processors** (`file_content.py:370-515`) as a new `reducer` variant for dependency-manifest tool output (`npm install`, `poetry lock`, `go mod`) — same "extract name+version, drop everything else" shape as `reduce_pytest`'s "extract FAILED ids, drop repeated tracebacks," genuinely new surface.
5. **Secret-redaction regex pair** (`env.py:60-80`) as a standalone content-redaction pass, if ForgeOS confirms it has no equivalent — the letter-boundary lookaround trick (avoiding `\bKEY\b` false negatives on `API_KEY`) is a real, tested fix worth copying verbatim rather than re-deriving.

## Anti-ports — correctness traded for cheapness

- **`compress_json_value` depth/list truncation** (`utils.py:20-64`): lists over 5 items keep only the first 3 + `"... (N more items)"`; dicts recurse only 4 levels deep. For a harness whose gate depends on evidence integrity, this can silently drop the one failing item at list index 4 of a JSON test report or `aws`/`kubectl` describe output, with no signal that anything specific was cut — unlike `reducer.py`'s pytest path, which guarantees every FAILED node id survives. Do not port this shape without adding reducer.py's "never drop, only summarize duplicates" discipline.
- **DB/tabular row truncation** (`db_query.py:73-202`): psql/mysql/CSV/generic-table paths all keep head N + tail N rows and drop the middle with a bare `"(N rows omitted)"` — no attempt to keep anomalous or error rows, unlike `reduce_generic`'s budgeted middle-scan for error/warning lines. Same failure class as above, worse for verification use since a wrong row anywhere in a result set is exactly what a DB-backed acceptance check would need to see.
- **Command rewriting via marker injection** (`wrap.py:58-75`, `inject_markers`): for chained commands, wrap.py doesn't just compress *output* — it rewrites the *command actually executed*, wrapping each segment in `{ echo 'MARKER'; segment; }`. That's a stronger, riskier intervention point than post-hoc reduction: it can interact with exit-code propagation, output-consuming pipelines, or scripts sensitive to exact stdout shape. ForgeOS's `reducer.py` only ever touches captured output after the fact, never the command being run — keep it that way; port the *classifier*, not the rewrite.
- **Unqualified savings percentages** (`stats.py`, README badge): fine for a local dev tool, wrong pattern for ForgeOS — any ported "X% saved" number needs a `Figure`/`Provenance` wrapper before it's allowed near a receipt.
