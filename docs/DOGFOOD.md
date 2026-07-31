# Dogfood log — the harness debugging itself, live

> **Provenance: this is a work log, not a receipt.** The dollar figures below
> were recorded by hand while the runs happened. No committed ledger export
> backs them, and the local ledgers on the machine that produced them no longer
> hold matching rows. Read them as a narrative of what was done, not as
> measurements — the numbers this project stands behind are the ones
> `forge receipts` and `forge forgebench --json-out` print from the ledger, with
> their provenance attached. Flagged here rather than quietly deleted, because
> a project whose rule is "measured, never asserted" does not get to keep
> unbacked figures just because they are its own.

One real job, run repeatedly through the full kernel with real headless
Claude seats, on 2026-07-31: *"add a tested retry helper to a scratch
repo."* Every run below spent real money, billed through the real ledger,
and was refused by the real merge gate until the whole evidence chain held.
Each refusal exposed one defect; each defect became a pinned, pushed fix
the same hour. This is the receipts culture applied to ForgeOS itself.

| Run | Outcome | What it exposed | Fix landed |
|---|---|---|---|
| 1 | 3 empty attempts, $0.18 tier-prior | omc team runtime dies without tmux on native Windows; `health()` said "ok" for a backend that cannot execute | `cli_team.health()` detects the missing tmux runtime |
| 2 | Worker wrote a **correct** `retry()` + tests; gate refused: "nothing was actually verified" | `LocalCommandAdapter` emitted CLI stdout as MESSAGE chatter, so a genuine pytest summary never reached the evidence path | CLI stdout is TOOL_CALL/TOOL_UPDATE evidence, with a real-subprocess pin test |
| 3 | Refused: no tests + 40 security findings | `run_security([])` scanned the whole cwd and attributed state-dir noise to a task that touched nothing | An empty diff has no security surface → SKIPPED, reason stated |
| 4 | Refused: no tests (security now clean) | `_looks_like_pytest` demanded pytest's `=====` ruler; a bare "3 passed in 0.52s" never parsed | Bare-summary grammar (count clauses + timing), prose still never matches |
| 5–6 | Refused: no tests | The worker **couldn't run pytest at all**: headless permission gate blocked Bash, and the worker refused to invent a summary — "No summary line exists to quote" | Scoped project allowlist for exactly the acceptance command |
| 7 | Refused: no tests | The operator's own token-saver hook rewrote pytest *inside the worker* and replaced the summary with "Pytest: 5 passed" — no timing, unparseable; the worker proved counts via `--junitxml` and reported the interference honestly | Wrapped-summary grammar (`pytest:`-prefixed counts) |
| 10 | **ACCEPTED — merged** | The full loop held: routed seat, machine-verified counts through the wrapper, independent review, $0.0030 billed | One warning left: reviewer and implementer shared a provider family |
| 16 | **ACCEPTED — merged, zero warnings** | Cross-family review (deepseek reviewing anthropic's work, picked automatically by the router's new family preference) + acceptance-derived per-task permissions | The clean receipt: `merge: [] warn: []` |

Between runs, crash-killed processes left orphaned path leases that
correctly deferred later jobs until their TTL reaped them — the safety
mechanism working, at the cost of wall-clock. Owner-liveness on leases
(reap instantly when the holder is provably dead) is the next measured
improvement this log demands.

Also found along the way, by the same method: the worktree merge commit
relied on the operator's git identity (fails on every identity-less
machine — the harness now merges as itself), and a per-task `HEAD` read
let thread timing decide whether two edits "conflict" (pinned to one SHA
per job).

## What this proves

- **No false green survived.** A correct implementation was refused four
  times because the *proof* wasn't machine-verified. The gate held against
  its own maker's impatience, which is the only kind of holding that counts.
- **The workers inherit the culture.** A blocked worker wrote "I did not
  run pytest, and I won't invent one" — then fetched junitxml evidence of
  the truth it couldn't print.
- **Every refusal was billed.** Unmetered attempts were charged their tier
  prior; nothing read as free. Total spend for the whole diagnostic chain:
  under $0.50.
- **Every fix is a test now.** Each row's defect has a regression pin in
  the suite; none of these can return silently.

## What it cost to learn

Seven runs, ~$0.50, one afternoon — against a class of bug (evidence-chain
gaps between a worker's tooling and a verifier's grammar) that silently
inflates every agent product that doesn't bill and gate itself. The
harness found them because refusing unproven work is its default, and
every refusal names its reason.
