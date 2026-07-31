# The no-interruption team architecture

N agents can work one project at once with zero interruptions — not because
they're polite about it, but because each interference class is removed by
construction before it can happen. Below: what each layer prevents, how, and
the test that proves it. Present tense means merged and load-bearing;
"landing" means the task is in flight right now.

## 1. Parallel-safety pre-flight

Prevents two decomposed tasks being marked safe to run together when they'd
actually corrupt each other: not just overlapping file scope, but *silent
divergence* — one task's text introduces a symbol/contract the other task's
text already assumes exists, a failure file-level isolation alone can't see.
`forgeos/core/manager.py::parallel_safety` runs two deterministic gates
(scope overlap, shared not-yet-materialized symbols), conservative by design
— a false positive costs wall-clock, a false negative corrupts a file.
`forgeos/compiler.py::_sequence_unsafe_pairs` turns every unsafe pair into a
`depends_on` edge at decompose time, or — if that would cycle — a recorded
`Mission.dependency_conflicts` entry for a human, never auto-resolved. Proof:
`tests/test_manager.py`, `tests/test_compiler.py`.

## 2. Dependency waves

Prevents a task starting before its dependency finished, and independent
tasks convoying behind each other inside one batch. `forge.py`'s `Forge.run`
wave loop submits the whole ready set (`Scheduler.ready_tasks`) to a thread
pool held at `max_parallel` each iteration; readiness, the halt flag,
heartbeat expiry, and the governor's ruling are re-checked only at wave
*boundaries*, never mid-wave, so a live thread's leases can't be reclaimed
from under it. Admission is layered underneath: `ResourceGovernor.may_start`
(live pressure) and a budget-headroom check both refuse *before* a task
starts, never after. Proof: `tests/test_forge_parallel.py`,
`tests/test_scheduler.py`, `tests/test_resources.py`.

## 3. Path leases

Prevents two tasks writing overlapping paths at the same instant, in the
same process, under real thread concurrency. `forgeos/leases.py::LeaseStore`
does glob-aware overlap (`patterns_overlap`); WRITE conflicts with any
overlapping WRITE or READ, READ+READ is the only pair that never conflicts.
Leases are taken atomically with capacity in `Scheduler.assign` (refuses
rather than runs on collision), TTL'd so a dead worker's lease expires
instead of blocking a path forever, and released at report time. Proof:
`tests/test_leases.py`, and live under real threads in
`tests/test_forge_parallel.py::test_overlapping_paths_never_run_concurrently`.

## 4. Awareness board

Prevents a worker refused a lease learning only "no" — no idea who holds the
path, what they're doing, or whether a narrower request would succeed —
which is exactly what drives a wasted retry or escalation.
`forgeos/core/awareness.py::TeamBoard` is read-only queries over the same
lease store and event log everything else already writes: `would_collide` /
`idle_paths` for programmatic checks, `board()` for an operator digest,
`context_for(task_id)` for a ~400-token-capped digest of what other live
tasks hold, folded into a worker's prompt in `forge.py`'s attempt loop —
always in the volatile tail, never the byte-stable cached prefix
(`forgeos/prompts/prefix.py`) — sorted deterministically, never by
timestamp, so the same board state always serializes to the same bytes.
Proof: `tests/test_awareness.py`;
`tests/test_forge.py::test_worker_prompt_carries_teammate_board_context`
proves the digest reaches the actual prompt, not just the unit.

## 5. Per-task worktrees — merged, opt-in

Prevents a worker's uncommitted edits ever touching another worker's files
on disk, even if that worker ignored its own lease — leases stop the
*decision* to overlap, worktrees stop the *filesystem effect* if a decision
turns out wrong anyway. `forgeos/worktrees.py` creates
`<repo>/.forgeos-worktrees/<task_id>` on branch `forgeos/task/<task_id>`,
both deterministic from `task_id` alone; `merge_check` runs a
non-destructive `git merge-tree --write-tree` before `merge_accepted`
attempts a real `--no-ff` merge, aborting cleanly rather than half-applying.
Both the module and its `Forge.run` wiring are merged: a per-task `cwd`
override, gate-accept triggering the merge back to the main branch, and an
honest refusal (`"merge conflict against main"`) plus worktree teardown with
the branch kept for forensics on conflict. It stays behind the opt-in
`isolate_worktrees` flag, so default behavior is byte-identical to a build
without it. Every task in a job bases its worktree off one SHA pinned at job
start, not a per-task `HEAD` read — otherwise thread scheduling, not the
edits themselves, would decide whether two changes "conflict". Proof:
`tests/test_worktrees.py` (`@pytest.mark.slow` — real git subprocesses, real
merges, real conflicts) and the `isolate_worktrees` tests in
`tests/test_forge.py`, including one proving a retry reuses its worktree.

## 6. Batch merge (Bors-style bisection)

Prevents throwing away the parallelism per-task worktrees buy by serializing
merge checks one candidate at a time when several finished
near-simultaneously — exactly the throughput worktrees exist to unlock.
`forgeos/core/batchmerge.py::verify_batch` checks a batch together first and
only bisects (split in half, recurse) once the combined check actually fails
— `O(k log n)` checks worst case for `k` real failures among `n` candidates.
A check that couldn't run at all is marked UNKNOWN and re-queued whole,
never bisected on a non-result and never blamed. Pure logic today: no git,
no subprocess, no filesystem — the real check is injected by the caller.
Status, honestly: standalone. Nothing calls `verify_batch` yet, and that
wiring (needs mechanism 5 first) has no task assigned — unlike mechanism 5,
it is not currently landing. Proof: `tests/test_batchmerge.py`.

## 7. Merge gate + independent, different-family review

Prevents trusting a worker's own claim of success, and a model reviewing its
own patch. Passing tests alone are not proof — SecureAgentBench shows agents
producing patches that are functionally correct and still insecure — and LLM
evaluators favor their own generations (Panickssery et al., NeurIPS 2024,
arXiv 2404.13076), worse exactly when the two candidates are close in
quality (arXiv 2406.07791). `forgeos/core/verify.py`'s `MergeGate.evaluate`
requires green tests backed by real command output, a clean scan of the
actual diff (a missing scanner is UNAVAILABLE, never clean), and an
independent review verdict: `reviewer_worker == implementer_worker` (or one
id derived from the other) is a hard block, same provider *family* is a WARN
only. `Forge._pick_reviewer` only ever names a real, different worker,
returning `""` — never a manufactured id — when the fleet has one capable
worker, which the gate then honestly refuses for. Status, honestly:
`Forge.run` and `python -m forgeos team` now build a routed reviewer by
default. It shares the Forge's ledger-owned gateway, so a gateway review is
attributed and budgeted in the same receipt rather than being an unmetered
second path. A fleet with only one capable worker is still refused by the
merge gate. Proof: `tests/test_cli.py`; `tests/test_verify.py`;
`tests/test_forge.py` (`test_missing_independent_review_blocks_the_merge`,
`test_a_rejecting_reviewer_blocks_the_merge`).

## 8. Test-tampering gate

Prevents a worker gaming a visible pass/fail loop by editing the test
instead of the code — deleting assertions, loosening bounds, rewriting the
check to match the implementation. Documented by SpecBench (arXiv
2605.21384) and the RLVR reward-hacking survey (arXiv 2604.15149); neither
the verification ladder nor the security scan above catches this, since both
trust whatever test the diff shipped with. `detect_test_tampering` flags any
diff touching a test and a non-test `.py` file together — a named pairing
(`tests/test_foo.py` + `foo.py`) is the strong signal, an unpaired
coincidence is reported at WARN — and the gate blocks on it unless a human
sets `tamper_reviewed=True`. Proof: `tests/test_verify.py`.

## 9. Per-worker ledger + budget contract

Prevents one agent spending the team's money invisibly or twice, and the
router learning from a worker's self-reported success instead of what the
gate actually ratified. `forgeos/ledger.py::record_spend` is integer
microdollars only, so a budget comparison can't drift; every call is
recorded before its result is used, and an unmetered adapter is charged its
tier prior rather than reading as free. `job_spend_micros`/`task_spend_micros`
feed the admission refusal in mechanism 2 — a tripped budget escalates to a
human, never widened silently. `worker_stats` deduplicates by task in SQL,
and `record_report` scores win rate off the gate's *ruling*, never the
worker's claim. Proof: `tests/test_billed_cost.py`, `tests/test_scheduler.py`;
surfaced read-only via `python -m forgeos receipts` (`tests/test_cli.py`).

## Running it

```
python -m forgeos team "<objective>" --dry-run
```

Compiles the objective into a task graph and prints it — including any
`dependency_conflicts` mechanism 1 couldn't resolve — without spending
anything. Drop `--dry-run` and add `--budget-usd <n>` (required; forgeos
never invents a spending cap) to run it end to end and print a receipt.

```
python -m forgeos watch --queue <dir> [--once] [--poll-interval N]
```

The unattended form: polls a directory for job-spec JSON files, runs each
through one long-lived `Forge`, files the spec plus a receipt into `done/` or
`failed/`. A malformed spec is refused to `failed/` with a reason attached
rather than crashing the loop; a missing `budget_usd` is refused rather than
silently defaulted. See `examples/README.md` for a walkthrough.
