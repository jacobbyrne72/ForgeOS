# Contributing to ForgeOS

ForgeOS is a cost-governed AI coding harness — domain-agnostic infrastructure,
not a project of its own. Read [AGENTS.md](AGENTS.md) first; it is the actual
contribution contract and it overrides anything below that conflicts with it.

## Dev setup

```bash
git clone <this-repo> ForgeOS && cd ForgeOS
pip install -e ".[dev]" ruff   # ruff is not in the dev extra; CI installs it separately
```

Python 3.11+. `semgrep`, `gitleaks`, and `ruff` are optional and auto-detected
at runtime for the security/lint gates the harness runs on *other* repos — none
of them are required to develop ForgeOS itself, only to run its full CI lint step.

## Lint gate

```bash
ruff check forgeos/ tests/ tools/
```

`pyproject.toml`'s `[tool.ruff]` block is the single source of what "clean"
means (line length 110, target `py311`). CI runs this before any test, on
purpose: a style or import error fails in two seconds, not after the slow
integration tests have spent their minutes.

## The two-speed test suite

```bash
# Per edit — fast path, no real subprocess scanners:
python -m pytest tests -q -m "not slow"

# At the merge boundary — everything, including real semgrep/gitleaks/ruff/git:
python -m pytest tests -q
```

`slow` marks tests that spawn real external processes. They are the ones that
genuinely prove the integrations work, and also the ones that make the suite
too slow to run on every edit — this is the same verification-ladder idea
`forgeos/economy/testselect.py` implements for the harness's own callers,
applied to ForgeOS's own suite. CI runs both stages as separate steps so a
fast-path failure is reported before the slow one even starts.

**RTK (or any pytest-wrapping tool) caveat.** A wrapper that filters or
reformats pytest's own output can misreport a run — on this machine, RTK's
matcher has reported "No tests collected" on an otherwise-passing run, and can
swallow warnings. Before trusting a wrapped result, check it against pytest's
own summary line (`N passed` / `N failed` / real collection errors), or run
the wrapper's raw-passthrough mode (e.g. `rtk proxy python -m pytest ...`) so
nothing between you and pytest can quietly turn a red run green.

## House style

- **WHY-docstrings, not narration.** A comment states a constraint the code
  itself can't show — why it has to be this way, what breaks if you "simplify"
  it — never what the next line obviously does. `# increment i` is noise;
  `# oldest-first: matches the order Anthropic's clear_tool_uses clears in`
  is not.
- **Match existing prose density.** Some modules carry a paragraph of WHY
  above a function; some carry one line. Match the file you're editing, don't
  import a house style from somewhere else in the tree.
- **Surgical changes.** Touch only what the task needs. Don't "improve"
  adjacent code, rename things you didn't have to, or refactor something that
  isn't broken as a drive-by.

## Hard rules a PR must not break

The full list is [AGENTS.md](AGENTS.md#hard-rules); these are the ones the
README also calls out as load-bearing:

- **Never widen a budget to make a job finish.** `max_usd`/`max_seconds`/
  `max_iterations` are a contract with the human. A tripped governor escalates
  to a person; it is never a number a PR edits to get a job past a trip.
- **Never bypass the ledger.** Every worker call records spend before its
  result is used. An unrecorded call is invisible, and the governor cannot
  stop what it cannot see.
- **Unavailable is not a pass.** A missing scanner or gate blocks a merge.
  "We could not check" must never read as "it is fine."
- **Task text is untrusted input to the router.** Routing reads deterministic
  features (capabilities, measured history, risk class) only — never
  instructions embedded in a task description or file content.
- **Only MODEL failures escalate the tier.** A broken venv, a missing binary,
  a config error fails identically at every price point; escalating those
  buys premium tokens for a problem no bigger model can fix.
- **Claims need receipts.** "N% cheaper," "M% faster" — anything asserted
  about savings or performance needs a measured, replayed, or explicitly
  `modelled` number behind it (`SavingsProof`), never a bare assertion.

## PR expectations

- **Tests for every change.** New behavior gets a new test; a bug fix gets a
  test that reproduces the bug before the fix, and passes after.
- **Paste real command output.** "Should work" is not evidence. Run the
  targeted test file, then the fast suite, and paste what actually printed —
  including the failures, if any remain.
- **No false greens.** If part of the task's scope was skipped, say which
  part and why in the PR description, rather than reporting a full pass that
  didn't happen.
- **State scope, don't guess it.** If a change needs to touch a file outside
  what the issue/task described, say so explicitly rather than quietly
  widening the diff.
