# Examples

A 5-minute tour: check what can run, preview a plan for free, run a queued job, read the receipt. Every command block below was actually run; nothing here is typed up by hand.

## 1. What can run right now

```
$ python -m forgeos doctor
```

Read-only, no network (see the module docstring in `forgeos/__main__.py`). Real output, trimmed to a few representative lines -- the full provider and worker lists are long and specific to whatever's installed on your machine; the omc entrypoint path is this machine's own and left out:

```
forgeos doctor
----------------------------------------------------------
machine        standard  6 cores, 15.9 GiB
gpu            GeForce GT 710 (1.0 GiB VRAM)
pressure       4.9 GiB free, cpu 0% -> proceed

providers        (14 checked -- trimmed here)
  claude         cli       ready
  ollama         local     ready
  gateway        gateway   needs auth (set $FORGEOS_GATEWAY_URL)

Registry: 10 worker(s)
  forgeos.executor         ok: 'node' on PATH and omc entrypoint present
  ollama.local             ok: ollama daemon reachable and model 'llava-phi3:3.8b' present
  gateway.free             unavailable: gateway.free needs a Gateway; pass gateway=...
Runnable now: 9/10

Price catalog: 6371 model(s), 5165 stale or unstamped (>30d)
  oldest stamped price: 51.4 day(s) old
```

## 2. Preview a plan for free

```
$ python -m forgeos team "add a retry helper with exponential backoff" --dry-run
```

Compiles the objective into a task graph and stops -- no budget, no `Forge` constructed, nothing spent. Exactly what running it produced:

```
Mission: add a retry helper with exponential backoff
Tasks: 1
  [task_d8665eb] add a retry helper with exponential backoff
    scope: .
```

## 3. `watch_job.json` -- a real queue spec

`watch_job.json` in this directory is the shape `watch_queue` (`forgeos/watch.py`) expects: an `objective`, a `cwd`, an explicit `budget_usd` (required -- forgeos never invents a spending cap), and one TaskSpec-shaped task (`subject`, `description`, `acceptance`, `capabilities`, `scope`).

Round-tripped through the exact validation `watch.py` runs on every queue file -- the real pydantic contracts, no test file, just a throwaway `python -c`:

```
$ python -c "
import json
from forgeos.watch import _parse_job_spec
raw = json.loads(open('examples/watch_job.json').read())
objective, cwd, budget, tasks = _parse_job_spec(raw)
print('OK:', objective, tasks[0].capabilities)
"
OK: add a retry helper with exponential backoff ['edit', 'implement', 'python']
```

## 4. Run it through the watch queue

```bash
mkdir -p myqueue/incoming
cp examples/watch_job.json myqueue/incoming/
python -m forgeos watch --queue myqueue --once
```

This is a real job: `watch_queue` builds a real `Forge`, which routes the task to whichever worker your machine has configured -- a real subprocess or a real API call, real money.

So this doc doesn't spend money on every machine that builds it, the block below ran that same command against a copy of `watch_job.json` with its `capabilities` changed to a tag no worker declares. The router genuinely has no candidate to route to, so it refuses *before* touching any worker -- at $0.0000 -- while the ledger, scheduling, and done/failed routing all still run for real:

```
$ python -m forgeos watch --queue demo_queue --state-dir demo_state --once
watch: 0 done, 1 failed
```

The file moved from `demo_queue/incoming/` to `demo_queue/failed/`, with a `.receipt.json` sidecar recording `"reason": "no worker has the required capabilities"` and `"usd_micros": 0`. Run the shipped `watch_job.json` as-is (real capabilities, matching a configured provider) and it will actually execute.

## 5. Read the receipt

```
$ python -m forgeos receipts --state-dir demo_state
```

Read-only summary of the ledger the run above left behind:

```
Jobs: 1
------------------------------------------------------------------------
  [job_d20376200f85] add a retry helper with exponential backoff
    state=done (closed)  tasks=1 done=0 failed=1  spend=$0.0000  $/accepted=n/a

Total: $0.0000 across 1 job(s), 0 accepted task(s), n/a/accepted
```

Omit `--state-dir` to read forgeos's default home (`~/.forgeos`), the same place a bare `Forge()` writes to.
