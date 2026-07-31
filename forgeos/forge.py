"""The Forge — the assembled machine.

Every other module in forgeos is a part. This is the only file that wires them into
one loop, and without it the parts are just a library. The order below IS the cost
thesis, executed:

    0. compile an immutable mission contract        (no model call)
    1. size the pools to this actual machine        (no model call)
    2. price every provider's quota as inventory    (no model call)
    3. per task:
         a. build the smallest sufficient capsule   (no model call)
         b. preflight — refuse before spending      (no model call)
         c. route: cheapest tier that can finish, priced against the market
         d. take path leases, then execute
         e. reduce the output before a model sees it
         f. climb the verification ladder
         g. merge gate: tests AND security AND evidence AND independent review
    4. emit a receipt separating measured from estimated

Steps 0-3b and 3e-3g involve no model at all. That is the point: the model is
called for the ambiguous middle, and everything around it is ordinary code.

Adapters are injected rather than imported so this file stays testable without a
network, a subscription, or a model.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from collections.abc import Callable, MutableMapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

from .contracts import (
    AttemptSummary,
    Budget,
    FailureClass,
    JobSpec,
    Scope,
    TaskSpec,
    TaskState,
    TestResults,
    Verdict,
    WorkerReport,
    from_micros,
)
from .core.awareness import TeamBoard
from .core.governor import Action, Governor
from .core.market import CapacityMarket
from .core.resources import ResourceGovernor, WorkerKind, sample_pressure
from .core.router import Router, Tier
from .core.quota import QuotaTracker
from .core.scheduler import Scheduler
from .core.timing import Phase, SpanStore
from .core.verify import GateResult, MergeGate, run_security
from .diagnostics import record_degradation
from .economy.avoidance import AvoidanceLog, AvoidanceMethod
from .economy.lowerer import Operation, classify, savings_estimate
from .economy.preflight import (
    Decision as PreflightDecision,
    check_repeat_work,
    count_tokens,
)
from .economy.reducer import reduce_generic, reduce_pytest

from .events import EventLog, EventType
from .hooks import HookResult, HookRunner, job_end_payload, task_payload
from .leases import LeaseStore
from .ledger import Ledger
from .registry import Adapter, CostTier, Registry, WorkerProfile, default_registry
from .settings import HookEvent, Settings
from .worktrees import (
    BRANCH_PREFIX as _WORKTREE_BRANCH_PREFIX,
    create_worktree,
    list_task_worktrees,
    merge_accepted,
    merge_check,
    remove_worktree,
)

DEFAULT_HOME = Path.home() / ".forgeos"

# Vendor/model substrings this fleet has actually used, mapped to the provider
# family the self-preference literature cares about (Panickssery et al.,
# NeurIPS 2024, arXiv 2404.13076: LLM evaluators recognize and favor their own
# generations) — same underlying model vendor, not merely a different worker
# id. Checked in order, so a more specific hint never loses to a coincidental
# substring match.
_FAMILY_HINTS: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("anthropic", "anthropic"),
    ("codex", "openai"),
    ("gpt", "openai"),
    ("openai", "openai"),
    ("gemini", "google"),
    ("deepseek", "deepseek"),
    ("qwen", "alibaba"),
    ("llama", "meta"),
    ("mistral", "mistralai"),
)


def _family_of(profile: WorkerProfile | None) -> str:
    """Normalize a worker profile to a provider family for the merge gate's
    same-family reviewer WARN.

    `vendor` ("which backing CLI actually runs it") is the strongest signal
    and is checked first. `model` is the fallback: a gateway/free-tier profile
    routes by model string rather than a fixed CLI vendor, so vendor alone
    would miss it. An adapter that never leaves this machine (Ollama, a local
    command) falls back to "local" — a local model reviewing its own local
    output is the same self-preference risk in miniature, not a different
    question. Anything left over is "" rather than a guess: an unrecognized
    profile must never register as matching another unrecognized one, or two
    unrelated unknown workers would silently trip the WARN on no evidence.

    No `family` field was added to `WorkerProfile` for this — every case above
    is already derivable from fields the profile already has.
    """
    if profile is None:
        return ""
    for haystack in (profile.vendor.lower(), profile.model.lower()):
        for needle, family in _FAMILY_HINTS:
            if needle in haystack:
                return family
    if profile.adapter in (Adapter.OLLAMA, Adapter.LOCAL_COMMAND):
        return "local"
    return ""


def _looks_like_pytest(output: str) -> bool:
    """Whether the pytest reducer can meaningfully parse this.

    The bare summary line — "3 passed in 0.02s" — must match on its own: a
    headless CLI worker's final message often carries exactly that line and
    nothing else pytest-shaped, and requiring "pytest"/`::`/`=====` alongside
    it sent a genuine summary through the generic reducer, which reports no
    test counts, which the merge gate correctly read as "nothing verified".
    A correct implementation was refused over the detector, not the evidence.
    Observed live on the headless-Claude dogfood run.
    """
    low = output.lower()
    if re.search(r"\b\d+ (passed|failed|error(?:s)?)\b", low):
        return True
    return any(m in low for m in ("passed", "failed", "error")) and (
        "pytest" in low or "::" in output or "=====" in output
    )


def _detect_git_repo_root(cwd: str) -> str | None:
    """Once-per-job check: is `cwd` inside a git repo `isolate_worktrees` can use?

    A quiet `None` (not a repo, git missing, timeout) means isolation is simply
    unavailable for this job -- every task then runs exactly as it did before
    this feature existed, never a crash.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _resolve_head(repo_root: str) -> str | None:
    """The commit `isolate_worktrees` bases every task's worktree on, resolved
    once at job start rather than read fresh (as bare `"HEAD"`) per task.

    Tasks in a wave create their worktrees at different real times as threads
    get scheduled — reading `"HEAD"` per task would mean a task whose attempt
    starts after a sibling's merge already landed silently rebases onto that
    merge instead of genuinely conflicting with it, so whether two edits
    "conflict" would depend on thread timing rather than on the edits
    themselves. Pinning one SHA up front gives every task in the job the same
    starting point; only merge time decides who lands and who doesn't.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


# Set by `Forge._run_task` immediately around the executor call, read by
# `current_task_cwd` below. A thread-local rather than a new `Executor`
# parameter or a shared attribute: tasks in one wave run concurrently on
# separate pool threads (see `Forge.run`), so a shared slot would race between
# them, and adding a parameter would break every `Executor` callable already
# in the wild -- production adapters and every existing test's fake alike.
_task_cwd_local = threading.local()


def current_task_cwd(default: str = ".") -> str:
    """The isolated worktree path for the task attempt running on this thread.

    Returns `default` (ordinarily the job's own cwd) when isolation is off, or
    when no task is currently attempting on this thread. Any executor -- the
    default routed one (rebuilt per isolated attempt, see `_run_task`) or a
    caller-supplied one that chooses to consult this -- can find out where to
    actually write without the `Executor` Callable type ever changing shape.
    """
    return getattr(_task_cwd_local, "value", None) or default


class ExecutionResult(BaseModel):
    """What a worker adapter must return. Deliberately small."""

    state: TaskState
    evidence: str = ""
    commands_run: list[str] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)
    tests: TestResults | None = None
    raw_output: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached_in: int = 0
    usd_micros: int = 0
    seconds: float = 0.0
    blocker: str = ""
    failure: FailureClass | None = None


# A worker executes one task and reports. Injected, so the Forge never imports a
# provider client and can be exercised end to end with a fake.
Executor = Callable[[TaskSpec, str], ExecutionResult]


class TaskOutcome(BaseModel):
    task_id: str
    subject: str
    accepted: bool
    worker_id: str = ""
    tier: int | None = None
    reason: str = ""
    merge_reasons: list[str] = Field(default_factory=list)
    merge_warnings: list[str] = Field(default_factory=list)
    usd_micros: int = 0
    attempts: int = 0


class ForgeResult(BaseModel):
    job_id: str
    objective: str
    accepted: int
    rejected: int
    outcomes: list[TaskOutcome] = Field(default_factory=list)
    spend_usd: float = 0.0
    cache_hit_pct: float = 0.0
    avoided_tokens: int = 0
    bottleneck: str | None = None
    halted_reason: str = ""

    @property
    def all_accepted(self) -> bool:
        return self.rejected == 0 and self.accepted > 0

    @property
    def cost_per_accepted(self) -> float | None:
        return round(self.spend_usd / self.accepted, 4) if self.accepted else None


# Hard cap, in tokens, on the WHOLE attempt-history block carried into a retry
# prompt. Carrying the full prior transcript would fix the "worker repeats the
# same mistake" problem too, but at unbounded cost; this caps it. Enforced by
# dropping whole older `AttemptSummary` entries (newest-first), never by
# editing the text kept in one -- see `_cap_attempt_history`.
ATTEMPT_HISTORY_MAX_TOKENS = 300


def _attempt_summary(attempt_no: int, result: ExecutionResult, evidence: str,
                      kept_lines: list[str]) -> AttemptSummary:
    """One failed attempt, compacted for the next retry -- every piece verbatim.

    `evidence` is already the reducer-augmented text `_run_task` builds before
    this is called; `kept_lines` are the reducer's own verbatim FAILED
    lines/test ids (see `economy.reducer.reduce_pytest`/`reduce_generic`).
    Nothing here re-parses raw output or rewrites a single word of it -- that
    would be a second reducer, and it would risk exactly what arXiv:2607.12161
    measured: rewording or trimming an error string/test node id makes the next
    attempt unable to match its fix against the failure, which raised billed
    cost AND lowered success rate in that study. Reuse, never re-derive.
    """
    parts: list[str] = []
    if result.blocker:
        parts.append(result.blocker.strip())
    stripped_evidence = evidence.strip() if evidence else ""
    if stripped_evidence and stripped_evidence not in parts:
        parts.append(stripped_evidence)
    for line in kept_lines:  # exact FAILED test ids / error text, already deduplicated
        if line and line not in parts:
            parts.append(line)
    if result.tests is not None and not result.tests.green:
        test_str = f"tests: {result.tests}"
        if test_str not in parts:
            parts.append(test_str)
    reason = "\n".join(parts) if parts else "no evidence reported"
    return AttemptSummary(
        attempt=attempt_no,
        failure_class=result.failure.value if result.failure else "",
        reason=reason,
    )


# Merge-gate refusals a worker could plausibly fix on a second attempt: it did
# the work but skipped the proof, or its own output failed. Matched against
# `MergeGate.evaluate`'s reason strings.
_FIXABLE_REFUSALS = (
    "nothing was actually verified",   # it never ran the tests
    "no evidence recorded",            # it ran them and did not report
    "no command was run",              # evidence with nothing behind it
    "test(s) failing",                 # its own output is wrong
    "security failed",                 # a finding in code it just wrote
)

# Refusals no retry can clear, because they are properties of the machine or the
# fleet rather than of the attempt. Retrying these buys a guaranteed second
# refusal at full price. Checked FIRST -- "could not be checked" contains no
# fixable substring, but "no independent review" must beat any loose match.
_STRUCTURAL_REFUSALS = (
    "no security gate was run",        # no scanner installed here
    "could not be checked",            # gate UNAVAILABLE, same cause
    "no independent review",           # no second worker exists to review
    "reviewer identity missing",       # fleet composition, not the worker
    "reviewer must not be the implementer",
    "is derived from implementer",
)


def _refusal_is_fixable(reasons: list[str]) -> bool:
    """Whether a second attempt could plausibly clear this refusal.

    Conservative in the expensive direction: ANY structural reason makes the
    whole refusal unfixable, even alongside fixable ones. A task refused for
    both "no tests passed" and "no independent review" would still be refused
    after the worker adds tests, so paying for that attempt is certain waste.
    """
    if not reasons:
        return False
    blob = " ".join(reasons).lower()
    if any(s in blob for s in _STRUCTURAL_REFUSALS):
        return False
    return any(f in blob for f in _FIXABLE_REFUSALS)


def _cap_attempt_history(
    history: list[AttemptSummary], *, max_tokens: int = ATTEMPT_HISTORY_MAX_TOKENS
) -> list[AttemptSummary]:
    """Keep as many of the newest (front of list) entries as fit under `max_tokens`.

    Always keeps at least the single newest entry, even if it alone exceeds
    the cap: a truncated anchor is worse than a large but intact one (the same
    finding as `_attempt_summary`'s docstring). Compression beyond that point
    means dropping whole older entries wholesale, never shortening one.
    """
    kept: list[AttemptSummary] = []
    used = 0
    for item in history:  # newest-first already
        cost = count_tokens(f"attempt {item.attempt} [{item.failure_class}]: {item.reason}").tokens
        if kept and used + cost > max_tokens:
            break
        kept.append(item)
        used += cost
    return kept


def _task_from_ledger_row(row) -> TaskSpec:
    """Rehydrate one persisted task for crash recovery.

    The ledger stores the original contract as JSON so a resumed job does not
    ask a model to reconstruct scope, budgets, or dependency edges from prose.
    Older ledgers predate the ``depends_on`` column and receive the honest empty
    dependency list from SQLite's migration default.
    """
    keys = set(row.keys())
    return TaskSpec(
        id=row["id"],
        job_id=row["job_id"],
        parent_id=row["parent_id"],
        subject=row["subject"],
        description=row["description"],
        acceptance=json.loads(row["acceptance"]),
        scope=Scope.model_validate(json.loads(row["scope"])),
        capabilities=json.loads(row["capabilities"]),
        budget=Budget.model_validate(json.loads(row["budget"])),
        depends_on=json.loads(row["depends_on"]) if "depends_on" in keys else [],
        created_at=float(row["created_at"]),
    )


class Forge:
    """The assembled harness. One object, one loop, every part wired."""

    def __init__(
        self,
        *,
        home: Path | None = None,
        registry: Registry | None = None,
        settings: Settings | None = None,
        gateway: object | None = None,
        market: CapacityMarket | None = None,
        max_attempts: int = 3,
        retry_fixable_refusals: bool = False,
    ):
        self.home = home or DEFAULT_HOME
        self.home.mkdir(parents=True, exist_ok=True)

        self.ledger = Ledger(self.home / "ledger.db")
        self.events = EventLog(self.home / "events.db")
        self.leases = LeaseStore(self.home / "leases.db")
        self.avoidance = AvoidanceLog(self.home / "avoidance.db")
        self.spans = SpanStore(self.home / "spans.db")

        self.settings = settings or Settings.load()
        self.hooks = HookRunner.from_settings(self.settings)
        self.registry = registry or default_registry()
        self._gateway = gateway
        self._owns_gateway = gateway is None
        self.market = market or CapacityMarket()
        self.quota_path = self.home / "quota.json"
        self.quota_persist_error = ""
        try:
            self.quota = QuotaTracker.load(self.quota_path)
            self.quota_load_error = ""
        except ValueError as exc:
            # A corrupt telemetry snapshot must not prevent the operator from
            # opening Forge. Start conservatively empty and surface the error
            # for diagnostics rather than treating stale data as truth.
            self.quota = QuotaTracker()
            self.quota_load_error = str(exc)

        self.resources = ResourceGovernor()
        self.governor = Governor(self.ledger, self.events)
        self.router = Router(self.registry)

        limits = self.resources.live_limits()
        self.scheduler = Scheduler(
            self.ledger, self.events, self.leases, self.registry, self.governor,
            # Sized to THIS machine. A fixed number either starves a workstation or
            # thrashes a laptop, and thrashing is slower than running sequentially.
            max_parallel=max(1, limits.execution),
        )
        self.board = TeamBoard(self.leases, self.events)
        self.merge_gate = MergeGate()
        self.max_attempts = max_attempts
        # Off by default, and that default is the honest one. Retrying a
        # merge-gate refusal costs a whole extra attempt, and nothing here has
        # yet measured that a worker handed "you did not run the tests" behaves
        # any differently on the second pass. Turning it on is a bet that it
        # does; leaving it off is the position this project takes everywhere
        # else — do not spend without evidence. Flip it when there are receipts
        # showing the retry pays for itself, not before.
        self.retry_fixable_refusals = retry_fixable_refusals

        # Concurrency plumbing for `run`. The SQLite stores are already opened
        # with check_same_thread=False; what they cannot protect is the
        # scheduler's in-memory `_active` dict and multi-step read-then-act
        # sequences (capacity check → lease acquisition, budget read → refusal).
        # One lock serialises all of that bookkeeping; worker executions — the
        # minutes-long part — run outside it, so the lock costs microseconds and
        # buys away every torn read. `_trip` is how the first thread to see a
        # governor trip stops the others from spending through it.
        self._sched_lock = threading.Lock()
        self._quota_lock = threading.Lock()
        self._trip = threading.Event()
        self._trip_reason = ""

    def close(self) -> None:
        self._persist_quota()
        if self._owns_gateway and self._gateway is not None:
            try:
                self._gateway.close()
            except Exception as exc:
                record_degradation(
                    "forge_close", "gateway close failed", exc,
                    consequence="gateway connections/resources may not have been released cleanly",
                )
        for store in (self.ledger, self.events, self.leases, self.avoidance, self.spans):
            try:
                store.close()
            except Exception as exc:
                record_degradation(
                    "forge_close", f"{store.__class__.__name__} close failed", exc,
                    consequence="a store did not close cleanly -- on-disk state may not be fully flushed",
                )

    def _persist_quota(self) -> None:
        """Best-effort persistence for telemetry; never replaces ledger truth."""
        with self._quota_lock:
            self._persist_quota_locked()

    def _persist_quota_locked(self) -> None:
        try:
            self.quota.save(self.quota_path)
            self.quota_persist_error = ""
        except OSError as exc:
            self.quota_persist_error = str(exc)

    def ingest_quota(self, observation):
        """Apply normalized provider telemetry and persist it immediately.

        Adapters own how they receive headers or CLI text; Forge owns the
        durable tracker and the routing decision. This seam keeps credentials
        and network policy outside the core while making a provider observation
        usable on the very next task.
        """
        from .core.quota_ingest import QuotaIngestor

        with self._quota_lock:
            state = QuotaIngestor.apply(self.quota, observation)
            self._persist_quota_locked()
            return state

    def default_executor(self, *, cwd: str = ".",
                         checkpoint_store: MutableMapping[str, dict] | None = None) -> Executor:
        """Build the standard routed executor with this Forge's ledger.

        The gateway is lazy so constructing a Forge for a dry-run or a purely
        local test does not load a model catalog or create network transports.
        Once a real default run needs it, the gateway shares this Forge's
        ledger, settings, and avoidance log; gateway spend therefore lands in
        the same receipt as every other worker.

        `checkpoint_store`, when given, is threaded straight through to
        `routed_executor` — see `run()` for the per-job store it creates once
        and reuses across every attempt and every isolated-worktree rebuild of
        this executor.
        """
        if self._gateway is None:
            from .catalog import default_catalog
            from .gateway.client import Gateway
            from .gateway.dead_models import DeadModelStore

            self._gateway = Gateway(
                catalog=default_catalog(),
                ledger=self.ledger,
                settings=self.settings,
                dead_models=DeadModelStore(self.home / "dead_models.db"),
                avoidance_log=self.avoidance,
            )
        from .adapters.routed import routed_executor

        return routed_executor(
            self.registry,
            self.ledger,
            gateway=self._gateway,
            cwd=cwd,
            checkpoint_store=checkpoint_store,
        )

    # ------------------------------------------------------------------ run

    def run(
        self,
        objective: str,
        tasks: list[TaskSpec],
        executor: Executor | None = None,
        *,
        cwd: str = ".",
        budget: Budget | None = None,
        dependencies: dict[str, list[str]] | None = None,
        reviewer: Executor | None = None,
        operations: dict[str, Operation] | None = None,
        isolate_worktrees: bool = False,
        allow_repeat_work: bool = False,
        resume_job_id: str | None = None,
    ) -> ForgeResult:
        """Run a job to completion.

        `operations` optionally describes a task as a bulk operation. The Lowerer
        then records whether a deterministic strategy exists — advice about HOW to
        do the work cheaply, never a substitute for doing it.

        With `executor=None` the Forge runs whichever backend the router picks,
        via the registry's adapter field — including the lazy, ledger-owned
        Gateway for API/free-pool profiles. Pass an executor explicitly only
        when embedding a different transport or a deterministic test double.

        `isolate_worktrees` (default False, so every existing caller is
        byte-for-byte unaffected) gives each file-editing task its own git
        worktree (forgeos/worktrees.py) instead of every parallel task sharing
        `cwd` directly. Requires `cwd` to be inside a git repo — detected once
        here, cheaply — and applies per task only when the task declares
        `scope.paths` and its routed worker `can_edit_files`. A task the merge
        gate allows is only truly accepted once its worktree branch actually
        merges into the repo's current branch; a conflict turns an
        otherwise-accepted task into an honest refusal instead of a false green.
        """
        resume_mode = resume_job_id is not None
        executor_is_default = executor is None
        # Per-job, in-memory: see `adapters.executor.CheckpointStore`. Created
        # unconditionally (a bare dict costs nothing) but only ever populated
        # or read along the default routed path — a caller-supplied executor
        # never sees it.
        checkpoint_store: dict[str, dict] = {}
        if executor is None:
            executor = self.default_executor(cwd=cwd, checkpoint_store=checkpoint_store)
        self._operations = operations or {}
        self._trip.clear()
        self._trip_reason = ""
        repo_root = _detect_git_repo_root(cwd) if isolate_worktrees else None
        job_base_ref = _resolve_head(repo_root) if repo_root is not None else None
        if resume_mode:
            row = self.ledger.job(resume_job_id)
            if row is None:
                raise ValueError(f"job {resume_job_id!r} not found")
            job = JobSpec(
                id=row["id"],
                objective=row["objective"],
                cwd=row["cwd"],
                budget=Budget(
                    max_usd=from_micros(int(row["max_usd_micros"])),
                    max_seconds=int(row["max_seconds"]),
                    max_iterations=int(row["max_iterations"]),
                ),
                created_at=float(row["created_at"]),
            )
            resumed_ids = set(self.scheduler.resume(job.id))
            if not resumed_ids:
                raise ValueError(f"job {job.id!r} has no non-terminal work to resume")
            self.ledger.reopen_job(job.id)
            self.events.append(job.id, EventType.MISSION_RESUMED,
                               resumed_tasks=sorted(resumed_ids))
            tasks = [
                _task_from_ledger_row(task_row)
                for task_row in self.ledger.tasks_for_job(job.id)
                if task_row["id"] in resumed_ids
            ]
        else:
            job = JobSpec(objective=objective, cwd=cwd,
                          budget=budget or Budget(max_usd=20.0, max_seconds=7200,
                                                  max_iterations=60))
        for t in tasks:
            t.job_id = job.id

        # Receipt-aware preflight: refuse to re-spend on a contract this exact
        # ledger already settled, BEFORE routing/leases/execution ever run —
        # the wiring `economy.preflight.check_repeat_work`'s docstring
        # specifies. A refusal here is an outcome, not an error, and carries
        # the prior receipt so the caller sees what it already paid.
        outcomes: list[TaskOutcome] = []
        if resume_mode:
            resumed_ids = {t.id for t in tasks}
            for task_row in self.ledger.tasks_for_job(job.id):
                if task_row["id"] in resumed_ids:
                    continue
                accepted = any(
                    event.type is EventType.TASK_ACCEPTED
                    for event in self.events.for_task(task_row["id"])
                )
                outcomes.append(TaskOutcome(
                    task_id=task_row["id"], subject=task_row["subject"],
                    accepted=accepted,
                    reason="already terminal before resume",
                ))
        submitted: list[TaskSpec] = []
        for t in tasks:
            verdict = check_repeat_work(self.ledger, t, skip=allow_repeat_work or resume_mode)
            if verdict.decision is PreflightDecision.REFUSE_DUPLICATE:
                outcomes.append(TaskOutcome(
                    task_id=t.id, subject=t.subject, accepted=False,
                    reason=verdict.reason,
                ))
                continue
            submitted.append(t)
        tasks = submitted
        if not resume_mode:
            self.scheduler.submit(job, tasks)

        halted = ""
        # `TaskSpec.depends_on` is the obvious place to declare a dependency, so it
        # must be honoured. Reading only the separate dict meant a caller who used
        # the schema field got their ordering silently ignored — and silent
        # wrong-order execution is the worst failure shape available.
        dependencies = dict(dependencies or {})
        for t in tasks:
            if t.depends_on:
                dependencies.setdefault(t.id, []).extend(t.depends_on)

        # The job runs in waves. Each wave submits the WHOLE ready set to a pool
        # holding `max_parallel` threads: the pool keeps exactly that many tasks in
        # flight and refills a slot the moment a task finishes, so independent
        # tasks saturate capacity with no convoy inside the wave. Readiness is
        # recomputed at wave boundaries — the only points where dependency states
        # change — and the wave boundary is also where the governor, the operator
        # halt flag and heartbeat expiry get their say. Expiry between waves (never
        # during one) means a live thread can never have its leases reclaimed out
        # from under it: anything still in `_active` after a wave drains is a
        # leaked assignment with no thread behind it, which is exactly what expiry
        # exists to collect.
        with ThreadPoolExecutor(
            max_workers=max(1, self.scheduler.max_parallel),
            thread_name_prefix="forge-wave",
        ) as pool:
            while True:
                # The operator's kill switch. The dashboard writes this flag; if
                # nothing reads it the button is decoration on a screen someone is
                # watching precisely because they want the spending to stop.
                if self._halted(job.id):
                    halted = "halted by operator"
                    break

                # Reclaim slots from workers that went silent. Without this a
                # single task reporting BLOCKED holds its assignment forever; on a
                # machine where execution capacity is 1, every later task then
                # fails "no capacity" and one approval-blocked task kills the job.
                with self._sched_lock:
                    self.scheduler.expire_heartbeats()

                job_decision = self.governor.check_job(job.id)
                if job_decision.action is Action.TRIP:
                    halted = f"governor: {job_decision.reason}"
                    break

                ready = self.scheduler.ready_tasks(job.id, dependencies)
                if not ready:
                    break

                wave: list[tuple[TaskSpec, Future]] = []
                for task_id in ready:
                    spec = next((t for t in tasks if t.id == task_id), None)
                    if spec is None:
                        continue
                    wave.append((spec, pool.submit(
                        self._run_task_guarded, job, spec, executor, reviewer,
                        isolate_worktrees=isolate_worktrees, repo_root=repo_root,
                        executor_is_default=executor_is_default, job_base_ref=job_base_ref,
                        checkpoint_store=checkpoint_store,
                    )))

                # Collect in SUBMISSION order, not completion order. Completion
                # order is scheduling noise; a result list that reorders between
                # identical runs makes every downstream comparison flaky.
                progressed = False
                for spec, future in wave:
                    outcome = future.result()
                    if outcome is None:
                        # Deferred: resource pressure, a lease held by a task
                        # still in flight, or a governor trip observed elsewhere.
                        # The task never ran, so it stays QUEUED for the next
                        # wave rather than being fabricated into a failure.
                        continue
                    outcomes.append(outcome)
                    progressed = True

                    # Every finished task MUST leave the ready set. Without this a
                    # task that ends unaccepted without a state change (no capable
                    # worker, a refused merge) is handed back by ready_tasks
                    # forever, and because an outcome was produced the loop reads
                    # it as progress — an infinite loop that looks like work.
                    task_row = self.ledger.task(spec.id)
                    if task_row is None:
                        raise RuntimeError(f"ledger task disappeared: {spec.id}")
                    state = TaskState(task_row["state"])
                    if state not in (TaskState.DONE, TaskState.FAILED, TaskState.PAUSED):
                        self.ledger.set_task_state(
                            spec.id, TaskState.DONE if outcome.accepted else TaskState.FAILED
                        )

                if self._trip.is_set():
                    # One thread observed a governor trip. The whole job stops:
                    # tasks in flight were drained above, tasks not yet started
                    # were refused by the same flag inside _run_task, and no new
                    # wave begins. A budget trip seen by one thread while the
                    # others kept spending would make the cap decorative.
                    halted = f"governor: {self._trip_reason}"
                    break

                if not progressed:
                    # Nothing ran: every remaining task is blocked on a lease, a
                    # capability gap, or a failed dependency. Spinning here would
                    # be a busy-loop, so report rather than retry forever.
                    blocked = self.scheduler.blocked_tasks(job.id, dependencies)
                    halted = f"no runnable task; blocked: {blocked}" if blocked else "no runnable task"
                    break

        self.ledger.close_job(job.id, TaskState.DONE if not halted else TaskState.PAUSED)
        result = self._result(job, objective, outcomes, halted)

        # --- hooks: job_end — observe only; never changes the result --------
        hook_result = self.hooks.run(
            HookEvent.JOB_END,
            job_end_payload(job, accepted=result.accepted, rejected=result.rejected,
                             spend_usd=result.spend_usd, halted_reason=result.halted_reason),
        )
        with self._sched_lock:
            self._log_hooks(job.id, None, HookEvent.JOB_END, hook_result)

        return result

    def resume(
        self,
        job_id: str,
        executor: Executor | None = None,
        *,
        reviewer: Executor | None = None,
        isolate_worktrees: bool = False,
    ) -> ForgeResult:
        """Resume one persisted job without redoing terminal tasks.

        The original objective, budget, task contracts, and dependency edges
        come from the ledger. Only tasks whose event projection was non-terminal
        are requeued; accepted or rejected work is carried into the receipt but
        never charged a second implementation attempt.
        """
        row = self.ledger.job(job_id)
        if row is None:
            raise ValueError(f"job {job_id!r} not found")
        return self.run(
            row["objective"], [], executor=executor, cwd=row["cwd"],
            reviewer=reviewer, isolate_worktrees=isolate_worktrees,
            resume_job_id=job_id,
        )

    def _run_task_guarded(self, job: JobSpec, spec: TaskSpec, executor: Executor,
                          reviewer: Executor | None, *, isolate_worktrees: bool = False,
                          repo_root: str | None = None, executor_is_default: bool = False,
                          job_base_ref: str | None = None,
                          checkpoint_store: MutableMapping[str, dict] | None = None,
                          ) -> TaskOutcome | None:
        """`_run_task` plus the two guarantees threading adds.

        An exception in a worker thread must surface as a FAILED outcome — a task
        that simply vanishes from the results is the worst failure shape a
        parallel wave can produce. And the crashed task must not abscond with its
        leases, or one bad adapter would wedge every later wave that touches the
        same paths. No retry here on purpose: a crash is a harness bug, not a
        model failure, and burning the remaining attempts on it buys nothing.
        """
        try:
            return self._run_task(job, spec, executor, reviewer,
                                  isolate_worktrees=isolate_worktrees, repo_root=repo_root,
                                  executor_is_default=executor_is_default,
                                  job_base_ref=job_base_ref,
                                  checkpoint_store=checkpoint_store)
        except Exception as exc:  # noqa: BLE001 — the alternative is a vanished task
            reason = f"worker thread raised {type(exc).__name__}: {exc}"
            try:
                with self._sched_lock:
                    self.scheduler.report(
                        WorkerReport(task_id=spec.id, worker_id="forge.thread",
                                     state=TaskState.FAILED, blocker=reason),
                        job_id=job.id, budget=spec.budget,
                    )
            except Exception as report_exc:
                # Release was best-effort; the outcome below still records the
                # crash. But a report that never landed means this task's path
                # leases were never released either, and a leaked lease blocks
                # every later task touching those paths until its TTL expires —
                # silence here is a wall-clock cost nobody can see the reason for.
                record_degradation(
                    "forge.scheduler", "crash report/lease release failed", report_exc,
                    consequence="path leases may be leaked; later tasks on those paths wait for TTL",
                )
            return TaskOutcome(task_id=spec.id, subject=spec.subject,
                               accepted=False, reason=reason)

    # ------------------------------------------------------------- one task

    def _run_task(self, job: JobSpec, spec: TaskSpec, executor: Executor,
                  reviewer: Executor | None, *, isolate_worktrees: bool = False,
                  repo_root: str | None = None, executor_is_default: bool = False,
                  job_base_ref: str | None = None,
                  checkpoint_store: MutableMapping[str, dict] | None = None,
                  ) -> TaskOutcome | None:
        attempts = 0
        tier_used: int | None = None
        worker_id = ""
        merge_reasons: list[str] = []
        merge_warnings: list[str] = []
        # Escalation state carried across attempts. Set only on a classified
        # MODEL failure; consumed (and cleared) at the next attempt's routing
        # step so one failure buys at most one rung, never a ratchet.
        escalate_route = None
        escalate_failure: FailureClass | None = None
        # Newest-first record of this task's own failed attempts, fed into the
        # next attempt's prompt (tail-only, see _build_prompt) so a retry does
        # not pay for and repeat the identical mistake. Empty on attempt 1,
        # which is what keeps attempt 1 byte-identical to a task with no retry
        # machinery at all.
        attempt_history: list[AttemptSummary] = []
        # Set once, on whichever attempt first needs it; reused by every later
        # attempt of THIS task (see the retry `continue`s below) so attempt 2
        # actually resumes attempt 1's edits — that is the point of isolation.
        worktree_info = None

        def _finish(outcome):
            """Every `_run_task` return funnels through here so a live worktree's
            working directory is torn down exactly once, right before the task
            truly finishes — never on a `continue`, which is what lets a retry
            reuse the same one. The branch is kept either way (forensics); only
            the checkout goes."""
            if worktree_info is not None:
                with self._sched_lock:
                    remove_worktree(repo_root, spec.id, keep_branch=True)
            return outcome

        # Rung 0: record whether a deterministic strategy exists. This never
        # short-circuits the task — see _note_lowering.
        with self._sched_lock:
            self._note_lowering(job, spec)

        while attempts < self.max_attempts:
            # A trip observed by ANY thread stops everyone. Budget exhaustion seen
            # by one worker must not be spent through by the rest of the wave, so
            # this is checked before every attempt begins. Deferring (None) rather
            # than fabricating a FAILED outcome keeps the task honestly QUEUED —
            # it never ran.
            if self._trip.is_set():
                return _finish(None)

            # --- pressure: never start local work the machine cannot hold ----
            pressure = sample_pressure()
            with self._sched_lock:
                active = self.scheduler.active_count
            if not self.resources.may_start(WorkerKind.EXECUTION, active, pressure):
                if active == 0:
                    # Nothing running and still no room: proceed rather than
                    # deadlock. Refusing forever is worse than one tight run.
                    pass
                else:
                    # Pressure refusals shrink concurrency, they never fail work:
                    # the task waits for a later wave instead of running anyway.
                    return _finish(None)

            # --- hooks: pre_route — may veto BEFORE any spend -----------------
            # A crashed/timed-out/malformed hook proceeds with a warning; only
            # a clean exit 2 + {"veto": reason} refuses the task (hooks.py).
            hook_result = self.hooks.run(HookEvent.PRE_ROUTE, task_payload(job, spec))
            with self._sched_lock:
                self._log_hooks(job.id, spec.id, HookEvent.PRE_ROUTE, hook_result)
            if hook_result.vetoed:
                return _finish(TaskOutcome(
                    task_id=spec.id, subject=spec.subject, accepted=False,
                    reason=f"hook veto (pre_route): {hook_result.veto_reason}",
                    attempts=attempts,
                ))

            # --- route: cheapest tier that can finish, priced by the market ---
            # Quota awareness: skip providers whose window is exhausted.
            # A subscription with 0% remaining is not "cheapest capable" —
            # routing there burns a retry and a rate-limit error. Fall through.
            with self._quota_lock:
                exhausted_providers = {
                    state.provider for state in self.quota.states() if not state.available()
                }
            with self._sched_lock:
                stats = {r["worker_id"]: r
                         for r in self.ledger.worker_stats(spec.capabilities)
                         if r["worker_id"].split(".", 1)[0] not in exhausted_providers}
            route = self.router.route(
                spec.capabilities,
                stats=stats,
                market=self.market,
                quota=self.quota,
                needs_file_edits=bool(spec.scope.paths),
            )
            # A MODEL failure on the previous attempt asks the router for the
            # next rung up. `escalate` refuses when a rung is not the problem
            # (non-model failures, already at the top, nothing stronger
            # available) — in which case the retry honestly re-runs the
            # cheapest route rather than pretending a stronger one exists.
            if escalate_failure is not None and escalate_route is not None:
                esc = self.router.escalate(
                    escalate_route, escalate_failure, spec.capabilities,
                    stats=stats,
                    market=self.market,
                    quota=self.quota,
                    needs_file_edits=bool(spec.scope.paths),
                )
                if esc is not None:
                    route = esc
            escalate_route = None
            escalate_failure = None
            if route is None:
                return _finish(TaskOutcome(task_id=spec.id, subject=spec.subject, accepted=False,
                                   reason="no worker has the required capabilities",
                                   attempts=attempts))
            tier_used = int(route.tier)

            if route.tier is Tier.DETERMINISTIC:
                # Rung 0: ordinary code did it. The largest saving available.
                with self._sched_lock:
                    self.ledger.set_task_state(spec.id, TaskState.DONE)
                    self.events.append(job.id, EventType.TASK_ACCEPTED, task_id=spec.id,
                                       reason="handled deterministically")
                return _finish(TaskOutcome(task_id=spec.id, subject=spec.subject, accepted=True,
                                   tier=tier_used, reason=route.reason, attempts=attempts))

            # --- assign: takes path leases, refuses on collision -------------
            # The router is the single decision-maker: the scheduler binds
            # capacity and leases to the router's worker rather than running a
            # second selection. Two independent selections recorded the router's
            # tier against the scheduler's worker — a (worker, tier) pair no
            # decision ever produced, poisoning every stat keyed on it.
            with self._sched_lock:
                contended = False
                asn = self.scheduler.assign(job.id, spec.id,
                                            needs_file_edits=bool(spec.scope.paths),
                                            worker_id=route.worker_id or None)
                if asn is None:
                    collision = self.board.would_collide(spec.scope.paths, "default",
                                                         task_id=spec.id)
                    contended = collision.collides or self.scheduler.at_capacity()
            if asn is None:
                if contended:
                    # Another task holds the lease (or every slot). The lease is
                    # the safety mechanism: a task that cannot get its paths WAITS
                    # for a later wave — it never runs anyway, and it is not a
                    # failure. The holder releases at report time, so the next
                    # wave retries against a settled board.
                    return _finish(None)
                return _finish(TaskOutcome(task_id=spec.id, subject=spec.subject, accepted=False,
                                   reason="scheduler refused the assignment",
                                   attempts=attempts))
            worker_id = asn.worker_id
            attempts += 1

            # --- admission control: refuse BEFORE the money is gone ----------
            # A ledger check after the fact can only report an overspend; it can
            # never prevent one. Checking here is the difference between a cap and
            # a tripwire.
            with self._sched_lock:
                job_row = self.ledger.job(job.id)
                if job_row is None:
                    raise RuntimeError(f"ledger job disappeared: {job.id}")
                remaining = int(job_row["max_usd_micros"]) - self.ledger.job_spend_micros(job.id)
                task_remaining = spec.budget.max_usd_micros - self.ledger.task_spend_micros(spec.id)
            headroom = min(remaining, task_remaining)
            if headroom <= 0:
                with self._sched_lock:
                    self.scheduler.report(
                        WorkerReport(task_id=spec.id, worker_id=worker_id,
                                     state=TaskState.PAUSED, blocker="budget exhausted"),
                        job_id=job.id, budget=spec.budget,
                    )
                return _finish(TaskOutcome(
                    task_id=spec.id, subject=spec.subject, accepted=False,
                    worker_id=worker_id, tier=tier_used, attempts=attempts,
                    reason="refused before spending: no budget headroom",
                    usd_micros=self.ledger.task_spend_micros(spec.id),
                ))

            # --- worktree isolation: create/reuse before this attempt --------
            # Additive and opt-in: when `isolate_worktrees` is False (the
            # default) or any gate below fails, `task_cwd` stays `job.cwd` and
            # every line after this block behaves exactly as it did before
            # this feature existed.
            task_cwd = job.cwd
            if isolate_worktrees and repo_root is not None and spec.scope.paths:
                edit_profile = self.registry.get(worker_id)
                if edit_profile is not None and edit_profile.can_edit_files:
                    if worktree_info is None:
                        # A retry (`continue`, below) reaches this same call
                        # again with `worktree_info` already set — reused, not
                        # recreated.
                        branch = f"{_WORKTREE_BRANCH_PREFIX}{spec.id}"
                        try:
                            with self._sched_lock:
                                matches = [w for w in list_task_worktrees(repo_root)
                                          if w.branch == branch]
                                worktree_info = (matches[0] if matches
                                                 else create_worktree(
                                                     repo_root, spec.id,
                                                     base=job_base_ref or "HEAD"))
                        except Exception as exc:
                            return _finish(TaskOutcome(
                                task_id=spec.id, subject=spec.subject, accepted=False,
                                worker_id=worker_id, tier=tier_used, attempts=attempts,
                                reason=f"environment: could not create isolated worktree: {exc}",
                                usd_micros=self.ledger.task_spend_micros(spec.id),
                            ))
                    task_cwd = worktree_info.path

            # --- execute -----------------------------------------------------
            # Attempt 1 sends `spec` itself, untouched, when there is no retry
            # history and no live teammate to report -- no copy, no possible
            # byte difference from a task with neither retry machinery nor a
            # team around it. Both the capped history and the teammate-board
            # digest are per-attempt, in-the-moment state, so they are folded
            # into a COPY's fields here -- in the prompt's volatile TAIL (see
            # `_build_prompt`, forgeos/adapters/executor.py) -- and never into
            # `spec` itself. Per-attempt state belongs in the tail, never the
            # byte-stable cached prefix (forgeos/prompts/prefix.py): mixing it
            # into the prefix would turn every cache hit into a miss. `spec`
            # (the canonical object used for routing, leases, ledger keys,
            # budget checks above and below) is never mutated.
            task_for_attempt = spec
            updates: dict = {}
            if attempt_history:
                updates["attempt_history"] = _cap_attempt_history(attempt_history)
            board_context = self.board.context_for(spec.id)
            if board_context:
                updates["description"] = f"{spec.description}\n\n{board_context}"
            if updates:
                task_for_attempt = spec.model_copy(update=updates)

            # --- hooks: pre_execute — last veto point before any spend --------
            # Leases and worktree are already held at this point, so a veto here
            # must release them exactly like the crash path below does, or the
            # lease would leak for the rest of the job.
            hook_result = self.hooks.run(
                HookEvent.PRE_EXECUTE,
                task_payload(job, spec, worker_id=worker_id, tier=tier_used),
            )
            with self._sched_lock:
                self._log_hooks(job.id, spec.id, HookEvent.PRE_EXECUTE, hook_result)
            if hook_result.vetoed:
                reason = f"hook veto (pre_execute): {hook_result.veto_reason}"
                with self._sched_lock:
                    self.scheduler.report(
                        WorkerReport(task_id=spec.id, worker_id=worker_id,
                                     state=TaskState.FAILED, blocker=reason),
                        job_id=job.id, budget=spec.budget,
                    )
                return _finish(TaskOutcome(
                    task_id=spec.id, subject=spec.subject, accepted=False,
                    worker_id=worker_id, tier=tier_used, reason=reason, attempts=attempts,
                    usd_micros=self.ledger.task_spend_micros(spec.id),
                ))

            # The default routed executor bakes its cwd in at construction
            # (adapters/routed.py); rebuilding it here, once per isolated
            # attempt, is how a per-task worktree reaches it without adding a
            # cwd parameter to the public `Executor` Callable. Forge never
            # swaps out a caller-supplied executor — it keeps running exactly
            # as given — but `current_task_cwd()` still tells it, if it
            # chooses to ask, where this attempt's worktree lives.
            call_executor = executor
            if worktree_info is not None and executor_is_default:
                call_executor = self.default_executor(cwd=task_cwd,
                                                       checkpoint_store=checkpoint_store)
            _task_cwd_local.value = task_cwd if worktree_info is not None else None
            try:
                with self.spans.measure(job.id, Phase.MODEL_REASONING, task_id=spec.id):
                    result = call_executor(task_for_attempt, worker_id)
            finally:
                _task_cwd_local.value = None

            # An unmetered adapter must never read as free. Subscription CLIs
            # (omc team, codex) expose no token or dollar figure at all, so
            # they report 0 — and a ledger of zeros makes the USD ceiling
            # unenforceable while the dashboard cheerfully shows $0.00. Charge the
            # tier prior instead, flagged as an estimate, so the governor still has
            # something to stop.
            #
            # Some transports write to the ledger *themselves* before returning
            # (`Gateway.complete`'s invariant 4). For those the call is already
            # banked, and recording it here bills the same money twice: the budget
            # trips at half the real ceiling and every receipt reads double.
            # Observed live — one DeepSeek task produced two identical 145 µ$ rows,
            # one written by the gateway and one written here.
            #
            # "Reported zero" and "reported a real figure that is already banked"
            # are different states, so the unmetered fallback has to be skipped for
            # the second one too. Otherwise a paid API worker is charged its true
            # spend by the transport AND a tier prior by us, on the same call.
            already_recorded = getattr(result, "spend_already_recorded", False)
            charged = result.usd_micros
            profile = self.registry.get(worker_id)
            estimated = False
            if not already_recorded:
                if charged == 0 and profile is not None and profile.tier is not CostTier.FREE:
                    charged = profile.prior_micros
                    estimated = True
                with self._sched_lock:
                    self.ledger.record_spend(
                        job.id, worker_id, route.worker_id or worker_id, charged,
                        task_id=spec.id, tokens_in=result.tokens_in,
                        tokens_out=result.tokens_out,
                        tokens_cached_in=result.tokens_cached_in,
                        kind="estimate" if estimated else "call",
                    )

            # --- reduce output BEFORE any model sees it ----------------------
            evidence = result.evidence
            kept_lines: list[str] = []  # exact FAILED/error lines, for a retry's history
            if result.raw_output:
                # Choose the reducer by what the output actually is. Running the
                # pytest reducer over cargo/jest/gradle output yields the sentinel
                # "no pytest summary line found", which is TRUTHY and would overwrite
                # the worker's real evidence with a notice saying evidence could not
                # be parsed — a false green the merge gate would then accept.
                raw = result.raw_output
                reduced = (reduce_pytest(raw) if _looks_like_pytest(raw)
                           else reduce_generic(raw))
                kept_lines = reduced.kept_lines
                # Never let a reducer summary replace real evidence; augment it.
                evidence = f"{evidence} | {reduced.summary}".strip(" |") if evidence \
                    else reduced.summary
                if reduced.saved_tokens > 0:
                    with self._sched_lock:
                        self.avoidance.record(
                            job_id=job.id, task_id=spec.id,
                            method=AvoidanceMethod.DETERMINISTIC,
                            baseline_tokens=reduced.original_tokens_estimate,
                            actual_tokens=reduced.reduced_tokens_estimate,
                            baseline_source="tiktoken estimate of the raw tool output",
                        )

            report = WorkerReport(
                task_id=spec.id, worker_id=worker_id, state=result.state,
                goal=spec.subject, evidence=evidence,
                commands_run=result.commands_run, files_touched=result.files_touched,
                tests=result.tests, blocker=result.blocker,
                tokens_in=result.tokens_in, tokens_out=result.tokens_out,
                usd_micros=result.usd_micros, seconds=result.seconds,
                verdict=Verdict.PASS if result.state is TaskState.DONE else None,
            )
            with self._sched_lock:
                decision = self.scheduler.report(report, job_id=job.id, budget=spec.budget)
                if decision.action is Action.TRIP and not self._trip.is_set():
                    # First observer wins; the reason recorded is the one that
                    # actually stopped the job. Every other thread reads the flag
                    # at its next attempt boundary and stands down.
                    self._trip_reason = decision.reason
                    self._trip.set()

            if decision.action is Action.TRIP:
                return _finish(TaskOutcome(task_id=spec.id, subject=spec.subject, accepted=False,
                                   worker_id=worker_id, tier=tier_used,
                                   reason=f"governor: {decision.reason}", attempts=attempts,
                                   usd_micros=self.ledger.task_spend_micros(spec.id)))

            if result.state is not TaskState.DONE:
                # Classify before reacting. Escalating an environment failure buys
                # premium tokens for a problem no model can fix.
                if result.failure and not self.governor.should_escalate_model(result.failure):
                    return _finish(TaskOutcome(
                        task_id=spec.id, subject=spec.subject, accepted=False,
                        worker_id=worker_id, tier=tier_used, attempts=attempts,
                        reason=f"{result.failure.value}: "
                               f"{self.governor.remedy_for(result.failure, attempts, self.max_attempts).value}",
                        usd_micros=self.ledger.task_spend_micros(spec.id),
                    ))
                # Quota learning: if the blocker mentions a rate limit or reset,
                # feed it to the tracker so routing skips this provider until
                # the window reopens. "resets in 2h 15m" → don't burn retries.
                if result.blocker:
                    provider = worker_id.split(".")[0] if "." in worker_id else worker_id
                    with self._quota_lock:
                        self.quota.record_report(provider, result.blocker)
                        self._persist_quota_locked()
                attempt_history.insert(0, _attempt_summary(attempts, result, evidence, kept_lines))
                if result.failure is not None:
                    escalate_route = route
                    escalate_failure = result.failure
                continue  # model failure: retry, the next routing step escalates the tier -- worktree (if any) stays

            # --- security: scan the real diff, never assert a pass ------------
            with self.spans.measure(job.id, Phase.TOOL_CALL, task_id=spec.id):
                gates: list[GateResult] = [
                    run_security(result.files_touched, cwd=task_cwd)
                ]

            # --- independent review by a DIFFERENT worker ---------------------
            review_verdict = None
            reviewer_worker = ""
            if reviewer is not None:
                # The reviewer must be a genuinely different worker. Synthesising an
                # id that merely cannot collide turns the independence check into a
                # string ritual — the same model reviews its own patch and passes.
                reviewer_worker = self._pick_reviewer(spec, exclude=worker_id)
                if reviewer_worker:
                    with self.spans.measure(job.id, Phase.MANAGER, task_id=spec.id):
                        rev = reviewer(spec, reviewer_worker)
                    # Record BEFORE the result is used. An unrecorded call is an
                    # invisible call, and review is often the most expensive worker
                    # in the job — omitting it understates cost per accepted task.
                    # Unless the transport already banked it, in which case
                    # recording again double-bills the review exactly as it would
                    # double-bill the implementation.
                    if not getattr(rev, "spend_already_recorded", False):
                        with self._sched_lock:
                            self.ledger.record_spend(
                                job.id, reviewer_worker, "review", rev.usd_micros,
                                task_id=spec.id, tokens_in=rev.tokens_in,
                                tokens_out=rev.tokens_out,
                                tokens_cached_in=rev.tokens_cached_in,
                            )
                    review_verdict = "pass" if rev.state is TaskState.DONE else "fail"

            implementer_profile = self.registry.get(worker_id)
            reviewer_profile = self.registry.get(reviewer_worker) if reviewer_worker else None

            verdict = self.merge_gate.evaluate(
                gates=gates,
                tests_passed=result.tests.passed if result.tests else 0,
                tests_failed=result.tests.failed if result.tests else 0,
                evidence=evidence, commands_run=result.commands_run,
                reviewer_verdict=review_verdict,
                reviewer_worker=reviewer_worker, implementer_worker=worker_id,
                reviewer_family=_family_of(reviewer_profile),
                implementer_family=_family_of(implementer_profile),
                # `result.files_touched` is passed as-is — never coerced to `None`
                # on empty. For the real production path, `BridgedExecutionResult`
                # (adapters/executor.py) builds this list by folding every
                # FILE_DIFF event the session actually emitted; an empty list
                # there means the session provably touched no files, the same
                # confirmed-clean signal `MergeGate.evaluate`'s `files_touched=[]`
                # already means (see verify.py) — not "we don't know". Coercing it
                # to None would throw away real, checked information the bridge
                # went to the trouble of collecting.
                files_touched=result.files_touched,
            )
            merge_reasons = verdict.reasons
            merge_warnings = verdict.warnings

            # --- worktree isolation: the gate ruled; can it actually land? ---
            # An accepted-but-unmergeable task is not accepted -- ratifying the
            # gate's ALLOW and then discovering main can't absorb the change
            # would be a false green recorded as a true one. A rejected task
            # skips this: there is no point checking mergeability of work the
            # gate already refused on its own merits.
            if worktree_info is not None and verdict.allowed:
                with self._sched_lock:
                    check = merge_check(repo_root, worktree_info.branch)
                    landed = check.clean and merge_accepted(
                        repo_root, worktree_info.branch, message=f"{spec.id}: {spec.subject}"
                    )
                if not landed:
                    detail = (", ".join(check.conflicts) if check.conflicts
                             else check.detail or "merge could not be completed")
                    verdict = verdict.model_copy(update={
                        "allowed": False,
                        "reasons": merge_reasons + [f"merge conflict against main: {detail}"],
                    })
                    merge_reasons = verdict.reasons

            # --- hooks: post_gate — observe only, on the FINAL ruling ---------
            # `hook_result.vetoed` is deliberately never read below. The merge
            # gate above (including the worktree merge-check it just went
            # through) is the only thing that may declare a task accepted
            # (AGENTS.md rules 1, 2) — a hook can log an opinion, never cast
            # the vote.
            hook_result = self.hooks.run(
                HookEvent.POST_GATE,
                task_payload(job, spec, worker_id=worker_id, tier=tier_used,
                             merge_allowed=verdict.allowed),
            )
            with self._sched_lock:
                self._log_hooks(job.id, spec.id, HookEvent.POST_GATE, hook_result)

            # The merge gate is the only thing that may declare a task accepted.
            # The scheduler emits TASK_COMPLETED when a worker claims success;
            # this is where that claim is either ratified or refused.
            with self._sched_lock:
                self.events.append(
                    job.id,
                    EventType.TASK_ACCEPTED if verdict.allowed else EventType.TASK_REJECTED,
                    task_id=spec.id,
                    reason="merge gate allowed" if verdict.allowed
                    else "; ".join(merge_reasons)[:400],
                )
                # And the gate gets the last word on the worker's record. Win rate
                # is scored from the FINAL report per (worker, task), and a worker
                # reporting DONE on work the gate then refused was counting as a
                # 100% win — which is the number the router learns from, so it
                # would keep preferring workers whose output never merges. Same
                # self-flattery as counting unmerged work as accepted, one level
                # down where it is harder to see.
                self.ledger.record_report(
                    WorkerReport(
                        task_id=spec.id,
                        worker_id=worker_id,
                        state=TaskState.DONE if verdict.allowed else TaskState.FAILED,
                        verdict=Verdict.PASS if verdict.allowed else Verdict.FAIL,
                        summary="merge gate ruling",
                        blocker="" if verdict.allowed else "; ".join(merge_reasons)[:300],
                        # Spend is already recorded above; repeating it here would
                        # double the per-task average this same table feeds.
                        usd_micros=0,
                    )
                )

            # A refusal is worth retrying only when the worker could plausibly
            # fix it. Blanket-retrying triples the bill for a guaranteed second
            # refusal -- observed exactly that way with a text-only gateway
            # worker refused for "no tests passed; no commands run", which it is
            # structurally incapable of ever satisfying. Blanket-never wastes the
            # case the retry context was built for: the gate's reasons are now
            # carried into the next prompt, so "you did not run the tests" is
            # actionable feedback rather than a dead end.
            if (self.retry_fixable_refusals
                    and not verdict.allowed
                    and _refusal_is_fixable(merge_reasons)):
                attempt_history.insert(
                    0, _attempt_summary(attempts, result, "; ".join(merge_reasons), [])
                )
                continue

            return _finish(TaskOutcome(
                task_id=spec.id, subject=spec.subject, accepted=verdict.allowed,
                worker_id=worker_id, tier=tier_used,
                reason="merged" if verdict.allowed else "merge gate refused",
                merge_reasons=merge_reasons, merge_warnings=merge_warnings, attempts=attempts,
                usd_micros=self.ledger.task_spend_micros(spec.id),
            ))

        return _finish(TaskOutcome(task_id=spec.id, subject=spec.subject, accepted=False,
                           worker_id=worker_id, tier=tier_used,
                           reason=f"exhausted {self.max_attempts} attempts",
                           merge_reasons=merge_reasons, merge_warnings=merge_warnings,
                           attempts=attempts,
                           usd_micros=self.ledger.task_spend_micros(spec.id)))

    def _halted(self, job_id: str) -> bool:
        """Whether the operator pressed halt in the dashboard.

        Read from the same `halts.json` the dashboard writes. A missing or corrupt
        file means "not halted" — a broken flag file must not wedge a running job,
        and the governor's ceilings remain the real safety net either way.
        """
        flag = self.home / "halts.json"
        if not flag.exists():
            return False
        try:
            data = json.loads(flag.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError):
            return False
        entry = data.get(job_id)
        if isinstance(entry, dict):
            return bool(entry.get("halted", True))
        return bool(entry)

    def _log_hooks(self, job_id: str, task_id: str | None,
                   event: HookEvent, result: HookResult) -> None:
        """Record what a lifecycle event's hooks said. Observe-only: this never
        decides anything, and it never runs at all when no hook fired (that
        is `result.outcomes` empty, which is exactly what `HookRunner.run`
        returns for an unconfigured event without spawning a subprocess).
        Caller holds `_sched_lock`.
        """
        if not result.outcomes:
            return
        self.events.append(
            job_id, EventType.HOOK_INVOKED, task_id=task_id,
            hook_event=event.value, vetoed=result.vetoed,
            veto_reason=result.veto_reason, warnings=result.warnings,
        )

    def _pick_reviewer(self, spec: TaskSpec, *, exclude: str) -> str:
        """A real second worker, or "" meaning no independent review happened.

        Returning "" is the honest outcome when the fleet has only one capable
        worker — the merge gate then refuses for lack of review, which is correct.
        Manufacturing a reviewer id would let one model approve its own patch.

        Among capable candidates, one whose provider family (`_family_of`)
        differs from the implementer's is preferred: the self-preference
        literature (Panickssery et al., see `_FAMILY_HINTS` above) says a
        different family reviews better, and a fleet that actually has a
        cross-family worker should use it instead of leaving the merge gate's
        same-family WARN to fire on an avoidable choice. This only reorders
        candidates into two partitions -- cross-family first, then the
        existing order -- it never invents a new ranking within either one.
        Same-family review stays the fallback when no cross-family candidate
        is capable: the WARN already covers that case, and refusing to review
        at all would be worse than a biased review. An unknown family ("")
        never counts as cross-family, matching `_family_of`'s contract that
        two unrecognized profiles must never be treated as differing on
        evidence neither carries.
        """
        ranked = self.registry.rank(["review"]) or self.registry.rank(spec.capabilities)
        candidates = [cand for cand in ranked if cand.worker.worker_id != exclude]
        implementer_family = _family_of(self.registry.get(exclude))
        cross_family = [
            cand for cand in candidates
            if (fam := _family_of(cand.worker)) and fam != implementer_family
        ]
        for cand in cross_family or candidates:
            return cand.worker.worker_id
        return ""

    def _note_lowering(self, job: JobSpec, spec: TaskSpec) -> None:
        """Skip the model when the work is algorithmic. Checked before routing.

        This function NEVER accepts a task. It only records that a deterministic
        strategy exists and returns None so the normal path runs.

        The first version of this accepted the task outright — marking it DONE with
        no command run, no evidence, and no merge gate. That let a caller describe N
        tasks as bulk operations and receive `accepted=N, spend=$0.00,
        cost_per_accepted=0.0` having performed no work at all: a perfect score on
        the objective function for doing nothing. It also broke the rule that a task
        is done only when its acceptance criteria pass against something that ran.

        Lowering is advice about HOW to do the work cheaply. Something still has to
        do it, and prove it did.
        """
        operation = getattr(self, "_operations", {}).get(spec.id)
        if operation is None:
            return None

        verdict = classify(operation)
        if not verdict.lowerable:
            return None

        est = savings_estimate(operation, verdict)
        self.events.append(
            job.id, EventType.ACTION_RECORDED, task_id=spec.id,
            lowering="available", strategy=verdict.strategy,
            requires_sample_validation=verdict.requires_sample_validation,
            estimated_tokens_saved=int(est.get("saved", 0) or 0),
            note="deterministic strategy available; the work still has to run and "
                 "produce evidence before it can be accepted",
        )
        return None

    # ---------------------------------------------------------------- report

    def _result(self, job: JobSpec, objective: str, outcomes: list[TaskOutcome],
                halted: str) -> ForgeResult:
        cache = self.ledger.cache_stats(job.id)
        totals = self.avoidance.totals(job.id)
        attribution = self.spans.attribution(job.id)
        return ForgeResult(
            job_id=job.id,
            objective=objective,
            accepted=sum(1 for o in outcomes if o.accepted),
            rejected=sum(1 for o in outcomes if not o.accepted),
            outcomes=outcomes,
            spend_usd=round(from_micros(self.ledger.job_spend_micros(job.id)), 6),
            cache_hit_pct=cache.get("cache_hit_pct", 0.0),
            avoided_tokens=int(totals.get("saved_tokens", 0) or 0),
            bottleneck=attribution.bottleneck.value if attribution.bottleneck else None,
            halted_reason=halted,
        )

    # ----------------------------------------------------------------- views

    def doctor(self) -> str:
        """What this machine can actually do right now."""
        pol = self.resources.policy()
        pressure = sample_pressure()
        lines = [
            "forgeos doctor",
            "-" * 58,
            f"machine        {pol['machine']['resource_class']}  "
            f"{pol['machine']['physical_cores']} cores, "
            f"{pol['machine']['ram_total_gib']} GiB",
            f"gpu            {pol['machine']['gpu'] or 'none'} "
            f"({pol['machine']['gpu_vram_gib']} GiB VRAM)",
            f"inference      {'remote preferred' if pol['inference']['prefer_remote'] else 'local viable'}",
            f"pools          reasoning={pol['limits']['reasoning_workers']} "
            f"execution={pol['limits']['execution_workers']}",
            f"pressure       {pressure.ram_available_gib:.1f} GiB free, "
            f"cpu {pressure.cpu_percent:.0f}% -> {pressure.action.value}",
            "",
            "providers",
        ]
        for p in sorted(self.settings.providers.values(), key=lambda x: x.name):
            lines.append(f"  {p.name:<14} {p.kind.value:<9} {p.status()}")
        with self._quota_lock:
            quota_states = self.quota.states()
        lines += ["", "quota telemetry"]
        if self.quota_load_error:
            lines.append(f"  invalid snapshot ({self.quota_load_error}) — using empty state")
        elif not quota_states:
            lines.append("  no provider-reported snapshots")
        else:
            for state in quota_states:
                remaining = (
                    f"{state.pct_remaining:.0f}% remaining" if state.pct_remaining is not None
                    else "remaining unknown"
                )
                reset = (
                    f", reset in {state.seconds_until_reset():.0f}s"
                    if state.resets_at is not None else ""
                )
                lines.append(
                    f"  {state.provider:<14} {state.model or '*':<18} {remaining}"
                    f" ({state.source.value}{reset})"
                )
        if self.market.resources():
            lines += ["", self.market.report()]
        return "\n".join(lines)


__all__ = ["ExecutionResult", "Executor", "Forge", "ForgeResult", "TaskOutcome",
           "DEFAULT_HOME"]
