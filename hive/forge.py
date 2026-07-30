"""The Forge — the assembled machine.

Every other module in hive is a part. This is the only file that wires them into
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
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from .contracts import (
    Budget,
    FailureClass,
    JobSpec,
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
from .core.scheduler import Scheduler
from .core.timing import Phase, SpanStore
from .core.verify import GateResult, MergeGate, run_security
from .economy.avoidance import AvoidanceLog, AvoidanceMethod
from .economy.lowerer import Operation, classify, savings_estimate
from .economy.reducer import reduce_generic, reduce_pytest

from .events import EventLog, EventType
from .leases import LeaseStore
from .ledger import Ledger
from .registry import CostTier, Registry, default_registry
from .settings import Settings

DEFAULT_HOME = Path.home() / ".hive"


def _looks_like_pytest(output: str) -> bool:
    """Whether the pytest reducer can meaningfully parse this."""
    low = output.lower()
    return any(m in low for m in ("passed", "failed", "error")) and (
        "pytest" in low or "::" in output or "=====" in output
    )



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


class Forge:
    """The assembled harness. One object, one loop, every part wired."""

    def __init__(
        self,
        *,
        home: Path | None = None,
        registry: Registry | None = None,
        settings: Settings | None = None,
        market: CapacityMarket | None = None,
        max_attempts: int = 3,
    ):
        self.home = home or DEFAULT_HOME
        self.home.mkdir(parents=True, exist_ok=True)

        self.ledger = Ledger(self.home / "ledger.db")
        self.events = EventLog(self.home / "events.db")
        self.leases = LeaseStore(self.home / "leases.db")
        self.avoidance = AvoidanceLog(self.home / "avoidance.db")
        self.spans = SpanStore(self.home / "spans.db")

        self.settings = settings or Settings.load()
        self.registry = registry or default_registry()
        self.market = market or CapacityMarket()

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

    def close(self) -> None:
        for store in (self.ledger, self.events, self.leases, self.avoidance, self.spans):
            try:
                store.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ run

    def run(
        self,
        objective: str,
        tasks: list[TaskSpec],
        executor: Executor,
        *,
        cwd: str = ".",
        budget: Budget | None = None,
        dependencies: dict[str, list[str]] | None = None,
        reviewer: Executor | None = None,
        operations: dict[str, Operation] | None = None,
    ) -> ForgeResult:
        """Run a job to completion.

        `operations` optionally describes a task as a bulk operation. The Lowerer
        then records whether a deterministic strategy exists — advice about HOW to
        do the work cheaply, never a substitute for doing it.
        """
        self._operations = operations or {}
        job = JobSpec(objective=objective, cwd=cwd,
                      budget=budget or Budget(max_usd=20.0, max_seconds=7200,
                                              max_iterations=60))
        for t in tasks:
            t.job_id = job.id
        self.scheduler.submit(job, tasks)

        outcomes: list[TaskOutcome] = []
        halted = ""
        # `TaskSpec.depends_on` is the obvious place to declare a dependency, so it
        # must be honoured. Reading only the separate dict meant a caller who used
        # the schema field got their ordering silently ignored — and silent
        # wrong-order execution is the worst failure shape available.
        dependencies = dict(dependencies or {})
        for t in tasks:
            if t.depends_on:
                dependencies.setdefault(t.id, []).extend(t.depends_on)

        while True:
            # The operator's kill switch. The dashboard writes this flag; if nothing
            # reads it the button is decoration on a screen someone is watching
            # precisely because they want the spending to stop.
            if self._halted(job.id):
                halted = "halted by operator"
                break

            # Reclaim slots from workers that went silent. Without this a single
            # task reporting BLOCKED holds its assignment forever; on a machine
            # where execution capacity is 1, every later task then fails "no
            # capacity" and one approval-blocked task kills the whole job.
            self.scheduler.expire_heartbeats()

            job_decision = self.governor.check_job(job.id)
            if job_decision.action is Action.TRIP:
                halted = f"governor: {job_decision.reason}"
                break

            ready = self.scheduler.ready_tasks(job.id, dependencies)
            if not ready:
                break

            progressed = False
            for task_id in ready:
                spec = next((t for t in tasks if t.id == task_id), None)
                if spec is None:
                    continue
                outcome = self._run_task(job, spec, executor, reviewer)
                if outcome is None:
                    continue  # deferred by resource pressure; try again next pass
                outcomes.append(outcome)
                progressed = True

                # Every finished task MUST leave the ready set. Without this a task
                # that ends unaccepted without a state change (no capable worker, a
                # refused merge) is handed back by ready_tasks forever, and because
                # an outcome was produced the loop reads it as progress — an
                # infinite loop that looks like work.
                state = TaskState(self.ledger.task(spec.id)["state"])
                if state not in (TaskState.DONE, TaskState.FAILED, TaskState.PAUSED):
                    self.ledger.set_task_state(
                        spec.id, TaskState.DONE if outcome.accepted else TaskState.FAILED
                    )

            if not progressed:
                # Nothing ran: every remaining task is blocked on a lease, a
                # capability gap, or a failed dependency. Spinning here would be a
                # busy-loop, so report rather than retry forever.
                blocked = self.scheduler.blocked_tasks(job.id, dependencies)
                halted = f"no runnable task; blocked: {blocked}" if blocked else "no runnable task"
                break

        self.ledger.close_job(job.id, TaskState.DONE if not halted else TaskState.PAUSED)
        return self._result(job, objective, outcomes, halted)

    # ------------------------------------------------------------- one task

    def _run_task(self, job: JobSpec, spec: TaskSpec, executor: Executor,
                  reviewer: Executor | None) -> TaskOutcome | None:
        attempts = 0
        tier_used: int | None = None
        worker_id = ""
        merge_reasons: list[str] = []

        # Rung 0: record whether a deterministic strategy exists. This never
        # short-circuits the task — see _note_lowering.
        self._note_lowering(job, spec)

        while attempts < self.max_attempts:
            # --- pressure: never start local work the machine cannot hold ----
            pressure = sample_pressure()
            if not self.resources.may_start(WorkerKind.EXECUTION, self.scheduler.active_count,
                                            pressure):
                if self.scheduler.active_count == 0:
                    # Nothing running and still no room: proceed rather than
                    # deadlock. Refusing forever is worse than one tight run.
                    pass
                else:
                    return None

            # --- route: cheapest tier that can finish, priced by the market ---
            stats = {r["worker_id"]: r
                     for r in self.ledger.worker_stats(spec.capabilities)}
            route = self.router.route(
                spec.capabilities,
                stats=stats,
                needs_file_edits=bool(spec.scope.paths),
            )
            if route is None:
                return TaskOutcome(task_id=spec.id, subject=spec.subject, accepted=False,
                                   reason="no worker has the required capabilities",
                                   attempts=attempts)
            tier_used = int(route.tier)

            if route.tier is Tier.DETERMINISTIC:
                # Rung 0: ordinary code did it. The largest saving available.
                self.ledger.set_task_state(spec.id, TaskState.DONE)
                self.events.append(job.id, EventType.TASK_ACCEPTED, task_id=spec.id,
                                   reason="handled deterministically")
                return TaskOutcome(task_id=spec.id, subject=spec.subject, accepted=True,
                                   tier=tier_used, reason=route.reason, attempts=attempts)

            # --- assign: takes path leases, refuses on collision -------------
            asn = self.scheduler.assign(job.id, spec.id,
                                        needs_file_edits=bool(spec.scope.paths))
            if asn is None:
                collision = self.board.would_collide(spec.scope.paths, "default",
                                                     task_id=spec.id)
                return TaskOutcome(task_id=spec.id, subject=spec.subject, accepted=False,
                                   reason=f"blocked: {collision.suggestion}"
                                          if collision.collides else "no capacity",
                                   attempts=attempts)
            worker_id = asn.worker_id
            attempts += 1

            # --- admission control: refuse BEFORE the money is gone ----------
            # A ledger check after the fact can only report an overspend; it can
            # never prevent one. Checking here is the difference between a cap and
            # a tripwire.
            job_row = self.ledger.job(job.id)
            remaining = int(job_row["max_usd_micros"]) - self.ledger.job_spend_micros(job.id)
            task_remaining = spec.budget.max_usd_micros - self.ledger.task_spend_micros(spec.id)
            headroom = min(remaining, task_remaining)
            if headroom <= 0:
                self.scheduler.report(
                    WorkerReport(task_id=spec.id, worker_id=worker_id,
                                 state=TaskState.PAUSED, blocker="budget exhausted"),
                    job_id=job.id, budget=spec.budget,
                )
                return TaskOutcome(
                    task_id=spec.id, subject=spec.subject, accepted=False,
                    worker_id=worker_id, tier=tier_used, attempts=attempts,
                    reason="refused before spending: no budget headroom",
                    usd_micros=self.ledger.task_spend_micros(spec.id),
                )

            # --- execute -----------------------------------------------------
            with self.spans.measure(job.id, Phase.MODEL_REASONING, task_id=spec.id):
                result = executor(spec, worker_id)

            # An unmetered adapter must never read as free. Subscription CLIs
            # (omc team, jcode, codex) expose no token or dollar figure at all, so
            # they report 0 — and a ledger of zeros makes the USD ceiling
            # unenforceable while the dashboard cheerfully shows $0.00. Charge the
            # tier prior instead, flagged as an estimate, so the governor still has
            # something to stop.
            charged = result.usd_micros
            profile = self.registry.get(worker_id)
            estimated = False
            if charged == 0 and profile is not None and profile.tier is not CostTier.FREE:
                charged = profile.prior_micros
                estimated = True
            self.ledger.record_spend(
                job.id, worker_id, route.worker_id or worker_id, charged,
                task_id=spec.id, tokens_in=result.tokens_in,
                tokens_out=result.tokens_out, tokens_cached_in=result.tokens_cached_in,
                kind="estimate" if estimated else "call",
            )

            # --- reduce output BEFORE any model sees it ----------------------
            evidence = result.evidence
            if result.raw_output:
                # Choose the reducer by what the output actually is. Running the
                # pytest reducer over cargo/jest/gradle output yields the sentinel
                # "no pytest summary line found", which is TRUTHY and would overwrite
                # the worker's real evidence with a notice saying evidence could not
                # be parsed — a false green the merge gate would then accept.
                raw = result.raw_output
                reduced = (reduce_pytest(raw) if _looks_like_pytest(raw)
                           else reduce_generic(raw))
                # Never let a reducer summary replace real evidence; augment it.
                evidence = f"{evidence} | {reduced.summary}".strip(" |") if evidence \
                    else reduced.summary
                if reduced.saved_tokens > 0:
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
            decision = self.scheduler.report(report, job_id=job.id, budget=spec.budget)

            if decision.action is Action.TRIP:
                return TaskOutcome(task_id=spec.id, subject=spec.subject, accepted=False,
                                   worker_id=worker_id, tier=tier_used,
                                   reason=f"governor: {decision.reason}", attempts=attempts,
                                   usd_micros=self.ledger.task_spend_micros(spec.id))

            if result.state is not TaskState.DONE:
                # Classify before reacting. Escalating an environment failure buys
                # premium tokens for a problem no model can fix.
                if result.failure and not self.governor.should_escalate_model(result.failure):
                    return TaskOutcome(
                        task_id=spec.id, subject=spec.subject, accepted=False,
                        worker_id=worker_id, tier=tier_used, attempts=attempts,
                        reason=f"{result.failure.value}: "
                               f"{self.governor.remedy_for(result.failure, attempts, self.max_attempts).value}",
                        usd_micros=self.ledger.task_spend_micros(spec.id),
                    )
                continue  # model failure: retry, router may escalate the tier

            # --- security: scan the real diff, never assert a pass ------------
            with self.spans.measure(job.id, Phase.TOOL_CALL, task_id=spec.id):
                gates: list[GateResult] = [
                    run_security(result.files_touched, cwd=job.cwd)
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
                    self.ledger.record_spend(
                        job.id, reviewer_worker, "review", rev.usd_micros,
                        task_id=spec.id, tokens_in=rev.tokens_in,
                        tokens_out=rev.tokens_out,
                        tokens_cached_in=rev.tokens_cached_in,
                    )
                    review_verdict = "pass" if rev.state is TaskState.DONE else "fail"

            verdict = self.merge_gate.evaluate(
                gates=gates,
                tests_passed=result.tests.passed if result.tests else 0,
                tests_failed=result.tests.failed if result.tests else 0,
                evidence=evidence, commands_run=result.commands_run,
                reviewer_verdict=review_verdict,
                reviewer_worker=reviewer_worker, implementer_worker=worker_id,
            )
            merge_reasons = verdict.reasons

            return TaskOutcome(
                task_id=spec.id, subject=spec.subject, accepted=verdict.allowed,
                worker_id=worker_id, tier=tier_used,
                reason="merged" if verdict.allowed else "merge gate refused",
                merge_reasons=merge_reasons, attempts=attempts,
                usd_micros=self.ledger.task_spend_micros(spec.id),
            )

        return TaskOutcome(task_id=spec.id, subject=spec.subject, accepted=False,
                           worker_id=worker_id, tier=tier_used,
                           reason=f"exhausted {self.max_attempts} attempts",
                           merge_reasons=merge_reasons, attempts=attempts,
                           usd_micros=self.ledger.task_spend_micros(spec.id))

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

    def _pick_reviewer(self, spec: TaskSpec, *, exclude: str) -> str:
        """A real second worker, or "" meaning no independent review happened.

        Returning "" is the honest outcome when the fleet has only one capable
        worker — the merge gate then refuses for lack of review, which is correct.
        Manufacturing a reviewer id would let one model approve its own patch.
        """
        ranked = self.registry.rank(["review"]) or self.registry.rank(spec.capabilities)
        for cand in ranked:
            if cand.worker.worker_id != exclude:
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
            "hive doctor",
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
        if self.market.resources():
            lines += ["", self.market.report()]
        return "\n".join(lines)


__all__ = ["ExecutionResult", "Executor", "Forge", "ForgeResult", "TaskOutcome",
           "DEFAULT_HOME"]
