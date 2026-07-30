"""The missing glue: routed worker_id → live adapter → Forge-shaped executor.

`build_adapter` turns a registry profile into a backend; `adapter_executor`
drives a backend to an `ExecutionResult`. Neither knows the other's caller, so
until now every `Forge.run` needed a hand-rolled executor and the default
experience was a library, not a tool. This module composes the two into the
callable `Forge.run` actually wants, resolving the profile from the
`worker_id` the router chose — the same id the ledger bills, so spend lands
on the worker that ran.

Unavailability stays honest: a profile this machine cannot run comes back as
a FAILED result with `FailureClass.ENVIRONMENT` — repair, never escalate —
and never a silent substitution of a different worker. Substituting would
attribute the work and its cost to the profile the router chose rather than
the one that ran, poisoning every measured win-rate the router later
depends on.
"""

from __future__ import annotations

from ..contracts import FailureClass, TaskSpec, TaskState
from .executor import adapter_executor
from .factory import build_adapter


def routed_executor(
    registry,
    ledger=None,
    *,
    gateway: object | None = None,
    cwd: str = ".",
    timeout_seconds: float = 900.0,
):
    """An `Executor` that runs whichever worker the router picked.

    `registry` resolves `worker_id` to a profile; `ledger` (optional) lets
    gateway-backed workers compute live budget headroom so their preflight can
    refuse an over-budget call. Without a `gateway`, gateway-backed profiles
    report unavailable with the factory's reason — reported, not substituted.

    Adapters are built per call, not cached: `adapter_executor` runs one full
    session lifecycle (start → send → close) per task, and a stale cached
    backend whose binary or key vanished mid-job would fail every later task
    with a reason recorded at the wrong time.
    """
    from ..forge import ExecutionResult  # local: forge imports this module's caller

    def _remaining_micros_for(spec: TaskSpec):
        def remaining() -> int:
            task_cap = spec.budget.max_usd_micros
            task_rem = task_cap - ledger.task_spend_micros(spec.id)
            job_rem = task_rem
            try:
                row = ledger.job(spec.job_id)
                job_rem = int(row["max_usd_micros"]) - ledger.job_spend_micros(spec.job_id)
            except Exception:
                # No job row (direct executor use in tests): the task budget
                # still caps the call, so the preflight keeps a real ceiling.
                pass
            return max(0, min(task_rem, job_rem))

        return remaining

    def execute(spec: TaskSpec, worker_id: str) -> ExecutionResult:
        profile = registry.get(worker_id)
        if profile is None:
            return ExecutionResult(
                state=TaskState.FAILED, failure=FailureClass.ENVIRONMENT,
                blocker=f"router chose {worker_id!r} but the registry has no such profile",
            )
        adapter, reason = build_adapter(
            profile,
            gateway=gateway,
            job_id=spec.job_id,
            remaining_micros=_remaining_micros_for(spec) if ledger is not None else None,
        )
        if adapter is None:
            return ExecutionResult(
                state=TaskState.FAILED, failure=FailureClass.ENVIRONMENT,
                blocker=reason,
            )
        run = adapter_executor(
            adapter, cwd=cwd, model_profile=profile.model or "",
            timeout_seconds=timeout_seconds,
        )
        return run(spec, worker_id)

    return execute


__all__ = ["routed_executor"]
