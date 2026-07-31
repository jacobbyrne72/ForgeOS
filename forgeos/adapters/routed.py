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

from pathlib import Path

from ..contracts import FailureClass, TaskSpec, TaskState
from ..registry import Adapter
from .executor import CheckpointStore, adapter_executor
from .factory import build_adapter

# Runner tokens that make an acceptance line shell-shaped enough to derive a
# `Bash(...)` allowlist pattern from. Deliberately small: a token not on this
# list means "don't guess", not "add more tokens speculatively".
_RUNNER_TOKENS = {"python", "pytest", "npm", "node", "cargo", "go", "make"}

# Trailing prose words stripped from the tail of an acceptance line before
# the pattern is cut — e.g. "python -m pytest tests -q passes" ends in verdict
# language, not part of the command. Only the tail is touched; a word in this
# set that appears mid-command (there is no such case in the grammar observed
# so far) is never removed.
_TRAILING_PROSE = {"passes", "pass", "succeeds", "succeed", "should", "must", "green", "ok"}


def _acceptance_tools(spec: TaskSpec) -> list[str]:
    """Derive `Bash(<command>:*)` allowlist patterns from shell-shaped acceptance criteria.

    An acceptance entry is shell-shaped only if it opens with a known runner
    token (see `_RUNNER_TOKENS`); anything else ("all tests green") derives
    nothing rather than guessing at a command that isn't there. The pattern is
    the first three whitespace-split tokens of what remains after stripping
    trailing verdict words (see `_TRAILING_PROSE`) from the tail — deterministic,
    and never wider than what the acceptance line actually says.
    """
    patterns: list[str] = []
    for entry in spec.acceptance:
        tokens = entry.split()
        if not tokens or tokens[0].lower() not in _RUNNER_TOKENS:
            continue
        while tokens and tokens[-1].lower().strip(".,!") in _TRAILING_PROSE:
            tokens.pop()
        if not tokens:
            continue
        pattern = f"Bash({' '.join(tokens[:3])}:*)"
        if pattern not in patterns:
            patterns.append(pattern)
    return patterns


def routed_executor(
    registry,
    ledger=None,
    *,
    gateway: object | None = None,
    cwd: str = ".",
    timeout_seconds: float = 900.0,
    checkpoint_store: CheckpointStore | None = None,
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

    `checkpoint_store`, when given, is forwarded to `adapter_executor`
    unchanged. That is what lets a task's checkpoint survive across attempts
    even though a fresh adapter is built per call above — the continuity
    lives in the store, never in the (deliberately never cached) adapter
    instance.
    """
    from ..forge import ExecutionResult  # local: forge imports this module's caller

    def _remaining_micros_for(spec: TaskSpec):
        def remaining() -> int:
            task_cap = spec.budget.max_usd_micros
            if ledger is None:
                return task_cap
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
        # claude-CLI-specific affordance: a headless `claude -p --permission-mode
        # acceptEdits` seat cannot run its own acceptance command non-interactively,
        # and a per-repo .claude/settings.json allowlist is ignored until the
        # workspace is interactively trusted (docs/DOGFOOD.md runs 5-7). Deriving
        # `--allowedTools` from the task's own acceptance criteria on the command
        # line is the deterministic fix: minimum privilege per run, zero repo
        # config, and it only ever widens this one process's own argv — never
        # any other command's profile.
        if profile.adapter is Adapter.LOCAL_COMMAND and Path(profile.command).stem.lower() == "claude":
            patterns = _acceptance_tools(spec)
            if patterns:
                profile = profile.model_copy(update={
                    "args": [*profile.args, "--allowedTools", ", ".join(patterns)],
                })
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
            timeout_seconds=timeout_seconds, checkpoint_store=checkpoint_store,
        )
        return run(spec, worker_id)

    return execute


__all__ = ["routed_executor"]
