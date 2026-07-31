"""Lifecycle hooks — deterministic, config-driven, strictly bounded.

Every harness feature map (docs/research/harness-features-*.md) names a
hooks/lifecycle-event system as ForgeOS's biggest missing organ: Claude
Code's Pre/PostToolUse, gemini's policy engine, codex's notify. This is
ForgeOS's version, and it is deliberately narrower than any of those:

- A hook attaches to one of four lifecycle events: `pre_route`, `pre_execute`,
  `post_gate`, `job_end`.
- `pre_route` and `pre_execute` may VETO a task — refuse it before any money
  moves. That is the entire power a hook has.
- `post_gate` and `job_end` are OBSERVE-ONLY. The merge gate
  (`forgeos/core/verify.py`) is the only thing that may declare a task
  accepted (AGENTS.md rules 1, 2). A hook can never widen a budget, forge
  evidence, or approve a merge, so `forge.py` never reads a `post_gate` or
  `job_end` hook's exit code for a decision — only logs it.
- Hooks are defined ONLY in `Settings.hooks` (`~/.forgeos/settings.json`),
  never in task text. Task text is untrusted input to the router (AGENTS.md
  rule 9) and must never gain the ability to name a command Forge executes.

A broken hook — crash, timeout, a non-zero exit, an exit 2 without the
documented veto shape — must never become a denial-of-service on the job.
Every outcome other than a clean `exit 2` with `{"veto": "<reason>"}` on
stdout degrades to "proceed, with a warning recorded in the event log."
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from .contracts import JobSpec, TaskSpec
from .settings import HookDef, HookEvent, Settings

__all__ = [
    "HookOutcome",
    "HookResult",
    "HookRunner",
    "job_end_payload",
    "task_payload",
]


@dataclass(frozen=True)
class HookOutcome:
    """What one hook invocation produced."""

    command: str
    vetoed: bool = False
    # The veto reason (vetoed=True), or a human-readable note on why an
    # abnormal outcome is being treated as "proceed" (ok=False).
    reason: str = ""
    ok: bool = True


@dataclass(frozen=True)
class HookResult:
    """Every hook registered for one event, already resolved.

    `vetoed`/`veto_reason` reflect only the one documented veto shape (a clean
    `exit 2` with `{"veto": reason}` on stdout). Every other abnormal outcome
    — crash, timeout, wrong exit code, malformed stdout — lands in `warnings`
    instead, never in `vetoed`. That split is what keeps a broken hook from
    stalling a task: only a hook that succeeded at producing the documented
    refusal can refuse anything.
    """

    outcomes: tuple[HookOutcome, ...] = ()

    @property
    def vetoed(self) -> bool:
        return any(o.vetoed for o in self.outcomes)

    @property
    def veto_reason(self) -> str:
        return next((o.reason for o in self.outcomes if o.vetoed), "")

    @property
    def warnings(self) -> list[str]:
        return [f"{o.command}: {o.reason}" for o in self.outcomes if not o.ok]


_EMPTY_RESULT = HookResult()


class HookRunner:
    """Loads hook definitions from `Settings` and invokes them at lifecycle events.

    The empty-config case — no `hooks` entries in `~/.forgeos/settings.json`,
    the default on every existing install — is the one that must cost nothing:
    `run()` returns the shared `_EMPTY_RESULT` before `subprocess` is ever
    touched, so a Forge with no hooks configured pays zero subprocess calls
    for hooks existing as a feature.
    """

    def __init__(self, hooks: list[HookDef] | None = None):
        self._by_event: dict[HookEvent, list[HookDef]] = {}
        for h in hooks or []:
            self._by_event.setdefault(h.event, []).append(h)

    @classmethod
    def from_settings(cls, settings: Settings) -> HookRunner:
        return cls(settings.hooks)

    def configured(self, event: HookEvent) -> bool:
        return bool(self._by_event.get(event))

    def run(self, event: HookEvent, payload: dict) -> HookResult:
        """Invoke every hook registered for `event`, in config order.

        Returns `_EMPTY_RESULT` — no subprocess, no allocation beyond the
        dict lookup — when nothing is registered for this event.
        """
        defs = self._by_event.get(event)
        if not defs:
            return _EMPTY_RESULT
        return HookResult(outcomes=tuple(self._invoke(h, payload) for h in defs))

    @staticmethod
    def _invoke(hook: HookDef, payload: dict) -> HookOutcome:
        stdin_bytes = json.dumps(payload).encode("utf-8")
        try:
            proc = subprocess.run(  # noqa: S603 — argv list, never shell=True
                [hook.command, *hook.args],
                input=stdin_bytes,
                capture_output=True,
                timeout=hook.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return HookOutcome(
                command=hook.command, ok=False,
                reason=f"timed out after {hook.timeout_seconds}s; proceeding",
            )
        except OSError as exc:
            return HookOutcome(
                command=hook.command, ok=False,
                reason=f"failed to start ({exc}); proceeding",
            )

        if proc.returncode == 2:
            reason = _parse_veto(proc.stdout)
            if reason:
                return HookOutcome(command=hook.command, vetoed=True, reason=reason)
            return HookOutcome(
                command=hook.command, ok=False,
                reason='exited 2 without a {"veto": reason} payload; proceeding',
            )

        if proc.returncode != 0:
            return HookOutcome(
                command=hook.command, ok=False,
                reason=f"exited {proc.returncode}; proceeding",
            )

        return HookOutcome(command=hook.command)


def _parse_veto(stdout: bytes) -> str:
    try:
        data = json.loads(stdout.decode("utf-8", "replace").strip() or "{}")
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    reason = data.get("veto")
    return reason if isinstance(reason, str) and reason else ""


# --------------------------------------------------------------------------
# Payload builders. The privacy floor lives here, once, so no call site in
# forge.py can widen it by forgetting to strip something.
# --------------------------------------------------------------------------

def task_payload(job: JobSpec, spec: TaskSpec, **extra) -> dict:
    """The bounded shape sent to every task-scoped hook (`pre_route`,
    `pre_execute`, `post_gate`).

    `subject` (a short title) travels; `description` — the full task body,
    and exactly the untrusted, potentially sensitive text a task author
    writes — never does. Secrets and full transcripts are never in scope
    either: nothing here is anything but an id, a short label, or a number.
    """
    payload: dict = {
        "job_id": job.id,
        "task_id": spec.id,
        "subject": spec.subject,
        "scope_paths": list(spec.scope.paths),
        "capabilities": list(spec.capabilities),
        "budget_max_usd": spec.budget.max_usd,
        "budget_max_seconds": spec.budget.max_seconds,
        "budget_max_iterations": spec.budget.max_iterations,
        "job_budget_max_usd": job.budget.max_usd,
    }
    payload.update(extra)
    return payload


def job_end_payload(job: JobSpec, *, accepted: int, rejected: int,
                     spend_usd: float, halted_reason: str) -> dict:
    """The bounded shape sent to the `job_end` hook.

    No objective text, for the same reason `task_payload` omits
    `description`: it is free-form text a caller wrote, not a bounded,
    structured figure.
    """
    return {
        "job_id": job.id,
        "accepted": accepted,
        "rejected": rejected,
        "spend_usd": spend_usd,
        "halted_reason": halted_reason,
    }
