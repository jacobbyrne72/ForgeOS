"""Turns a routed `WorkerProfile` into a live `WorkerAdapter`.

The registry has always carried an `adapter` field on every profile, and until
now nothing read it. The router would pick `omniroute.free`, the scheduler would
lease its paths, and there was no code anywhere that could answer "so what object
actually runs this?". The enum was documentation.

This module closes that half of the gap. The other half — driving an adapter's
async event stream to produce an `ExecutionResult` — is the executor bridge.
Together they form the whole path:

    router picks a profile
      → build_adapter(profile)      (here)
      → adapter_executor(adapter)   (the bridge)
      → Forge.run

## Unavailable is reported, never substituted

`build_adapter` returns `(adapter, reason)` with `adapter=None` when a backend
cannot be constructed on this machine, and it never raises and never quietly
falls back to a different worker. Silently substituting would be the worst
possible behaviour here: the ledger would attribute the work — and its cost — to
the profile the router chose rather than the one that ran, and every measured
win-rate the router later depends on would be evidence about the wrong worker.

Refusing to build is also not a failure of the task. It is information the caller
needs in order to route somewhere else honestly.
"""

from __future__ import annotations

from collections.abc import Callable

from ..registry import Adapter, WorkerProfile
from .base import WorkerAdapter


def build_adapter(
    profile: WorkerProfile,
    *,
    gateway: object | None = None,
    job_id: str = "",
    remaining_micros: Callable[[], int] | None = None,
    check_health: bool = True,
) -> tuple[WorkerAdapter | None, str]:
    """Construct the backend a profile names.

    Returns `(adapter, reason)`. `adapter` is None when this machine cannot run
    that backend; `reason` always explains, whether or not construction worked.

    `gateway` / `job_id` / `remaining_micros` are only consulted for
    gateway-backed profiles. They are passed rather than constructed here because
    a `Gateway` owns a ledger and a settings object, and an adapter factory that
    invented its own would produce a worker whose spend lands somewhere nobody is
    reading.
    """
    adapter, reason = _construct(
        profile, gateway=gateway, job_id=job_id, remaining_micros=remaining_micros
    )
    if adapter is None:
        return None, reason

    if not check_health:
        return adapter, reason

    # A backend that imports fine but has no binary, no auth, or no transport is
    # not usable, and finding that out at spawn time would waste a lease and a
    # scheduling slot. `health()` is contractually forbidden from raising, but a
    # third-party adapter could still break that promise, so it is guarded.
    try:
        ok, health_reason = adapter.health()
    except Exception as exc:
        return None, f"{profile.worker_id}: health check raised {exc.__class__.__name__}: {exc}"
    if not ok:
        return None, f"{profile.worker_id}: {health_reason}"
    return adapter, health_reason


def _construct(
    profile: WorkerProfile,
    *,
    gateway: object | None,
    job_id: str,
    remaining_micros: Callable[[], int] | None,
) -> tuple[WorkerAdapter | None, str]:
    # Imports are local so that one unimportable backend cannot stop the other
    # backends from being built. A missing optional dependency should cost you
    # that one worker, not the whole fleet.
    if profile.adapter is Adapter.OLLAMA:
        try:
            from .ollama import OllamaAdapter
        except Exception as exc:
            return None, f"ollama adapter unimportable: {exc}"
        model = profile.model or ""
        return (OllamaAdapter(model) if model else OllamaAdapter()), "ollama adapter built"

    if profile.adapter is Adapter.LOCAL_COMMAND:
        if not profile.command:
            return None, f"{profile.worker_id} has no local command configured"
        try:
            from .local_command import LocalCommandAdapter
        except Exception as exc:
            return None, f"local command adapter unimportable: {exc}"
        return LocalCommandAdapter(profile.command, profile.args), "local command adapter built"

    if profile.adapter is Adapter.ACP:
        if not profile.command:
            return None, f"{profile.worker_id} has no ACP vendor CLI command configured"
        try:
            from .acp import ACPAdapter
        except Exception as exc:
            return None, f"acp adapter unimportable: {exc}"
        # Missing SDK / missing vendor CLI are both surfaced by `health()`
        # (the caller runs it right after `_construct`, unless check_health is
        # off) — that is the one place those reasons are worded, so they are
        # not duplicated here.
        return ACPAdapter(profile.command, profile.args), "acp adapter built"

    if profile.adapter is Adapter.CLI_TEAM:
        try:
            from .cli_team import OMCTeamAdapter
        except Exception as exc:
            return None, f"omc team adapter unimportable: {exc}"
        # `vendor` is the backing CLI the omc runtime spawns; `agent_type` is
        # forgeos's role vocabulary. The runtime's --agent flag takes vendor names
        # and rejects roles, so only vendor may be forwarded as the default.
        kwargs = {}
        if profile.vendor:
            kwargs["default_agent_type"] = profile.vendor
        return OMCTeamAdapter(**kwargs), "omc team adapter built"

    if profile.adapter is Adapter.GATEWAY:
        if gateway is None:
            return None, (
                f"{profile.worker_id} needs a Gateway; pass gateway=... "
                "(it owns the ledger the spend is recorded against)"
            )
        if remaining_micros is None:
            return None, (
                f"{profile.worker_id} needs remaining_micros; without a live budget "
                "the preflight cannot refuse an over-budget call"
            )
        try:
            from .gateway_worker import GatewayWorkerAdapter
        except Exception as exc:
            return None, f"gateway worker adapter unimportable: {exc}"
        return (
            GatewayWorkerAdapter(
                gateway,  # type: ignore[arg-type]
                job_id=job_id,
                worker_id=profile.worker_id,
                remaining_micros=remaining_micros,
                default_model_ref=profile.model,
            ),
            "gateway worker adapter built",
        )

    # An enum member with no branch is a profile nobody can run. Saying so beats
    # returning something plausible.
    return None, f"no implementation registered for adapter {profile.adapter.value!r}"


def runnable_workers(
    profiles: list[WorkerProfile],
    *,
    gateway: object | None = None,
    job_id: str = "",
    remaining_micros: Callable[[], int] | None = None,
) -> dict[str, str]:
    """Which registered profiles this machine can actually run, and why not.

    Answers the question `doctor()` should be asking: not "what is in the
    registry" but "what would actually execute if the router picked it". A fleet
    listing workers that cannot run is a fleet that loses tasks at assignment
    time, after a lease and a scheduling slot have already been spent on them.
    """
    out: dict[str, str] = {}
    for p in profiles:
        adapter, reason = build_adapter(
            p, gateway=gateway, job_id=job_id, remaining_micros=remaining_micros
        )
        out[p.worker_id] = ("ok: " if adapter is not None else "unavailable: ") + reason
    return out


__all__ = ["build_adapter", "runnable_workers"]
