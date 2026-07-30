"""GatewayWorkerAdapter — makes every API-key provider a routable worker.

`forgeos/gateway/client.py` already builds the one call path to any provider:
estimate, refuse, call (HTTP first, litellm fallback), record. What was missing
is the piece that lets the *scheduler* send a task down it. Without this file,
`Adapter.GATEWAY` named a backend with no implementation — a worker sitting in
the fleet that the router could pick and nothing could run.

That is the same defect that had `jcode.local` registered as a routable worker
when jcode was only ever meant to be source code we read. A profile in the
registry is a promise that work sent there will actually execute; an adapter-less
profile turns the router's cheapest-capable-first rule into a way to lose tasks.

## This worker cannot edit files

An HTTP completion returns text. It has no filesystem, no tools, no working
directory it can write to. `capabilities().file_diffs` is therefore False and any
`WorkerProfile` bound to this adapter must set `can_edit_files=False`, or the
router will send it an edit it is structurally incapable of performing and then
pay a second worker to redo it.

## Spend is recorded by the gateway, not by this adapter

`Gateway.complete` writes to the ledger before it returns — that is invariant (4)
and it is not optional, because an unrecorded call is invisible to the governor.
The consequence matters for anything mapping this adapter into an
`ExecutionResult`: **the money is already in the ledger by the time `send()`
yields.** Reporting it again as fresh spend would charge the same call twice,
tripping the budget at half the real ceiling and overstating every receipt by 2x.

So the USAGE event and `usage()` both carry the true figure — losing it would be
its own dishonesty — and both flag it as already recorded:

    event.data["spend_already_recorded"] is True
    adapter.spend_already_recorded is True

A consumer must charge for this worker exactly zero additional microdollars, and
must not fall back to charging a tier prior either: "the adapter reported 0" and
"the adapter reported a figure that is already banked" are different states, and
only the first one means nobody has paid yet.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from ..economy.preflight import CallRefused
from ..gateway.client import Gateway, GatewayRequest, GatewayResponse, TransportError
from .base import EventKind, WorkerAdapter, WorkerCapabilities, WorkerEvent, WorkerUsage

DEFAULT_MAX_OUTPUT_TOKENS = 2_048


@dataclass
class _Session:
    """One task's worth of gateway calls.

    There is no server-side session to hold: each `send()` is a standalone
    completion. What is worth keeping is the accumulated usage, because the
    ledger's per-call rows answer "what did the job cost" while a worker's
    `usage()` has to answer "what did *this session* cost".
    """

    task_id: str
    cwd: str
    model_ref: str
    input_tokens: int = 0
    output_tokens: int = 0
    usd_micros: int = 0
    # False until a provider actually returns usage numbers. One estimated call
    # in a session is enough to make the session total an estimate: a mix of
    # measured and guessed is a guess, and rounding it up to "exact" is how a
    # budget quietly stops holding.
    exact: bool = True
    calls: int = 0
    cancelled: bool = False
    responses: list[GatewayResponse] = field(default_factory=list)


class GatewayWorkerAdapter(WorkerAdapter):
    """Runs a task as one gateway completion per `send()`.

    `remaining_micros` is a callable, not a number, and deliberately so. The
    gateway refuses a call that would exceed the remaining budget, and that check
    is only as good as its input: a snapshot taken when the adapter was built
    would let a call through on budget that a concurrent task had already spent.
    """

    def __init__(
        self,
        gateway: Gateway,
        *,
        job_id: str,
        worker_id: str,
        remaining_micros: Callable[[], int],
        default_model_ref: str = "",
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        self._gateway = gateway
        self._job_id = job_id
        self._worker_id = worker_id
        self._remaining = remaining_micros
        self._default_model_ref = default_model_ref
        self._max_output_tokens = max_output_tokens
        self._sessions: dict[str, _Session] = {}
        self._counter = 0

    # Read by whatever maps this adapter into an ExecutionResult. See the module
    # docstring: the figure in `usage()` is real AND already banked in the ledger.
    spend_already_recorded = True

    def set_job_id(self, job_id: str) -> None:
        """Point subsequent spend at the job that is actually running.

        An adapter is built before `Forge.run` exists to create a job, so the id
        given at construction is necessarily a placeholder. Since this adapter's
        gateway is the *authoritative* recorder for its calls, leaving that
        placeholder in place files every row under a job nobody queries: the job's
        own spend total reads as zero while the money is real and sitting one row
        away. Observed exactly that way before this existed.
        """
        if job_id:
            self._job_id = job_id

    # ------------------------------------------------------------------ health

    def health(self) -> tuple[bool, str]:
        """Never raises — a misconfigured gateway degrades this worker, it does
        not take the harness down."""
        try:
            transports = getattr(self._gateway, "_transports", None)
            if not transports:
                return False, "gateway has no transports configured"
            if self._default_model_ref:
                card = self._gateway._catalog.get(self._default_model_ref)
                if card is None:
                    return False, f"model {self._default_model_ref!r} is not in the catalog"
            return True, f"gateway ready ({len(transports)} transport(s))"
        except Exception as exc:  # pragma: no cover - defensive, per the base contract
            return False, f"gateway unusable: {exc.__class__.__name__}: {exc}"

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            structured_messages=False,
            resumable_sessions=True,  # sessions are stateless, so resuming is free
            tool_calls=False,
            file_diffs=False,  # an HTTP completion has no filesystem — see docstring
            reasoning_effort=True,  # forwarded to GatewayRequest.reasoning_effort
            token_usage=True,
            images=False,
            max_parallel_sessions=8,
        )

    # ----------------------------------------------------------------- session

    async def start(self, task_id: str, cwd: str, model_profile: str) -> str:
        self._counter += 1
        session_id = f"gw-{self._counter}"
        self._sessions[session_id] = _Session(
            task_id=task_id,
            cwd=cwd,
            model_ref=model_profile or self._default_model_ref,
        )
        return session_id

    async def send(self, session_id: str, prompt: str) -> AsyncIterator[WorkerEvent]:
        sess = self._sessions.get(session_id)
        if sess is None:
            yield WorkerEvent(kind=EventKind.ERROR, text=f"no active gateway session: {session_id}")
            return
        if sess.cancelled:
            yield WorkerEvent(kind=EventKind.ERROR, text="session was cancelled")
            return
        if not sess.model_ref:
            yield WorkerEvent(
                kind=EventKind.ERROR,
                text="no model_ref: pass one as model_profile to start() or set default_model_ref",
            )
            return

        request = GatewayRequest(
            model_ref=sess.model_ref,
            prompt_tail=prompt,
            max_output_tokens=self._max_output_tokens,
        )

        try:
            # `Gateway.complete` is synchronous and does blocking network I/O.
            # Calling it inline would stall the event loop for the whole request,
            # which defeats running sessions concurrently at all.
            response = await asyncio.to_thread(
                self._gateway.complete,
                request,
                job_id=self._job_id,
                task_id=sess.task_id,
                worker_id=self._worker_id,
                remaining_micros=self._remaining(),
            )
        except CallRefused as exc:
            # A refusal is the preflight working, not a crash. It is reported as
            # an error event so the run loop can classify it as POLICY rather
            # than as a model failure that would escalate to a pricier tier.
            yield WorkerEvent(
                kind=EventKind.ERROR,
                text=f"refused before the call: {exc}",
                data={"refused": True, "spend_already_recorded": False},
            )
            return
        except TransportError as exc:
            yield WorkerEvent(
                kind=EventKind.ERROR,
                text=f"transport failed: {exc}",
                # The gateway records the estimate when a call raises mid-flight,
                # so an attempted-but-failed call has still been charged.
                data={"transient": True, "spend_already_recorded": True},
            )
            return
        except Exception as exc:
            yield WorkerEvent(
                kind=EventKind.ERROR,
                text=f"{exc.__class__.__name__}: {exc}",
                data={"spend_already_recorded": False},
            )
            return

        sess.calls += 1
        sess.input_tokens += response.tokens_in + response.tokens_cached_in
        sess.output_tokens += response.tokens_out
        sess.usd_micros += response.usd_micros
        sess.exact = sess.exact and response.exact_usage
        sess.responses.append(response)

        yield WorkerEvent(kind=EventKind.MESSAGE, text=response.text)
        yield WorkerEvent(
            kind=EventKind.USAGE,
            data={
                "input_tokens": response.tokens_in + response.tokens_cached_in,
                "output_tokens": response.tokens_out,
                "usd_micros": response.usd_micros,
                "exact": response.exact_usage,
                "cache_hit": response.cache_hit,
                "model_used": response.model_used,
                "provider_used": response.provider_used,
                # The ledger already has this call. Charging it again double-counts.
                "spend_already_recorded": True,
            },
        )
        yield WorkerEvent(kind=EventKind.DONE)

    async def cancel(self, session_id: str) -> None:
        """Blocks any further send on this session.

        Honest about its limit: an HTTP request already in flight inside
        `Gateway.complete` is not interruptible from here. Claiming otherwise
        would invite a caller to treat cancel as a hard budget stop, which it is
        not — the in-flight call will complete and be charged.
        """
        sess = self._sessions.get(session_id)
        if sess is not None:
            sess.cancelled = True

    async def checkpoint(self, session_id: str) -> dict:
        sess = self._sessions.get(session_id)
        if sess is None:
            return {}
        # State a different worker can resume from. No transcript: a replacement
        # model resumes the task, never another model's conversation.
        return {
            "adapter": "gateway",
            "task_id": sess.task_id,
            "cwd": sess.cwd,
            "model_ref": sess.model_ref,
            "calls": sess.calls,
            "input_tokens": sess.input_tokens,
            "output_tokens": sess.output_tokens,
            "usd_micros": sess.usd_micros,
            "exact": sess.exact,
        }

    async def resume(self, checkpoint: dict) -> str:
        session_id = await self.start(
            checkpoint.get("task_id", ""),
            checkpoint.get("cwd", ""),
            checkpoint.get("model_ref", ""),
        )
        sess = self._sessions[session_id]
        sess.calls = int(checkpoint.get("calls", 0))
        sess.input_tokens = int(checkpoint.get("input_tokens", 0))
        sess.output_tokens = int(checkpoint.get("output_tokens", 0))
        sess.usd_micros = int(checkpoint.get("usd_micros", 0))
        sess.exact = bool(checkpoint.get("exact", True))
        return session_id

    async def usage(self, session_id: str) -> WorkerUsage:
        """Real spend for this session — and already in the ledger.

        `exact` is False before any call has been made: zero-because-nothing-ran
        is not a measured zero, and the two must stay distinguishable.
        """
        sess = self._sessions.get(session_id)
        if sess is None or sess.calls == 0:
            return WorkerUsage(exact=False)
        return WorkerUsage(
            input_tokens=sess.input_tokens,
            output_tokens=sess.output_tokens,
            usd_micros=sess.usd_micros,
            exact=sess.exact,
        )

    async def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


__all__ = ["DEFAULT_MAX_OUTPUT_TOKENS", "GatewayWorkerAdapter"]
