"""Tests for `forgeos/adapters/gateway_worker.py`.

No network and no real `Gateway`: a fake stands in for it, because what needs
proving here is the adapter's honesty about money and failure, not httpx's.

The load-bearing property is the one in the module docstring — the gateway has
already written the spend to the ledger by the time an event is yielded, so this
adapter must report the figure *and* flag it as banked. Getting that wrong in
either direction is a real cost bug: drop the figure and the session looks free,
report it as fresh and the same call is charged twice.
"""

from __future__ import annotations

import asyncio

import pytest

from forgeos.adapters.base import EventKind
from forgeos.adapters.gateway_worker import GatewayWorkerAdapter
from forgeos.economy.preflight import CallEstimate, CallRefused, Decision, PreflightVerdict
from forgeos.gateway.client import GatewayResponse, TransportError


def _refusal(reason: str = "estimated 5000000 > remaining 100") -> CallRefused:
    """A real refusal, built the way the preflight builds one.

    Constructed from an actual `PreflightVerdict` rather than a bare string so the
    test exercises the exception the gateway genuinely raises.
    """
    return CallRefused(
        PreflightVerdict(
            decision=Decision.REFUSE_BUDGET,
            reason=reason,
            estimate=CallEstimate(
                model_ref="openrouter/free-model",
                tokens_in=1_000,
                tokens_out=2_048,
                usd_micros=5_000_000,
                fits_context=True,
                exact=True,
            ),
        )
    )


class _FakeCatalog:
    def __init__(self, known: set[str] | None = None) -> None:
        self._known = known if known is not None else {"openrouter/free-model"}

    def get(self, ref: str):
        return object() if ref in self._known else None


class _FakeGateway:
    """Records what it was asked, returns what it was told to."""

    def __init__(self, responses=None, raises=None, *, transports=(1,), catalog=None):
        self._transports = list(transports)
        self._catalog = catalog if catalog is not None else _FakeCatalog()
        self._responses = list(responses or [])
        self._raises = raises
        self.calls: list[dict] = []

    def complete(self, request, *, job_id, task_id, worker_id, remaining_micros):
        self.calls.append(
            {
                "model_ref": request.model_ref,
                "prompt": request.prompt,
                "job_id": job_id,
                "task_id": task_id,
                "worker_id": worker_id,
                "remaining_micros": remaining_micros,
            }
        )
        if self._raises is not None:
            raise self._raises
        return self._responses.pop(0)


def _response(**kw) -> GatewayResponse:
    base = dict(
        text="done",
        tokens_in=100,
        tokens_out=20,
        tokens_cached_in=0,
        usd_micros=1_500,
        model_used="openrouter/free-model",
        provider_used="openrouter",
        exact_usage=True,
        cache_hit=False,
    )
    base.update(kw)
    return GatewayResponse(**base)


def _adapter(gateway, *, remaining=lambda: 10_000_000, model="openrouter/free-model"):
    return GatewayWorkerAdapter(
        gateway,
        job_id="job-1",
        worker_id="gateway.free",
        remaining_micros=remaining,
        default_model_ref=model,
    )


async def _drain(adapter, session_id, prompt="do the thing"):
    return [ev async for ev in adapter.send(session_id, prompt)]


def _run(coro):
    return asyncio.run(coro)


# ================================================================ happy path


def test_send_yields_message_then_usage_then_done():
    gw = _FakeGateway([_response(text="hello")])
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        return await _drain(ad, sid)

    events = _run(scenario())
    assert [e.kind for e in events] == [EventKind.MESSAGE, EventKind.USAGE, EventKind.DONE]
    assert events[0].text == "hello"


def test_usage_event_flags_the_spend_as_already_recorded():
    """The single most important assertion in this file.

    The gateway writes to the ledger before returning. A consumer that reads this
    event and charges the amount again bills the same call twice.
    """
    gw = _FakeGateway([_response(usd_micros=4_200)])
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        return await _drain(ad, sid)

    usage = next(e for e in _run(scenario()) if e.kind is EventKind.USAGE)
    assert usage.data["usd_micros"] == 4_200
    assert usage.data["spend_already_recorded"] is True
    assert GatewayWorkerAdapter.spend_already_recorded is True


def test_the_call_carries_the_job_task_and_worker_identity_through():
    gw = _FakeGateway([_response()])
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("task-77", "/repo", "openrouter/free-model")
        await _drain(ad, sid, "prompt text")

    _run(scenario())
    assert gw.calls[0]["job_id"] == "job-1"
    assert gw.calls[0]["task_id"] == "task-77"
    assert gw.calls[0]["worker_id"] == "gateway.free"
    assert "prompt text" in gw.calls[0]["prompt"]


def test_remaining_budget_is_read_fresh_on_every_send_not_snapshotted():
    """A stale budget would let a call through on money another task already spent."""
    budget = [10_000_000]
    gw = _FakeGateway([_response(), _response()])
    ad = _adapter(gw, remaining=lambda: budget[0])

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        await _drain(ad, sid)
        budget[0] = 25  # something else spent the budget between calls
        await _drain(ad, sid)

    _run(scenario())
    assert [c["remaining_micros"] for c in gw.calls] == [10_000_000, 25]


# ================================================================== capability


def test_this_worker_declares_that_it_cannot_edit_files():
    """An HTTP completion has no filesystem. Claiming file_diffs would get it
    routed edits it cannot perform, and then paid for twice."""
    ad = _adapter(_FakeGateway())
    caps = ad.capabilities()
    assert caps.file_diffs is False
    assert caps.tool_calls is False
    assert caps.token_usage is True


# ===================================================================== usage


def test_usage_is_not_exact_before_any_call_has_been_made():
    """Zero-because-nothing-ran must stay distinguishable from a measured zero."""
    ad = _adapter(_FakeGateway())

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        return await ad.usage(sid)

    u = _run(scenario())
    assert u.usd_micros == 0
    assert u.exact is False


def test_usage_accumulates_across_calls_in_a_session():
    gw = _FakeGateway([_response(usd_micros=1_000), _response(usd_micros=2_500)])
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        await _drain(ad, sid)
        await _drain(ad, sid)
        return await ad.usage(sid)

    u = _run(scenario())
    assert u.usd_micros == 3_500
    assert u.exact is True


def test_one_estimated_call_makes_the_whole_session_total_an_estimate():
    """A mix of measured and guessed is a guess. Rounding it up to exact is how a
    budget quietly stops holding."""
    gw = _FakeGateway([_response(exact_usage=True), _response(exact_usage=False)])
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        await _drain(ad, sid)
        await _drain(ad, sid)
        return await ad.usage(sid)

    assert _run(scenario()).exact is False


def test_cached_input_is_counted_as_input_not_dropped():
    gw = _FakeGateway([_response(tokens_in=40, tokens_cached_in=160, tokens_out=10)])
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        await _drain(ad, sid)
        return await ad.usage(sid)

    u = _run(scenario())
    assert u.input_tokens == 200
    assert u.output_tokens == 10


# ==================================================================== failure


def test_a_preflight_refusal_is_an_error_event_not_a_raise():
    gw = _FakeGateway(raises=_refusal())
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        return await _drain(ad, sid)

    events = _run(scenario())
    assert [e.kind for e in events] == [EventKind.ERROR]
    assert events[0].data["refused"] is True


def test_a_refused_call_reports_that_nothing_was_spent():
    """The preflight refuses *before* any transport is touched, so no money moved
    and no ledger row exists. Flagging it as already-recorded would make a
    consumer skip a charge that was never made — harmless here, but the same
    field is what stops a real call being charged twice, so it has to mean
    exactly one thing."""
    gw = _FakeGateway(raises=_refusal("too expensive"))
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        return await _drain(ad, sid)

    assert _run(scenario())[0].data["spend_already_recorded"] is False


def test_a_transport_failure_reports_that_the_attempt_was_already_charged():
    """`Gateway.complete` records the estimate when the call raises mid-flight —
    an attempted call cost something even though it produced nothing."""
    gw = _FakeGateway(raises=TransportError("upstream 502"))
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        return await _drain(ad, sid)

    events = _run(scenario())
    assert events[0].kind is EventKind.ERROR
    assert events[0].data["transient"] is True
    assert events[0].data["spend_already_recorded"] is True


def test_an_unexpected_exception_becomes_an_error_event_not_a_crash():
    gw = _FakeGateway(raises=ValueError("something odd"))
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        return await _drain(ad, sid)

    events = _run(scenario())
    assert events[0].kind is EventKind.ERROR
    assert "ValueError" in events[0].text


def test_a_failed_call_does_not_pollute_the_session_usage():
    gw = _FakeGateway(raises=TransportError("boom"))
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        await _drain(ad, sid)
        return await ad.usage(sid)

    u = _run(scenario())
    assert u.usd_micros == 0
    assert u.exact is False


def test_send_on_an_unknown_session_is_an_error_event_not_a_keyerror():
    ad = _adapter(_FakeGateway())
    events = _run(_drain(ad, "nope"))
    assert events[0].kind is EventKind.ERROR
    assert "nope" in events[0].text


def test_send_without_any_model_ref_refuses_before_calling_the_gateway():
    gw = _FakeGateway([_response()])
    ad = _adapter(gw, model="")

    async def scenario():
        sid = await ad.start("t1", "/repo", "")
        return await _drain(ad, sid)

    events = _run(scenario())
    assert events[0].kind is EventKind.ERROR
    assert gw.calls == []


# ================================================================ lifecycle


def test_cancel_blocks_any_further_send_on_that_session():
    gw = _FakeGateway([_response()])
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        await ad.cancel(sid)
        return await _drain(ad, sid)

    events = _run(scenario())
    assert events[0].kind is EventKind.ERROR
    assert gw.calls == []


def test_checkpoint_carries_state_and_no_transcript():
    gw = _FakeGateway([_response(usd_micros=900)])
    ad = _adapter(gw)

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        await _drain(ad, sid)
        return await ad.checkpoint(sid)

    cp = _run(scenario())
    assert cp["task_id"] == "t1"
    assert cp["usd_micros"] == 900
    banned = {"transcript", "messages", "history", "conversation", "text", "prompt"}
    assert not banned & set(cp), f"checkpoint leaks conversation state: {banned & set(cp)}"


def test_resume_restores_the_accumulated_spend():
    ad = _adapter(_FakeGateway())

    async def scenario():
        sid = await ad.resume(
            {
                "task_id": "t9",
                "cwd": "/repo",
                "model_ref": "openrouter/free-model",
                "calls": 2,
                "input_tokens": 500,
                "output_tokens": 60,
                "usd_micros": 7_777,
                "exact": False,
            }
        )
        return await ad.usage(sid)

    u = _run(scenario())
    assert u.usd_micros == 7_777
    assert u.exact is False


def test_close_is_idempotent():
    ad = _adapter(_FakeGateway())

    async def scenario():
        sid = await ad.start("t1", "/repo", "openrouter/free-model")
        await ad.close(sid)
        await ad.close(sid)
        return await ad.usage(sid)

    assert _run(scenario()).exact is False


# =================================================================== health


def test_health_is_false_when_the_gateway_has_no_transports():
    ad = _adapter(_FakeGateway(transports=()))
    ok, reason = ad.health()
    assert ok is False
    assert "transport" in reason


def test_health_is_false_when_the_model_is_not_in_the_catalog():
    ad = _adapter(_FakeGateway(catalog=_FakeCatalog(known=set())))
    ok, reason = ad.health()
    assert ok is False
    assert "catalog" in reason


def test_health_never_raises_even_on_a_broken_gateway():
    class Exploding:
        @property
        def _transports(self):
            raise RuntimeError("gateway is broken")

    ok, reason = _adapter(Exploding()).health()
    assert ok is False
    assert "unusable" in reason


def test_health_is_true_for_a_working_gateway():
    ok, reason = _adapter(_FakeGateway()).health()
    assert ok is True
    assert "ready" in reason


# ========================================================== registry contract


def test_no_registered_profile_claims_this_adapter_can_edit_files():
    """Guards the pairing the docstring warns about: routing an edit to a worker
    that returns text and cannot touch the filesystem."""
    from forgeos.registry import Adapter, default_registry

    offenders = [
        w.worker_id
        for w in default_registry().all()
        if w.adapter is Adapter.GATEWAY and w.can_edit_files
    ]
    assert not offenders, f"gateway-backed profiles cannot edit files: {offenders}"


@pytest.mark.parametrize("method", ["start", "send", "cancel", "checkpoint", "resume",
                                    "usage", "close", "health", "capabilities"])
def test_the_full_worker_contract_is_implemented(method):
    assert callable(getattr(GatewayWorkerAdapter, method, None))
