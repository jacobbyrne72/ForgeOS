"""Job-scoped cache affinity: pinning a job's later calls to the seat its
first successful call actually used, so a warm provider-side cache prefix
(see forgeos/prompts/prefix.py and `_content_blocks`'s `cache_control`
breakpoint in forgeos/gateway/client.py) doesn't go unused because a later
call in the same job independently re-resolved its transport.

The pin is a PREFERENCE, never a cage: every test below that exercises a
release rule sets up a scenario where, without the pin, the natural
health/dead-model-filtered order would prefer a DIFFERENT transport than the
pinned one -- so a passing assertion proves the release rule actually fired,
not just that the naturally-preferred transport happened to win anyway.

Transport is always a scripted double, never real HTTP or litellm -- same
convention as test_gateway.py / test_gateway_resilience.py.
"""

from __future__ import annotations

import pytest

from forgeos.catalog import Catalog, ModelCard
from forgeos.core.quota import QuotaTracker
from forgeos.gateway.client import (
    AFFINITY_TTL_SECONDS,
    Gateway,
    GatewayRequest,
    ModelUnavailableError,
    RawCallResult,
    TransportError,
)
from forgeos.gateway.dead_models import DeadModelStore
from forgeos.gateway.health import HealthTracker
from forgeos.ledger import Ledger

CARD = ModelCard(
    model_id="test-model",
    provider="testprov",
    input_cost_per_1m=1.0,
    output_cost_per_1m=2.0,
    context=1000,
)


def _request(**overrides):
    kwargs = dict(model_ref="testprov/test-model", prompt_prefix="p", prompt_tail="", max_output_tokens=50)
    kwargs.update(overrides)
    return GatewayRequest(**kwargs)


def _catalog() -> Catalog:
    return Catalog([CARD])


@pytest.fixture()
def ledger():
    led = Ledger(":memory:")
    yield led
    led.close()


class FakeTransport:
    """A transport double whose behavior is fully scripted; never touches a network."""

    def __init__(self, name: str, result: RawCallResult | None = None, error: Exception | None = None):
        self.name = name
        self._result = result
        self._error = error
        self.calls: list[dict] = []

    def complete(self, *, model_id, prompt, max_output_tokens, reasoning_effort, tools_schema):
        self.calls.append(dict(model_id=model_id))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class SwitchableTransport:
    """A transport double whose failure can be toggled between calls, so a
    test can prove what a later, unconstrained call WOULD have preferred
    without the pin in play -- rather than merely observing a transport that
    was excluded (or included) for an unrelated reason the whole time.
    """

    def __init__(self, name: str, *, fail_error: Exception | None = None):
        self.name = name
        self.should_fail = False
        self.calls = 0
        self._fail_error = fail_error or TransportError(f"{name} down")

    def complete(self, **kwargs):
        self.calls += 1
        if self.should_fail:
            raise self._fail_error
        return RawCallResult(text=f"from {self.name}", tokens_in=1, tokens_out=1)


# ------------------------------------------------------- byte-identical default


def test_no_affinity_key_leaves_affinity_state_and_response_untouched(ledger):
    """Omitting `affinity_key` (the default) must behave exactly as it did
    before cache affinity existed: nothing recorded, nothing preferred."""
    transport = FakeTransport("only", result=RawCallResult(text="hi", tokens_in=10, tokens_out=5))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request()

    resp = gw.complete(req, job_id="j-plain", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert resp.affinity_applied is False
    assert gw._affinity == {}
    assert len(transport.calls) == 1


# ----------------------------------------------------------------- preference


def test_second_call_with_same_key_prefers_the_first_calls_seat(ledger):
    """`hot` is deliberately made to recover between calls so it would
    naturally win call 2 again on its own merits (list order, zero recorded
    latency) -- the only thing that can keep `cool` in the seat is the pin
    from call 1."""
    hot = SwitchableTransport("hot")
    hot.should_fail = True
    cool = FakeTransport("cool", result=RawCallResult(text="from cool", tokens_in=1, tokens_out=1))
    health = HealthTracker()
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[hot, cool], health=health)
    req = _request()

    resp1 = gw.complete(
        req, job_id="j1", task_id=None, worker_id="w1", remaining_micros=1_000_000, affinity_key="job-x"
    )
    assert resp1.provider_used == "cool"
    assert resp1.affinity_applied is False, "nothing to prefer yet on a key's first call"

    hot.should_fail = False
    health.record_success("hot", latency_ms=0.0)  # hot "recovers" -- plain failures don't self-heal

    resp2 = gw.complete(
        req, job_id="j2", task_id=None, worker_id="w1", remaining_micros=1_000_000, affinity_key="job-x"
    )

    assert resp2.provider_used == "cool"
    assert resp2.affinity_applied is True
    assert hot.calls == 1, "hot's only call is the failed one from call 1 -- the pin kept it out of rotation"


# -------------------------------------------------------------- release rules


def test_terminally_dead_seat_drops_the_pin_without_forcing_it(ledger):
    seatX = FakeTransport("seatX", result=RawCallResult(text="x", tokens_in=1, tokens_out=1))
    seatY = FakeTransport("seatY", result=RawCallResult(text="y", tokens_in=1, tokens_out=1))
    dead_models = DeadModelStore()
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[seatX, seatY], dead_models=dead_models)
    req = _request()

    resp1 = gw.complete(
        req, job_id="j1", task_id=None, worker_id="w1", remaining_micros=1_000_000, affinity_key="job-dead"
    )
    assert resp1.provider_used == "seatX"  # natural tie -> first in the transport list

    # seatX goes terminally dead for this exact model between calls (a 404,
    # say). Existing dead-model filtering already excludes it from
    # `candidates` -- affinity must never fight that exclusion.
    dead_models.mark_dead("seatX", CARD.ref, reason="retired")

    resp2 = gw.complete(
        req, job_id="j2", task_id=None, worker_id="w1", remaining_micros=1_000_000, affinity_key="job-dead"
    )
    assert resp2.provider_used == "seatY"
    assert resp2.affinity_applied is False


def test_still_cooling_seat_drops_the_pin_without_forcing_it(ledger):
    seatX = FakeTransport("seatX", result=RawCallResult(text="x", tokens_in=1, tokens_out=1))
    seatY = FakeTransport("seatY", result=RawCallResult(text="y", tokens_in=1, tokens_out=1))
    dead_models = DeadModelStore(clock=lambda: 1_000.0)
    gw = Gateway(
        catalog=_catalog(), ledger=ledger, transports=[seatX, seatY],
        dead_models=dead_models, clock=lambda: 1_000.0,
    )
    req = _request()

    resp1 = gw.complete(
        req, job_id="j1", task_id=None, worker_id="w1", remaining_micros=1_000_000, affinity_key="job-cool"
    )
    assert resp1.provider_used == "seatX"

    # seatX enters a temporary cooldown between calls -- still inside its own
    # retry_after window, not even HALF_OPEN yet.
    dead_models.mark_dead("seatX", CARD.ref, reason="glitch", retry_after=5_000.0)

    resp2 = gw.complete(
        req, job_id="j2", task_id=None, worker_id="w1", remaining_micros=1_000_000, affinity_key="job-cool"
    )
    assert resp2.provider_used == "seatY"
    assert resp2.affinity_applied is False


def test_exhausted_quota_drops_the_pin(ledger):
    hot = SwitchableTransport("hot")
    hot.should_fail = True
    cool = FakeTransport("cool", result=RawCallResult(text="from cool", tokens_in=1, tokens_out=1))
    health = HealthTracker()
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[hot, cool], health=health)
    req = _request()
    quota = QuotaTracker()

    resp1 = gw.complete(
        req, job_id="j1", task_id=None, worker_id="w1",
        remaining_micros=1_000_000, affinity_key="job-q", quota=quota,
    )
    assert resp1.provider_used == "cool"

    hot.should_fail = False
    health.record_success("hot", latency_ms=0.0)  # hot would naturally win call 2 again
    quota.record_exhaustion(CARD.provider, model=CARD.ref)  # but its provider's window is out

    resp2 = gw.complete(
        req, job_id="j2", task_id=None, worker_id="w1",
        remaining_micros=1_000_000, affinity_key="job-q", quota=quota,
    )
    assert resp2.provider_used == "hot", "quota exhaustion dropped the pin; natural order won instead"
    assert resp2.affinity_applied is False


def test_low_headroom_drops_the_pin(ledger):
    hot = SwitchableTransport("hot")
    hot.should_fail = True
    cool = FakeTransport("cool", result=RawCallResult(text="from cool", tokens_in=1, tokens_out=1))
    health = HealthTracker()
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[hot, cool], health=health)
    req = _request()
    job_id = "j-headroom"

    resp1 = gw.complete(
        req, job_id=job_id, task_id=None, worker_id="w1",
        remaining_micros=1_000_000, affinity_key=job_id,
    )
    assert resp1.provider_used == "cool"

    hot.should_fail = False
    health.record_success("hot", latency_ms=0.0)

    # Inflate the job's recorded spend directly so a modest remaining budget
    # on call 2 is well under AFFINITY_MIN_HEADROOM_PCT (0.15) of the total --
    # cheaper and more precise than accumulating it through real call costs.
    ledger.record_spend(job_id, "w1", "synthetic-prior-spend", 900_000)

    resp2 = gw.complete(
        req, job_id=job_id, task_id=None, worker_id="w1",
        remaining_micros=50_000, affinity_key=job_id,  # 50k / (900k+50k) ~= 5.3% headroom
    )
    assert resp2.provider_used == "hot", "low headroom dropped the pin; natural order won instead"
    assert resp2.affinity_applied is False


def test_a_different_model_ref_is_not_affected_by_an_older_pin(ledger):
    """The pin is scoped to (transport, model_ref), which is also the
    mechanism by which a router's own tier/model decision always overrides
    affinity: Gateway has no tier concept of its own (tier/model selection
    happens upstream, in Router, before `complete` is ever called), so the
    only way affinity could contradict that decision is by leaking a
    preference across model_refs. It must not."""
    card_a = ModelCard(model_id="model-a", provider="testprov",
                        input_cost_per_1m=1.0, output_cost_per_1m=2.0, context=1000)
    card_b = ModelCard(model_id="model-b", provider="testprov",
                        input_cost_per_1m=1.0, output_cost_per_1m=2.0, context=1000)
    catalog = Catalog([card_a, card_b])

    seatA = SwitchableTransport("seatA", fail_error=ModelUnavailableError("seatA doesn't have model-a"))
    seatA.should_fail = True
    seatB = FakeTransport("seatB", result=RawCallResult(text="b", tokens_in=1, tokens_out=1))
    health = HealthTracker()
    # Give seatA a healthy, zero-latency track record up front so, once its
    # model-a failure is over, it is what pick_healthy naturally prefers for
    # a DIFFERENT model -- otherwise "never yet measured" would rank it
    # below whatever answered call 1, and the test couldn't tell correct
    # (scoped) behaviour apart from a pin that incorrectly leaked across
    # model_refs (both would land on the same transport).
    health.record_success("seatA", latency_ms=0.0)
    gw = Gateway(catalog=catalog, ledger=ledger, transports=[seatA, seatB], health=health)

    req_a = GatewayRequest(model_ref="testprov/model-a", prompt_prefix="p", max_output_tokens=50)
    resp1 = gw.complete(
        req_a, job_id="j1", task_id=None, worker_id="w1", remaining_micros=1_000_000, affinity_key="job-x"
    )
    assert resp1.provider_used == "seatB"  # seatA is model-unavailable for model-a; pin becomes (seatB, model-a)

    seatA.should_fail = False  # seatA is fine in general -- just was never asked about model-b
    # seatB's real (nonzero) latency from answering call 1 would otherwise
    # tie or beat seatA's -- pin it deliberately worse so seatA's zero-latency
    # record is the unambiguous natural winner for call 2.
    health.record_success("seatB", latency_ms=50.0)

    req_b = GatewayRequest(model_ref="testprov/model-b", prompt_prefix="p", max_output_tokens=50)
    resp2 = gw.complete(
        req_b, job_id="j2", task_id=None, worker_id="w1", remaining_micros=1_000_000, affinity_key="job-x"
    )
    # The (seatB, model-a) pin must not apply to model-b: natural health
    # ordering (seatA's better record) wins instead.
    assert resp2.provider_used == "seatA"
    assert resp2.affinity_applied is False


# ------------------------------------------------------------------- bounding


def test_affinity_pin_expires_after_its_ttl(ledger):
    hot = SwitchableTransport("hot")
    hot.should_fail = True
    cool = FakeTransport("cool", result=RawCallResult(text="from cool", tokens_in=1, tokens_out=1))
    health = HealthTracker()
    clock = {"t": 0.0}
    gw = Gateway(
        catalog=_catalog(), ledger=ledger, transports=[hot, cool],
        health=health, clock=lambda: clock["t"],
    )
    req = _request()

    gw.complete(
        req, job_id="j1", task_id=None, worker_id="w1", remaining_micros=1_000_000, affinity_key="job-ttl"
    )
    hot.should_fail = False
    health.record_success("hot", latency_ms=0.0)

    clock["t"] += AFFINITY_TTL_SECONDS + 1

    resp = gw.complete(
        req, job_id="j2", task_id=None, worker_id="w1", remaining_micros=1_000_000, affinity_key="job-ttl"
    )
    assert resp.provider_used == "hot", "an expired pin must not be preferred"
    assert resp.affinity_applied is False


def test_affinity_map_is_bounded_by_max_entries(ledger, monkeypatch):
    import forgeos.gateway.client as client_mod

    monkeypatch.setattr(client_mod, "_MAX_AFFINITY_ENTRIES", 3)

    transport = FakeTransport("only", result=RawCallResult(text="hi", tokens_in=1, tokens_out=1))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request()

    for i in range(10):
        gw.complete(
            req, job_id=f"j{i}", task_id=None, worker_id="w1",
            remaining_micros=1_000_000, affinity_key=f"key-{i}",
        )

    assert len(gw._affinity) <= 3
