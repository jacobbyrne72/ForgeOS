"""Resilience-feature tests for forgeos/gateway:

- a ledger billing gap where a response that fails to price after a
  *successful* transport call was left unrecorded (invariant 4 broken)
- proactive rate-limit saturation parsed from response headers, feeding
  transport ordering so a nearly-exhausted provider is deprioritised before
  it starts refusing
- terminal-vs-temporary memory of dead (transport, model_ref) pairs, so a
  retired free-tier slug is not rediscovered at the cost of a network call
  on every run
- concurrent identical-request dedup, opt-in only, sharing one upstream call
  and one ledger entry across waiters without ever bypassing a caller's own
  budget check

Transport is always a scripted double, or (for the one test that proves
`HttpTransport` itself parses headers) a directly-constructed `httpx.Response`
-- never a real network call. Same convention as test_gateway.py /
test_gateway_transports.py.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from forgeos.catalog import Catalog, ModelCard
from forgeos.economy.avoidance import AvoidanceLog
from forgeos.economy.preflight import CallRefused
from forgeos.gateway.client import (
    Gateway,
    GatewayRequest,
    HttpTransport,
    ModelUnavailableError,
    RateLimitError,
    RawCallResult,
    TransportError,
    rate_limit_saturation,
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


class BlockingTransport:
    """Forces genuine thread overlap for dedup tests -- no sleep-based timing
    races. By default every call blocks on `release`; with
    `block_every_call=False` only the first call blocks, so a later,
    non-concurrent call (e.g. a timeout fallback) returns immediately.
    """

    def __init__(self, name: str, result=None, error=None, block_every_call: bool = True):
        self.name = name
        self._result = result
        self._error = error
        self._block_every_call = block_every_call
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def complete(self, *, model_id, prompt, max_output_tokens, reasoning_effort, tools_schema):
        self.calls += 1
        if self._block_every_call or self.calls == 1:
            self.started.set()
            self.release.wait(timeout=5)
        if self._error is not None:
            raise self._error
        return self._result


# =============================================================== billing gap
# (reported by the coordinator: `_to_response` raising after a successful
# transport call left that call's spend unrecorded)


def test_spend_recorded_when_response_parsing_fails_after_a_successful_call(ledger):
    """The upstream call already happened -- and already cost money -- even
    though turning its result into a priced `GatewayResponse` is what raised.
    Must be charged exactly like a transport failure is."""
    transport = FakeTransport("clean", result=RawCallResult(text="hi", tokens_in=100, tokens_out=20))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    gw._to_response = lambda *a, **kw: (_ for _ in ()).throw(ValueError("pricing exploded"))
    req = _request()

    with pytest.raises(ValueError, match="pricing exploded"):
        gw.complete(
            req, job_id="j-parsefail", task_id="t-parsefail", worker_id="w1", remaining_micros=1_000_000
        )

    assert ledger.job_spend_micros("j-parsefail") > 0
    assert ledger.task_spend_micros("t-parsefail") > 0


def test_spend_still_recorded_exactly_once_on_the_happy_path(ledger):
    """Regression guard on the fix above: the normal success path must record
    the response's cost exactly once, not the estimate and then the response."""
    transport = FakeTransport("clean", result=RawCallResult(text="hi", tokens_in=100, tokens_out=20))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])

    resp = gw.complete(_request(), job_id="j-once", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert ledger.job_spend_micros("j-once") == resp.usd_micros


# ============================================== feature A: saturation parsing


def test_saturation_from_anthropic_token_headers():
    headers = {
        "anthropic-ratelimit-tokens-limit": "1000",
        "anthropic-ratelimit-tokens-remaining": "250",
    }
    assert rate_limit_saturation(headers) == pytest.approx(0.75)


def test_saturation_from_openai_style_token_headers():
    headers = {"x-ratelimit-limit-tokens": "500", "x-ratelimit-remaining-tokens": "500"}
    assert rate_limit_saturation(headers) == pytest.approx(0.0)


def test_saturation_falls_back_to_anthropic_input_output_token_headers():
    headers = {
        "anthropic-ratelimit-input-tokens-limit": "200",
        "anthropic-ratelimit-input-tokens-remaining": "50",
    }
    assert rate_limit_saturation(headers) == pytest.approx(0.75)


def test_saturation_falls_back_to_request_headers_when_no_token_headers():
    headers = {"x-ratelimit-limit-requests": "60", "x-ratelimit-remaining-requests": "3"}
    assert rate_limit_saturation(headers) == pytest.approx(1 - 3 / 60)


def test_saturation_prefers_token_headers_over_request_headers():
    headers = {
        "anthropic-ratelimit-tokens-limit": "1000",
        "anthropic-ratelimit-tokens-remaining": "999",  # ~0.001 saturation
        "x-ratelimit-limit-requests": "10",
        "x-ratelimit-remaining-requests": "0",  # would be 1.0 saturation alone
    }
    assert rate_limit_saturation(headers) < 0.01


def test_saturation_is_none_when_headers_carry_nothing_usable():
    assert rate_limit_saturation({}) is None
    assert rate_limit_saturation({"content-type": "application/json"}) is None


def test_saturation_is_none_when_limit_is_zero_or_malformed():
    assert rate_limit_saturation(
        {"x-ratelimit-limit-tokens": "0", "x-ratelimit-remaining-tokens": "0"}
    ) is None
    assert rate_limit_saturation(
        {"x-ratelimit-limit-tokens": "not-a-number", "x-ratelimit-remaining-tokens": "5"}
    ) is None


def test_saturation_is_clamped_when_remaining_exceeds_limit():
    """A provider bug or a race between the header snapshot and reality must
    not produce a negative (or >1) saturation value."""
    headers = {"x-ratelimit-limit-tokens": "100", "x-ratelimit-remaining-tokens": "150"}
    assert rate_limit_saturation(headers) == 0.0


def test_httptransport_populates_saturation_from_a_real_response(monkeypatch):
    """Proves the wiring through `HttpTransport.complete` end to end, using a
    directly-constructed `httpx.Response` -- object construction, no network."""
    fake_response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ok"}}],
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        headers={"x-ratelimit-limit-tokens": "1000", "x-ratelimit-remaining-tokens": "100"},
    )
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: fake_response)

    transport = HttpTransport(base_url="https://example.invalid", name="fakehttp")
    raw = transport.complete(
        model_id="test-model", prompt="hi", max_output_tokens=10, reasoning_effort="none", tools_schema=None
    )

    assert raw.rate_limit_saturation == pytest.approx(0.9)


# ======================================== feature A: HealthTracker ordering


def test_health_tracker_saturation_is_none_until_recorded():
    assert HealthTracker().saturation("x") is None


def test_health_tracker_ignores_a_none_saturation_recording():
    health = HealthTracker()
    health.record_saturation("x", 0.8)
    health.record_saturation("x", None)  # must not overwrite a real reading with "unknown"
    assert health.saturation("x") == pytest.approx(0.8)


def test_health_tracker_saturation_reading_goes_stale():
    clock = {"t": 1_000.0}
    health = HealthTracker(clock=lambda: clock["t"])
    health.record_saturation("x", 0.9)
    assert health.saturation("x") == pytest.approx(0.9)

    clock["t"] += 121  # past the 120s TTL
    assert health.saturation("x") is None


def test_pick_healthy_deprioritizes_the_more_saturated_transport_without_excluding_it():
    health = HealthTracker()
    health.record_saturation("hot", 0.95)
    health.record_saturation("cool", 0.05)

    order = health.pick_healthy(["hot", "cool"])

    assert order == ["cool", "hot"]  # deprioritised
    assert set(order) == {"hot", "cool"}  # never excluded outright


def test_pick_healthy_never_penalizes_a_transport_with_no_saturation_data():
    """Missing headers must mean unknown, never saturated -- a provider that
    publishes nothing must not be sorted into permanent last place for it."""
    health = HealthTracker()
    health.record_saturation("chatty", 0.9)
    order = health.pick_healthy(["chatty", "quiet"])
    assert order == ["quiet", "chatty"]


# ==================================== feature A: through Gateway end to end


def test_gateway_prefers_the_less_saturated_transport(ledger):
    hot = FakeTransport("hot", result=RawCallResult(text="from hot", tokens_in=1, tokens_out=1))
    cool = FakeTransport("cool", result=RawCallResult(text="from cool", tokens_in=1, tokens_out=1))
    health = HealthTracker()
    health.record_saturation("hot", 0.95)
    health.record_saturation("cool", 0.02)
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[hot, cool], health=health)

    resp = gw.complete(_request(), job_id="j-sat", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert resp.text == "from cool"
    assert len(cool.calls) == 1
    assert len(hot.calls) == 0


def test_gateway_still_tries_a_saturated_transport_when_it_is_the_only_one(ledger):
    hot = FakeTransport("hot", result=RawCallResult(text="from hot", tokens_in=1, tokens_out=1))
    health = HealthTracker()
    health.record_saturation("hot", 0.99)
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[hot], health=health)

    resp = gw.complete(
        _request(), job_id="j-sat-only", task_id=None, worker_id="w1", remaining_micros=1_000_000
    )

    assert resp.text == "from hot"


def test_gateway_records_saturation_from_a_successful_call(ledger):
    transport = FakeTransport(
        "http", result=RawCallResult(text="hi", tokens_in=1, tokens_out=1, rate_limit_saturation=0.42)
    )
    health = HealthTracker()
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport], health=health)

    gw.complete(_request(), job_id="j-record-sat", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert health.saturation("http") == pytest.approx(0.42)


def test_gateway_records_no_saturation_when_transport_reports_none(ledger):
    transport = FakeTransport("http", result=RawCallResult(text="hi", tokens_in=1, tokens_out=1))
    health = HealthTracker()
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport], health=health)

    gw.complete(_request(), job_id="j-no-sat", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert health.saturation("http") is None


# ==================================================== feature B: dead models


def test_dead_model_store_terminal_entry_never_self_heals():
    clock = {"t": 1_000.0}
    store = DeadModelStore(clock=lambda: clock["t"])
    store.mark_dead("openrouter", "openrouter/gone:free", reason="404")

    assert store.is_dead("openrouter", "openrouter/gone:free") is True
    clock["t"] += 10_000_000
    assert store.is_dead("openrouter", "openrouter/gone:free") is True


def test_dead_model_store_temporary_entry_self_heals_after_retry_after():
    clock = {"t": 1_000.0}
    store = DeadModelStore(clock=lambda: clock["t"])
    store.mark_dead("openrouter", "openrouter/flaky", reason="temporary glitch", retry_after=1_060.0)

    assert store.is_dead("openrouter", "openrouter/flaky") is True
    clock["t"] = 1_061.0
    assert store.is_dead("openrouter", "openrouter/flaky") is False


def test_dead_model_store_unknown_pair_is_not_dead():
    assert DeadModelStore().is_dead("openrouter", "openrouter/never-seen") is False


def test_dead_model_store_clear_resets_a_terminal_entry():
    store = DeadModelStore()
    store.mark_dead("openrouter", "openrouter/gone", reason="404")
    assert store.is_dead("openrouter", "openrouter/gone") is True

    store.clear("openrouter", "openrouter/gone")
    assert store.is_dead("openrouter", "openrouter/gone") is False


def test_dead_model_store_is_scoped_per_transport():
    store = DeadModelStore()
    store.mark_dead("openrouter", "shared/model", reason="404 on openrouter specifically")
    assert store.is_dead("openrouter", "shared/model") is True
    assert store.is_dead("omniroute", "shared/model") is False


def test_dead_model_store_persists_across_reopening_the_same_file(tmp_path):
    """The actual point of this feature: survives a process restart."""
    db_path = tmp_path / "dead_models.db"
    store = DeadModelStore(db_path)
    store.mark_dead("openrouter", "openrouter/gone:free", reason="404: retired")
    store.close()

    reopened = DeadModelStore(db_path)
    try:
        assert reopened.is_dead("openrouter", "openrouter/gone:free") is True
        entry = reopened.get("openrouter", "openrouter/gone:free")
        assert entry is not None
        assert entry.terminal is True
        assert "404" in entry.reason
    finally:
        reopened.close()


# ========================================= feature B: through Gateway end to end


def _dead_card() -> ModelCard:
    return ModelCard(model_id="gone:free", provider="openrouter", input_cost_per_1m=0.0,
                      output_cost_per_1m=0.0, context=1000)


def test_gateway_marks_a_model_dead_after_a_model_unavailable_error(ledger):
    dead = FakeTransport("openrouter", error=ModelUnavailableError("openrouter HTTP 404: gone"))
    gw = Gateway(catalog=Catalog([_dead_card()]), ledger=ledger, transports=[dead])
    req = GatewayRequest(model_ref="openrouter/gone:free", prompt_tail="hi", max_output_tokens=10)

    with pytest.raises(TransportError):
        gw.complete(req, job_id="j-dead1", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert gw._dead_models.is_dead("openrouter", "openrouter/gone:free") is True


def test_gateway_skips_a_known_dead_model_without_touching_the_transport_again(ledger):
    dead = FakeTransport("openrouter", error=ModelUnavailableError("404"))
    gw = Gateway(catalog=Catalog([_dead_card()]), ledger=ledger, transports=[dead])
    req = GatewayRequest(model_ref="openrouter/gone:free", prompt_tail="hi", max_output_tokens=10)

    with pytest.raises(TransportError):
        gw.complete(req, job_id="j-dead2a", task_id=None, worker_id="w1", remaining_micros=1_000_000)
    assert len(dead.calls) == 1

    with pytest.raises(ModelUnavailableError):
        gw.complete(req, job_id="j-dead2b", task_id=None, worker_id="w1", remaining_micros=1_000_000)
    assert len(dead.calls) == 1, "no second network attempt for a model already known dead"


def test_a_model_dead_on_one_transport_still_succeeds_via_another(ledger):
    dead = FakeTransport("openrouter", error=ModelUnavailableError("404"))
    fanout = FakeTransport("omniroute", result=RawCallResult(text="ok", tokens_in=1, tokens_out=1))
    card = ModelCard(model_id="m", provider="openrouter", input_cost_per_1m=0.0,
                      output_cost_per_1m=0.0, context=1000)
    gw = Gateway(catalog=Catalog([card]), ledger=ledger, transports=[dead, fanout])
    req = GatewayRequest(model_ref="openrouter/m", prompt_tail="hi", max_output_tokens=10)

    resp1 = gw.complete(req, job_id="j-fan1", task_id=None, worker_id="w1", remaining_micros=1_000_000)
    assert resp1.text == "ok"
    assert gw._dead_models.is_dead("openrouter", "openrouter/m") is True
    assert gw._dead_models.is_dead("omniroute", "openrouter/m") is False

    resp2 = gw.complete(req, job_id="j-fan2", task_id=None, worker_id="w1", remaining_micros=1_000_000)
    assert resp2.text == "ok"
    assert len(dead.calls) == 1, "openrouter tried once, ever"
    assert len(fanout.calls) == 2, "omniroute tried both times"


# ==================================== feature B2: half-open probe wiring
# (`_call_transports` claiming/resolving a HALF_OPEN trial for a TEMPORARY
# dead entry whose `retry_after` has passed -- `DeadModelStore.claim_probe`/
# `report_probe` themselves are covered in test_dead_models_probe.py; these
# prove Gateway actually calls them at the right moments.)


def _gateway_with_clock(transports, ledger, dead_models, t=1_000.0):
    clock = {"t": t}
    gw = Gateway(
        catalog=Catalog([_dead_card()]), ledger=ledger, transports=transports,
        dead_models=dead_models, clock=lambda: clock["t"],
    )
    return gw, clock


def test_gateway_skips_a_still_cooling_down_temporary_entry_without_claiming(ledger):
    """`retry_after` in the future: not HALF_OPEN yet, so no claim attempt at
    all -- the transport is never touched and no probe slot is spent."""
    dead_models = DeadModelStore(clock=lambda: 1_000.0)
    dead_models.mark_dead("openrouter", "openrouter/gone:free", reason="glitch", retry_after=2_000.0)
    transport = FakeTransport("openrouter", result=RawCallResult(text="hi", tokens_in=1, tokens_out=1))
    gw, _ = _gateway_with_clock([transport], ledger, dead_models, t=1_000.0)
    req = GatewayRequest(model_ref="openrouter/gone:free", prompt_tail="hi", max_output_tokens=10)

    with pytest.raises(ModelUnavailableError):
        gw.complete(req, job_id="j-cooldown", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert len(transport.calls) == 0
    assert dead_models.get("openrouter", "openrouter/gone:free").probe_claimed_at is None


def test_gateway_claims_and_clears_a_successful_half_open_probe(ledger):
    dead_models = DeadModelStore(clock=lambda: 2_000.0)
    dead_models.mark_dead("openrouter", "openrouter/gone:free", reason="glitch", retry_after=2_000.0)
    transport = FakeTransport("openrouter", result=RawCallResult(text="back", tokens_in=1, tokens_out=1))
    gw, _ = _gateway_with_clock([transport], ledger, dead_models, t=2_000.0)
    req = GatewayRequest(model_ref="openrouter/gone:free", prompt_tail="hi", max_output_tokens=10)

    resp = gw.complete(req, job_id="j-probe-ok", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert resp.text == "back"
    assert len(transport.calls) == 1
    # The successful trial call resolved the probe: entry gone entirely, not
    # just "not dead right now".
    assert dead_models.get("openrouter", "openrouter/gone:free") is None


def test_gateway_reopens_after_a_failed_half_open_probe(ledger):
    dead_models = DeadModelStore(clock=lambda: 2_000.0)
    dead_models.mark_dead("openrouter", "openrouter/gone:free", reason="glitch", retry_after=2_000.0)
    transport = FakeTransport("openrouter", error=TransportError("still down"))
    gw, _ = _gateway_with_clock([transport], ledger, dead_models, t=2_000.0)
    req = GatewayRequest(model_ref="openrouter/gone:free", prompt_tail="hi", max_output_tokens=10)

    with pytest.raises(TransportError):
        gw.complete(req, job_id="j-probe-fail", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    entry = dead_models.get("openrouter", "openrouter/gone:free")
    assert entry is not None
    assert entry.terminal is False, "a failed probe backs off, it does not promote to terminal"
    assert entry.retry_after > 2_000.0
    assert dead_models.is_dead("openrouter", "openrouter/gone:free") is True


def test_gateway_reopens_after_a_rate_limited_half_open_probe(ledger):
    dead_models = DeadModelStore(clock=lambda: 2_000.0)
    dead_models.mark_dead("openrouter", "openrouter/gone:free", reason="glitch", retry_after=2_000.0)
    transport = FakeTransport("openrouter", error=RateLimitError("429", retry_after=5.0))
    gw, _ = _gateway_with_clock([transport], ledger, dead_models, t=2_000.0)
    req = GatewayRequest(model_ref="openrouter/gone:free", prompt_tail="hi", max_output_tokens=10)

    with pytest.raises(TransportError):
        gw.complete(req, job_id="j-probe-429", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    entry = dead_models.get("openrouter", "openrouter/gone:free")
    assert entry is not None and entry.terminal is False
    # Exact, not just "later than before": the provider's own Retry-After
    # signal is what set this, not `report_probe`'s generic backoff default.
    assert entry.retry_after == 2_000.0 + 5.0


def test_gateway_a_model_unavailable_probe_result_goes_terminal_not_backoff(ledger):
    """A 404 during the trial call is a stronger signal than an ordinary
    probe failure -- it should still promote straight to TERMINAL, the same
    as it would for a never-dead model, not just reschedule another trial."""
    dead_models = DeadModelStore(clock=lambda: 2_000.0)
    dead_models.mark_dead("openrouter", "openrouter/gone:free", reason="glitch", retry_after=2_000.0)
    transport = FakeTransport("openrouter", error=ModelUnavailableError("openrouter HTTP 404: gone"))
    gw, _ = _gateway_with_clock([transport], ledger, dead_models, t=2_000.0)
    req = GatewayRequest(model_ref="openrouter/gone:free", prompt_tail="hi", max_output_tokens=10)

    with pytest.raises(TransportError):
        gw.complete(req, job_id="j-probe-404", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    entry = dead_models.get("openrouter", "openrouter/gone:free")
    assert entry is not None
    assert entry.terminal is True
    assert entry.probe_claimed_at is None  # mark_dead always clears the claim


def test_gateway_only_one_concurrent_caller_gets_the_half_open_trial(ledger):
    """Two callers racing the same HALF_OPEN window: only the claim-winner
    ever reaches the transport. The loser gets a clean 'nothing to try'
    failure instead of also spending a network call on the same trial."""
    dead_models = DeadModelStore(clock=lambda: 2_000.0)
    dead_models.mark_dead("openrouter", "openrouter/gone:free", reason="glitch", retry_after=2_000.0)
    transport = BlockingTransport(
        "openrouter", result=RawCallResult(text="back", tokens_in=1, tokens_out=1)
    )
    gw, _ = _gateway_with_clock([transport], ledger, dead_models, t=2_000.0)
    req = GatewayRequest(model_ref="openrouter/gone:free", prompt_tail="hi", max_output_tokens=10)
    results: dict[str, object] = {}
    errors: dict[str, Exception] = {}

    def call(name, job_id):
        try:
            results[name] = gw.complete(
                req, job_id=job_id, task_id=None, worker_id="w1", remaining_micros=1_000_000
            )
        except Exception as e:
            errors[name] = e

    t_a = threading.Thread(target=call, args=("a", "j-race-a"))
    t_a.start()
    assert transport.started.wait(timeout=5), "claim-winner never reached the transport"

    # By now A has already claimed the trial and is blocked inside the
    # transport call -- B's claim attempt must lose and never call in.
    call("b", "j-race-b")
    assert isinstance(errors.get("b"), ModelUnavailableError)
    assert transport.calls == 1, "the loser must not also touch the transport"

    transport.release.set()
    t_a.join(timeout=5)

    assert not errors.get("a")
    assert results["a"].text == "back"
    assert dead_models.get("openrouter", "openrouter/gone:free") is None


def test_gateway_probes_at_most_one_temporary_dead_transport_per_request(ledger):
    """Two different transports both happen to be due for a HALF_OPEN trial
    for the same model_ref: only one gets probed this request, not both --
    one experimental call per request, not a burst across every dead
    transport on the ladder (see the comment above the candidate-building
    loop in `_call_transports`)."""
    dead_models = DeadModelStore(clock=lambda: 2_000.0)
    dead_models.mark_dead("openrouter", "openrouter/gone:free", reason="glitch", retry_after=2_000.0)
    dead_models.mark_dead("omniroute", "openrouter/gone:free", reason="glitch", retry_after=2_000.0)
    openrouter = FakeTransport("openrouter", result=RawCallResult(text="a", tokens_in=1, tokens_out=1))
    omniroute = FakeTransport("omniroute", result=RawCallResult(text="b", tokens_in=1, tokens_out=1))
    gw, _ = _gateway_with_clock([openrouter, omniroute], ledger, dead_models, t=2_000.0)
    req = GatewayRequest(model_ref="openrouter/gone:free", prompt_tail="hi", max_output_tokens=10)

    resp = gw.complete(req, job_id="j-cap", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert resp.text in ("a", "b")
    assert len(openrouter.calls) + len(omniroute.calls) == 1, "only one transport probed this request"
    remaining = [
        e for e in (
            dead_models.get("openrouter", "openrouter/gone:free"),
            dead_models.get("omniroute", "openrouter/gone:free"),
        )
        if e is not None
    ]
    assert len(remaining) == 1, "the tried one resolved (cleared); the other was never touched"
    assert remaining[0].probe_claimed_at is None, "never claimed -- this request's one slot was already spent"


# ======================================================= feature C: dedup


def test_dedup_is_off_by_default(ledger):
    transport = FakeTransport("http", result=RawCallResult(text="hi", tokens_in=10, tokens_out=5))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request()  # dedup=False

    gw.complete(req, job_id="j-a", task_id=None, worker_id="w1", remaining_micros=1_000_000)
    gw.complete(req, job_id="j-b", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert len(transport.calls) == 2


def test_concurrent_identical_dedup_calls_share_one_upstream_call_and_one_spend(ledger):
    transport = BlockingTransport("http", result=RawCallResult(text="hi", tokens_in=10, tokens_out=5))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request(dedup=True)
    results: dict[str, object] = {}
    errors: dict[str, Exception] = {}

    def call(name, job_id):
        try:
            results[name] = gw.complete(
                req, job_id=job_id, task_id=None, worker_id="w1", remaining_micros=1_000_000
            )
        except Exception as e:  # captured for the assertion below, any type is fine here
            errors[name] = e

    t_a = threading.Thread(target=call, args=("a", "j-lead"))
    t_a.start()
    assert transport.started.wait(timeout=5), "leader never reached the transport"

    t_b = threading.Thread(target=call, args=("b", "j-wait"))
    t_b.start()
    time.sleep(0.05)  # best-effort head start for B to reach the wait state

    transport.release.set()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    assert not errors, errors
    assert transport.calls == 1
    assert results["a"].text == results["b"].text == "hi"
    assert results["a"].usd_micros == results["b"].usd_micros
    # Only the leader's job paid; the waiter's job rode the same upstream
    # call for free -- a deliberate consequence of sharing, see report.
    assert ledger.job_spend_micros("j-lead") == results["a"].usd_micros
    assert ledger.job_spend_micros("j-wait") == 0


def test_dedup_waiter_gets_the_leaders_failure_with_spend_recorded_once(ledger):
    transport = BlockingTransport("http", error=TransportError("upstream 500"))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request(dedup=True)
    results: dict[str, object] = {}
    errors: dict[str, Exception] = {}

    def call(name, job_id):
        try:
            results[name] = gw.complete(
                req, job_id=job_id, task_id=None, worker_id="w1", remaining_micros=1_000_000
            )
        except Exception as e:
            errors[name] = e

    t_a = threading.Thread(target=call, args=("a", "j-lead-fail"))
    t_a.start()
    assert transport.started.wait(timeout=5)

    t_b = threading.Thread(target=call, args=("b", "j-wait-fail"))
    t_b.start()
    time.sleep(0.05)

    transport.release.set()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    assert not results, results
    assert set(errors) == {"a", "b"}, "a failed leader must fail every waiter too"
    assert "upstream 500" in str(errors["a"])
    assert "upstream 500" in str(errors["b"])
    assert transport.calls == 1
    assert ledger.job_spend_micros("j-lead-fail") > 0
    assert ledger.job_spend_micros("j-wait-fail") == 0, "must not be cached and separately charged"


def test_dedup_inflight_entry_is_cleaned_up_after_success(ledger):
    transport = FakeTransport("http", result=RawCallResult(text="hi", tokens_in=1, tokens_out=1))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request(dedup=True)

    gw.complete(req, job_id="j1", task_id=None, worker_id="w1", remaining_micros=1_000_000)
    assert gw._inflight == {}

    # A later, non-concurrent identical call must be a fresh leader, not join
    # a stale entry -- the only way to observe that cleanup really happened.
    gw.complete(req, job_id="j2", task_id=None, worker_id="w1", remaining_micros=1_000_000)
    assert len(transport.calls) == 2


def test_dedup_inflight_entry_is_cleaned_up_after_failure(ledger):
    """A crashed leader must not wedge every future identical request behind
    it. Proven by the second call being a genuinely independent evaluation:
    if the first call's map entry had leaked, this second call would instead
    join it as a waiter and replay the exact same cached "boom" error
    without ever reaching `_call_transports` again."""
    transport = FakeTransport("http", error=TransportError("boom"))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request(dedup=True)

    with pytest.raises(TransportError, match="boom"):
        gw.complete(req, job_id="j1", task_id=None, worker_id="w1", remaining_micros=1_000_000)
    assert gw._inflight == {}, "a crashed leader must not wedge every future identical request"
    assert len(transport.calls) == 1

    # The transport is now unhealthy for the unrelated, pre-existing reason
    # that a plain TransportError marks it so (HealthTracker has no
    # auto-recovery for that, only for a rate limit) -- so this second call
    # is refused by `_call_transports` before ever reaching the transport
    # again. That is expected and orthogonal to what this test checks: the
    # error must be a FRESH "no healthy transport" verdict, not a replay of
    # the first call's cached "boom" from a leaked map entry.
    with pytest.raises(TransportError) as excinfo:
        gw.complete(req, job_id="j2", task_id=None, worker_id="w1", remaining_micros=1_000_000)
    assert "boom" not in str(excinfo.value)
    assert len(transport.calls) == 1


def test_dedup_does_not_let_a_broke_caller_ride_on_someone_elses_budget(ledger):
    transport = BlockingTransport("http", result=RawCallResult(text="hi", tokens_in=1, tokens_out=1))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request(dedup=True)

    t_a = threading.Thread(
        target=lambda: gw.complete(
            req, job_id="j-rich", task_id=None, worker_id="w1", remaining_micros=1_000_000
        )
    )
    t_a.start()
    assert transport.started.wait(timeout=5)

    # Identical request, but this caller's own budget cannot cover even the
    # estimate -- must be refused on its own terms, even while a leader for
    # the exact same key is actively in flight.
    with pytest.raises(CallRefused):
        gw.complete(req, job_id="j-broke", task_id=None, worker_id="w2", remaining_micros=0)
    assert ledger.job_spend_micros("j-broke") == 0

    transport.release.set()
    t_a.join(timeout=5)


def test_dedup_waiter_records_avoidance_when_an_avoidance_log_is_provided(ledger):
    transport = BlockingTransport("http", result=RawCallResult(text="hi", tokens_in=10, tokens_out=5))
    avoidance = AvoidanceLog(":memory:")
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport], avoidance_log=avoidance)
    req = _request(dedup=True)

    def call(job_id):
        gw.complete(req, job_id=job_id, task_id=None, worker_id="w1", remaining_micros=1_000_000)

    t_a = threading.Thread(target=call, args=("j-lead-av",))
    t_a.start()
    assert transport.started.wait(timeout=5)

    t_b = threading.Thread(target=call, args=("j-wait-av",))
    t_b.start()
    time.sleep(0.05)

    transport.release.set()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    waiter_totals = avoidance.totals("j-wait-av")["by_method"]
    assert waiter_totals.get("dedup", {}).get("count") == 1
    assert waiter_totals["dedup"]["saved_tokens"] > 0
    # The leader paid for real; it must not also show up as a dedup avoidance.
    assert avoidance.totals("j-lead-av")["by_method"] == {}
    avoidance.close()


def test_dedup_waiter_falls_back_to_an_independent_call_if_the_leader_never_finishes(ledger):
    transport = BlockingTransport(
        "http", result=RawCallResult(text="hi", tokens_in=1, tokens_out=1), block_every_call=False
    )
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport], dedup_wait_timeout=0.05)
    req = _request(dedup=True)

    t_a = threading.Thread(
        target=lambda: gw.complete(
            req, job_id="j-hang", task_id=None, worker_id="w1", remaining_micros=1_000_000
        )
    )
    t_a.start()
    assert transport.started.wait(timeout=5)

    # B waits at most 0.05s for the (still-blocked) leader, then must make
    # its own independent call rather than hang forever.
    resp_b = gw.complete(req, job_id="j-fallback", task_id=None, worker_id="w2", remaining_micros=1_000_000)

    assert resp_b.text == "hi"
    assert transport.calls == 2  # the still-blocked leader's call, plus B's own

    transport.release.set()
    t_a.join(timeout=5)
