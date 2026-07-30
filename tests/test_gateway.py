"""Gateway tests: transport is always a scripted double, never real HTTP or
litellm — no network calls anywhere in this file."""

from __future__ import annotations

import pytest

from hive.catalog import Catalog, ModelCard
from hive.economy.preflight import CallRefused
from hive.gateway.client import (
    Gateway,
    RateLimitError,
    RawCallResult,
    TransportError,
    assemble_prompt,
)
from hive.gateway.health import HealthTracker
from hive.ledger import Ledger

CARD = ModelCard(
    model_id="test-model",
    provider="testprov",
    input_cost_per_1m=1.0,
    output_cost_per_1m=2.0,
    context=1000,
)


def _request(**overrides):
    from hive.gateway.client import GatewayRequest

    kwargs = dict(model_ref="testprov/test-model", prompt_prefix="p", prompt_tail="", max_output_tokens=50)
    kwargs.update(overrides)
    return GatewayRequest(**kwargs)


class FakeTransport:
    """A transport double whose behavior is fully scripted; never touches a network."""

    def __init__(self, name: str, result: RawCallResult | None = None, error: Exception | None = None):
        self.name = name
        self._result = result
        self._error = error
        self.calls: list[dict] = []

    def complete(self, *, model_id, prompt, max_output_tokens, reasoning_effort, tools_schema):
        self.calls.append(
            dict(
                model_id=model_id,
                prompt=prompt,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                tools_schema=tools_schema,
            )
        )
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class NeverCallTransport:
    """Fails the test outright if invoked — proves a refusal (or a skipped,
    rate-limited transport) never reaches the transport layer."""

    def __init__(self, name: str = "never"):
        self.name = name

    def complete(self, **kwargs):
        pytest.fail(f"transport {self.name!r} was called when it should have been blocked")


@pytest.fixture()
def ledger():
    led = Ledger(":memory:")
    yield led
    led.close()


def _catalog() -> Catalog:
    return Catalog([CARD])


# ---------------------------------------------------------------- ordering


def test_budget_refusal_happens_before_any_transport_call(ledger):
    """A prompt that fits the context window but costs more than remains must
    be refused without ever invoking a transport."""
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[NeverCallTransport()])
    req = _request(prompt_prefix="short prompt", max_output_tokens=100)

    with pytest.raises(CallRefused):
        gw.complete(req, job_id="j-budget", task_id=None, worker_id="w1", remaining_micros=0)

    assert ledger.job_spend_micros("j-budget") == 0


def test_context_refusal_happens_before_any_transport_call(ledger):
    """A prompt too large for the model's context window is refused
    regardless of how much budget remains — and still never touches transport."""
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[NeverCallTransport()])
    req = _request(prompt_prefix="x" * 10_000, max_output_tokens=100)

    with pytest.raises(CallRefused):
        gw.complete(req, job_id="j-ctx", task_id=None, worker_id="w1", remaining_micros=10_000_000)

    assert ledger.job_spend_micros("j-ctx") == 0


def test_unknown_model_ref_raises_before_touching_transport(ledger):
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[NeverCallTransport()])
    req = _request(model_ref="nope/does-not-exist")

    with pytest.raises(LookupError):
        gw.complete(req, job_id="j-unknown", task_id=None, worker_id="w1", remaining_micros=1_000_000)


def test_spend_recorded_even_when_call_raises_mid_flight(ledger):
    """The call was attempted and presumably cost something even though it
    blew up before returning a usable result — the ledger must still see it."""
    transport = FakeTransport("flaky", error=TransportError("connection reset mid-stream"))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request(prompt_prefix="hello", max_output_tokens=50)

    with pytest.raises(TransportError):
        gw.complete(req, job_id="j-midflight", task_id="t-midflight", worker_id="w1", remaining_micros=1_000_000)

    assert ledger.job_spend_micros("j-midflight") > 0
    assert ledger.task_spend_micros("t-midflight") > 0


def test_ledger_records_before_returning_result(ledger):
    """By the time complete() returns, its cost is already visible in the ledger."""
    transport = FakeTransport("clean", result=RawCallResult(text="hi", tokens_in=100, tokens_out=20))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request()

    resp = gw.complete(req, job_id="j-order", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert ledger.job_spend_micros("j-order") == resp.usd_micros


# ------------------------------------------------------------------- prefix


def test_prefix_bytes_identical_across_calls_with_different_tails():
    prefix = "SYSTEM PROMPT -- do not vary this text.\n"
    p1 = assemble_prompt(prefix, "task A")
    p2 = assemble_prompt(prefix, "task B, a much longer tail than the first one")

    assert p1[: len(prefix)] == prefix
    assert p2[: len(prefix)] == prefix
    assert p1[: len(prefix)].encode("utf-8") == p2[: len(prefix)].encode("utf-8")


# ------------------------------------------------------------------- health


def test_rate_limited_provider_is_excluded_until_available_at_passes():
    clock = {"t": 1_000.0}
    health = HealthTracker(clock=lambda: clock["t"])

    health.record_rate_limit("flaky", retry_after_seconds=60)
    assert health.healthy("flaky") is False
    assert health.pick_healthy(["flaky", "other"]) == ["other"]

    clock["t"] += 61
    assert health.healthy("flaky") is True
    assert "flaky" in health.pick_healthy(["flaky", "other"])


def test_gateway_skips_rate_limited_transport_without_calling_it(ledger):
    blocked = NeverCallTransport("blocked")
    good = FakeTransport("good", result=RawCallResult(text="ok", tokens_in=10, tokens_out=5))

    health = HealthTracker()
    health.record_rate_limit("blocked", retry_after_seconds=60)

    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[blocked, good], health=health)
    req = _request()

    resp = gw.complete(req, job_id="j-skip", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert resp.provider_used == "good"
    assert len(good.calls) == 1


def test_falls_back_to_second_transport_when_first_is_unreachable(ledger):
    first = FakeTransport("http", error=TransportError("omniroute unreachable"))
    second = FakeTransport("litellm", result=RawCallResult(text="hi", tokens_in=4, tokens_out=2))

    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[first, second])
    req = _request(prompt_prefix="p", prompt_tail="q")

    resp = gw.complete(req, job_id="j-fallback", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert resp.provider_used == "litellm"
    assert resp.text == "hi"
    assert len(first.calls) == 1
    assert len(second.calls) == 1


def test_first_transport_rate_limit_moves_it_out_of_rotation_for_next_call(ledger):
    """A 429 from the primary transport must not be retried in a tight loop —
    the health tracker records it, and the second call in the same window
    skips straight to the fallback without touching the rate-limited one again."""
    clock = {"t": 0.0}
    health = HealthTracker(clock=lambda: clock["t"])
    first = FakeTransport("http", error=RateLimitError("slow down", retry_after=60))
    second = FakeTransport("litellm", result=RawCallResult(text="ok", tokens_in=1, tokens_out=1))

    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[first, second], health=health)
    req = _request()

    gw.complete(req, job_id="j1", task_id=None, worker_id="w1", remaining_micros=1_000_000)
    assert len(first.calls) == 1

    # Second call, still inside the backoff window: "http" must be skipped entirely.
    gw.complete(req, job_id="j2", task_id=None, worker_id="w1", remaining_micros=1_000_000)
    assert len(first.calls) == 1  # unchanged -- not retried
    assert len(second.calls) == 2


# ------------------------------------------------------------------- usage


def test_exact_usage_false_when_provider_returns_no_usage_numbers(ledger):
    transport = FakeTransport("noisy", result=RawCallResult(text="some output text", tokens_in=None, tokens_out=None))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request(prompt_prefix="prompt text here", max_output_tokens=20)

    resp = gw.complete(req, job_id="j-noexact", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert resp.exact_usage is False
    assert resp.tokens_in > 0
    assert resp.tokens_out > 0


def test_exact_usage_true_and_cache_split_when_provider_reports_usage(ledger):
    transport = FakeTransport(
        "clean", result=RawCallResult(text="hi", tokens_in=100, tokens_out=20, tokens_cached_in=30)
    )
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request()

    resp = gw.complete(req, job_id="j-exact", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert resp.exact_usage is True
    assert resp.tokens_cached_in == 30
    assert resp.tokens_in == 70  # fresh = total (100) - cached (30)
    assert resp.cache_hit is True


def test_no_cache_hit_reported_as_false(ledger):
    transport = FakeTransport("clean", result=RawCallResult(text="hi", tokens_in=10, tokens_out=5))
    gw = Gateway(catalog=_catalog(), ledger=ledger, transports=[transport])
    req = _request()

    resp = gw.complete(req, job_id="j-nocache", task_id=None, worker_id="w1", remaining_micros=1_000_000)

    assert resp.tokens_cached_in == 0
    assert resp.cache_hit is False
