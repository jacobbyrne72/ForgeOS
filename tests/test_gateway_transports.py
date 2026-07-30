"""Transport selection, provider binding, and what counts as a health event.

Every test here corresponds to something that actually broke while connecting a
real DeepSeek and OpenRouter key, and each failure was invisible in a different
way:

- a configured, authenticated provider was simply unreachable, because only one
  transport was ever built;
- transports were tried in health order with no regard for which vendor they
  serve, so a `deepseek/*` model could be sent to OpenRouter's endpoint;
- one retired free-tier slug returned 404 and took the entire provider offline
  for every other model in the same run;
- when several transports failed, only the last error survived, so the vendor's
  own explanation was replaced by a generic fallback message.
"""

from __future__ import annotations

import pytest

from forgeos.catalog import Catalog, ModelCard
from forgeos.contracts import Budget, JobSpec
from forgeos.gateway.client import (
    Gateway,
    GatewayRequest,
    ModelUnavailableError,
    RawCallResult,
    TransportError,
    default_transports,
)
from forgeos.ledger import open_ledger
from forgeos.settings import AuthMode, Provider, ProviderKind, Settings, default_settings


def _card(ref: str) -> ModelCard:
    provider, model_id = ref.split("/", 1)
    return ModelCard(
        model_id=model_id, provider=provider, name=model_id,
        input_cost_per_1m=1.0, output_cost_per_1m=2.0, context=100_000, max_output=4_096,
    )


def _catalog(*refs: str) -> Catalog:
    return Catalog([_card(r) for r in refs])


class _Recorder:
    """A transport that records what it was asked and does what it was told."""

    def __init__(self, name: str, *, serves=None, raises=None):
        self.name = name
        self.serves = serves or set()
        self._raises = raises
        self.seen: list[str] = []

    def complete(self, *, model_id, prompt, max_output_tokens, reasoning_effort, tools_schema):
        self.seen.append(model_id)
        if self._raises is not None:
            raise self._raises
        return RawCallResult(text="ok", tokens_in=10, tokens_out=2, model_used=model_id)


def _gateway(catalog, transports):
    ledger = open_ledger(":memory:")
    job_id = ledger.open_job(JobSpec(objective="t", cwd=".", budget=Budget(max_usd=1.0)))
    gw = Gateway(catalog=catalog, ledger=ledger, settings=default_settings(),
                 transports=transports)
    return gw, ledger, job_id


def _call(gw, job_id, ref):
    return gw.complete(
        GatewayRequest(model_ref=ref, prompt_tail="hi", max_output_tokens=16),
        job_id=job_id, task_id=None, worker_id="w", remaining_micros=1_000_000,
    )


# ===================================================== a key that does nothing


def _settings_with(**providers) -> Settings:
    s = Settings(providers={}, roles={})
    for name, p in providers.items():
        s.providers[name] = p
    return s


def _api(name: str, env_key: str, url: str = "https://api.example.com") -> Provider:
    return Provider(name=name, kind=ProviderKind.API, auth=AuthMode.API_KEY,
                    env_key=env_key, base_url=url)


def test_a_configured_api_provider_gets_its_own_transport(monkeypatch):
    """Without this, entering a key changed nothing: the provider showed as
    'ready' in settings and was completely unreachable at call time."""
    monkeypatch.setenv("FAKE_KEY", "x")
    s = _settings_with(fakeprov=_api("fakeprov", "FAKE_KEY"))
    names = [t.name for t in default_transports(s)]
    assert "fakeprov" in names


def test_an_unauthenticated_provider_gets_no_transport(monkeypatch):
    """Reported once in settings rather than failing on every call at spend time."""
    monkeypatch.delenv("FAKE_KEY", raising=False)
    s = _settings_with(fakeprov=_api("fakeprov", "FAKE_KEY"))
    assert "fakeprov" not in [t.name for t in default_transports(s)]


def test_a_disabled_provider_gets_no_transport(monkeypatch):
    """Disabling is a gate, not a hint — otherwise 'disabled' silently means
    'still spending money'."""
    monkeypatch.setenv("FAKE_KEY", "x")
    p = _api("fakeprov", "FAKE_KEY")
    p.enabled = False
    assert "fakeprov" not in [t.name for t in default_transports(_settings_with(fakeprov=p))]


def test_litellm_is_always_last(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    s = _settings_with(fakeprov=_api("fakeprov", "FAKE_KEY"))
    assert [t.name for t in default_transports(s)][-1] == "litellm"


def test_the_real_default_settings_build_transports_for_both_live_providers(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    names = [t.name for t in default_transports(default_settings())]
    assert "deepseek" in names and "openrouter" in names


# ========================================================== provider binding


def test_a_model_is_never_sent_to_another_vendors_endpoint():
    """A 400 at best, a wrong-model charge at worst."""
    ds = _Recorder("deepseek", serves={"deepseek"})
    orr = _Recorder("openrouter", serves={"openrouter"})
    gw, _, job = _gateway(_catalog("deepseek/chat"), [orr, ds])

    _call(gw, job, "deepseek/chat")
    assert ds.seen == ["chat"]
    assert orr.seen == [], "openrouter was handed a deepseek model id"


def test_a_fan_out_gateway_serves_any_provider():
    """`serves` empty is the property of a gateway, not an oversight."""
    gateway_t = _Recorder("omniroute")
    gw, _, job = _gateway(_catalog("deepseek/chat"), [gateway_t])
    _call(gw, job, "deepseek/chat")
    assert gateway_t.seen == ["chat"]


def test_a_provider_with_no_transport_says_so_plainly():
    bound = _Recorder("openrouter", serves={"openrouter"})
    gw, _, job = _gateway(_catalog("deepseek/chat"), [bound])
    with pytest.raises(TransportError) as e:
        _call(gw, job, "deepseek/chat")
    assert "deepseek" in str(e.value)
    assert bound.seen == []


# ================================================ a dead slug is not a dead endpoint


def test_a_model_404_does_not_mark_the_endpoint_unhealthy():
    """The failure that took OpenRouter offline for a whole run.

    A retired free-tier slug 404s. If that counts as a transport health event,
    the endpoint is blacklisted and the *next* model — which works fine — is
    never attempted. Free slugs rot constantly, so this is the common case.
    """
    dead = _Recorder("openrouter", serves={"openrouter"},
                     raises=ModelUnavailableError("openrouter HTTP 404: no endpoints found"))
    gw, _, job = _gateway(_catalog("openrouter/gone:free", "openrouter/alive"), [dead])

    with pytest.raises(TransportError):
        _call(gw, job, "openrouter/gone:free")

    # The endpoint must still be considered healthy for a different model.
    assert gw._health.pick_healthy(["openrouter"]) == ["openrouter"]


def test_a_real_transport_failure_does_mark_the_endpoint_unhealthy():
    """The distinction has to cut both ways, or the health tracker is decorative."""
    broken = _Recorder("openrouter", serves={"openrouter"},
                       raises=TransportError("connection refused"))
    gw, _, job = _gateway(_catalog("openrouter/m"), [broken])
    with pytest.raises(TransportError):
        _call(gw, job, "openrouter/m")
    assert gw._health.pick_healthy(["openrouter"]) != ["openrouter"]


def test_a_dead_model_on_one_endpoint_still_lets_another_endpoint_answer():
    dead = _Recorder("openrouter", serves={"openrouter"},
                     raises=ModelUnavailableError("404"))
    fanout = _Recorder("omniroute")
    gw, _, job = _gateway(_catalog("openrouter/m"), [dead, fanout])
    resp = _call(gw, job, "openrouter/m")
    assert resp.text == "ok"
    assert fanout.seen == ["m"]


# ==================================================== every failure is reported


def test_all_transport_errors_are_reported_not_just_the_last():
    """The vendor's own explanation is the useful one, and it was being replaced
    by whatever the final fallback happened to say."""
    a = _Recorder("openrouter", serves={"openrouter"},
                  raises=ModelUnavailableError("openrouter HTTP 404: model retired"))
    b = _Recorder("litellm", raises=TransportError("LLM Provider NOT provided"))
    gw, _, job = _gateway(_catalog("openrouter/m"), [a, b])

    with pytest.raises(TransportError) as e:
        _call(gw, job, "openrouter/m")
    msg = str(e.value)
    assert "model retired" in msg, "the vendor's explanation was dropped"
    assert "LLM Provider NOT provided" in msg


def test_a_failed_call_is_still_charged_to_the_ledger():
    """An attempted call cost something even though it produced nothing, and the
    governor cannot stop what it cannot see."""
    broken = _Recorder("openrouter", serves={"openrouter"}, raises=TransportError("boom"))
    gw, ledger, job = _gateway(_catalog("openrouter/m"), [broken])
    with pytest.raises(TransportError):
        _call(gw, job, "openrouter/m")
    assert ledger.job_spend_micros(job) > 0
