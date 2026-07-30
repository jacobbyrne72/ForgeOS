"""The one call path to any provider.

Four invariants, in order, inside `Gateway.complete`:

1. **Estimate before calling.** `estimate_call` prices the prospective call
   against the model's own tokeniser where available.
2. **Refuse before calling.** `preflight.check` turns a too-expensive or
   too-large call into a `CallRefused` — raised before any transport is
   touched. A refusal that still reached the network would be the preflight
   check lying about what it prevents.
3. **Call through one interface.** An HTTP-first, litellm-fallback pair
   behind a shared `Transport` protocol, so nothing above this module ever
   branches on which transport answered.
4. **Record before returning.** Spend is written to the ledger before the
   result reaches the caller — and also when the call raises mid-flight,
   because a call that was attempted already cost something even if it
   produced nothing usable. An unrecorded call is an invisible call, and the
   governor cannot stop what it cannot see.

The prompt is always assembled prefix-then-tail (`assemble_prompt`), and the
prefix bytes never change shape across calls: provider prompt-caching keys off
an exact prefix match, so reordering or recompressing the prefix would quietly
turn off caching for every call that follows.
"""

from __future__ import annotations

import os
import time
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, Field

from ..catalog import Catalog, ModelCard
from ..economy.preflight import check, count_tokens, estimate_call
from ..ledger import Ledger
from ..settings import Settings, default_settings
from .health import HealthTracker


def assemble_prompt(prefix: str, tail: str) -> str:
    """Prefix-then-tail, always. This is the only place a prompt is assembled —
    no reordering, no trimming, no recompressing the prefix. Provider prompt
    caching only fires on an exact byte match against a previously-seen
    prefix, so this concatenation has to be identical every time it is called
    with the same prefix, whatever the tail is.
    """
    return prefix + tail


# --------------------------------------------------------------------------
# requests / responses
# --------------------------------------------------------------------------


class GatewayRequest(BaseModel):
    """One prospective call, addressed at a specific model."""

    model_ref: str
    prompt_prefix: str = ""
    prompt_tail: str = ""
    max_output_tokens: int = Field(gt=0)
    reasoning_effort: Literal["none", "low", "medium", "high", "max"] = "medium"
    tools_schema: list | dict | None = None

    @property
    def prompt(self) -> str:
        return assemble_prompt(self.prompt_prefix, self.prompt_tail)


class GatewayResponse(BaseModel):
    """What a completed call produced, priced and attributed.

    `tokens_in` is the *fresh* (non-cached) input token count and
    `tokens_cached_in` the cached portion — matching `Ledger.record_spend`'s
    own convention, so a response can be forwarded to the ledger without
    re-deriving anything. `exact_usage` is True only when the provider's
    response carried real usage numbers; otherwise the counts were estimated
    with the same tokeniser-or-chars/4 fallback the preflight estimate uses.
    """

    text: str
    tokens_in: int
    tokens_out: int
    tokens_cached_in: int = 0
    usd_micros: int
    model_used: str
    provider_used: str
    exact_usage: bool
    cache_hit: bool


class RawCallResult(BaseModel):
    """What a transport hands back, before the gateway prices and attributes it.

    `tokens_in`/`tokens_out` are `None` exactly when the provider's response
    carried no usage block at all — that absence is the one signal
    `GatewayResponse.exact_usage` is built from, so a transport must not
    invent numbers here to fill the gap.
    """

    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_cached_in: int = 0
    model_used: str = ""


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


class TransportError(RuntimeError):
    """A transport could not complete the call.

    The gateway tries the next transport in order; if none are left, the
    preflight estimate is recorded as spend and this (or a wrapping
    `TransportError`) is re-raised, because a call that was attempted is not
    a call that gets to be invisible to the ledger.
    """


class RateLimitError(TransportError):
    """A 429 or provider-reported rate limit. Carries how long to back off."""

    def __init__(self, message: str, retry_after: float = 30.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class Transport(Protocol):
    """Shared shape for every way of actually reaching a model. Callers of
    `Gateway` never see this — only `Gateway._call_transports` branches on
    which one answered, and only to record health, never to change behavior.
    """

    name: str

    def complete(
        self,
        *,
        model_id: str,
        prompt: str,
        max_output_tokens: int,
        reasoning_effort: str,
        tools_schema: list | dict | None,
    ) -> RawCallResult: ...


class HttpTransport:
    """An OpenAI-compatible HTTP endpoint. OmniRoute by default — a gateway
    that fans out to many upstream providers and free tiers behind one URL,
    so this single transport covers all of them without hive ever branching
    on which upstream provider actually answered.
    """

    def __init__(
        self,
        base_url: str,
        api_key_env: str = "",
        timeout: float = 60.0,
        name: str = "omniroute",
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._timeout = timeout

    def complete(
        self,
        *,
        model_id: str,
        prompt: str,
        max_output_tokens: int,
        reasoning_effort: str,
        tools_schema: list | dict | None,
    ) -> RawCallResult:
        headers = {"Content-Type": "application/json"}
        if self._api_key_env:
            # Read the env var NAME only; the value never appears in a log or error.
            key = os.environ.get(self._api_key_env, "").strip()
            if key:
                headers["Authorization"] = f"Bearer {key}"

        payload: dict = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
        }
        if reasoning_effort and reasoning_effort != "none":
            payload["reasoning_effort"] = reasoning_effort
        if tools_schema:
            payload["tools"] = tools_schema

        try:
            resp = httpx.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
        except httpx.HTTPError as e:
            raise TransportError(f"{self.name} unreachable: {e}") from e

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 30))
            raise RateLimitError(f"{self.name} rate limited", retry_after=retry_after)
        if resp.status_code >= 400:
            raise TransportError(f"{self.name} HTTP {resp.status_code}")

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content")) or ""
        usage = data.get("usage") or {}
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        return RawCallResult(
            text=text,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            tokens_cached_in=cached,
            model_used=str(data.get("model") or model_id),
        )


class LiteLLMTransport:
    """Fallback reached only when the HTTP endpoint is unreachable. litellm
    owns per-provider auth internally; hive still never reads a key value
    here, it only ever hands litellm a model id and lets it resolve auth.
    """

    name = "litellm"

    def complete(
        self,
        *,
        model_id: str,
        prompt: str,
        max_output_tokens: int,
        reasoning_effort: str,
        tools_schema: list | dict | None,
    ) -> RawCallResult:
        try:
            import litellm
        except ImportError as e:
            raise TransportError("litellm not installed (install the 'gateway' extra)") from e

        kwargs: dict = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
        }
        if tools_schema:
            kwargs["tools"] = tools_schema

        try:
            resp = litellm.completion(**kwargs)
        except Exception as e:  # litellm raises many provider-specific error types
            raise TransportError(f"litellm call failed: {e}") from e

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", None) if usage else None
        tokens_out = getattr(usage, "completion_tokens", None) if usage else None
        return RawCallResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model_used=str(getattr(resp, "model", "") or model_id),
        )


def default_transports(settings: Settings) -> list[Transport]:
    """HTTP first (OmniRoute), litellm as the fallback. Order is fixed and
    deliberate — callers never branch on transport, so the ordering here is
    the only place "prefer HTTP, fall back to litellm" actually lives.
    """
    omni = settings.providers.get("omniroute")
    base_url = omni.base_url if omni else "http://127.0.0.1:8787"
    env_key = omni.env_key if omni else ""
    return [HttpTransport(base_url=base_url, api_key_env=env_key), LiteLLMTransport()]


# --------------------------------------------------------------------------
# gateway
# --------------------------------------------------------------------------


class Gateway:
    """One unified call path. See module docstring for the four invariants
    `complete` enforces — the ordering is mandatory, not stylistic.
    """

    def __init__(
        self,
        *,
        catalog: Catalog,
        ledger: Ledger,
        settings: Settings | None = None,
        transports: list[Transport] | None = None,
        health: HealthTracker | None = None,
        max_context: int = 0,
    ) -> None:
        self._catalog = catalog
        self._ledger = ledger
        self._settings = settings if settings is not None else default_settings()
        self._transports = transports if transports is not None else default_transports(self._settings)
        self._health = health if health is not None else HealthTracker()
        self._max_context = max_context

    def complete(
        self,
        request: GatewayRequest,
        *,
        job_id: str,
        task_id: str | None,
        worker_id: str,
        remaining_micros: int,
    ) -> GatewayResponse:
        card = self._catalog.get(request.model_ref)
        if card is None:
            raise LookupError(f"unknown model_ref: {request.model_ref!r} (not in catalog)")

        # (1) estimate, before anything is touched.
        prompt = assemble_prompt(request.prompt_prefix, request.prompt_tail)
        estimate = estimate_call(prompt, request.max_output_tokens, card)

        # (2) refuse-or-allow. A refusal raises here; no transport is ever touched.
        verdict = check(estimate, remaining_micros, max_context=self._max_context)
        verdict.raise_if_refused()

        # (3) make the call.
        try:
            raw, provider_used = self._call_transports(card, prompt, request)
        except Exception:
            # The call was attempted even though it produced no usable result.
            # That still cost something the governor must be able to see.
            self._ledger.record_spend(
                job_id,
                worker_id,
                card.ref,
                estimate.usd_micros,
                task_id=task_id,
                tokens_in=estimate.tokens_in,
                tokens_out=0,
            )
            raise

        response = self._to_response(card, raw, provider_used, prompt)

        # (4) record before returning.
        self._ledger.record_spend(
            job_id,
            worker_id,
            response.model_used,
            response.usd_micros,
            task_id=task_id,
            kind="call",
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            tokens_cached_in=response.tokens_cached_in,
        )
        return response

    # ---------------------------------------------------------------- internals

    def _call_transports(
        self, card: ModelCard, prompt: str, request: GatewayRequest
    ) -> tuple[RawCallResult, str]:
        """Try transports in health-filtered order. A rate-limited transport
        is skipped without being invoked at all — that is what keeps a 429
        from turning into a retry storm.
        """
        by_name = {t.name: t for t in self._transports}
        ordered = self._health.pick_healthy(list(by_name.keys()))

        last_err: Exception | None = None
        for name in ordered:
            transport = by_name[name]
            start = time.monotonic()
            try:
                raw = transport.complete(
                    model_id=card.model_id,
                    prompt=prompt,
                    max_output_tokens=request.max_output_tokens,
                    reasoning_effort=request.reasoning_effort,
                    tools_schema=request.tools_schema,
                )
            except RateLimitError as e:
                self._health.record_rate_limit(name, retry_after_seconds=e.retry_after)
                last_err = e
                continue
            except Exception as e:  # any other transport failure — try the next one
                self._health.record_failure(name, str(e))
                last_err = e
                continue
            self._health.record_success(name, latency_ms=(time.monotonic() - start) * 1000)
            return raw, name

        if last_err is not None:
            raise TransportError(f"all transports failed for {card.ref}: {last_err}") from last_err
        raise TransportError(f"no healthy transport available for {card.ref}")

    def _to_response(
        self, card: ModelCard, raw: RawCallResult, provider_used: str, prompt: str
    ) -> GatewayResponse:
        exact_usage = raw.tokens_in is not None and raw.tokens_out is not None

        total_in = raw.tokens_in if raw.tokens_in is not None else count_tokens(prompt, card.model_id).tokens
        tokens_out = (
            raw.tokens_out if raw.tokens_out is not None else count_tokens(raw.text, card.model_id).tokens
        )
        tokens_cached_in = max(0, min(raw.tokens_cached_in, total_in))
        fresh_in = total_in - tokens_cached_in

        # Cached input is priced at the cache rate. Charging it at the fresh rate
    # made prompt caching -- the reason this module exists -- produce exactly
    # zero measured savings, and put the ledger's token and dollar columns in
    # permanent disagreement.
        # separate discounted cache rate, so pricing on the full count is the
        # conservative (never-under-counts) choice available with today's
        # catalog data. tokens_cached_in is still reported separately so
        # Ledger.cache_stats can see the real hit rate for observability.
        usd_micros = card.cost_micros(fresh_in, tokens_out,
                                  tokens_cached_in=tokens_cached_in)

        return GatewayResponse(
            text=raw.text,
            tokens_in=fresh_in,
            tokens_out=tokens_out,
            tokens_cached_in=tokens_cached_in,
            usd_micros=usd_micros,
            model_used=raw.model_used or card.model_id,
            provider_used=provider_used,
            exact_usage=exact_usage,
            cache_hit=tokens_cached_in > 0,
        )


__all__ = [
    "Gateway",
    "GatewayRequest",
    "GatewayResponse",
    "HttpTransport",
    "LiteLLMTransport",
    "RateLimitError",
    "RawCallResult",
    "Transport",
    "TransportError",
    "assemble_prompt",
    "default_transports",
]
