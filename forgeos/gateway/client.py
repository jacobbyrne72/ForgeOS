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

import hashlib
import json
import os
import threading
import time
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, Field

from ..catalog import Catalog, ModelCard
from ..economy.avoidance import AvoidanceLog, AvoidanceMethod
from ..economy.preflight import check, count_tokens, estimate_call
from ..ledger import Ledger
from ..settings import ProviderKind, Settings, default_settings
from .dead_models import DeadModelStore
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
    """One prospective call, addressed at a specific model.

    `dedup` is opt-in only, default off. Concurrent identical requests (same
    model_ref, prompt, max_output_tokens, reasoning_effort, tools_schema) can
    share one upstream call instead of each paying for it — see
    `Gateway._complete_deduped`.

    This is a caller-declared flag rather than an inferred one. OmniRoute's
    reference implementation gates dedup on `temperature <= 0.1`, because low
    temperature is what actually makes a call's output deterministic. forgeos's
    `GatewayRequest` has no temperature knob at all, and `reasoning_effort` is
    not a proxy for it: reasoning_effort controls how much internal reasoning
    budget a model spends, not how randomly it samples its final tokens — a
    "low" reasoning-effort call can still be high-temperature, and the
    field's own default here is "medium", so gating on
    `reasoning_effort in ("none", "low")` would silently opt most ordinary
    calls in or out based on a field that says nothing about determinism.
    Guessing wrong here is not a wasted optimization, it is two callers who
    each expected an independent sample getting the same text back — so the
    caller who actually knows the request is idempotent says so explicitly
    instead.
    """

    model_ref: str
    prompt_prefix: str = ""
    prompt_tail: str = ""
    max_output_tokens: int = Field(gt=0)
    reasoning_effort: Literal["none", "low", "medium", "high", "max"] = "medium"
    tools_schema: list | dict | None = None
    dedup: bool = False

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

    `rate_limit_saturation` is 0..1 for how close the provider says it is to
    its advertised quota on *this* response, or None when the response
    carried no usable rate-limit headers. Parsed from `resp.headers` by
    `HttpTransport.complete` (the only place that sees them);
    `Gateway._call_transports` forwards it to
    `HealthTracker.record_saturation` so a nearly-exhausted provider is
    deprioritised before it ever gets a 429, not just skipped after one.
    """

    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_cached_in: int = 0
    model_used: str = ""
    rate_limit_saturation: float | None = None


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

# Token-quota header pairs, tried in priority order; the first pair present on
# the response wins. Anthropic and OpenAI-compatible endpoints (OpenRouter,
# most gateways) both publish these on EVERY response, successes included --
# the proactive signal `getTokenHeaderSaturation` in the reference
# implementation is built on, because it lets a caller back off before the
# 429 instead of only reacting to one.
_TOKEN_HEADER_PAIRS: tuple[tuple[str, str], ...] = (
    ("anthropic-ratelimit-tokens-limit", "anthropic-ratelimit-tokens-remaining"),
    ("x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens"),
    ("anthropic-ratelimit-input-tokens-limit", "anthropic-ratelimit-input-tokens-remaining"),
    ("anthropic-ratelimit-output-tokens-limit", "anthropic-ratelimit-output-tokens-remaining"),
)

# Request-count (RPM) header pairs -- the fallback for an endpoint that only
# publishes a requests-per-minute budget, not a token-per-minute one.
_REQUEST_HEADER_PAIRS: tuple[tuple[str, str], ...] = (
    ("anthropic-ratelimit-requests-limit", "anthropic-ratelimit-requests-remaining"),
    ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests"),
    ("x-ratelimit-limit", "x-ratelimit-remaining"),
)


def _saturation_from_pairs(headers, pairs: tuple[tuple[str, str], ...]) -> float | None:
    for limit_key, remaining_key in pairs:
        limit_str = headers.get(limit_key)
        remaining_str = headers.get(remaining_key)
        if limit_str is None or remaining_str is None:
            continue
        try:
            limit = float(limit_str)
            remaining = float(remaining_str)
        except (TypeError, ValueError):
            continue
        if not limit > 0:
            continue
        used = limit - remaining
        return min(1.0, max(0.0, used / limit))
    return None


def rate_limit_saturation(headers) -> float | None:
    """0..1 how close a provider is to its advertised quota, from response
    headers alone -- or None when nothing usable was present.

    `headers` is anything with a case-insensitive-or-not `.get(key)` (an
    `httpx.Headers` is case-insensitive by construction; a plain dict works
    too for tests). Token headers are tried first -- present on every
    response for both Anthropic and OpenAI-compatible endpoints, so they are
    the universal proactive signal. Request-count headers are the fallback
    for an endpoint that only publishes RPM, not TPM. Missing or malformed
    headers of either kind mean "unknown", returned as None -- never
    interpreted as fully saturated, which would penalise a provider for
    simply not publishing this data.
    """
    saturation = _saturation_from_pairs(headers, _TOKEN_HEADER_PAIRS)
    if saturation is not None:
        return saturation
    return _saturation_from_pairs(headers, _REQUEST_HEADER_PAIRS)


class TransportError(RuntimeError):
    """A transport could not complete the call.

    The gateway tries the next transport in order; if none are left, the
    preflight estimate is recorded as spend and this (or a wrapping
    `TransportError`) is re-raised, because a call that was attempted is not
    a call that gets to be invisible to the ledger.
    """


class ModelUnavailableError(TransportError):
    """This endpoint cannot serve this *model*. The endpoint itself is fine.

    The distinction is load-bearing. A catalogued free slug that the provider has
    since retired returns 404, and treating that as a transport failure marks the
    whole endpoint unhealthy — so the next model, which would have worked, is
    never even attempted and the entire provider drops out of the pool. Observed
    exactly that way: one stale `:free` slug took OpenRouter offline for every
    other model in the same run.

    Free-tier slugs rot constantly, so this is the common case, not the edge case.
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


# Anthropic only caches a block you explicitly mark, and it refuses to cache one
# shorter than this (1024 tokens for most models; 2048 for the smallest). Marking
# a shorter prefix is not merely useless — a cache WRITE is billed at ~1.25x
# normal input, so marking something that will never be read back is a straight
# surcharge. The floor is the whole point of the guard.
CACHE_CONTROL_MIN_TOKENS = 1024

# Providers whose API honours `cache_control`. Anthropic-shaped only: sending
# structured content blocks with this key to an endpoint that does not expect it
# is at best ignored and at worst a 400, so this is an allowlist, never a guess.
CACHE_CONTROL_PROVIDERS = frozenset({"anthropic", "openrouter", "gateway"})


def _content_blocks(prompt: str, prefix: str, *, mark_cache: bool) -> str | list[dict]:
    """The `content` field: a plain string, or blocks with a cache breakpoint.

    Returns a bare string whenever the marker cannot help, so every provider that
    does not do explicit caching sees exactly the payload it saw before — this
    must not change the wire format for OpenAI, DeepSeek or a local model.

    Splitting on the prefix boundary the caller already computed is the whole
    trick: `assemble_prompt` builds `prefix + tail`, and the prefix is byte-stable
    by construction, which is precisely what a cache breakpoint needs.
    """
    if not mark_cache or not prefix or not prompt.startswith(prefix):
        return prompt
    if count_tokens(prefix, "").tokens < CACHE_CONTROL_MIN_TOKENS:
        # Too short to be cached; marking it would buy a write surcharge and no
        # read discount.
        return prompt

    tail = prompt[len(prefix):]
    blocks: list[dict] = [
        {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}
    ]
    if tail:
        blocks.append({"type": "text", "text": tail})
    return blocks


class HttpTransport:
    """An OpenAI-compatible HTTP endpoint. OmniRoute by default — a gateway
    that fans out to many upstream providers and free tiers behind one URL,
    so this single transport covers all of them without forgeos ever branching
    on which upstream provider actually answered.
    """

    def __init__(
        self,
        base_url: str,
        api_key_env: str = "",
        timeout: float = 60.0,
        name: str = "omniroute",
        serves: set[str] | None = None,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._timeout = timeout
        # Which catalog providers this endpoint can answer for. Empty means "any"
        # — the property of a fan-out gateway, not of a direct vendor endpoint.
        # Without this, transports were tried in health order and a
        # `deepseek/deepseek-chat` call could be sent to OpenRouter's endpoint
        # carrying a model id that endpoint has never heard of.
        self.serves: set[str] = serves or set()
        # Whether this endpoint honours Anthropic-style `cache_control` markers.
        # Derived from what it serves rather than configured separately, so a new
        # provider cannot silently get markers it will reject. A fan-out gateway
        # (empty `serves`) is included because it may route to Anthropic, and the
        # min-token floor keeps a wrong guess from costing anything.
        self.supports_cache_control = bool(
            (self.serves and self.serves <= CACHE_CONTROL_PROVIDERS)
            or (not self.serves and name in CACHE_CONTROL_PROVIDERS)
        )

    def complete(
        self,
        *,
        model_id: str,
        prompt: str,
        max_output_tokens: int,
        reasoning_effort: str,
        tools_schema: list | dict | None,
        prompt_prefix: str = "",
    ) -> RawCallResult:
        headers = {"Content-Type": "application/json"}
        if self._api_key_env:
            # Read the env var NAME only; the value never appears in a log or error.
            key = os.environ.get(self._api_key_env, "").strip()
            if key:
                headers["Authorization"] = f"Bearer {key}"

        payload: dict = {
            "model": model_id,
            "messages": [{"role": "user", "content": _content_blocks(
                prompt, prompt_prefix,
                mark_cache=self.supports_cache_control,
            )}],
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
            # The body says *why* — unknown model, no credits, bad key. A bare
            # status code sends the operator to guess at a problem the provider
            # already described. Truncated, and never includes the request
            # headers, so the key cannot ride along into a log.
            body = (resp.text or "").strip().replace("\n", " ")[:300]
            if resp.status_code in (400, 404):
                # Addressed to a model this endpoint does not serve. The endpoint
                # answered correctly; it is the model id that is wrong.
                raise ModelUnavailableError(f"{self.name} HTTP {resp.status_code}: {body}")
            raise TransportError(f"{self.name} HTTP {resp.status_code}: {body}")

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
            rate_limit_saturation=rate_limit_saturation(resp.headers),
        )


class LiteLLMTransport:
    """Fallback reached only when the HTTP endpoint is unreachable. litellm
    owns per-provider auth internally; forgeos still never reads a key value
    here, it only ever hands litellm a model id and lets it resolve auth.
    """

    name = "litellm"
    # litellm resolves auth and routing per provider itself, so it can answer for
    # anything the catalog names.
    serves: set[str] = set()
    # litellm owns its own provider-specific payload shaping, so forgeos must not
    # hand it Anthropic content blocks — it would double-encode them.
    supports_cache_control = False

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
    """One transport per reachable provider: gateways first, then direct vendor
    endpoints, then litellm as the universal fallback.

    Ordering is fixed and deliberate — callers never branch on transport, so this
    is the only place the preference actually lives. Fan-out gateways come first
    because they are the ones that can reach a free tier; a direct vendor endpoint
    always bills.

    This used to build a single OmniRoute transport and nothing else, which meant
    configuring a DeepSeek or OpenRouter key in settings had no effect whatsoever:
    the provider was enabled, authenticated and completely unreachable, and the
    only symptom was every call falling through to litellm or failing outright.
    A key you can enter but cannot use is worse than no field at all.

    Only *usable* providers get a transport — enabled, installed, and holding a
    non-empty key. An unauthenticated provider is left out rather than added and
    left to fail per call, so the failure shows up once in settings rather than
    repeatedly at spend time.
    """
    transports: list[Transport] = []

    gateways = [p for p in settings.providers.values()
                if p.kind is ProviderKind.GATEWAY and p.enabled and p.base_url]
    for p in sorted(gateways, key=lambda x: x.name):
        # A gateway is reachable without a key (a local OmniRoute needs none), so
        # it is not held to the usable() check the direct endpoints are.
        transports.append(
            HttpTransport(base_url=p.base_url, api_key_env=p.env_key, name=p.name)
        )

    apis = [p for p in settings.providers.values()
            if p.kind is ProviderKind.API and p.base_url and p.usable]
    for p in sorted(apis, key=lambda x: x.name):
        transports.append(
            HttpTransport(
                base_url=p.base_url,
                api_key_env=p.env_key,
                name=p.name,
                # Bound to its own provider. Trying `deepseek-chat` against
                # OpenRouter would be a confusing 400 at best and a wrong-model
                # charge at worst.
                serves={p.name},
            )
        )

    transports.append(LiteLLMTransport())
    return transports


# --------------------------------------------------------------------------
# concurrent request dedup
# --------------------------------------------------------------------------

# Bounds the in-flight map's memory under pathological load (many distinct
# dedup-eligible requests all outstanding at once). Carried over from the
# reference implementation's MAX_INFLIGHT; oldest-first eviction only ever
# matters in that pathological case, since a normal entry is removed by its
# own `finally` the moment its call finishes.
_MAX_INFLIGHT = 1000


class _InFlightCall:
    """One shared outcome for a group of coalesced identical requests.

    `done` is set exactly once, by whichever thread is the leader, in a
    `finally` -- so it fires whether the leader's call succeeded or raised.
    Waiters block on it instead of the network.
    """

    __slots__ = ("done", "response", "error")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.response: GatewayResponse | None = None
        self.error: Exception | None = None


def _dedup_key(request: GatewayRequest) -> str:
    """Hash of the fields that determine the call's OUTPUT.

    job_id/task_id/worker_id/remaining_micros are deliberately excluded: they
    identify who is asking and how much they can spend, not what the model
    would produce -- mirroring the reference implementation excluding
    stream/user/metadata as fields that "don't affect LLM output". Full
    SHA-256 (not truncated, unlike the reference's 16-hex-char slice): this
    key gates whether two callers silently share a result, so collision
    resistance matters more here than a shorter key for log readability.
    """
    canonical = {
        "model_ref": request.model_ref,
        "prompt": request.prompt,
        "max_output_tokens": request.max_output_tokens,
        "reasoning_effort": request.reasoning_effort,
        "tools_schema": request.tools_schema,
    }
    blob = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# gateway
# --------------------------------------------------------------------------


class Gateway:
    """One unified call path. See module docstring for the four invariants
    `complete` enforces — the ordering is mandatory, not stylistic. Dedup
    (see `_complete_deduped`) shares the *call and record* steps (3 and 4)
    across concurrent identical requests; it never shares or skips a
    caller's own estimate-and-refuse (steps 1 and 2), so a call one caller
    could not afford never rides in on another's already-approved budget.
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
        dead_models: DeadModelStore | None = None,
        avoidance_log: AvoidanceLog | None = None,
        dedup_wait_timeout: float = 90.0,
    ) -> None:
        self._catalog = catalog
        self._ledger = ledger
        self._settings = settings if settings is not None else default_settings()
        self._transports = transports if transports is not None else default_transports(self._settings)
        self._health = health if health is not None else HealthTracker()
        self._max_context = max_context
        self._dead_models = dead_models if dead_models is not None else DeadModelStore()
        # Optional: when set, a waiter that avoided a real call records it
        # honestly via AvoidanceMethod.DEDUP instead of the saving going
        # unmeasured. None is a fully valid default -- dedup itself works
        # without it, this only affects observability.
        self._avoidance = avoidance_log
        # How long a waiter blocks on the leader before giving up and making
        # its own independent call. Generous relative to HttpTransport's own
        # default 60s timeout, but bounded: a waiter must never hang forever
        # on a leader that itself hangs forever.
        self._dedup_wait_timeout = dedup_wait_timeout
        self._inflight: dict[str, _InFlightCall] = {}
        self._inflight_lock = threading.Lock()

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
        # Always per-caller, even for a dedup-eligible request that ends up
        # sharing steps (3)/(4) below with another call: two callers can send
        # byte-identical prompts while holding very different remaining
        # budgets, and a call one of them could not afford must never ride in
        # on the other's already-approved one.
        verdict = check(estimate, remaining_micros, max_context=self._max_context)
        verdict.raise_if_refused()

        if not request.dedup:
            return self._call_and_record(
                card, prompt, request,
                job_id=job_id, task_id=task_id, worker_id=worker_id, estimate=estimate,
            )
        return self._complete_deduped(
            card, prompt, request,
            job_id=job_id, task_id=task_id, worker_id=worker_id, estimate=estimate,
        )

    # ---------------------------------------------------------------- internals

    def _call_and_record(
        self,
        card: ModelCard,
        prompt: str,
        request: GatewayRequest,
        *,
        job_id: str,
        task_id: str | None,
        worker_id: str,
        estimate,
    ) -> GatewayResponse:
        """Steps (3) and (4): make the call, then record before returning.

        Both the transport call AND turning its raw result into a priced
        response are inside the same guard. A transport that answers fine
        but whose response then fails to price or parse (a malformed usage
        block, a pricing edge case) already cost money upstream just as
        surely as a transport that raised outright -- guarding only the
        transport call and not this step left that spend invisible to the
        ledger, which is the one thing invariant (4) exists to prevent.
        """
        try:
            raw, provider_used = self._call_transports(card, prompt, request)
            response = self._to_response(card, raw, provider_used, prompt)
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

    def _complete_deduped(
        self,
        card: ModelCard,
        prompt: str,
        request: GatewayRequest,
        *,
        job_id: str,
        task_id: str | None,
        worker_id: str,
        estimate,
    ) -> GatewayResponse:
        """Coalesce concurrent identical requests into one `_call_and_record`.

        The first caller for a given key becomes the leader and does the
        real work; every other caller for the same key while the leader is
        still in flight becomes a waiter and shares its outcome -- success or
        failure -- without a second transport call and without a second
        ledger entry. The in-flight entry is removed in a `finally` on the
        leader's path, success or failure, so a crashed leader can never wedge
        every future identical request behind it forever.
        """
        key = _dedup_key(request)

        with self._inflight_lock:
            leader_call = self._inflight.get(key)
            if leader_call is None:
                if len(self._inflight) >= _MAX_INFLIGHT:
                    oldest_key = next(iter(self._inflight), None)
                    if oldest_key is not None:
                        del self._inflight[oldest_key]
                leader_call = _InFlightCall()
                self._inflight[key] = leader_call
                is_leader = True
            else:
                is_leader = False

        if not is_leader:
            if leader_call.done.wait(timeout=self._dedup_wait_timeout):
                if leader_call.error is not None:
                    raise leader_call.error
                assert leader_call.response is not None
                if self._avoidance is not None:
                    self._record_dedup_avoidance(leader_call.response, job_id, task_id)
                return leader_call.response
            # The leader never finished inside the wait window. Never hang a
            # caller forever on someone else's call -- fall back to an
            # independent attempt so this request is not silently lost.
            return self._call_and_record(
                card, prompt, request,
                job_id=job_id, task_id=task_id, worker_id=worker_id, estimate=estimate,
            )

        try:
            response = self._call_and_record(
                card, prompt, request,
                job_id=job_id, task_id=task_id, worker_id=worker_id, estimate=estimate,
            )
        except Exception as e:
            leader_call.error = e
            raise
        else:
            leader_call.response = response
            return response
        finally:
            leader_call.done.set()
            with self._inflight_lock:
                if self._inflight.get(key) is leader_call:
                    del self._inflight[key]

    def _record_dedup_avoidance(
        self, response: GatewayResponse, job_id: str, task_id: str | None
    ) -> None:
        """A waiter that avoided a real call gets that saving measured
        honestly instead of it going unmeasured -- `AvoidanceMethod.DEDUP`
        exists in `forgeos.economy.avoidance` for exactly this, previously
        unused anywhere in the codebase. Baseline is what an independent
        call would have cost: the total tokens the shared response actually
        used; actual is 0, since this waiter's own call never happened.
        """
        assert self._avoidance is not None
        self._avoidance.record(
            job_id,
            AvoidanceMethod.DEDUP,
            baseline_tokens=response.tokens_in + response.tokens_cached_in + response.tokens_out,
            actual_tokens=0,
            baseline_source=(
                "gateway dedup: identical concurrent request; the coalesced "
                "leader call already recorded this spend"
            ),
            task_id=task_id,
        )

    def _call_transports(
        self, card: ModelCard, prompt: str, request: GatewayRequest
    ) -> tuple[RawCallResult, str]:
        """Try transports in health-filtered order. A rate-limited transport
        is skipped without being invoked at all — that is what keeps a 429
        from turning into a retry storm. A transport remembered dead for this
        exact model_ref (see `forgeos.gateway.dead_models`) is skipped the same
        way, for the same reason: a retired free-tier slug should not cost a
        network round trip to rediscover on every call.
        """
        # Only transports that can actually answer for this model's provider. A
        # transport bound to one vendor must never be handed another vendor's
        # model id; `serves` empty means a fan-out gateway that can take anything.
        by_name = {
            t.name: t
            for t in self._transports
            if not getattr(t, "serves", set()) or card.provider in t.serves
        }
        if not by_name:
            raise TransportError(
                f"no transport serves provider {card.provider!r} for {card.ref} — "
                "the provider is probably disabled or missing its API key"
            )
        ordered = self._health.pick_healthy(list(by_name.keys()))

        # A model remembered dead on a given transport (a previous 404 for
        # this exact model_ref) is skipped without a network call -- that is
        # what stops a retired free-tier slug from being rediscovered at the
        # cost of a request on every single run. This is independent of
        # `_health`: a dead model says nothing about the transport's own
        # reachability, so it is filtered here rather than folded into
        # `pick_healthy`.
        candidates = [name for name in ordered if not self._dead_models.is_dead(name, card.ref)]

        if not candidates and ordered:
            raise ModelUnavailableError(
                f"{card.ref}: no transport tried -- all of {ordered} are remembered "
                "dead for this model (use DeadModelStore.clear to reconfigure)"
            )

        # Every transport's failure, not just the last one. When a vendor endpoint
        # fails and the litellm fallback then fails differently, reporting only
        # the final error hides the one that actually explains the problem — the
        # vendor's own 4xx — behind a generic fallback message.
        errors: list[str] = []
        last_err: Exception | None = None
        for name in candidates:
            transport = by_name[name]
            start = time.monotonic()
            try:
                # The prefix travels alongside the assembled prompt, not instead
                # of it: a transport that ignores it sends exactly what it sent
                # before, and one that supports cache breakpoints knows where the
                # byte-stable boundary is without re-deriving it.
                raw = transport.complete(
                    model_id=card.model_id,
                    prompt=prompt,
                    max_output_tokens=request.max_output_tokens,
                    reasoning_effort=request.reasoning_effort,
                    tools_schema=request.tools_schema,
                    prompt_prefix=request.prompt_prefix,
                )
            except ModelUnavailableError as e:
                # NOT a health event. The endpoint answered; this model is simply
                # not on it. Penalising the transport here would drop the whole
                # provider over one stale slug. Remembered instead, terminally,
                # so the next call for this exact model_ref does not have to
                # pay a network round trip to rediscover the same 404.
                self._dead_models.mark_dead(name, card.ref, reason=str(e))
                errors.append(f"{name}: {e}")
                last_err = e
                continue
            except RateLimitError as e:
                self._health.record_rate_limit(name, retry_after_seconds=e.retry_after)
                errors.append(f"{name}: rate limited ({e})")
                last_err = e
                continue
            except Exception as e:  # any other transport failure — try the next one
                self._health.record_failure(name, str(e))
                errors.append(f"{name}: {e.__class__.__name__}: {e}")
                last_err = e
                continue
            self._health.record_success(name, latency_ms=(time.monotonic() - start) * 1000)
            self._health.record_saturation(name, raw.rate_limit_saturation)
            return raw, name

        if last_err is not None:
            detail = " | ".join(errors)
            raise TransportError(f"all transports failed for {card.ref} -> {detail}") from last_err
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
    "ModelUnavailableError",
    "TransportError",
    "assemble_prompt",
    "default_transports",
    "rate_limit_saturation",
]
