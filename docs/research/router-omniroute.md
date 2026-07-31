# OmniRoute deep-mine — routing core vs ForgeOS gateway

**Scope:** `vendor/OmniRoute` (v3.8.50, Next.js/TS, 290 providers, `open-sse/` streaming
engine + `src/lib/`) read via `AGENTS.md`/`CLAUDE.md` (already very precise — this doc
trusts their line/file claims and verifies the routing-core ones directly) plus targeted
reads of `open-sse/services/{combo,accountFallback,autoCombo,rateLimitManager,tokenRefresh}/`,
`src/lib/{proxyHealth,credentialHealth,quota,resilience,freeProxyProviders,proxySubscription,combos}/`,
`src/shared/utils/circuitBreaker.ts`. Cross-checked against ForgeOS's
`forgeos/gateway/{client,health,dead_models,free_pool}.py` and `forgeos/core/{quota,router,market}.py`.

One framing correction up front: ForgeOS's own resilience stack (typed `TransportError`
subclasses, `DeadModelStore`'s terminal/temporary split with single-claim half-open probing,
`CapacityMarket`'s shadow pricing) is not a toy next to OmniRoute's — in several places it is
the more principled design. This mine is not one-way admiration; see "OmniRoute does worse" below.

## Inventory

| Mechanism | OmniRoute approach | ForgeOS | Target module | Source (OmniRoute → ForgeOS) |
|---|---|---|---|---|
| Multi-seat-per-provider ranking | `rankByHeadroom`: `headroom = 1 − max(util_5h, util_7d)`, proactively spreads load across a provider's *multiple accounts* before any 429 | **Missing primitive** — `QuotaTracker`/`Router` key by `(provider, model)`, no concept of >1 credential per provider | `core/quota.py`, `core/market.py` | `open-sse/services/combo/headroomRanking.ts:54-91` → n/a (no seat dimension yet) |
| Reset-window quota scoring | `scoreResetAwareQuota`: blends session(5h)+weekly(7d) remaining % against reset-proximity pressure into one continuous score | Partial — `Entitlement.urgency()` is single-window; `CapacityMarket.shadow_price` prices scarcity/surplus but doesn't blend two simultaneous windows (Codex's 5h+7d) | `core/market.py::shadow_price` | `open-sse/services/combo/quotaScoring.ts:222-277` → `forgeos/core/market.py:121-128` |
| Provider circuit breaker | 4-state (`CLOSED/DEGRADED/HALF_OPEN/OPEN`), per-failure-kind thresholds, SQLite-persisted, backoff escalates by open-cycle count (`resetTimeout × 2^(cycles−3)`, capped 16×) | Partial — `HealthTracker` is in-memory only (lost on restart), binary reachable/rate_limited, flat `retry_after_seconds`, no cycle-escalating backoff | `gateway/health.py` | `src/shared/utils/circuitBreaker.ts:68-250` → `forgeos/gateway/health.py` |
| Model-level dead memory | Model-lockout map: `(provider, connection, model)` keyed, exponential backoff, eviction-cap 1000, **no single-claim probe** — just time-based reopen | **Have, arguably better**: `DeadModelStore` terminal-vs-temporary + `claim_probe`/`report_probe` single-claim half-open (prevents stampede) | — (already own) | `open-sse/services/resilience/modelLockoutSettings.ts` ↔ `forgeos/gateway/dead_models.py` |
| Fleet-wide incident detection | `SelfHealingManager.updateIncidentMode`: >50% of pool circuit-OPEN → exploitation-only mode | **Missing** — `HealthTracker`/`Router` reason per-candidate only, no system-wide "most of the fleet is down, stop speculating" signal | `gateway/health.py` | `open-sse/services/autoCombo/selfHealing.ts:125-135` → n/a |
| Non-retryable edge-block detection | `isNonRetryableCloudflareError`: parses CF error 1010 / `browser_signature_banned` / `retryable:false`, short-circuits to 0-cooldown next-target instead of the standard 3× retry wait | **Missing** — `ModelUnavailableError`/`RateLimitError`/`TransportError` split exists but has no WAF-ban-specific case; a Cloudflare edge ban currently reads as a generic `TransportError` and gets retried | `gateway/client.py::_call_transports` | `open-sse/services/accountFallback/nonRetryableUpstream.ts` → `forgeos/gateway/client.py:1052-1074` |
| Session/prompt-cache stickiness | `sessionStickiness.ts`+`promptCacheAffinity.ts`: pins a multi-turn conversation to one connection so provider prompt-caching actually hits across turns, with a `STICKINESS_HEADROOM_THRESHOLD=0.15` escape hatch that breaks the pin before the pinned seat exhausts | **Missing** — `Gateway._dedup_key`/`_complete_deduped` coalesces *exact-duplicate concurrent* requests only; nothing keeps a multi-turn job's later calls on the same transport/seat that already holds the warm cache prefix | `gateway/client.py` (new: seat continuity) | `open-sse/services/combo/sessionStickiness.ts:65-429` → `forgeos/gateway/client.py:296-365` (cache_control machinery exists, seat pinning doesn't) |
| Local admission control | `checkQueueAdmission`: opt-in cap on the *local* rate-limit queue depth, fast-rejects before Bottleneck grows unboundedly under burst | **Missing** — `Gateway` has no queue/admission concept at all; `_inflight` only bounds dedup map size (`_MAX_INFLIGHT=1000`), not incoming request admission | `gateway/client.py` | `open-sse/services/rateLimitManager/admission.ts:26-48` → `forgeos/gateway/client.py:601-694` |
| Pluggable strategy registry | 5 swappable `RouterStrategy` impls (rules/cost/latency/sla-aware/lkgp) behind one interface, runtime-selectable per combo | Partial — `Router.route` is one fixed tiered-escalation algorithm (cheapest-tier-clearing-win-rate-threshold + optional market override); no registry, no per-job strategy swap | `core/router.py` | `open-sse/services/autoCombo/routerStrategy.ts:47-378` → `forgeos/core/router.py:119-337` |
| 12-factor weighted scoring | `calculateFactors`: quota, health, cost⁻¹, latency⁻¹, taskFit, stability(stddev), tier, cacheAffinity, resetWindowAffinity, connectionDensity — one blended score | Have narrower — `Router.tier_of`+`win_rate` (success-rate) + `CapacityMarket.effective_cost` (cash + shadow price); no latency-stddev/task-fitness/cache-affinity dimensions | `core/router.py`, `core/market.py` | `open-sse/services/autoCombo/scoring.ts:118-260` → `forgeos/core/router.py:195-224` |
| Rate-limit header parsing | Per-provider header maps (`STANDARD_HEADERS`/`ANTHROPIC_HEADERS`) parsed reactively (mostly on error path) | **Have, arguably better** — `rate_limit_saturation()` parses the *same* header pairs proactively on **every** response (success included), feeding `HealthTracker.record_saturation` before any 429 | — (already own) | `open-sse/services/rateLimitManager/headers.ts:1-95` ↔ `forgeos/gateway/client.py:196-253` |
| Free-tier residential-proxy rotation | `freeProxyProviders/*` (iplocate/proxifly/oneproxy/webshare) + tri-state health (`ok/fail/inconclusive`, never auto-mutates operator status) | N/A — different problem (IP/stealth proxy rotation, not LLM API routing); ForgeOS has no anti-bot proxy need in the gateway | — | `src/lib/freeProxyProviders/scheduler.ts` → not applicable |
| OAuth token rotation/CAS guard | `tokenRotationMap`+`casGuardShouldSkipPersist`: dedups stale refresh_token callers, CAS-guards concurrent DB persist to stop token-family invalidation storms | Not present in `gateway/`/`core/` as reviewed — ForgeOS's `HttpTransport` reads a static env-var key, no OAuth refresh flow visible in this scope | (adapters, out of scope) | `open-sse/services/tokenRefresh/{rotationMap,casGuard}.ts` → n/a in gateway/core |

## Top 5 ports (novelty × demand ÷ effort)

1. **Session/prompt-cache stickiness with headroom-based auto-release.** ForgeOS's own
   `client.py` already goes to real lengths for prompt-cache economics (`assemble_prompt`
   prefix-then-tail, `cache_control` breakpoints, `CACHE_CONTROL_MIN_TOKENS` floor) but has
   no mechanism to keep a multi-turn job's later calls landing on the *same* transport/seat
   that holds the warm prefix — every call re-resolves independently. This is the single
   biggest miss given how much ForgeOS already invests in caching. Port the pin + 15%-headroom
   escape hatch, not the full session-hash/DB layer. Effort: medium (needs a job-scoped seat
   pin in `Gateway`, no new data model).

2. **Multi-seat headroom ranking.** `computeHeadroom`/`rankByHeadroom` is ~90 lines, pure,
   already a standalone leaf. The blocker is that ForgeOS's `QuotaTracker`/`CapacityMarket`
   have no "N credentials for one provider" concept at all — `select_seat` picks among
   different *providers*, not different *accounts of the same provider*. Worth porting the
   ranking formula now and wiring the seat dimension in as ForgeOS grows beyond one credential
   per provider (it already tracks per-model windows, this is the natural next axis). Effort:
   small for the function, medium for the prerequisite data model.

3. **Circuit-breaker `DEGRADED` state + persisted, cycle-escalating backoff.** `HealthTracker`
   resets to a blank slate every process restart and never escalates backoff for a
   repeat-offender provider — `resetTimeout` is static forever. For a 24/7 daemon this is a
   real gap: a provider that keeps flapping gets probed at the same cadence indefinitely
   instead of backing off further. Port the escalation math (`src/shared/utils/circuitBreaker.ts:241-250`,
   ~10 lines) plus a `DEGRADED` early-warning tier before full `OPEN`. Effort: medium
   (escalation math is trivial; SQLite persistence needs `forgeos/_sqlite.py` wiring, same
   pattern `DeadModelStore` already uses).

4. **Fleet-wide incident mode.** `updateIncidentMode` (~15 lines, pure) stops speculative
   exploration when most of the pool is simultaneously down — a shared outage shouldn't burn
   requests re-discovering the obvious. `Router`/`HealthTracker` currently reason per-candidate
   only. Cheap, high-value for the 24/7 use case. Effort: small.

5. **Cloudflare/WAF-ban non-retryable detection.** `isNonRetryableCloudflareError` is a ~90-line
   pure predicate. `client.py`'s `HttpTransport` is explicitly OmniRoute-shaped ("OmniRoute by
   default"), so a WAF ban surfacing through that exact fan-out path is a live scenario, not
   hypothetical — right now it's indistinguishable from an ordinary `TransportError` and pays
   the full retry ladder before failing. Effort: small, drop-in predicate in `_call_transports`'s
   generic-exception branch.

## Where OmniRoute does worse (for the record)

- **Model dead-memory:** OmniRoute's lockout map is a bare exponential-backoff timer with no
  stampede protection when it reopens. ForgeOS's `claim_probe`/`report_probe` single-claim
  half-open is the more correct circuit-breaker pattern — grants exactly one trial call, not
  every concurrent caller retrying the instant the timer expires.
- **Economic reasoning:** OmniRoute's headroom/reset-window scores rank candidates but never
  express *value* — there's no "this quota will expire unused, spend it before midnight"
  signal. `CapacityMarket.shadow_price`'s negative-price/use-it-or-lose-it framing (`core/market.py:14-22`)
  has no OmniRoute analogue in what was reviewed.
- **Error taxonomy:** OmniRoute's non-retryable detection is regex/JSON-sniffing on error text
  (`isNonRetryableCloudflareError`). ForgeOS's `ModelUnavailableError`/`RateLimitError`/`TransportError`
  are typed exceptions carrying their own retry semantics — structurally sounder than
  string-matching a body.
- **Rotation discipline:** ForgeOS's `ExhaustionSignal` (OK/VENDOR_EXHAUSTED/RATE_LIMITED/NETWORK_ERROR)
  is a single typed gate for "should I rotate seats," documented as converged-on by three
  independent tools. OmniRoute spreads the same decision across cooldown/circuit-breaker/lockout
  layers that each independently infer retryability from status codes — more surface area for
  the three mechanisms to disagree.
