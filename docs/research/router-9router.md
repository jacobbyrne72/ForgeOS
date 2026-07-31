# Adjacent territory mined from decolua/9router

Source: `vendor/9router` (full local checkout, `9router-app` 0.5.45 — Next.js
dashboard + Express/undici gateway + a separate `mitm/` interception layer +
`open-sse/` translation engine). Read-only survey, nothing built, changed, or
run. ForgeOS status read against `docs/ROADMAP.md` §3 ("Subscription
orchestration" is already the #1 novel-tool lever) plus a live grep of
`forgeos/gateway/{client,health,dead_models,free_pool}.py`,
`forgeos/core/{quota,router}.py`, `forgeos/settings.py`.

9Router's own pitch (`README.md`) is explicit: pool OAuth logins across dozens
of consumer subscription/free-tier accounts and round-robin them to "maximize
subscriptions" and "use every bit before reset." That framing shapes most of
what follows — several mechanisms below are well-built engineering in service
of a goal ForgeOS deliberately doesn't pursue (`settings.py`: "forgeos never
stores a secret... it never sees, copies, or logs the token").

## Inventory

| Mechanism | 9router implementation (receipt) | ForgeOS status | Target |
|---|---|---|---|
| Proxy/interception — explicit endpoint | `/v1/*` OpenAI-compatible route, client points its own base_url at it. `next.config.mjs` rewrite → `src/app/api/v1/*`; `custom-server.js` derives real client IP from the TCP socket, strips attacker-controlled `X-Forwarded-For` | **HAVE** — `gateway/client.py:368` `HttpTransport`, `:483` `LiteLLMTransport` hit explicitly configured endpoints the same way | — |
| Proxy/interception — transparent MITM | Locally-generated root CA installed as OS-trusted (`mitm/cert/rootCA.js:26,98`); hosts-file rewrite + DNS flush per tool (`mitm/dns/dnsConfig.js:104,146`); TLS-terminating proxy with SNI callback + HTTP/2 passthrough (`mitm/server.js:41,178,250`); per-tool intercept/rewrite/forward handlers (`mitm/handlers/{antigravity,copilot,cursor,kiro}.js`) targeting `cloudcode-pa.googleapis.com`, `api.individual.githubcopilot.com`, `codewhisperer.us-east-1.amazonaws.com`/`runtime.us-east-1.kiro.dev`, `api2.cursor.sh` (`mitm/config.js:16-23`); requires OS elevation (`mitm/winElevated.js`) | **MISSING, by design** — see DO-NOT-PORT #1 | N/A |
| Account/seat pooling + rotation | `providerConnections` SQLite table w/ `priority` (`db/schema.js:34`); OAuth-token pool per provider, `connectionsRepo.js` CRUD/reorder; fallback `while(true)` loop excludes tried `connectionId`s (`sse/handlers/chat.js:195-282`); `stickyRoundRobinLimit`/`comboStickyRoundRobinLimit` settings | **HAVE — different shape, arguably cleaner** — `core/quota.py:146` `SeatChoice`/`:547` `select_seat` picks the available seat whose window closes soonest first, over the operator's *own* accounts under vendor-CLI auth, not a harvested cross-account pool | — |
| Quota exhaustion detection | `open-sse/services/accountFallback.js:4-49` `checkFallbackError(status, errorText, backoffLevel)` → per-provider rule table + flat exponential backoff; separate model-level lock (`:147` `buildModelLockUpdate`) from account-level lock | **HAVE — stronger** — `core/quota.py:67` typed `ExhaustionSignal` (VENDOR_EXHAUSTED/RATE_LIMITED/NETWORK_ERROR) with `:78` `should_rotate` branching on signal, not raw status code; quota tracked per (provider, model) already | see Port #1 |
| Health/cooldown | `open-sse/services/usage/claude.js:19` `OAUTH_429_COOLDOWN_MS` isolates a noisy usage-polling endpoint from the main call path | **HAVE — stronger** — `gateway/health.py:40` `HealthTracker` (reachability + header-saturation deprioritization); `gateway/dead_models.py:86-87` circuit breaker distinguishing TERMINAL vs TEMPORARY, exponential backoff 30s→3600s cap | see Port #2 |
| Request translation | `open-sse/translator/*` pivots through OpenAI as intermediate format; translators self-register via `register(from,to,reqFn,resFn)` import side-effect; exact `source:target` pairs get a **direct route** skipping the lossy double-hop for fragile fields (thinking blocks, tool ids, non-base64 images, `is_error`); binary/protobuf upstreams (Kiro EventStream, Cursor protobuf, NDJSON) live in their own executor, not the translator (`CLAUDE.md:67-70,90`) | **PARTIAL** — `gateway/client.py:197-243` handles Anthropic-vs-OpenAI-compat shape differences ad hoc inside `HttpTransport`; no general N-way engine, delegates most translation to vendored LiteLLM/OmniRoute | see Port #3 |
| Streaming handling | `mitm/handlers/base.js:39` `pipeSSE` (raw passthrough); `:76` `pipeTransformedSSE` (parse `data:` lines → `transformFn(parsed, state)` → re-emit, explicit flush on `null`); `:158` `pipeTransformedEventStream` (same contract, emits binary AWS EventStream frames for Kiro's wire format) | **PARTIAL/unverified** — `gateway/client.py` streaming path not deep-read this pass | see Port #3 (shares surface) |
| Auth/session management | Dashboard session cookie (`JWT_SECRET`), `INITIAL_PASSWORD` default `123456`, API keys bound to `machineId` (`node-machine-id`, `MACHINE_ID_SALT`); per-provider OAuth PKCE catalog mimicking each vendor's own CLI login (`oauth/providers/*.js` × 20, generic `oauth/services/oauth.js:10` `OAuthService`) — tokens land in 9router's own DB, not the vendor CLI's | **Different by design** — `settings.py:1-13` hard rule: never store a secret, shell out to the vendor's own browser OAuth, only observe *whether* it succeeded | see Port #4 (catalog as reference only) |
| Config/UI surface | Next.js dashboard (~100 API routes under `src/app/api/`); SQLite 4-tier adapter fallback `bun:sqlite → better-sqlite3 → node:sqlite → sql.js` so install never hard-fails without build tools (`db/driver.js`); one-click relay-pool deploy to Cloudflare/Deno/Vercel (`app/api/proxy-pools/{cloudflare,deno,vercel}-deploy`) | **MISSING** — ForgeOS is CLI/library-first, no dashboard, by product shape | N/A |
| "Staying within ToS" | **Nothing found.** No disclaimer, no safety valve anywhere in README/CHANGELOG/docs limiting cross-account pooling or MITM scope | **N/A** — ForgeOS routes only through operator-owned seats + paid APIs | — |

## Top 5 ports (novelty × demand ÷ effort)

1. **Per-provider error-text → signal corpus.** 9router's `accountFallback.js`
   rule table is, across dozens of providers, a maintained answer to "what
   does provider X's 429/403 body actually say when it's quota-exhausted vs.
   merely rate-limited." ForgeOS's `ExhaustionSignal` classifier
   (`core/quota.py:67-82`) already makes the right typed distinction but has
   to infer it per-provider; mining this table as seed patterns for
   `parse_usage_report` (`core/quota.py:180`) directly raises `should_rotate`
   accuracy for near-zero effort. Receipts: `open-sse/services/accountFallback.js:1-49`.
2. **Isolate side-channel cooldowns from main-path health.** `usage/claude.js:19`'s
   pattern — a metadata/usage-polling call gets its own per-token cooldown so
   its 429 never touches the primary chat-completion path's availability —
   is a one-line-of-discipline check worth applying to `core/quota_ingest.py`:
   confirm a noisy usage-report poll can't flip `HealthTracker`/`DeadModelStore`
   state for the actual serving path. Receipts: `open-sse/services/usage/claude.js:17-86`.
3. **Stream transform-in-flight contract.** `transformFn(parsedChunk, state) => output|null`
   with an explicit flush-on-`null` end signal, and a *separate* code path for
   non-SSE binary wire formats (AWS EventStream) — worth keeping as the shape
   if ForgeOS ever needs to re-frame a vendor stream itself (e.g. for a live
   ledger/dashboard event feed) instead of leaning on LiteLLM's normalization.
   Not urgent; no current ForgeOS consumer. Receipts: `mitm/handlers/base.js:76-141,152-224`.
4. **OAuth-parameter catalog as a read-only cross-check.** The 20 files under
   `oauth/providers/*.js` are current, exact PKCE parameters (client_id,
   scopes, authorize/token URLs) for every CLI ForgeOS already treats as
   "shell out to its own login" (Claude, Codex, Cursor, Gemini-cli, GitHub,
   Kiro, ...). Useful only to sanity-check `forge doctor`'s provider-detection
   logic against — never to run the flow. Receipts: `oauth/providers/claude.js:1-60`
   (representative), `oauth/providers/index.js`.
5. **Dependency-fallback-chain principle.** Not code (`bun:sqlite → better-sqlite3
   → node:sqlite → sql.js` is npm-specific) but the general pattern — an
   ordered list of persistence backends so install never hard-fails on a
   missing native/build toolchain — worth keeping in mind the next time
   ForgeOS adds an optional compiled dependency. Speculative, lowest priority.
   Receipts: `src/lib/db/driver.js`, `package.json` `optionalDependencies` comment.

## DO-NOT-PORT (illegitimate or ToS-questionable — flagged explicitly)

1. **MITM root-CA + hosts-file interception of native subscription apps**
   (`mitm/manager.js`, `mitm/cert/rootCA.js`, `mitm/dns/dnsConfig.js`,
   `mitm/handlers/*`). Installs a locally-trusted CA and rewrites the OS
   hosts file so Antigravity/Copilot/Cursor/Kiro's own official traffic is
   transparently terminated and rewritten to route through pooled
   third-party credentials — subverting the vendor's own TLS trust model
   without the user's client knowing its request never reached the vendor.
   ForgeOS's entire auth posture (`settings.py:1-13`) is the opposite of
   this on purpose; do not narrow that gap.
2. **Fingerprint spoofing.** `CHANGELOG.md:109` — "Antigravity: align
   provider fingerprint with IDE Desktop 2.1.1" — deliberately matches the
   exact official client version string so the vendor's abuse detection
   can't distinguish MITM-proxied traffic from the real app. Textbook
   anti-detection evasion.
3. **Cross-account OAuth harvesting into a shared pool** (`oauth/providers/*.js`
   × ~20, `providerConnections` table) explicitly marketed to "maximize
   subscriptions" across many accounts. Most consumer subscription ToS
   (Claude Pro/Max, GitHub Copilot Individual, Cursor, AWS Kiro) prohibit
   sharing/proxying/automating a seat this way. The `machineId`-bound API
   key (`apiKeys.machineId`, `MACHINE_ID_SALT`) even suggests awareness of
   per-device licensing enforcement it's simultaneously routing around.
4. **Serverless relay-pool deploy targets**
   (`app/api/proxy-pools/{cloudflare,deno,vercel}-deploy`). One-click
   deploy of a router instance to rotate egress IP across edge platforms
   reads as IP-based rate-limit/abuse-detection evasion, not a legitimate
   multi-region need.
5. **`codex-reset-credits` route** (`app/api/usage/[connectionId]/codex-reset-credits/route.js`) —
   ambiguous rather than clearly illegitimate; it wraps what looks like an
   official "rate-limit reset credit" redemption (server-generated one-time
   redeem id, `getCodexRateLimitResetCredits`/`consumeCodexRateLimitResetCredit`
   naming). Automating that redemption across a harvested account pool
   amplifies whatever single-account allowance it represents. Flagged as
   unverified/borderline, not confirmed abuse.
