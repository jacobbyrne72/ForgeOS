# Security Policy

ForgeOS is a pre-1.0 research/dev harness for cost-governed AI coding tasks —
scheduling, budgets, file leases, and verification around a model call, not a
hosted service. Read this policy with that status in mind: it describes what
the harness actually does, not aspirational guarantees.

## Scope honesty

ForgeOS is deliberately domain-agnostic (see [AGENTS.md](AGENTS.md)): it has
no trading, finance, or other domain logic baked in, and no capability to
take any domain-specific real-world action at all. The actual safety boundary
is narrower and harness-wide: **no irreversible or outward-facing action
without explicit human approval.** That covers pushes, merges to a default
branch, deploys, deletes, secret access, paid-API escalation, and any command
that leaves this machine — ForgeOS doesn't know what the repo it's pointed at
controls, so it treats the blast radius of any of those as real, with no
override flag.

**Dashboard** (`forgeos/dashboard/app.py`) binds `127.0.0.1` only. It is
read-mostly over four stores the harness already writes (`Ledger`,
`EventLog`, `AvoidanceLog`, `LeaseStore`); the one write endpoint is a
per-job **halt flag** — never a delete, never a budget edit. Known
limitation, stated plainly rather than glossed over: binding to loopback does
not fully isolate the dashboard from other software on the same machine — a
WebSocket handshake is exempt from same-origin policy, so another page open
in the same browser could in principle reach `ws://127.0.0.1:8899/ws` while
ForgeOS is running. `TrustedHostMiddleware` restricts the `Host` header, but
don't expose this port beyond localhost, and don't treat "binds loopback" as
"fully isolated from the local machine."

## Secrets

ForgeOS never stores a secret. Provider configs (`forgeos/settings.py`) hold
only the **name** of an environment variable (`env_key`), never a value —
either the key is read from that env var at call time, or auth is delegated
entirely to a vendor CLI that owns its own credential store (subscription
OAuth). Nothing in the ledger, event log, receipts, or dashboard output is
meant to carry a secret value; if you find one, that's a bug, not by design.

**If a key or token was ever exposed** — pasted into chat, committed, logged,
or echoed to a config file — rotate it. Assume it's compromised; don't wait
for confirmation of misuse.

## Reporting a vulnerability

Prefer a **GitHub Security Advisory** on this repository (Security tab →
"Report a vulnerability") over a public issue, so a fix can land before
disclosure. There is no dedicated security email yet — saying so plainly
rather than inventing an address nobody reads. As a pre-1.0 project there is
no formal SLA; expect acknowledgment on a best-effort basis, faster for
anything that looks actively exploitable.

## What counts as a security issue here

In scope:

- **Budget-cap bypass** — any path where total spend can exceed
  `max_usd`/`max_seconds`/`max_iterations` without the governor tripping, or
  where a trip is silently cleared instead of escalated.
- **Lease-safety violation** — two workers granted the same write lease
  concurrently (this has happened once, found and fixed after a full green
  suite — see the README's "Known gaps"; a regression here is a security bug,
  not just a correctness one).
- **Prompt injection that moves routing or budget** — task text, file
  content, or tool output that changes which tier/model a task routes to, or
  that raises a budget ceiling, rather than the router acting only on
  deterministic features (capabilities, measured history, risk class).
- **Secret leakage** — an env var *value* (not just its name) logged,
  echoed into a receipt/report, persisted to the ledger, or exposed via the
  dashboard.

Out of scope (for now, honestly, not evasively): the dashboard has no
authentication layer — don't expose it beyond localhost; live-adapter
backends (vendor CLIs driven as workers) inherit whatever security posture
that vendor CLI itself has, which this project does not audit.
