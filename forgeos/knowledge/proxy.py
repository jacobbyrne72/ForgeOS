"""Proxies, for when a fetch is blocked — and only then.

A proxy is a last resort in this design, not a default. Routing every request
through one is slower, costs money on a paid pool, and turns an ordinary fetch
into someone else's traffic. The order that matters:

    1. Fetch directly. Most pages answer.
    2. Blocked? Try another crawler backend first — a different client
       fingerprint fixes more blocks than a different IP does.
    3. Still blocked? Free proxies, if a source is configured.
    4. Still blocked? A paid pool, if the operator configured one.

CREDENTIALS COME FROM THE ENVIRONMENT AND NOWHERE ELSE.

`ProxyEndpoint` has no field capable of holding a password. The URL is
assembled only inside `connection_url()`, at the moment of use, from values
read out of `os.environ` — so a proxy record can be logged, serialised into a
receipt, or printed in a traceback without carrying a secret, because it never
held one. Same discipline as `gateway/keyring.py`, and for the same reason: not
leaking a credential should be a property of the type, not of everyone who
touches it remembering to be careful.

This module never reads a credentials file, never writes one, and never accepts
a password as an argument.

WHAT THIS IS NOT FOR: defeating access controls. A proxy here rotates an IP
that a rate limiter has throttled. It is not a way around a paywall, a login,
or a site that has said no — `crawl.py` reports a CAPTCHA as a block rather
than working around it, and nothing here changes that.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum

# Environment variable NAMES only. No value ever appears in this file.
ENV_PROXY_URL = "FORGEOS_PROXY_URL"            # a single fully-formed proxy URL
ENV_PROXY_HOST = "FORGEOS_PROXY_HOST"
ENV_PROXY_PORT = "FORGEOS_PROXY_PORT"
ENV_PROXY_USER = "FORGEOS_PROXY_USER"
ENV_PROXY_PASS = "FORGEOS_PROXY_PASS"          # read at use, never stored
ENV_PROXY_LIST = "FORGEOS_PROXY_LIST"          # comma-separated URLs

# A proxy that failed recently is skipped rather than retried into the ground.
DEFAULT_COOLDOWN_SECONDS = 300.0


class ProxyKind(str, Enum):
    DIRECT = "direct"    # no proxy — always tried first
    FREE = "free"        # public list; unreliable, unauthenticated, disposable
    PAID = "paid"        # operator's own pool, credentials in the environment


@dataclass
class ProxyEndpoint:
    """One route to the internet.

    Deliberately has NO password field. `connection_url()` reads the secret from
    the environment at call time and returns it to the caller for immediate use;
    the object itself stays safe to log, serialise and print.
    """

    kind: ProxyKind
    label: str                       # human-facing name, never a credential
    host: str = ""
    port: int = 0
    scheme: str = "http"
    user_env: str = ""               # NAME of the env var holding the username
    pass_env: str = ""               # NAME of the env var holding the password
    url_env: str = ""                # NAME of an env var holding a whole URL
    failures: int = 0
    cooled_until: float = 0.0

    def available(self, *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return now >= self.cooled_until

    def connection_url(self) -> str | None:
        """The URL to hand a client, assembled now and not retained.

        Returns None when the environment does not carry what this endpoint
        needs -- a configured-but-unset proxy must fail closed rather than
        silently connect unauthenticated.
        """
        if self.kind is ProxyKind.DIRECT:
            return None
        if self.url_env:
            return os.environ.get(self.url_env, "").strip() or None
        if not self.host or not self.port:
            return None
        user = os.environ.get(self.user_env, "").strip() if self.user_env else ""
        secret = os.environ.get(self.pass_env, "").strip() if self.pass_env else ""
        if self.user_env and not user:
            return None
        if self.pass_env and not secret:
            return None
        auth = f"{user}:{secret}@" if user else ""
        return f"{self.scheme}://{auth}{self.host}:{self.port}"

    def render(self) -> str:
        """Safe by construction: there is no secret in this object to print."""
        where = f"{self.host}:{self.port}" if self.host else (self.url_env or "-")
        state = "ready" if self.available() else f"cooling {self.cooled_until - time.time():.0f}s"
        return f"[{self.kind.value}] {self.label} {where} ({state}, {self.failures} failure(s))"


@dataclass
class ProxyPool:
    """Endpoints in escalation order, with per-endpoint cooldown.

    Always begins with DIRECT. A pool that starts at a proxy has already lost
    the argument -- most fetches do not need one, and the ones that do should
    have earned it by being refused first.
    """

    endpoints: list[ProxyEndpoint] = field(default_factory=list)
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS

    @classmethod
    def from_environment(cls, *, include_direct: bool = True) -> ProxyPool:
        """Build from env var NAMES. Reads no file, accepts no argument secret."""
        endpoints: list[ProxyEndpoint] = []
        if include_direct:
            endpoints.append(ProxyEndpoint(ProxyKind.DIRECT, "direct"))

        if os.environ.get(ENV_PROXY_URL, "").strip():
            endpoints.append(ProxyEndpoint(ProxyKind.PAID, "env-url", url_env=ENV_PROXY_URL))

        host = os.environ.get(ENV_PROXY_HOST, "").strip()
        port = os.environ.get(ENV_PROXY_PORT, "").strip()
        if host and port.isdigit():
            endpoints.append(ProxyEndpoint(
                ProxyKind.PAID, "env-host", host=host, port=int(port),
                user_env=ENV_PROXY_USER if os.environ.get(ENV_PROXY_USER) else "",
                pass_env=ENV_PROXY_PASS if os.environ.get(ENV_PROXY_PASS) else "",
            ))

        listed = os.environ.get(ENV_PROXY_LIST, "")
        for i, raw in enumerate(p.strip() for p in listed.split(",")):
            if not raw:
                continue
            # Stored under a per-entry env name rather than inline, so even a
            # list entry carrying credentials never lands in the object.
            name = f"{ENV_PROXY_LIST}__{i}"
            os.environ.setdefault(name, raw)
            endpoints.append(ProxyEndpoint(ProxyKind.PAID, f"list-{i}", url_env=name))

        return cls(endpoints=endpoints)

    def add_free(self, entries, *, limit: int = 20) -> int:
        """Add unauthenticated public proxies, e.g. from a free-list source.

        `entries` are `host:port` strings. Free proxies are unreliable by
        nature, so they sort AFTER paid ones and are capped: a thousand dead
        endpoints is a thousand timeouts, not resilience.
        """
        added = 0
        for entry in entries:
            if added >= limit:
                break
            host, _, port = str(entry).strip().partition(":")
            if not host or not port.isdigit():
                continue
            self.endpoints.append(
                ProxyEndpoint(ProxyKind.FREE, f"free-{host}", host=host, port=int(port))
            )
            added += 1
        return added

    def _ordered(self) -> list[ProxyEndpoint]:
        rank = {ProxyKind.DIRECT: 0, ProxyKind.PAID: 1, ProxyKind.FREE: 2}
        return sorted(
            (e for e in self.endpoints if e.available()),
            key=lambda e: (rank[e.kind], e.failures, e.label),
        )

    def next_endpoint(self) -> ProxyEndpoint | None:
        """The best available route. Deterministic: same state, same choice.

        No random pick -- a cost or success-rate regression has to be
        attributable to something, and a shuffled route makes that impossible.
        """
        ordered = self._ordered()
        return ordered[0] if ordered else None

    def record_failure(self, endpoint: ProxyEndpoint) -> None:
        """Cool an endpoint off. DIRECT never cools: losing it would mean every
        later fetch is forced through a proxy because one page was blocked."""
        endpoint.failures += 1
        if endpoint.kind is not ProxyKind.DIRECT:
            endpoint.cooled_until = time.time() + self.cooldown_seconds * endpoint.failures

    def record_success(self, endpoint: ProxyEndpoint) -> None:
        endpoint.failures = 0
        endpoint.cooled_until = 0.0

    def render(self) -> str:
        if not self.endpoints:
            return "no routes configured (direct fetch only)"
        return "\n".join(e.render() for e in self.endpoints)


def shuffled_free(entries, *, seed: int) -> list[str]:
    """A reproducible shuffle of free-proxy entries.

    Free lists are ordered, and everyone hammers the top of them. Spreading out
    helps -- but with a CALLER-SUPPLIED seed, so a run can be replayed. An
    unseeded shuffle would make a failed crawl impossible to reproduce.
    """
    items = [str(e).strip() for e in entries if str(e).strip()]
    rng = random.Random(seed)
    rng.shuffle(items)
    return items
