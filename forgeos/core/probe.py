"""Does this provider actually WORK, or does it merely exist?

`Provider.usable` answers "enabled, installed, and an env var is set". That is
not the same question as "will this serve a request", and the gap is where a run
dies:

    a key with no credit left        -> env var present -> "ready"
    a CLI installed but signed out   -> binary on PATH  -> "ready"
    ollama with no models pulled     -> daemon running  -> "ready"
    a plan whose quota reset is due  -> env var present -> "ready"

`Provider.authenticated`'s own docstring concedes it: "being installed is as
much as forgeos can honestly claim without running the CLI". This module runs
the check.

EVERY PROBE HERE IS FREE. No completion is ever requested: API providers are
asked to LIST MODELS (an unbilled endpoint on every OpenAI-compatible API),
ollama is asked for its local tags, and a CLI is asked for its version. A
readiness check that costs money is one people turn off, and then it stops
being a readiness check.

Credential values are never read, logged, printed, or returned. The key is
handed to the transport by name; what comes back here is a status, never a
secret.

The statuses deliberately distinguish CAN'T from DIDN'T:
`INSTALLED_UNVERIFIED` is not `READY`. A subscription CLI holds its credential
in a vendor-owned store this code cannot inspect without spending tokens, so
claiming it is ready would be exactly the overstatement this module exists to
delete.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..settings import AuthMode, Provider, ProviderKind


class ProbeStatus(str, Enum):
    READY = "ready"                          # verified: it answered a real request
    INSTALLED_UNVERIFIED = "installed?"      # present, but nothing proves it can serve
    NO_AUTH = "no auth"                      # credential rejected or absent
    NO_CREDIT = "no credit"                  # authenticated, out of quota or balance
    NO_MODEL = "no model"                    # reachable, serves nothing usable
    NOT_INSTALLED = "not installed"
    UNREACHABLE = "unreachable"              # network or binary failure
    DISABLED = "disabled"

    @property
    def routable(self) -> bool:
        """Whether the router may send real work here.

        `INSTALLED_UNVERIFIED` counts: a subscription CLI cannot be verified for
        free, and refusing to route to it would disable the flat-rate seats that
        are the cheapest capacity available. It is surfaced as distinct from
        READY so a person can see which is which -- the point is to stop
        claiming certainty, not to stop using the provider.
        """
        return self in (ProbeStatus.READY, ProbeStatus.INSTALLED_UNVERIFIED)


@dataclass
class ProbeResult:
    provider: str
    status: ProbeStatus
    detail: str = ""
    models_seen: int = 0
    seconds: float = 0.0
    checked_at: float = 0.0

    @property
    def routable(self) -> bool:
        return self.status.routable

    def render(self) -> str:
        mark = {
            ProbeStatus.READY: "[x]",
            ProbeStatus.INSTALLED_UNVERIFIED: "[~]",
        }.get(self.status, "[ ]")
        models = f"  {self.models_seen} model(s)" if self.models_seen else ""
        detail = f"  -- {self.detail}" if self.detail else ""
        return f"  {mark} {self.provider:<14} {self.status.value:<14}{models}{detail}"


@dataclass
class ProbeReport:
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def routable(self) -> list[ProbeResult]:
        return [r for r in self.results if r.routable]

    @property
    def verified(self) -> list[ProbeResult]:
        return [r for r in self.results if r.status is ProbeStatus.READY]

    def render(self) -> str:
        lines = [
            f"Provider probe: {len(self.verified)} verified, "
            f"{len(self.routable)} routable, {len(self.results)} configured",
            "",
        ]
        lines += [r.render() for r in sorted(self.results, key=lambda r: r.provider)]
        unverifiable = [r for r in self.results
                        if r.status is ProbeStatus.INSTALLED_UNVERIFIED]
        if unverifiable:
            lines += [
                "",
                "  [~] means present but unproven: a subscription CLI keeps its",
                "      credential in a vendor store, and confirming it would cost",
                "      tokens. Routable, just not claimed as verified.",
            ]
        dead = [r for r in self.results if not r.routable and r.status is not ProbeStatus.DISABLED]
        if dead:
            names = ", ".join(r.provider for r in dead)
            lines += ["", f"  Not routable: {names}. The router will skip these."]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "schema": "forgeos.provider_probe.v1",
            "results": [
                {
                    "provider": r.provider, "status": r.status.value, "detail": r.detail,
                    "models_seen": r.models_seen, "seconds": round(r.seconds, 3),
                    "checked_at": r.checked_at,
                }
                for r in sorted(self.results, key=lambda r: r.provider)
            ],
        }


# HTTP status -> what it actually means for routing. Kept as data because the
# difference between "your key is wrong" and "your key is fine but you are out
# of credit" is the entire value of this module, and burying it in if/elif is
# how the two get conflated back together.
_STATUS_MEANING: dict[int, tuple[ProbeStatus, str]] = {
    401: (ProbeStatus.NO_AUTH, "credential rejected"),
    403: (ProbeStatus.NO_AUTH, "credential lacks permission"),
    402: (ProbeStatus.NO_CREDIT, "payment required -- out of balance"),
    429: (ProbeStatus.NO_CREDIT, "rate limited or quota exhausted"),
}


def _probe_http(provider: Provider, *, timeout: float) -> ProbeResult:
    """List models over HTTP. Unbilled on every OpenAI-compatible API."""
    import httpx

    key = os.environ.get(provider.env_key, "").strip()
    if provider.auth is AuthMode.API_KEY and not key:
        return ProbeResult(provider.name, ProbeStatus.NO_AUTH,
                           f"{provider.env_key} is not set")

    base = (provider.base_url or "").rstrip("/")
    if not base:
        return ProbeResult(provider.name, ProbeStatus.UNREACHABLE, "no base_url configured")

    url = f"{base}/models"
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    started = time.monotonic()
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    except Exception as exc:
        return ProbeResult(provider.name, ProbeStatus.UNREACHABLE,
                           f"{type(exc).__name__}", seconds=time.monotonic() - started)
    seconds = time.monotonic() - started

    known = _STATUS_MEANING.get(resp.status_code)
    if known:
        status, detail = known
        return ProbeResult(provider.name, status, detail, seconds=seconds)
    if resp.status_code >= 400:
        # Never echo a response body: some providers reflect request headers,
        # and a key must not ride into a log through an error message.
        return ProbeResult(provider.name, ProbeStatus.UNREACHABLE,
                           f"HTTP {resp.status_code}", seconds=seconds)

    try:
        payload = resp.json()
        models = payload.get("data") if isinstance(payload, dict) else payload
        count = len(models) if isinstance(models, list) else 0
    except Exception:
        count = 0
    if count == 0:
        return ProbeResult(provider.name, ProbeStatus.NO_MODEL,
                           "authenticated but serves no models", seconds=seconds)
    return ProbeResult(provider.name, ProbeStatus.READY, models_seen=count, seconds=seconds)


def _probe_ollama(provider: Provider, *, timeout: float) -> ProbeResult:
    """Local daemon. Running with nothing pulled is the common trap -- the
    daemon answers, so every liveness check passes, and the first real call
    fails because there is no model to serve it."""
    import httpx

    base = (provider.base_url or "http://localhost:11434").rstrip("/")
    started = time.monotonic()
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=timeout)
        resp.raise_for_status()
        models = resp.json().get("models") or []
    except Exception as exc:
        return ProbeResult(provider.name, ProbeStatus.UNREACHABLE, f"{type(exc).__name__}",
                           seconds=time.monotonic() - started)
    seconds = time.monotonic() - started
    if not models:
        return ProbeResult(provider.name, ProbeStatus.NO_MODEL,
                           "daemon is up but no models are pulled", seconds=seconds)
    return ProbeResult(provider.name, ProbeStatus.READY, models_seen=len(models), seconds=seconds)


def _probe_cli(provider: Provider, *, timeout: float) -> ProbeResult:
    """Confirm the binary runs. Deliberately does NOT claim the session is
    authenticated: proving that needs a real call, and a readiness check that
    spends tokens gets switched off."""
    binary = provider.command or provider.name
    path = shutil.which(binary)
    if path is None:
        return ProbeResult(provider.name, ProbeStatus.NOT_INSTALLED, f"{binary} not on PATH")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [path, "--version"], capture_output=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ProbeResult(provider.name, ProbeStatus.UNREACHABLE, type(exc).__name__,
                           seconds=time.monotonic() - started)
    seconds = time.monotonic() - started
    if proc.returncode != 0:
        return ProbeResult(provider.name, ProbeStatus.UNREACHABLE,
                           f"--version exited {proc.returncode}", seconds=seconds)
    return ProbeResult(provider.name, ProbeStatus.INSTALLED_UNVERIFIED,
                       "binary runs; sign-in not checked (would cost tokens)",
                       seconds=seconds)


def probe_provider(provider: Provider, *, timeout: float = 10.0) -> ProbeResult:
    """One provider, one free check."""
    if not provider.enabled:
        return ProbeResult(provider.name, ProbeStatus.DISABLED, "disabled in settings")
    if provider.name == "ollama" or "ollama" in (provider.base_url or ""):
        result = _probe_ollama(provider, timeout=timeout)
    elif provider.kind is ProviderKind.CLI:
        result = _probe_cli(provider, timeout=timeout)
    else:
        result = _probe_http(provider, timeout=timeout)
    result.checked_at = time.time()
    return result


def probe_all(providers, *, timeout: float = 10.0) -> ProbeReport:
    """Probe every provider. Serial on purpose: this runs rarely, and a burst of
    parallel auth failures against one vendor is how an IP gets throttled."""
    return ProbeReport([probe_provider(p, timeout=timeout) for p in providers])


def save_report(report: ProbeReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_report(path: Path) -> ProbeReport | None:
    """Last probe, or None. Callers must treat None as "never checked" rather
    than as "nothing works"."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("results") or []
    except (OSError, ValueError):
        return None
    results = []
    for row in rows:
        try:
            results.append(ProbeResult(
                provider=str(row["provider"]), status=ProbeStatus(row["status"]),
                detail=str(row.get("detail", "")), models_seen=int(row.get("models_seen", 0)),
                seconds=float(row.get("seconds", 0.0)),
                checked_at=float(row.get("checked_at", 0.0)),
            ))
        except (KeyError, TypeError, ValueError):
            continue  # one malformed row must not discard the whole report
    return ProbeReport(results)
