"""Settings — which providers are in play, and who does which job.

Two hard properties:

1. **forgeos never stores a secret.** Providers reference an env var *name* or delegate
   entirely to a vendor CLI that owns its own credential store. Subscription
   providers (Claude, Codex) authenticate through their own browser OAuth flow;
   forgeos shells out to that flow and afterwards can only observe *whether* a
   provider is authenticated. It never sees, copies, or logs the token.
2. **A disabled provider is unreachable, not deprioritised.** Disabling is a gate,
   not a hint — otherwise "disabled" silently means "still spending money".

Roles can be bound manually (you pick the worker) or left automatic (the router
picks from measured history). Manual always wins; that is the point of manual.
"""

from __future__ import annotations

import os
import shutil
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .diagnostics import record_degradation

SETTINGS_PATH = Path(os.path.expanduser("~/.forgeos/settings.json"))


class ProviderKind(str, Enum):
    CLI = "cli"          # a vendor CLI on PATH; owns its own auth
    API = "api"          # HTTP API reached with a key from an env var
    LOCAL = "local"      # runs on this machine, no account, no metering
    GATEWAY = "gateway"  # fans out to many upstream providers


class AuthMode(str, Enum):
    SUBSCRIPTION = "subscription"  # browser OAuth owned by the vendor CLI
    API_KEY = "api_key"            # env var holds the key
    NONE = "none"                  # nothing to authenticate


class Role(str, Enum):
    """Jobs in the company. Bound to workers manually or automatically."""

    MANAGER = "manager"
    PLANNER = "planner"
    IMPLEMENTER = "implementer"
    REVIEWER = "reviewer"
    VERIFIER = "verifier"
    EXPLORER = "explorer"
    SUMMARIZER = "summarizer"


class Provider(BaseModel):
    name: str
    kind: ProviderKind
    auth: AuthMode = AuthMode.NONE
    enabled: bool = True
    command: str = ""          # CLI binary name, resolved against PATH
    login_command: str = ""    # what the *vendor* runs to authenticate
    env_key: str = ""          # NAME of the env var only — never the value
    base_url: str = ""
    capabilities: set[str] = Field(default_factory=set)
    notes: str = ""

    @property
    def installed(self) -> bool:
        if self.kind is ProviderKind.CLI:
            return shutil.which(self.command or self.name) is not None
        return True

    @property
    def authenticated(self) -> bool:
        """Best-effort observation. Never reads a secret's value.

        For API providers, presence of a non-empty env var is the signal. For
        subscription CLIs the vendor owns the credential store, so being installed
        is as much as forgeos can honestly claim without running the CLI.
        """
        if self.auth is AuthMode.NONE:
            return self.installed
        if self.auth is AuthMode.API_KEY:
            return bool(os.environ.get(self.env_key, "").strip())
        return self.installed

    @property
    def usable(self) -> bool:
        return self.enabled and self.installed and self.authenticated

    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        if not self.installed:
            return "not installed"
        if not self.authenticated:
            hint = f" (set ${self.env_key})" if self.env_key else ""
            if self.auth is AuthMode.SUBSCRIPTION and self.login_command:
                hint = f" (run: {self.login_command})"
            return f"needs auth{hint}"
        return "ready"


class RoleBinding(BaseModel):
    """Who does this job. `worker_id` empty means let the router decide."""

    role: Role
    worker_id: str = ""

    @property
    def automatic(self) -> bool:
        return not self.worker_id


class HookEvent(str, Enum):
    """The four lifecycle events a hook may attach to. See `forgeos/hooks.py`
    for the veto/observe split between them."""

    PRE_ROUTE = "pre_route"
    PRE_EXECUTE = "pre_execute"
    POST_GATE = "post_gate"
    JOB_END = "job_end"


class HookDef(BaseModel):
    """One lifecycle hook: an external command Forge invokes at `event`.

    Config-driven only — this model is reachable exclusively through
    `Settings.hooks`, loaded from `~/.forgeos/settings.json`. A task
    description can never define a hook: task text is untrusted input to the
    router (AGENTS.md rule 9), and letting it name a command Forge executes
    would hand untrusted text direct execution.
    """

    event: HookEvent
    command: str
    args: list[str] = Field(default_factory=list)
    timeout_seconds: float = 30.0


class Settings(BaseModel):
    providers: dict[str, Provider] = Field(default_factory=dict)
    roles: dict[Role, RoleBinding] = Field(default_factory=dict)
    max_parallel_workers: int = 4
    default_job_usd: float = 20.0
    default_task_usd: float = 2.0
    hooks: list[HookDef] = Field(default_factory=list)

    # ------------------------------------------------------------- providers

    def enable(self, name: str, on: bool = True) -> None:
        if name not in self.providers:
            raise KeyError(f"unknown provider: {name}")
        self.providers[name].enabled = on

    def disable(self, name: str) -> None:
        self.enable(name, False)

    def usable_providers(self) -> list[Provider]:
        return [p for p in self.providers.values() if p.usable]

    def providers_for(self, capability: str) -> list[Provider]:
        return [p for p in self.usable_providers() if capability in p.capabilities]

    # ----------------------------------------------------------------- roles

    def bind_role(self, role: Role, worker_id: str) -> None:
        """Pin a role to a specific worker. Manual assignment overrides the router."""
        self.roles[role] = RoleBinding(role=role, worker_id=worker_id)

    def unbind_role(self, role: Role) -> None:
        """Hand the role back to automatic routing."""
        self.roles[role] = RoleBinding(role=role)

    def bound_worker(self, role: Role) -> str | None:
        b = self.roles.get(role)
        return b.worker_id if b and not b.automatic else None

    # ------------------------------------------------------------------- io

    def save(self, path: str | Path = SETTINGS_PATH) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path = SETTINGS_PATH) -> Settings:
        """Load settings, falling back to defaults. Never crash on a bad file.

        A corrupt settings file must not brick the harness — it falls back to
        defaults so the operator can fix it from a working system.
        """
        p = Path(path)
        if not p.exists():
            return default_settings()
        try:
            return cls.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception as exc:
            record_degradation(
                "settings", "settings load failed", exc,
                consequence="default settings were used; operator configuration was not applied",
            )
            return default_settings()


# --------------------------------------------------------------------------
# Defaults. CLI list verified present on this machine 2026-07-30 by probing PATH.
# --------------------------------------------------------------------------

_CLI_DEFAULTS: tuple[tuple[str, str, AuthMode, str, set[str]], ...] = (
    # (name, command, auth, login_command, capabilities)
    ("claude", "claude", AuthMode.SUBSCRIPTION, "claude",
     {"edit", "implement", "review", "plan", "debug", "tools"}),
    ("codex", "codex", AuthMode.SUBSCRIPTION, "codex login",
     {"edit", "implement", "review", "debug", "tools"}),
    ("opencode", "opencode", AuthMode.SUBSCRIPTION, "opencode auth login",
     {"edit", "implement"}),
    ("qwen", "qwen", AuthMode.SUBSCRIPTION, "qwen",
     {"edit", "implement", "summarize"}),
    ("kimi", "kimi", AuthMode.SUBSCRIPTION, "kimi",
     {"edit", "implement", "summarize"}),
    ("grok", "grok", AuthMode.SUBSCRIPTION, "grok",
     {"plan", "review", "summarize"}),
    ("pi", "pi", AuthMode.SUBSCRIPTION, "pi",
     {"edit", "implement"}),
    ("droid", "droid", AuthMode.SUBSCRIPTION, "droid",
     {"edit", "implement", "review"}),
    ("copilot", "copilot", AuthMode.SUBSCRIPTION, "copilot",
     {"edit", "review"}),
    ("hermes", "hermes", AuthMode.NONE, "",
     {"triage", "summarize", "search"}),
)


def default_settings() -> Settings:
    providers: dict[str, Provider] = {}

    for name, command, auth, login, caps in _CLI_DEFAULTS:
        providers[name] = Provider(
            name=name,
            kind=ProviderKind.CLI,
            auth=auth,
            command=command,
            login_command=login,
            capabilities=caps,
            notes="Vendor CLI owns its own credentials; forgeos never reads the token.",
        )

    providers["ollama"] = Provider(
        name="ollama",
        kind=ProviderKind.LOCAL,
        auth=AuthMode.NONE,
        command="ollama",
        capabilities={"summarize", "classify", "triage"},
        notes="Local, unmetered. Preferred for cheap preprocessing.",
    )
    providers["gateway"] = Provider(
        name="gateway",
        kind=ProviderKind.GATEWAY,
        auth=AuthMode.API_KEY,
        env_key="FORGEOS_GATEWAY_URL",
        base_url="http://127.0.0.1:8787",
        capabilities={"summarize", "classify", "triage", "review", "plan"},
        notes="OpenAI-compatible gateway fanning out to many providers and free tiers.",
    )
    providers["openrouter"] = Provider(
        name="openrouter",
        kind=ProviderKind.API,
        auth=AuthMode.API_KEY,
        env_key="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        capabilities={"summarize", "classify", "triage", "plan", "review"},
    )
    providers["deepseek"] = Provider(
        name="deepseek",
        kind=ProviderKind.API,
        auth=AuthMode.API_KEY,
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        capabilities={"plan", "review", "summarize", "implement"},
        notes="Manager fallback when the free pool rate-limits.",
    )

    # Every role starts automatic. Bind one only when you want to override.
    roles = {r: RoleBinding(role=r) for r in Role}
    return Settings(providers=providers, roles=roles)


__all__ = [
    "AuthMode",
    "HookDef",
    "HookEvent",
    "Provider",
    "ProviderKind",
    "Role",
    "RoleBinding",
    "SETTINGS_PATH",
    "Settings",
    "default_settings",
]


if __name__ == "__main__":  # pragma: no cover
    s = Settings.load()
    print(f"{'provider':<12} {'kind':<10} {'status'}")
    print("-" * 46)
    for p in sorted(s.providers.values(), key=lambda x: (x.kind.value, x.name)):
        print(f"{p.name:<12} {p.kind.value:<10} {p.status()}")
    print(f"\nusable now: {len(s.usable_providers())}/{len(s.providers)}")
