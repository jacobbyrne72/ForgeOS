"""Turn one operator sentence into an immutable contract.

A `MissionContract` is the thing every worker, the manager and the governor agree
to honour. `compile_mission()` builds one deterministically — zero model calls —
by looking at what is actually present in the target repo (config filenames and
small manifests only, per AGENTS.md rule 3, never a large file).

Immutability is the point, not a nicety. A manager that can quietly edit
acceptance criteria mid-run can make any task "pass", which destroys the only
real completion signal the harness has (AGENTS.md rule 6). So a contract is
mutable up until `mark_started()` is called, and frozen after: any further
attribute assignment raises. The only sanctioned way to change a started
contract is `revise()`, which never mutates `self` — it returns a brand new
contract, one revision higher, with the reason for the change recorded for the
ledger and the human.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from ..contracts import Budget, new_id, now

# Detection never opens a file bigger than this. A bloated "manifest" is not a
# manifest; treat it like any other large file hive refuses to blind-read.
_MAX_MANIFEST_BYTES = 100_000

_ESLINT_CONFIG_NAMES = (
    ".eslintrc",
    ".eslintrc.js",
    ".eslintrc.cjs",
    ".eslintrc.json",
    ".eslintrc.yml",
    ".eslintrc.yaml",
    "eslint.config.js",
    "eslint.config.mjs",
    "eslint.config.cjs",
)


class RepoRef(BaseModel):
    """One repository (or worktree) the mission touches."""

    path: str
    branch: str = "main"


class MissionContract(BaseModel):
    """An objective plus everything needed to know when it is honestly done.

    Mutable until `mark_started()` is called; frozen after. `revise()` is the
    only way to change a contract past that point, and it always returns a new
    object — see module docstring.
    """

    id: str = Field(default_factory=lambda: new_id("mission"))
    objective: str
    repositories: list[RepoRef] = Field(default_factory=list)
    acceptance_commands: list[str] = Field(default_factory=list)
    acceptance_requirements: list[str] = Field(default_factory=list)
    acceptance_unverifiable: bool = False
    constraints: dict[str, Any] = Field(default_factory=dict)
    budget: Budget = Field(default_factory=Budget)
    # role name (see settings.Role) -> preferred worker id. Empty means the
    # router decides every role automatically; this never widens a budget or
    # skips an approval gate, it only hints which worker to prefer.
    routing_preferences: dict[str, str] = Field(default_factory=dict)
    # e.g. {"max_files_open": 20, "max_read_bytes": 100_000}. Advisory caps for
    # workers assembling their own context; not a substitute for scope.
    context_limits: dict[str, int] = Field(default_factory=dict)
    created_at: float = Field(default_factory=now)
    revision: int = 0
    revision_reason: str = ""
    parent_id: str | None = None

    _started: bool = PrivateAttr(default=False)

    @field_validator("objective")
    @classmethod
    def _non_empty_objective(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("objective must not be blank")
        return v.strip()

    @field_validator("revision")
    @classmethod
    def _non_negative_revision(cls, v: int) -> int:
        if v < 0:
            raise ValueError("revision must be >= 0")
        return v

    # ------------------------------------------------------------- lifecycle

    @property
    def started(self) -> bool:
        return self._started

    def mark_started(self) -> None:
        """Freeze the contract at the moment execution begins. Idempotent.

        Bypasses the mutation guard deliberately (this is the one sanctioned
        transition); every field assignment after this raises.
        """
        object.__setattr__(self, "_started", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self).model_fields and self._started:
            raise RuntimeError(
                f"MissionContract {self.id!r} is immutable once started "
                f"(revision {self.revision}); cannot set {name!r}. "
                "Call revise(reason) to get a new contract instead."
            )
        super().__setattr__(name, value)

    # ---------------------------------------------------------------- revise

    def revise(self, reason: str, **overrides: Any) -> MissionContract:
        """Return a NEW contract, one revision higher, reason recorded.

        Never mutates `self` — `self` is unaffected regardless of whether it
        has started. This is the only sanctioned path to change acceptance
        criteria, budget, scope, or anything else on a contract already in
        flight.
        """
        if not reason.strip():
            raise ValueError("revise() requires a non-blank reason")

        data = self.model_dump(exclude={"id", "revision", "revision_reason", "parent_id", "created_at"})
        data.update(overrides)
        return MissionContract(
            **data,
            parent_id=self.id,
            revision=self.revision + 1,
            revision_reason=reason.strip(),
            created_at=now(),
        )


# ----------------------------------------------------------------------------
# Deterministic acceptance-command detection. Reads filenames and small
# manifests only. Never opens anything over _MAX_MANIFEST_BYTES.
# ----------------------------------------------------------------------------


def _small_enough(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size <= _MAX_MANIFEST_BYTES
    except OSError:
        return False


def _read_small(path: Path) -> str | None:
    if not _small_enough(path):
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _detect_python_commands(root: Path) -> list[str]:
    commands: list[str] = []

    pyproject = root / "pyproject.toml"
    has_pytest = False
    has_ruff = False
    if pyproject.exists():
        has_pytest = True  # a Python project manifest; pytest is the deterministic default
        text = _read_small(pyproject)
        if text is not None and "[tool.ruff]" in text:
            has_ruff = True
    elif (root / "pytest.ini").exists() or (root / "setup.cfg").exists():
        has_pytest = True

    if (root / "ruff.toml").exists() or (root / ".ruff.toml").exists():
        has_ruff = True

    if has_pytest:
        commands.append("python -m pytest -q")
    if has_ruff:
        commands.append("ruff check .")
    return commands


def _detect_node_commands(root: Path) -> list[str]:
    package_json = root / "package.json"
    if not package_json.exists():
        return []
    text = _read_small(package_json)
    if text is None:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    commands: list[str] = []
    scripts = data.get("scripts") or {}
    test_script = scripts.get("test") or ""
    if test_script and "no test specified" not in test_script:
        commands.append("npm test")

    if scripts.get("lint"):
        commands.append("npm run lint")
    elif any((root / name).exists() for name in _ESLINT_CONFIG_NAMES):
        commands.append("npx eslint .")

    return commands


def _detect_rust_commands(root: Path) -> list[str]:
    if (root / "Cargo.toml").exists():
        return ["cargo test"]
    return []


def detect_acceptance_commands(root: Path) -> list[str]:
    """What is actually present in the repo, deterministically, zero model calls.

    Filenames and small manifests only — see module docstring. Order is stable
    (python, then node, then rust) but every detector that matches contributes;
    a polyglot repo gets every command it has evidence for, not just the first.
    """
    if not root.is_dir():
        return []
    commands: list[str] = []
    for detected in (
        _detect_python_commands(root),
        _detect_node_commands(root),
        _detect_rust_commands(root),
    ):
        for cmd in detected:
            if cmd not in commands:
                commands.append(cmd)
    return commands


def compile_mission(objective: str, cwd: str, **overrides: Any) -> MissionContract:
    """Deterministic. Zero model calls.

    Detects acceptance commands from what is actually on disk at `cwd`. If
    nothing can be detected, the contract is marked `acceptance_unverifiable`
    rather than silently claiming completion is checkable — a mission with no
    way to verify completion cannot honestly be called done.

    Any field on `MissionContract` can be overridden via keyword argument.
    """
    root = Path(cwd)
    detected = overrides.pop("acceptance_commands", None)
    if detected is None:
        detected = detect_acceptance_commands(root)

    acceptance_unverifiable = overrides.pop("acceptance_unverifiable", None)
    if acceptance_unverifiable is None:
        acceptance_unverifiable = not detected

    repositories = overrides.pop("repositories", None)
    if repositories is None:
        repositories = [RepoRef(path=cwd)]

    return MissionContract(
        objective=objective,
        repositories=repositories,
        acceptance_commands=detected,
        acceptance_unverifiable=acceptance_unverifiable,
        **overrides,
    )


__all__ = [
    "MissionContract",
    "RepoRef",
    "compile_mission",
    "detect_acceptance_commands",
]
