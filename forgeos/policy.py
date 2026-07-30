"""Mechanical enforcement of the read rules.

Instruction alone does not work. "Prefer bounded reads" gets rationalised past —
*this file is probably small*, *I'll just peek* — and the rationalisation is
invisible until the bill arrives. A gate that returns an error cannot be talked
past, so the wasteful path is made impossible rather than discouraged.

The escape hatch is load-bearing: **targeted reads always pass.** A gate that hides
data an agent genuinely needs is worse than the waste it prevents, because the agent
will find a more expensive way around it. So the rule is not "read less", it is
"say which part you want".
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

# Above this, an unbounded read must justify itself. Chosen so ordinary source
# files pass untouched and only genuine bulk hits the gate.
MAX_UNBOUNDED_BYTES = 100 * 1024

# Never worth reading into a model's context. Generated, vendored, or enormous.
DENY_DIR_PARTS = frozenset(
    {
        "node_modules", ".git", "__pycache__", ".venv", "venv", "site-packages",
        "dist", "build", ".next", ".nuxt", "target", ".pytest_cache", ".mypy_cache",
        ".ruff_cache", "coverage", ".tox", "vendor",
    }
)

DENY_SUFFIXES = frozenset(
    {
        ".lock", ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin", ".wasm",
        ".zip", ".tar", ".gz", ".7z", ".rar", ".jar", ".pdf", ".png", ".jpg",
        ".jpeg", ".gif", ".webp", ".mp4", ".mov", ".mp3", ".wav", ".sqlite",
        ".db-wal", ".db-shm", ".pack", ".idx",
    }
)

DENY_FILENAMES = frozenset(
    {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
        "Cargo.lock", "composer.lock", "Gemfile.lock", "uv.lock",
    }
)

# Reading these is a credential-exposure risk regardless of size. Denied outright
# rather than size-gated — a small secret is still a secret.
SECRET_HINTS = (
    ".env", "id_rsa", "id_ed25519", ".pem", ".key", ".keystore", "credentials",
    "secrets", ".npmrc", ".pypirc", ".netrc", "wallet", "seed.txt", "private_key",
)


class Verdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class Decision(BaseModel):
    verdict: Verdict
    reason: str = ""
    # A denial that does not say what to do instead just gets worked around.
    suggestion: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    @property
    def exit_code(self) -> int:
        """2 is the hard-deny signal a PreToolUse hook must return."""
        return 0 if self.allowed else 2


class ReadRequest(BaseModel):
    path: str
    offset: int | None = None
    limit: int | None = None
    size_bytes: int | None = Field(default=None, description="None means: stat it")

    @property
    def bounded(self) -> bool:
        """A read that names a range. Always permitted — this is the escape hatch."""
        return self.offset is not None or self.limit is not None

    def resolved_size(self) -> int:
        if self.size_bytes is not None:
            return self.size_bytes
        try:
            return os.path.getsize(self.path)
        except OSError:
            # Cannot stat it, so cannot judge it on size. Do not invent a reason
            # to block; other rules still apply.
            return 0


def _posix_parts(path: str) -> list[str]:
    return [p for p in path.replace("\\", "/").split("/") if p and p != "."]


def looks_like_secret(path: str) -> bool:
    low = path.replace("\\", "/").lower()
    name = low.rsplit("/", 1)[-1]
    return any(h in name or low.endswith(h) for h in SECRET_HINTS)


# Optional external digest tool. Detected, never assumed — forgeos must be useful on a
# machine that has never heard of codex-brain, which is every machine but one.
DIGEST_TOOL_ENV = "FORGEOS_DIGEST_TOOL"
PACK_TOOL_ENV = "FORGEOS_PACK_TOOL"


def _external_tool(env_var: str, filename: str) -> str | None:
    """Locate an optional helper: explicit env var first, then a conventional path."""
    configured = os.environ.get(env_var, "").strip()
    if configured and Path(configured).exists():
        return configured
    candidate = Path(os.path.expanduser("~/.codex-brain")) / filename
    return str(candidate) if candidate.exists() else None


def ripper_command(path: str, budget: int = 5000) -> str:
    """What to run instead of an unbounded read.

    Falls back to forgeos's own bounded-read advice when no external digest tool is
    installed. Suggesting a command that does not exist on the user's machine is
    worse than suggesting nothing — it reads as a broken tool rather than a rule.
    """
    tool = _external_tool(DIGEST_TOOL_ENV, "digest.py")
    if tool:
        return f'python "{tool}" file "{path}" --budget {budget}'
    return (
        f'Read "{path}" with offset/limit for the lines you need, '
        f'or search it first: rg -n "<pattern>" "{path}"'
    )


def pack_command(objective: str, budget: int = 12000) -> str:
    tool = _external_tool(PACK_TOOL_ENV, "atlas.py")
    if tool:
        return f'python "{tool}" pack "{objective}" --budget {budget}'
    return f'rg -n "{objective}" --glob "!node_modules" --glob "!.git" -m 20'


def check_read(req: ReadRequest) -> Decision:
    """The gate. Order matters: secrets and generated paths outrank size."""
    parts = _posix_parts(req.path)
    name = parts[-1] if parts else ""
    suffix = Path(name).suffix.lower()

    if looks_like_secret(req.path):
        return Decision(
            verdict=Verdict.DENY,
            reason=f"{name} looks like a credential file; reading it risks exposing a secret",
            suggestion="If a value is genuinely needed, read the env var NAME only — never the file.",
        )

    hit = next((p for p in parts[:-1] if p in DENY_DIR_PARTS), None)
    if hit:
        return Decision(
            verdict=Verdict.DENY,
            reason=f"path is inside '{hit}' — generated or vendored, never worth model context",
            suggestion="Search it with rg if you must, but do not read it into context.",
        )

    if name in DENY_FILENAMES:
        return Decision(
            verdict=Verdict.DENY,
            reason=f"{name} is a lockfile; thousands of pinned versions and no reasoning value",
            suggestion="Read the manifest (package.json / pyproject.toml) instead.",
        )

    if suffix in DENY_SUFFIXES:
        return Decision(
            verdict=Verdict.DENY,
            reason=f"'{suffix}' is binary or generated content",
            suggestion="Use markitdown for documents, or a targeted tool for binaries.",
        )

    # The escape hatch, checked last among size concerns: a request that names a
    # range has already done the thinking the gate exists to force.
    if req.bounded:
        return Decision(verdict=Verdict.ALLOW, reason="bounded read (offset/limit given)")

    size = req.resolved_size()
    if size > MAX_UNBOUNDED_BYTES:
        kb = size // 1024
        return Decision(
            verdict=Verdict.DENY,
            reason=(
                f"unbounded read of {kb} KB (limit {MAX_UNBOUNDED_BYTES // 1024} KB) — "
                f"this would spend roughly {size // 4:,} tokens on one file"
            ),
            suggestion=(
                f"{ripper_command(req.path)}\n"
                f"…then Read with offset/limit for the exact lines you need."
            ),
        )

    return Decision(verdict=Verdict.ALLOW, reason=f"{size} bytes, under the unbounded limit")


def check_tool_call(tool_name: str, tool_input: dict) -> Decision:
    """Dispatch a PreToolUse payload. Unknown tools are allowed.

    Defaulting to allow is deliberate: a gate that blocks anything it does not
    recognise turns every new tool into an outage.
    """
    if tool_name not in ("Read", "NotebookRead"):
        return Decision(verdict=Verdict.ALLOW, reason=f"{tool_name} is not size-gated")

    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if not path:
        return Decision(verdict=Verdict.ALLOW, reason="no path in payload")

    return check_read(
        ReadRequest(
            path=str(path),
            offset=tool_input.get("offset"),
            limit=tool_input.get("limit"),
        )
    )


__all__ = [
    "DENY_DIR_PARTS",
    "DENY_FILENAMES",
    "DENY_SUFFIXES",
    "Decision",
    "MAX_UNBOUNDED_BYTES",
    "ReadRequest",
    "SECRET_HINTS",
    "Verdict",
    "check_read",
    "check_tool_call",
    "looks_like_secret",
    "pack_command",
    "ripper_command",
]
