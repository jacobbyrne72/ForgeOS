"""Trusted binary resolution. One resolver, used everywhere hive spawns a process.

The bug this exists to prevent, confirmed against CPython's own source
(`Lib/shutil.py`: "The current directory takes precedence on Windows"):

    shutil.which("ruff")  ->  .\\ruff.bat   # a file sitting in the repo

hive is pointed at repositories it does not trust. A hostile repo that ships
`ruff.bat`, `semgrep.exe`, or `node.bat` in its root gets that file executed by
hive's own security scanner, as the operator, with every credential on the machine.
No model is involved and no prompt injection is required — the attacker only needs
the operator to run hive against the repo.

So: never spawn a bare name, and never accept a binary that lives in the working
directory. Resolve to an absolute path outside the cwd, or report the tool as
absent — which callers already treat as UNAVAILABLE, and UNAVAILABLE blocks a merge.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["ToolNotTrusted", "resolve_tool", "trusted_or_none"]


class ToolNotTrusted(RuntimeError):
    """A binary was found, but not somewhere hive is willing to execute from."""


def _untrusted_dirs() -> set[Path]:
    """Directories a resolved binary must not live in.

    The working directory is the repo under analysis. A binary there is not "on
    PATH", it is bait.
    """
    out: set[Path] = set()
    try:
        out.add(Path.cwd().resolve())
    except OSError:
        pass
    return out


def resolve_tool(name: str, *, cwd: str | os.PathLike[str] | None = None) -> str | None:
    """Absolute path to a trusted `name`, or None.

    Returning None rather than raising is deliberate: every caller already has an
    honest "tool unavailable" branch, and UNAVAILABLE is treated as blocking. A
    missing tool must never silently become a pass, but it also must not crash a
    run — refusing to execute is the safe outcome either way.
    """
    if not name:
        return None

    found = shutil.which(name)
    if found is None:
        return None

    try:
        resolved = Path(found).resolve()
    except OSError:
        return None

    if not resolved.is_file():
        return None

    untrusted = _untrusted_dirs()
    if cwd is not None:
        try:
            untrusted.add(Path(cwd).resolve())
        except OSError:
            pass

    # Reject the binary itself sitting in an untrusted directory, and reject it
    # living anywhere beneath one — a repo can plant `.bin/ruff` just as easily.
    for bad in untrusted:
        if resolved.parent == bad:
            return None
        try:
            if resolved.is_relative_to(bad):
                return None
        except (AttributeError, ValueError):
            pass

    return str(resolved)


def trusted_or_none(name: str, *, cwd: str | os.PathLike[str] | None = None) -> str | None:
    """Alias kept for call sites that read better as a question."""
    return resolve_tool(name, cwd=cwd)
