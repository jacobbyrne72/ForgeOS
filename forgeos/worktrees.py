"""Per-task git worktree isolation. Stops parallel file-editing workers from
ever touching the same working directory.

`leases.LeaseStore` is the in-process guard: it blocks two tasks from
declaring overlapping path claims. This module is the filesystem guard: each
task gets its own checkout, so even a worker that ignores its lease cannot
physically collide with another worker's edits. The two are complementary,
not a replacement for each other -- leases stop the *decision* to overlap,
worktrees stop the *filesystem effect* if a decision is wrong anyway.

Every worktree lives at `<repo_root>/.forgeos-worktrees/<task_id>` on branch
`forgeos/task/<task_id>`, both deterministic from `task_id` alone. Nothing
here persists a path->task mapping; `remove_worktree` recomputes where to
look the same way `create_worktree` computed where to put it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

WORKTREE_DIR = ".forgeos-worktrees"
BRANCH_PREFIX = "forgeos/task/"

# task_id becomes both a directory segment and a branch-name suffix. An
# allowlist of safe characters is simpler and safer than trying to blocklist
# every spelling of ".." or a path separator.
_SAFE_TASK_ID = re.compile(r"[A-Za-z0-9_-]+")


class WorktreeInfo(BaseModel):
    path: str
    branch: str


class MergeCheck(BaseModel):
    clean: bool
    conflicts: list[str] = Field(default_factory=list)
    # Set when clean=False for a reason other than a real conflict (e.g. git
    # too old to understand `merge-tree --write-tree`). Cleanliness is
    # unknown in that case, and unknown must never be reported as clean.
    detail: str | None = None


def _sanitize_task_id(task_id: str) -> str:
    if not task_id or not _SAFE_TASK_ID.fullmatch(task_id):
        raise ValueError(f"unsafe task_id for worktree/branch naming: {task_id!r}")
    return task_id


def _worktree_path(repo_root: str | Path, task_id: str) -> Path:
    return Path(repo_root) / WORKTREE_DIR / task_id


def _branch_name(task_id: str) -> str:
    return f"{BRANCH_PREFIX}{task_id}"


def _run(cmd: list[str], cwd: str | Path, *, check: bool = True,
         timeout: float = 120) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)!r} failed (rc={result.returncode}): {result.stderr.strip()}")
    return result


def _branch_exists(repo_root: str | Path, branch: str) -> bool:
    result = _run(["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
                  repo_root, check=False)
    return result.returncode == 0


def create_worktree(repo_root: str | Path, task_id: str, base: str = "HEAD") -> WorktreeInfo:
    """Create an isolated worktree + branch for `task_id`.

    `git worktree add` creates every missing directory in the destination
    path itself, so there is nothing to `mkdir` first.
    """
    task_id = _sanitize_task_id(task_id)
    repo_root = Path(repo_root)
    path = _worktree_path(repo_root, task_id)
    branch = _branch_name(task_id)
    _run(["git", "worktree", "add", "-b", branch, str(path), base], repo_root)
    return WorktreeInfo(path=str(path), branch=branch)


def remove_worktree(repo_root: str | Path, task_id: str, *, keep_branch: bool = False) -> None:
    """Tear down a task's worktree and, unless `keep_branch`, its branch.

    Idempotent: a task_id that was never created, or already removed, is a
    no-op rather than an error. `git worktree remove` on a path it isn't
    tracking exits fatal, so existence is checked first rather than letting
    git fail and swallowing that failure.
    """
    task_id = _sanitize_task_id(task_id)
    repo_root = Path(repo_root)
    path = _worktree_path(repo_root, task_id)
    branch = _branch_name(task_id)

    if path.exists():
        _run(["git", "worktree", "remove", "--force", str(path)], repo_root)
    else:
        _run(["git", "worktree", "prune"], repo_root, check=False)

    if not keep_branch and _branch_exists(repo_root, branch):
        _run(["git", "branch", "-D", branch], repo_root)


def list_task_worktrees(repo_root: str | Path) -> list[WorktreeInfo]:
    """Every worktree this module manages, for cleanup/doctor tooling.

    Parses `git worktree list --porcelain` and keeps only entries under
    `.forgeos-worktrees/` -- the main checkout is always the first entry and
    is never one of ours.
    """
    repo_root = Path(repo_root)
    result = _run(["git", "worktree", "list", "--porcelain"], repo_root)
    managed_root = (repo_root / WORKTREE_DIR).resolve()

    infos: list[WorktreeInfo] = []
    path: str | None = None
    branch: str | None = None
    for line in [*result.stdout.splitlines(), ""]:
        if line.startswith("worktree "):
            path = line[len("worktree "):].strip()
        elif line.startswith("branch "):
            branch = line[len("branch "):].strip().removeprefix("refs/heads/")
        elif line == "" and path is not None:
            resolved = Path(path).resolve()
            if resolved == managed_root or managed_root in resolved.parents:
                infos.append(WorktreeInfo(path=path, branch=branch or ""))
            path = None
            branch = None
    return infos


def merge_check(repo_root: str | Path, branch: str, into: str = "HEAD") -> MergeCheck:
    """Non-destructive check: would merging `branch` into `into` conflict?

    Uses `git merge-tree --write-tree` (git >= 2.38), which runs a real merge
    entirely in memory and touches neither the working tree nor HEAD. On any
    git too old to understand the flag, or any other outcome that isn't a
    clean result or a definite conflict list, cleanliness is unknown -- and
    unknown is reported as not-clean, since that's the one wrong answer a
    caller could act on.
    """
    result = subprocess.run(
        ["git", "merge-tree", "--write-tree", "--name-only", "--no-messages", into, branch],
        cwd=str(repo_root), capture_output=True, text=True, timeout=60,
    )
    if result.returncode == 0:
        return MergeCheck(clean=True)

    lines = result.stdout.strip("\n").splitlines()
    if result.returncode == 1 and lines:
        return MergeCheck(clean=False, conflicts=lines[1:])  # lines[0] is the discarded tree oid

    return MergeCheck(
        clean=False,
        detail=result.stderr.strip() or "git merge-tree did not report a definite result",
    )


def merge_accepted(repo_root: str | Path, branch: str, *, message: str) -> bool:
    """`--no-ff` merge `branch` into the current HEAD of `repo_root`.

    On any failure the merge is aborted so the repo is left exactly as it
    was -- a half-applied merge is worse than a rejected one.
    """
    # The merge commit is authored by the harness, explicitly. Relying on the
    # operator's global git identity meant a machine without one (every CI
    # runner) failed the merge with nothing but a False — the harness is the
    # merger here, so it signs as itself and works everywhere.
    result = subprocess.run(
        ["git", "-c", "user.name=ForgeOS merge gate",
         "-c", "user.email=merge-gate@forgeos.invalid",
         "merge", "--no-ff", "-m", message, branch],
        cwd=str(repo_root), capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        return True
    subprocess.run(["git", "merge", "--abort"], cwd=str(repo_root), capture_output=True, text=True, timeout=30)
    return False


__all__ = [
    "WorktreeInfo",
    "MergeCheck",
    "create_worktree",
    "remove_worktree",
    "list_task_worktrees",
    "merge_check",
    "merge_accepted",
]
