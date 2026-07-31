"""Worktree isolation tests: real git subprocesses, real worktrees, real merges.
No LLM calls, no network."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forgeos.worktrees import (
    create_worktree,
    list_task_worktrees,
    merge_accepted,
    merge_check,
    remove_worktree,
)


def _git(args: list[str], cwd) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _commit_all(cwd, message: str) -> None:
    _git(["add", "-A"], cwd)
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message], cwd)


@pytest.fixture
def repo(tmp_path):
    _git(["init", "-q", "-b", "main"], tmp_path)
    # Mirrors the real repo's .gitignore: worktree dirs are never tracked.
    (tmp_path / ".gitignore").write_text(".forgeos-worktrees/\n", encoding="utf-8")
    (tmp_path / "shared.txt").write_text("base\n", encoding="utf-8")
    _commit_all(tmp_path, "init")
    return tmp_path


# --------------------------------------------------------- create / merge_accepted


@pytest.mark.slow
def test_create_edit_merge_lands_the_commit(repo):
    info = create_worktree(repo, "task1")
    assert Path(info.path).is_dir()
    assert info.branch == "forgeos/task/task1"

    (Path(info.path) / "shared.txt").write_text("base\nfrom task1\n", encoding="utf-8")
    _commit_all(info.path, "task1 change")

    check = merge_check(repo, info.branch)
    assert check.clean is True
    assert check.conflicts == []

    assert merge_accepted(repo, info.branch, message="merge task1") is True
    assert (repo / "shared.txt").read_text(encoding="utf-8") == "base\nfrom task1\n"


@pytest.mark.slow
def test_conflicting_edits_are_detected_and_merge_is_rejected(repo):
    info_a = create_worktree(repo, "task_a")
    info_b = create_worktree(repo, "task_b")

    (Path(info_a.path) / "shared.txt").write_text("base\nfrom A\n", encoding="utf-8")
    _commit_all(info_a.path, "A change")

    (Path(info_b.path) / "shared.txt").write_text("base\nfrom B\n", encoding="utf-8")
    _commit_all(info_b.path, "B change")

    # Land A first so B's merge is a genuine conflict against current HEAD.
    assert merge_accepted(repo, info_a.branch, message="merge A") is True

    check = merge_check(repo, info_b.branch)
    assert check.clean is False
    assert "shared.txt" in check.conflicts

    before = (repo / "shared.txt").read_text(encoding="utf-8")
    assert merge_accepted(repo, info_b.branch, message="merge B") is False

    # Repo left exactly as it was: no half-applied merge, no dangling MERGE_HEAD.
    assert (repo / "shared.txt").read_text(encoding="utf-8") == before
    assert _git(["status", "--porcelain"], repo).stdout.strip() == ""
    assert not (repo / ".git" / "MERGE_HEAD").exists()


# --------------------------------------------------------------------- remove


@pytest.mark.slow
def test_remove_worktree_is_idempotent(repo):
    info = create_worktree(repo, "task_gone")
    remove_worktree(repo, "task_gone")
    assert list_task_worktrees(repo) == []

    remove_worktree(repo, "task_gone")  # already gone -- must not raise

    branch_check = _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{info.branch}"], repo)
    assert branch_check.returncode != 0


@pytest.mark.slow
def test_remove_worktree_can_keep_the_branch(repo):
    info = create_worktree(repo, "task_keep")
    remove_worktree(repo, "task_keep", keep_branch=True)
    assert list_task_worktrees(repo) == []

    branch_check = _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{info.branch}"], repo)
    assert branch_check.returncode == 0


@pytest.mark.slow
def test_list_task_worktrees_excludes_the_main_checkout(repo):
    create_worktree(repo, "task_x")
    create_worktree(repo, "task_y")
    branches = {w.branch for w in list_task_worktrees(repo)}
    assert branches == {"forgeos/task/task_x", "forgeos/task/task_y"}


# ------------------------------------------------------------- task_id sanitization


def test_create_worktree_rejects_path_traversal_in_task_id(tmp_path):
    with pytest.raises(ValueError):
        create_worktree(tmp_path, "../../etc")


def test_create_worktree_rejects_path_separators_in_task_id(tmp_path):
    with pytest.raises(ValueError):
        create_worktree(tmp_path, "sub/dir")


def test_remove_worktree_rejects_path_traversal_in_task_id(tmp_path):
    with pytest.raises(ValueError):
        remove_worktree(tmp_path, "../escape")


def test_create_worktree_rejects_empty_task_id(tmp_path):
    with pytest.raises(ValueError):
        create_worktree(tmp_path, "")
