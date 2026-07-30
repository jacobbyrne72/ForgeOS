"""Path-lease tests: overlap detection, conflict rules, expiry. No LLM calls, no network."""

from __future__ import annotations

import pytest

from hive.leases import LeaseStore, LeaseType, patterns_overlap


# ------------------------------------------------------------ patterns_overlap


def test_recursive_glob_overlaps_nested_file():
    assert patterns_overlap("src/**", "src/router/x.py")


def test_single_level_glob_overlaps_matching_file():
    assert patterns_overlap("src/router/*.py", "src/router/x.py")


def test_sibling_recursive_globs_do_not_overlap():
    assert not patterns_overlap("src/a/**", "src/b/**")


def test_identical_patterns_overlap():
    assert patterns_overlap("src/router/x.py", "src/router/x.py")


def test_overlap_normalizes_windows_backslashes():
    assert patterns_overlap("src\\router\\**", "src/router/x.py")


def test_overlap_normalizes_dot_slash_prefix():
    assert patterns_overlap("./src/router/**", "src/router/x.py")


def test_bare_directory_pattern_is_an_implicit_prefix():
    # No trailing "**" needed -- a directory pattern still covers everything
    # below it, mirroring contracts.Scope.covers' startswith convention.
    assert patterns_overlap("src/router", "src/router/sub/x.py")


def test_disjoint_files_in_same_directory_do_not_overlap():
    assert not patterns_overlap("src/router/x.py", "src/router/y.py")


# ------------------------------------------------------------------- fixtures


@pytest.fixture
def store():
    s = LeaseStore(":memory:")
    yield s
    s.close()


# --------------------------------------------------------------- acquire/conflicts


def test_two_write_leases_on_overlapping_paths_conflict(store):
    first = store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=60)
    assert first is not None
    second = store.acquire("task_b", "repo1", "src/router/x.py", LeaseType.WRITE, ttl_seconds=60)
    assert second is None  # exclusive write must never be granted twice on overlapping paths


def test_write_then_read_on_overlapping_paths_conflict(store):
    store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=60)
    reader = store.acquire("task_b", "repo1", "src/router/x.py", LeaseType.READ, ttl_seconds=60)
    assert reader is None


def test_read_then_write_on_overlapping_paths_conflict(store):
    store.acquire("task_a", "repo1", "src/router/**", LeaseType.READ, ttl_seconds=60)
    writer = store.acquire("task_b", "repo1", "src/router/x.py", LeaseType.WRITE, ttl_seconds=60)
    assert writer is None


def test_read_read_never_conflicts(store):
    first = store.acquire("task_a", "repo1", "src/router/**", LeaseType.READ, ttl_seconds=60)
    second = store.acquire("task_b", "repo1", "src/router/x.py", LeaseType.READ, ttl_seconds=60)
    assert first is not None and second is not None  # shared reads must coexist


def test_non_overlapping_writes_do_not_conflict(store):
    a = store.acquire("task_a", "repo1", "src/a/**", LeaseType.WRITE, ttl_seconds=60)
    b = store.acquire("task_b", "repo1", "src/b/**", LeaseType.WRITE, ttl_seconds=60)
    assert a is not None and b is not None


def test_conflict_is_scoped_to_repo(store):
    store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=60)
    other_repo = store.acquire("task_b", "repo2", "src/router/x.py", LeaseType.WRITE, ttl_seconds=60)
    assert other_repo is not None  # same pattern, different repo -- no shared namespace


def test_same_task_reacquiring_overlapping_pattern_is_a_noop(store):
    first = store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=60)
    second = store.acquire("task_a", "repo1", "src/router/x.py", LeaseType.WRITE, ttl_seconds=60)
    assert second is not None
    # No-op means the *same* lease comes back, not a fresh row -- a task can't
    # accidentally stack duplicate leases on itself by calling acquire twice.
    assert second.id == first.id
    assert len(store.active(repo_id="repo1")) == 1


def test_acquire_accepts_plain_string_lease_type(store):
    lease = store.acquire("task_a", "repo1", "src/**", "write", ttl_seconds=60)
    assert lease is not None
    assert lease.lease_type is LeaseType.WRITE


# ------------------------------------------------------------------------ expiry


def test_expired_lease_conflicts_with_nothing(store):
    # ttl_seconds <= 0 is the deterministic way to manufacture an already-dead
    # lease without sleeping or mocking the clock.
    dead = store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=-1)
    assert dead is not None
    assert store.conflicts("repo1", "src/router/x.py", LeaseType.WRITE) == []
    revived = store.acquire("task_b", "repo1", "src/router/x.py", LeaseType.WRITE, ttl_seconds=60)
    assert revived is not None  # a dead worker's lease must not block the queue forever


def test_expire_stale_reaps_past_ttl_and_releases_the_conflict(store):
    store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=-1)
    reaped = store.expire_stale()
    assert reaped == 1
    assert store.active(repo_id="repo1") == []
    granted = store.acquire("task_b", "repo1", "src/router/x.py", LeaseType.WRITE, ttl_seconds=60)
    assert granted is not None


def test_expire_stale_ignores_leases_still_within_ttl(store):
    store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=60)
    assert store.expire_stale() == 0
    assert len(store.active(repo_id="repo1")) == 1


# ----------------------------------------------------------------------- release


def test_release_frees_the_path_for_another_task(store):
    lease = store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=60)
    store.release(lease.id)
    granted = store.acquire("task_b", "repo1", "src/router/x.py", LeaseType.WRITE, ttl_seconds=60)
    assert granted is not None


def test_release_is_idempotent(store):
    lease = store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=60)
    store.release(lease.id)
    store.release(lease.id)  # releasing twice must not raise or resurrect anything
    assert store.active(repo_id="repo1") == []


def test_release_of_unknown_id_is_a_silent_noop(store):
    store.release("lease_does_not_exist")  # must not raise


# ------------------------------------------------------------------------ active


def test_active_excludes_released_and_expired(store):
    live = store.acquire("task_a", "repo1", "src/live/**", LeaseType.WRITE, ttl_seconds=60)
    released = store.acquire("task_b", "repo1", "src/rel/**", LeaseType.WRITE, ttl_seconds=60)
    store.release(released.id)
    store.acquire("task_c", "repo1", "src/dead/**", LeaseType.WRITE, ttl_seconds=-1)
    ids = {l.id for l in store.active(repo_id="repo1")}
    assert ids == {live.id}


def test_active_filters_by_repo_and_defaults_to_all(store):
    store.acquire("task_a", "repo1", "src/**", LeaseType.WRITE, ttl_seconds=60)
    store.acquire("task_b", "repo2", "src/**", LeaseType.WRITE, ttl_seconds=60)
    assert len(store.active(repo_id="repo1")) == 1
    assert len(store.active()) == 2


# ------------------------------------------------------------------------ holder


def test_holder_returns_the_owning_task_lease(store):
    lease = store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=60)
    found = store.holder("repo1", "src/router/x.py")
    assert found is not None
    assert found.id == lease.id


def test_holder_returns_none_when_nothing_covers_the_path(store):
    store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=60)
    assert store.holder("repo1", "src/other/y.py") is None


def test_holder_ignores_expired_leases(store):
    store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=-1)
    assert store.holder("repo1", "src/router/x.py") is None


def test_conflicts_lists_matching_lease_with_task_id(store):
    store.acquire("task_a", "repo1", "src/router/**", LeaseType.WRITE, ttl_seconds=60)
    hits = store.conflicts("repo1", "src/router/x.py", LeaseType.READ)
    assert len(hits) == 1
    assert hits[0].task_id == "task_a"
