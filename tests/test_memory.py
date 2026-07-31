"""Tests for the persistent memory layer (`forgeos/knowledge/memory.py`).

Covers the contract from the top of that module: a write/read round trip,
budgeted retrieval that never dumps raw rows, provenance that survives and can
never silently read as a fact, honest FTS5-degradation reporting, bounded growth
via pruning, the secret-shaped-value guard, and concurrent writes through the
guarded connection.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from forgeos.contracts import now
from forgeos.knowledge import MemoryKind, MemoryStore, Provenance
from forgeos.knowledge.memory import MemoryStore as _MemoryStoreType
from forgeos.knowledge.memory import _secret_hit
from forgeos.knowledge.vault import Vault


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def _remember(store, **kw):
    kw.setdefault("kind", MemoryKind.FACT)
    kw.setdefault("subject", "vault.py write_page")
    kw.setdefault("body", "Vault.write_page(folder, title, summary, body) indexes into index.md")
    kw.setdefault("provenance", Provenance.MEASURED)
    kw.setdefault("source_ref", "job:job_abc/task:task_1")
    return store.remember(**kw)


# ------------------------------------------------------------------ round trip


def test_write_read_round_trip(store):
    item = _remember(store)
    got = store.get(item.id)
    assert got is not None
    assert got.id == item.id
    assert got.subject == item.subject
    assert got.body == item.body
    assert got.provenance == Provenance.MEASURED
    assert got.source_ref == item.source_ref
    assert store.count() == 1


def test_round_trip_preserves_topics_and_correlation_ids(store):
    item = _remember(store, topics=["vault", "knowledge"], job_id="job_1", task_id="task_1")
    got = store.get(item.id)
    assert got.topics == ["vault", "knowledge"]
    assert got.job_id == "job_1"
    assert got.task_id == "task_1"


def test_unknown_id_returns_none_not_a_crash(store):
    assert store.get("mem_nope") is None


# ------------------------------------------------------------- budgeted recall


def test_recall_never_exceeds_the_token_budget(store):
    for i in range(30):
        _remember(
            store,
            kind=MemoryKind.FILE_RELEVANCE,
            subject=f"file relevance {i}",
            body=f"path forgeos/module_{i}.py mattered because reason number {i} " * 5,
            source_ref=f"job:job_{i}",
            topics=["relevance"],
        )
    capsule = store.recall("relevance path module", budget=200)
    assert capsule.total_tokens <= 200
    # Precondition: there really was more available than fit, so this exercises
    # the budget rather than trivially passing on a tiny result set.
    assert store.count() == 30


def test_recall_returns_empty_capsule_for_empty_store(store):
    capsule = store.recall("anything", budget=500)
    assert capsule.items == []
    assert capsule.total_tokens == 0


def test_recall_rejects_a_blank_query(store):
    with pytest.raises(ValueError):
        store.recall("   ", budget=500)


def test_recall_has_a_usable_default_budget_when_the_caller_omits_one(store):
    _remember(store)
    capsule = store.recall("vault write_page")  # no budget kwarg
    assert capsule.total_tokens <= 2_000
    assert capsule.items, "the default budget must be large enough to admit at least one card"


def test_recall_result_is_ranked_card_refs_not_a_raw_dump(store):
    """The load-bearing rule: recall() returns CapsuleBuilder-admitted CARD
    refs, never the raw MemoryItem rows pasted into a prompt."""
    _remember(store)
    capsule = store.recall("vault write_page", budget=5000)
    assert len(capsule.items) >= 1
    ref = capsule.items[0]
    assert ref.ref.startswith("card://")
    assert ref.kind.value == "card"


# --------------------------------------------------------------- provenance


def test_provenance_survives_the_round_trip(store):
    measured = _remember(store, provenance=Provenance.MEASURED, subject="measured fact")
    asserted = _remember(
        store,
        provenance=Provenance.ASSERTED,
        subject="asserted guess",
        body="an agent guessed this; nobody has checked it",
    )
    assert store.get(measured.id).provenance is Provenance.MEASURED
    assert store.get(asserted.id).provenance is Provenance.ASSERTED


def test_an_asserted_guess_never_renders_as_measured_in_recall(store):
    """The hard guarantee: provenance is baked into the text a worker actually
    reads, not just stored as metadata a caller could drop or ignore."""
    _remember(
        store,
        provenance=Provenance.ASSERTED,
        subject="unverified guess about routing",
        body="the router probably prefers cheap workers first",
        source_ref="agent:inference",
    )
    capsule = store.recall("unverified guess routing", budget=5000)
    assert capsule.items, "precondition: the item must actually have been admitted"
    assert any("asserted" in item.reason for item in capsule.items)
    assert not any("measured" in item.reason for item in capsule.items)


def test_provenance_is_a_mandatory_keyword_argument(store):
    """No default exists -- a caller cannot forget to say measured or asserted."""
    with pytest.raises(TypeError):
        store.remember(MemoryKind.FACT, "s", "b", source_ref="x")  # type: ignore[call-arg]


# --------------------------------------------------------------- FTS5 honesty


def test_health_reports_fts5_mode_when_available(store):
    health = store.health()
    assert health["fts5_available"] is True
    assert health["search_mode"] == "fts5_bm25"


def test_try_create_fts_returns_false_when_the_module_is_absent():
    """Unit-test the detection path itself: a build without FTS5 raises
    'no such module: fts5' on CREATE VIRTUAL TABLE, and the store must catch
    exactly that and report degraded mode rather than crashing or lying."""
    probe = _MemoryStoreType.__new__(_MemoryStoreType)

    class _NoFts5Conn:
        def execute(self, sql, params=()):
            if "fts5" in sql:
                raise sqlite3.OperationalError("no such module: fts5")
            raise AssertionError("unexpected SQL in this stub")

    probe._conn = _NoFts5Conn()
    assert probe._try_create_fts() is False


def test_search_is_still_correct_when_fts5_is_unavailable(store):
    """Simulate a runtime without the FTS5 extension by forcing the flag
    `_try_create_fts` would have set. `health()` must say so, and `search()`
    must still return the right rows via the LIKE fallback -- degraded
    ranking, not degraded correctness."""
    store.fts5_available = False
    a = _remember(store, subject="alpha gizmo", body="the alpha gizmo lives in core/alpha.py")
    _remember(store, subject="unrelated widget", body="nothing to do with the query")

    health = store.health()
    assert health["fts5_available"] is False
    assert health["search_mode"] == "like_fallback_recency_order"

    hits = store.search("alpha gizmo")
    assert a.id in [h.id for h in hits]
    assert all("unrelated" not in h.subject for h in hits)


# ---------------------------------------------------------------------- prune


def test_prune_by_age_removes_old_entries_and_keeps_recent(store):
    old = _remember(store, subject="old fact", created_at=now() - 30 * 86400)
    recent = _remember(store, subject="recent fact", created_at=now())
    deleted = store.prune(max_age_days=7)
    assert deleted == 1
    assert store.get(old.id) is None
    assert store.get(recent.id) is not None


def test_prune_by_keep_latest_keeps_only_the_newest_n(store):
    items = [_remember(store, subject=f"fact {i}", created_at=1_700_000_000.0 + i) for i in range(5)]
    deleted = store.prune(keep_latest=2)
    assert deleted == 3
    remaining_ids = {r.id for r in store.by_kind(MemoryKind.FACT, limit=100)}
    assert remaining_ids == {items[3].id, items[4].id}


def test_prune_with_no_bounds_is_a_no_op(store):
    _remember(store)
    assert store.prune() == 0
    assert store.count() == 1


def test_pruning_also_removes_the_fts_mirror_row(store):
    """A stale fts row for a deleted memory id would let search() surface an
    id that get() can no longer resolve."""
    item = _remember(store, subject="prunable fact", created_at=now() - 30 * 86400)
    store.prune(max_age_days=7)
    hits = store._conn.execute("SELECT id FROM memory_fts WHERE id=?", (item.id,)).fetchall()
    assert hits == []


def test_forget_deletes_one_row(store):
    item = _remember(store)
    assert store.forget(item.id) is True
    assert store.get(item.id) is None
    assert store.forget(item.id) is False  # already gone, nothing to delete


# ------------------------------------------------------------------ secrets


@pytest.mark.parametrize("field", ["subject", "body", "source_ref"])
def test_secret_shaped_values_are_refused_in_every_text_field(store, field):
    secret = "AKIAABCDEFGHIJKLMNOP"  # AWS-access-key shaped  # gitleaks:allow
    kw = dict(
        kind=MemoryKind.FACT,
        subject="ok subject",
        body="ok body text that is unremarkable",
        provenance=Provenance.MEASURED,
        source_ref="job:job_1",
    )
    kw[field] = secret if field != "body" else f"the key is {secret} apparently"
    with pytest.raises(ValueError):
        store.remember(**kw)
    assert store.count() == 0


@pytest.mark.parametrize(
    "secret",
    [
        "AKIAABCDEFGHIJKLMNOP",  # gitleaks:allow
        "ghp_" + "a" * 40,  # gitleaks:allow
        "sk-ant-" + "b" * 30,  # gitleaks:allow
        "sk-" + "c" * 30,  # gitleaks:allow
        "xoxb-" + "1" * 20,  # gitleaks:allow
        "api_key: 0123456789abcdef",  # gitleaks:allow
        "-----BEGIN RSA PRIVATE KEY-----",  # gitleaks:allow
    ],
)
def test_a_variety_of_secret_shapes_are_all_caught(secret):
    assert _secret_hit(secret) is not None


def test_ordinary_technical_prose_mentioning_password_or_token_is_not_flagged(store):
    """False-positive guard: the pattern must not make the store useless for
    legitimate notes that use these words in ordinary prose."""
    item = _remember(
        store,
        subject="auth flow note",
        body=(
            "the login form requires a password field with client-side validation; "
            "the session token is refreshed every 15 minutes and the password_policy "
            "constant lives in forgeos/settings.py"
        ),
        source_ref="job:job_1",
    )
    assert store.get(item.id) is not None


def test_nothing_is_persisted_when_a_secret_is_refused(store):
    with pytest.raises(ValueError):
        _remember(store, source_ref="token=abcdefghijklmnop")
    assert store.count() == 0
    if store.fts5_available:
        n = store._conn.execute("SELECT COUNT(*) AS n FROM memory_fts").fetchone()["n"]
        assert n == 0


# --------------------------------------------------------------------- reads


def test_by_kind_filters_and_orders_recent_first(store):
    _remember(store, kind=MemoryKind.FACT, subject="f1", created_at=1.0)
    _remember(store, kind=MemoryKind.FAILURE_FIX, subject="ff1", created_at=2.0)
    _remember(store, kind=MemoryKind.FACT, subject="f2", created_at=3.0)
    facts = store.by_kind(MemoryKind.FACT)
    assert [f.subject for f in facts] == ["f2", "f1"]


def test_for_job_returns_only_that_jobs_items_in_order(store):
    _remember(store, job_id="job_a", subject="a1", created_at=1.0)
    _remember(store, job_id="job_b", subject="b1", created_at=2.0)
    _remember(store, job_id="job_a", subject="a2", created_at=3.0)
    items = store.for_job("job_a")
    assert [i.subject for i in items] == ["a1", "a2"]


# --------------------------------------------------------------- vault export


def test_export_to_vault_writes_through_the_existing_vault_writer(store, tmp_path):
    item = _remember(store, subject="fact about routing")
    vault = Vault(tmp_path / "vault")
    page = store.export_to_vault(item, vault)
    assert page.exists()
    text = page.read_text(encoding="utf-8")
    assert "measured" in text
    assert item.source_ref in text
    assert any("fact about routing" in ln for ln in vault.index_entries())


# ---------------------------------------------------------------- concurrency

THREADS = 8
PER_THREAD = 15


def _fan_out(fn, n: int = THREADS):
    """Run `fn(i)` on n threads, re-raising the first failure in the caller."""
    errors: list[BaseException] = []

    def guarded(i: int):
        try:
            fn(i)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(guarded, range(n)))
    if errors:
        raise errors[0]


def test_concurrent_remembers_from_several_threads_lose_nothing(tmp_path):
    """Mirrors tests/test_sqlite_concurrency.py: the property that matters is
    that every write from every thread survives, which only holds because
    MemoryStore opens its connection through the guarded `_sqlite.connect`."""
    store = MemoryStore(tmp_path / "concurrent.db")

    def writer(i: int):
        for j in range(PER_THREAD):
            store.remember(
                MemoryKind.FACT,
                f"fact from thread {i} item {j}",
                f"body {i}-{j}",
                provenance=Provenance.MEASURED,
                source_ref=f"thread:{i}",
            )

    try:
        _fan_out(writer)
        assert store.count() == THREADS * PER_THREAD
        # Every row is retrievable, not just counted -- a torn write could
        # inflate rowcount without every row actually being readable.
        rows = store.by_kind(MemoryKind.FACT, limit=THREADS * PER_THREAD + 10)
        assert len(rows) == THREADS * PER_THREAD
    finally:
        store.close()
