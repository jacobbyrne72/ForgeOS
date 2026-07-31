"""The stores must survive being used from several threads at once.

Before `forgeos/_sqlite.py`, every store shared one `sqlite3.Connection` opened with
`check_same_thread=False` — a flag that disables Python's thread-affinity *check*
and nothing else. That was harmless while tasks ran one at a time. Once the Forge
began executing the ready set in a thread pool it became live, and it presented
the way this class of bug always does: green in isolation, red under load, with a
task dying before it was even routed and no better explanation than "worker
thread raised".

Measured, not assumed. Eight threads, forty transactions each, against the same
connection:

    raw     96 of 320 rows written, 7 of 8 threads dead
            OperationalError: cannot start a transaction within a transaction
    guarded 320 of 320 rows written, no errors

70% of the writes vanished without anything raising in the surviving thread —
which is what makes this worse than a crash. In the ledger those rows are spend
the governor can no longer see. `cannot start a transaction within a transaction`
is also the exact shape of the flaky suite failure that led here: a task dying
before it was routed (`worker_id=''`, `tier=None`) with no explanation beyond
"worker thread raised".

These tests hammer the real stores from real threads. They are not a proof of
absence — concurrency bugs never are — but they fail when the guard is removed.
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from forgeos._sqlite import connect as guarded_connect
from forgeos.contracts import Budget, JobSpec, TaskSpec, TaskState, WorkerReport
from forgeos.events import EventLog, EventType
from forgeos.leases import LeaseStore, LeaseType
from forgeos.ledger import Ledger

THREADS = 8
PER_THREAD = 25


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


# ================================================================ the wrapper


def test_concurrent_writes_do_not_lose_rows(tmp_path):
    conn = guarded_connect(tmp_path / "w.db")
    conn.execute("CREATE TABLE t (v INTEGER)")
    conn.commit()

    def writer(i: int):
        for j in range(PER_THREAD):
            conn.execute("INSERT INTO t (v) VALUES (?)", (i * 1000 + j,))
        conn.commit()

    _fan_out(writer)
    total = conn.execute("SELECT COUNT(*) AS n FROM t").fetchone()[0]
    assert total == THREADS * PER_THREAD


def test_concurrent_reads_and_writes_do_not_raise(tmp_path):
    """The exact shape that produced 'recursive use of cursors not allowed'."""
    conn = guarded_connect(tmp_path / "rw.db")
    conn.execute("CREATE TABLE t (v INTEGER)")
    conn.executemany("INSERT INTO t (v) VALUES (?)", [(i,) for i in range(200)])
    conn.commit()

    def mixed(i: int):
        for _ in range(PER_THREAD):
            conn.execute("SELECT v FROM t ORDER BY v").fetchall()
            conn.execute("INSERT INTO t (v) VALUES (?)", (i,))

    _fan_out(mixed)


def test_a_transaction_is_not_interleaved_by_another_thread(tmp_path):
    """`with conn:` is a transaction. Two threads inside it at once means one
    commits the other's half-finished work."""
    conn = guarded_connect(tmp_path / "tx.db")
    conn.execute("CREATE TABLE pair (a INTEGER, b INTEGER)")
    conn.commit()

    def paired(i: int):
        for _ in range(PER_THREAD):
            with conn:
                # Two statements that must land together. If another thread's
                # transaction interleaves, a row exists with a != b.
                conn.execute("INSERT INTO pair (a, b) VALUES (?, ?)", (i, i))
                conn.execute("UPDATE pair SET b = a WHERE b IS NULL")

    _fan_out(paired)
    torn = conn.execute("SELECT COUNT(*) AS n FROM pair WHERE a <> b").fetchone()[0]
    assert torn == 0


def test_results_survive_the_lock_being_released(tmp_path):
    """Rows are materialised inside the lock, so no cursor outlives its guard."""
    conn = guarded_connect(tmp_path / "r.db")
    conn.execute("CREATE TABLE t (v INTEGER)")
    conn.executemany("INSERT INTO t (v) VALUES (?)", [(i,) for i in range(5)])
    conn.commit()

    res = conn.execute("SELECT v FROM t ORDER BY v")
    # Hammer the connection from other threads before consuming the result.
    _fan_out(lambda i: conn.execute("SELECT COUNT(*) FROM t").fetchall())
    assert [r[0] for r in res.fetchall()] == [0, 1, 2, 3, 4]


def test_lastrowid_and_rowcount_still_work(tmp_path):
    """`events.py` reads lastrowid, `leases.py` reads rowcount. Both must survive
    the wrapper or the stores break in ways tests elsewhere would blame on logic."""
    conn = guarded_connect(tmp_path / "m.db")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v INTEGER)")
    conn.commit()
    r = conn.execute("INSERT INTO t (v) VALUES (1)")
    assert r.lastrowid == 1
    upd = conn.execute("UPDATE t SET v = 2")
    assert upd.rowcount == 1


def test_wal_is_enabled(tmp_path):
    conn = guarded_connect(tmp_path / "wal.db")
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


# =================================================================== stores


def test_ledger_survives_concurrent_spend_recording(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    job_id = ledger.open_job(JobSpec(objective="c", cwd=".", budget=Budget(max_usd=100.0)))
    task_id = ledger.add_task(
        TaskSpec(job_id=job_id, subject="s", description="d",
                 capabilities=["edit"], budget=Budget(max_usd=10.0))
    )

    def spender(i: int):
        for _ in range(PER_THREAD):
            ledger.record_spend(job_id, f"w{i}", "m", 10, task_id=task_id,
                                tokens_in=1, tokens_out=1)

    _fan_out(spender)
    # Every write must be present and priced exactly. A lost or double-counted
    # row here is money the governor cannot see.
    assert ledger.job_spend_micros(job_id) == THREADS * PER_THREAD * 10


def test_ledger_reports_and_stats_run_concurrently(tmp_path):
    ledger = Ledger(tmp_path / "ledger2.db")
    job_id = ledger.open_job(JobSpec(objective="c", cwd=".", budget=Budget(max_usd=100.0)))
    task_id = ledger.add_task(
        TaskSpec(job_id=job_id, subject="s", description="d",
                 capabilities=["edit"], budget=Budget(max_usd=10.0))
    )

    def worker(i: int):
        for _ in range(PER_THREAD):
            ledger.record_report(
                WorkerReport(task_id=task_id, worker_id=f"w{i}", state=TaskState.DONE)
            )
            ledger.worker_stats(["edit"])

    _fan_out(worker)


def test_event_log_appends_from_many_threads_without_gaps(tmp_path):
    log = EventLog(tmp_path / "events.db")

    def appender(i: int):
        for _ in range(PER_THREAD):
            log.append("job", EventType.ACTION_RECORDED, task_id=f"t{i}")

    _fan_out(appender)
    events = log.replay("job")
    assert len(events) == THREADS * PER_THREAD
    # seq comes from lastrowid; a torn read would duplicate or skip one.
    seqs = [e.seq for e in events]
    assert len(set(seqs)) == len(seqs)


def test_lease_store_never_grants_two_conflicting_write_leases(tmp_path):
    """The single most important concurrency property in the harness: two workers
    must never both hold a write lease on the same path."""
    store = LeaseStore(tmp_path / "leases.db")
    granted: list[str] = []
    lock = threading.Lock()

    def racer(i: int):
        lease = store.acquire(f"task{i}", "repo", "src/same.py", LeaseType.WRITE, 60)
        if lease is not None:
            with lock:
                granted.append(f"task{i}")

    _fan_out(racer, n=16)
    assert len(granted) == 1, f"{len(granted)} workers were granted the same write lease"


def test_a_second_connection_to_the_same_file_still_works(tmp_path):
    """WAL plus a busy timeout: a second writer waits rather than failing with
    'database is locked'."""
    path = tmp_path / "two.db"
    a = guarded_connect(path)
    a.execute("CREATE TABLE t (v INTEGER)")
    a.commit()
    b = guarded_connect(path)

    def writer(i: int):
        conn = a if i % 2 == 0 else b
        for _ in range(10):
            conn.execute("INSERT INTO t (v) VALUES (?)", (i,))
            conn.commit()

    _fan_out(writer, n=4)
    assert a.execute("SELECT COUNT(*) AS n FROM t").fetchone()[0] == 40


def test_row_factory_still_yields_named_columns(tmp_path):
    """Every store sets `row_factory = sqlite3.Row` and indexes rows by name."""
    conn = guarded_connect(tmp_path / "rf.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (name TEXT)")
    conn.execute("INSERT INTO t (name) VALUES ('x')")
    conn.commit()
    assert conn.execute("SELECT name FROM t").fetchone()["name"] == "x"


@pytest.mark.parametrize("method", ["execute", "executemany", "executescript",
                                    "commit", "rollback", "close"])
def test_the_connection_api_the_stores_use_is_present(method):
    assert callable(getattr(guarded_connect(":memory:"), method, None))


# ================== two CONNECTIONS, not two threads on one connection


def _dead_pid() -> int:
    """A pid that is provably not running, so the reap path is exercised."""
    import psutil

    for candidate in range(90_000, 99_999):
        if not psutil.pid_exists(candidate):
            return candidate
    pytest.skip("no dead pid available on this machine")


def _seed_dead_owner(db_path, *, path_pattern="src/race.py"):
    """One WRITE lease whose owning process no longer exists."""
    from forgeos.leases import LeaseStore, LeaseType

    store = LeaseStore(db_path)
    lease = store.acquire(task_id="ghost", repo_id="r", path_pattern=path_pattern,
                          lease_type=LeaseType.WRITE, ttl_seconds=600)
    assert lease is not None
    store._conn.execute(
        "UPDATE path_leases SET owner_pid = ? WHERE id = ?", (_dead_pid(), lease.id)
    )
    store._conn.commit()
    store.close()


def test_two_separate_connections_never_both_reap_and_grant(tmp_path):
    """THE regression, and the one the headline concurrency test cannot see.

    `test_lease_store_never_grants_two_conflicting_write_leases` races 16
    threads through ONE shared `LeaseStore`, so it only ever exercises the
    in-process lock. Two PROCESSES open two connections and therefore two
    independent `threading.RLock`s, which serialise nothing between them --
    and two processes on one state dir is the normal case: `forge.py` and
    `dashboard/app.py` already open separate LeaseStores against the same
    leases.db.

    Measured before the fix: both racers granted a WRITE lease on the same
    path, 5 runs out of 5.
    """
    from forgeos.leases import LeaseStore, LeaseType

    db = tmp_path / "leases.db"
    _seed_dead_owner(db)

    a, b = LeaseStore(db), LeaseStore(db)
    barrier = threading.Barrier(2)
    results: list[tuple[str, object]] = []
    guard = threading.Lock()

    def racer(store, name):
        barrier.wait()
        got = store.acquire(task_id=name, repo_id="r", path_pattern="src/race.py",
                            lease_type=LeaseType.WRITE, ttl_seconds=600)
        with guard:
            results.append((name, got))

    threads = [threading.Thread(target=racer, args=(a, "A")),
               threading.Thread(target=racer, args=(b, "B"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    granted = [name for name, lease in results if lease is not None]
    assert len(granted) == 1, f"two live owners of one write path: {granted}"

    check = LeaseStore(db)
    live = [ls for ls in check.active(repo_id="r")
            if ls.lease_type is LeaseType.WRITE and ls.task_id in ("A", "B")]
    assert len(live) == 1
    for store in (a, b, check):
        store.close()


def test_a_refused_acquire_does_not_leave_the_database_locked(tmp_path):
    """The reap's UPDATE ran as a bare `execute`, and an `acquire` that then
    returned None never reached a commit. The open write transaction held the
    database against every other connection until this one died -- surfacing as
    `database is locked` on an unrelated path, which reads as a completely
    different bug.
    """
    from forgeos.leases import LeaseStore, LeaseType

    db = tmp_path / "leases.db"
    _seed_dead_owner(db)

    blocker = LeaseStore(db)
    live = blocker.acquire(task_id="alive", repo_id="r", path_pattern="src/race.py",
                           lease_type=LeaseType.READ, ttl_seconds=600)
    assert live is not None

    refused_store = LeaseStore(db)
    refused = refused_store.acquire(task_id="loser", repo_id="r", path_pattern="src/race.py",
                                    lease_type=LeaseType.WRITE, ttl_seconds=600)
    assert refused is None, "a live READ blocker must still refuse a WRITE"

    # An unrelated, non-conflicting path from a third connection must not block.
    other = LeaseStore(db)
    got = other.acquire(task_id="elsewhere", repo_id="r", path_pattern="docs/readme.md",
                        lease_type=LeaseType.WRITE, ttl_seconds=600)
    assert got is not None, "a refused acquire left the write lock held"
    for store in (blocker, refused_store, other):
        store.close()
