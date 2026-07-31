"""Persistent memory -- what a session already learned, retrievable by a later one.

The problem this exists to prevent: when a session's context resets, everything it
learned is gone and gets rediscovered by re-reading the same files and re-running
the same searches. That is money spent twice for the same knowledge.

Design, matching the pattern the other guarded stores in this repo already use
(`ClaimStore`, `AvoidanceLog`, `SpanStore`, `Ledger`, `EventLog`, `LeaseStore`):

- **SQLite is the store**, opened through `forgeos._sqlite.connect()` -- never a
  raw `sqlite3.connect()`. A second unguarded connection is exactly how this repo
  lost 70% of its writes before `_sqlite.py` existed (see that module's docstring).
- **FTS5 for retrieval when the runtime's sqlite build has it**, LIKE-matching
  when it does not. `health()` reports which mode is active rather than silently
  pretending full-text ranking is running when it is really a recency-ordered
  substring scan.
- **The Obsidian vault is an export, never a dependency.** `export_to_vault()`
  writes a human-readable note through the existing `Vault.write_page()`. The
  store itself never constructs a `Vault` and works identically when the caller
  never calls that method.
- **Retrieval is budgeted, never dumped.** `recall()` ranks candidates and admits
  them through `CapsuleBuilder.fit()`, the same graduated-degradation assembler
  every other worker handoff in forgeos goes through. A memory layer that pastes
  its rows into a prompt would recreate exactly the context bloat this exists to
  eliminate -- and `fit()`'s budget is a hard ceiling, never exceeded.
- **Provenance is mandatory**, matching `claims.py`'s spirit without importing its
  corroboration machinery: this is the harness's own telemetry (a gate ruled, a
  test ran, a worker reported), not an outside claim that needs promotion. Every
  row records where it came from (`source_ref`, never blank) and whether it was
  `MEASURED` (observed directly) or `ASSERTED` (inferred, unchecked). `recall()`
  bakes the provenance tag into the first line of every rendered item, so a guess
  can never silently read back as a fact.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from .._sqlite import connect as _sql_connect
from ..contracts import new_id, now
from ..economy.capsule import Capsule, CapsuleBuilder, RefKind
from .vault import Vault


class MemoryKind(str, Enum):
    """What a remembered row is about."""

    TASK_OUTCOME = "task_outcome"              # subject + outcome + the merge gate's ruling
    CAPABILITY_RESULT = "capability_result"     # a worker succeeded/failed at a capability
    FILE_RELEVANCE = "file_relevance"           # a path that turned out to matter to an objective
    FAILURE_FIX = "failure_fix"                 # a failure signature and what fixed it
    FACT = "fact"                               # a resolved fact about the codebase (where X lives)


class Provenance(str, Enum):
    """How a row is known.

    Mirrors claims.py's measured-vs-asserted distinction without its
    corroboration/promotion gate -- that gate exists for OUTSIDE claims (a video,
    a blog post); this store is the harness's own first-party observations.
    """

    MEASURED = "measured"    # observed directly: exit code, gate verdict, worker report
    ASSERTED = "asserted"    # inferred or guessed by an agent; nobody has checked


# ------------------------------------------------------------- secret guard
#
# MUST NOT: store secrets or API key values (see forgeos/knowledge/memory.py
# callers' contract). This is a coarse, precision-first heuristic -- known
# secret shapes (prefixed tokens, private-key headers, JWTs) plus a generic
# "keyword: long-value" assignment idiom. It is static pattern matching, not a
# real scanner (see security_diff.py for the gitleaks/semgrep-backed diff
# scan) -- it exists to catch the obvious case at the point of write, not to
# replace a real secret-scanning tool.

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|access[_-]?key|auth[_-]?token|token|password|passwd)\b"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9/+_.\-]{12,}['\"]?"
        ),
    ),
)


def _secret_hit(*texts: str) -> str | None:
    """Return the name of the first secret-shaped pattern found, or None."""
    for text in texts:
        if not text:
            continue
        for name, pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                return name
    return None


# ------------------------------------------------------------------- schema

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    provenance TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    topics TEXT NOT NULL DEFAULT '[]',
    job_id TEXT,
    task_id TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory(kind);
CREATE INDEX IF NOT EXISTS idx_memory_created ON memory(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_job ON memory(job_id);
"""


class MemoryItem(BaseModel):
    """One distilled, provenance-tagged fact the harness already learned."""

    id: str = Field(default_factory=lambda: new_id("mem"))
    kind: MemoryKind
    subject: str
    body: str
    provenance: Provenance
    source_ref: str
    topics: list[str] = Field(default_factory=list)
    job_id: str | None = None
    task_id: str | None = None
    created_at: float = Field(default_factory=now)

    @field_validator("subject", "body", "source_ref")
    @classmethod
    def _clean(cls, v: str, info: ValidationInfo) -> str:
        v = v.strip()
        if not v:
            raise ValueError(f"{info.field_name} must not be blank")
        hit = _secret_hit(v)
        if hit:
            raise ValueError(
                f"refusing to store {info.field_name}: matches secret pattern '{hit}'"
            )
        return v

    @field_validator("topics")
    @classmethod
    def _clean_topics(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v if t.strip()]
        hit = _secret_hit(*cleaned)
        if hit:
            raise ValueError(f"refusing to store topics: matches secret pattern '{hit}'")
        return cleaned


def _render(item: MemoryItem) -> str:
    """The text actually sent to a worker for one memory item.

    Provenance is on the first line, not just in metadata a caller could drop --
    that is what makes "a remembered guess must never read as a remembered fact"
    an actual guarantee rather than a convention someone can forget to check.
    """
    lines = [
        f"[{item.kind.value} | {item.provenance.value}] {item.subject}",
        item.body,
        f"source: {item.source_ref}",
    ]
    if item.topics:
        lines.append(f"topics: {', '.join(item.topics)}")
    return "\n".join(lines)


_FTS_TOKEN_RE = re.compile(r"\w+")


def _fts_query(text: str) -> str | None:
    """Build a safe FTS5 MATCH expression from free text.

    Every token is double-quoted so FTS5 operators and punctuation in the
    original text (``"``, ``-``, ``:``, ``*``) can never be parsed as query
    syntax -- a caller's query is just words, not a mini-language they have to
    know about. Tokens are ORed: recall favours returning something relevant
    over returning nothing because one word out of several did not match.
    Returns None when the text has no word-characters at all (nothing to search).
    """
    tokens = _FTS_TOKEN_RE.findall(text)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


class MemoryStore:
    """Append-mostly SQLite store of distilled session knowledge.

    One guarded connection (`forgeos._sqlite.connect`), same shape as
    `ClaimStore`/`AvoidanceLog`/`SpanStore`. FTS5 is used for `search()`/
    `recall()` when the runtime's sqlite build has it; `health()` reports
    honestly when it does not rather than silently degrading to a slower but
    still-correct LIKE scan.
    """

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = _sql_connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(MEMORY_SCHEMA)
        self.fts5_available = self._try_create_fts()
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _try_create_fts(self) -> bool:
        """Detect FTS5 the only reliable way: try to create it.

        `sqlite3.OperationalError: no such module: fts5` is how a build without
        the FTS5 extension announces itself; there is no capability flag to
        check ahead of time.
        """
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
                "USING fts5(id UNINDEXED, subject, body, topics)"
            )
            return True
        except sqlite3.OperationalError:
            return False

    # ---------------------------------------------------------------- write

    def remember(
        self,
        kind: MemoryKind,
        subject: str,
        body: str,
        *,
        provenance: Provenance,
        source_ref: str,
        topics: list[str] | None = None,
        job_id: str | None = None,
        task_id: str | None = None,
        created_at: float | None = None,
    ) -> MemoryItem:
        """Persist one distilled item.

        `provenance` and `source_ref` are mandatory keyword arguments with no
        default -- there is no way to call this without saying where the fact
        came from and whether it was measured or asserted. Validation (blank
        fields, secret-shaped values) happens in `MemoryItem` before anything
        touches the connection, so a refusal never leaves a half-written row.
        """
        item = MemoryItem(
            kind=kind,
            subject=subject,
            body=body,
            provenance=provenance,
            source_ref=source_ref,
            topics=topics or [],
            job_id=job_id,
            task_id=task_id,
            created_at=created_at if created_at is not None else now(),
        )
        with self._conn:
            self._conn.execute(
                "INSERT INTO memory (id, kind, subject, body, provenance, source_ref,"
                " topics, job_id, task_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    item.id,
                    item.kind.value,
                    item.subject,
                    item.body,
                    item.provenance.value,
                    item.source_ref,
                    json.dumps(item.topics),
                    item.job_id,
                    item.task_id,
                    item.created_at,
                ),
            )
            if self.fts5_available:
                self._conn.execute(
                    "INSERT INTO memory_fts (id, subject, body, topics) VALUES (?,?,?,?)",
                    (item.id, item.subject, item.body, " ".join(item.topics)),
                )
        return item

    # ----------------------------------------------------------------- read

    def _row_to_item(self, r: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=r["id"],
            kind=MemoryKind(r["kind"]),
            subject=r["subject"],
            body=r["body"],
            provenance=Provenance(r["provenance"]),
            source_ref=r["source_ref"],
            topics=json.loads(r["topics"]),
            job_id=r["job_id"],
            task_id=r["task_id"],
            created_at=r["created_at"],
        )

    def get(self, item_id: str) -> MemoryItem | None:
        r = self._conn.execute("SELECT * FROM memory WHERE id=?", (item_id,)).fetchone()
        return self._row_to_item(r) if r else None

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS n FROM memory").fetchone()["n"])

    def by_kind(self, kind: MemoryKind, limit: int = 100) -> list[MemoryItem]:
        rows = self._conn.execute(
            "SELECT * FROM memory WHERE kind=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (kind.value, limit),
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def for_job(self, job_id: str) -> list[MemoryItem]:
        rows = self._conn.execute(
            "SELECT * FROM memory WHERE job_id=? ORDER BY created_at", (job_id,)
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def search(
        self,
        query: str,
        *,
        kinds: Sequence[MemoryKind] = (),
        limit: int = 20,
    ) -> list[MemoryItem]:
        """Ranked candidates, unbudgeted -- the raw material `recall()` feeds
        through `CapsuleBuilder.fit()`. Exposed on its own because sometimes a
        caller wants the rows directly (tests, `forget()` callers) rather than
        a token-priced capsule.

        FTS5 ranks by bm25 (its built-in `rank` column, best match first). The
        LIKE fallback ranks by recency, since a substring hit-count is a poor
        relevance signal and pretending otherwise would be worse than admitting
        the degradation in `health()`.
        """
        stripped = query.strip()
        if not stripped:
            return []
        kind_vals = [k.value for k in kinds]

        if self.fts5_available:
            fts_expr = _fts_query(stripped)
            if not fts_expr:
                return []
            ranked_ids = [
                r["id"]
                for r in self._conn.execute(
                    "SELECT id FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank",
                    (fts_expr,),
                ).fetchall()
            ]
            if not ranked_ids:
                return []
            placeholders = ",".join("?" * len(ranked_ids))
            rows = self._conn.execute(
                f"SELECT * FROM memory WHERE id IN ({placeholders})", ranked_ids
            ).fetchall()
            by_id = {r["id"]: self._row_to_item(r) for r in rows}
            items = [by_id[i] for i in ranked_ids if i in by_id]
            if kind_vals:
                items = [i for i in items if i.kind.value in kind_vals]
            return items[:limit]

        like = f"%{stripped}%"
        sql = "SELECT * FROM memory WHERE (subject LIKE ? OR body LIKE ?)"
        args: list = [like, like]
        if kind_vals:
            sql += f" AND kind IN ({','.join('?' * len(kind_vals))})"
            args += kind_vals
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(limit)
        rows = self._conn.execute(sql, args).fetchall()
        return [self._row_to_item(r) for r in rows]

    def recall(
        self,
        query: str,
        *,
        budget: int = 2_000,
        kinds: Sequence[MemoryKind] = (),
        limit: int = 20,
        model: str = "",
    ) -> Capsule:
        """Ranked, token-bounded recall -- the only retrieval path meant to feed
        a prompt. Every hit is admitted through `CapsuleBuilder.fit()`, so a
        block that ranks well but arrives after the budget is spent is trimmed
        to its head rather than lost outright (see economy/capsule.py), and the
        budget is a hard ceiling `fit()` itself enforces -- never a raw dump,
        never exceeded.

        The default of 2,000 tokens is deliberately smaller than
        `CapsuleBuilder.build()`'s 8,000-token default for primary-symbol
        capsules: memory cards are already-distilled notes, not file slices, so
        a "what do we already know about X" query should cost a small slice of
        a worker's context, not compete with the code it is there to inform.
        Always overridable -- a caller with more budget to spend, or a task
        where prior context matters more than usual, can pass its own number.
        """
        if not query.strip():
            raise ValueError("recall() requires a non-blank query")

        builder = CapsuleBuilder(budget=budget)
        for item in self.search(query, kinds=kinds, limit=limit):
            builder.fit(
                RefKind.CARD,
                f"card://{item.id}",
                _render(item),
                f"{item.kind.value} ({item.provenance.value}): {item.source_ref}",
                model=model,
            )
        return builder.finish(objective=query, acceptance=(), write_scope=())

    # -------------------------------------------------------------- pruning

    def prune(self, *, max_age_days: float | None = None, keep_latest: int | None = None) -> int:
        """Delete old rows so the store does not grow without bound.

        Both bounds are optional and additive: an age cutoff removes anything
        older than it, then a count cutoff keeps only the most recent
        `keep_latest` rows. Calling with neither set is a no-op -- pruning is
        opt-in, never implicit, so a caller cannot lose history by accident.
        """
        deleted = 0
        with self._conn:
            if max_age_days is not None:
                cutoff = now() - max_age_days * 86400
                r = self._conn.execute("DELETE FROM memory WHERE created_at < ?", (cutoff,))
                deleted += r.rowcount
            if keep_latest is not None:
                r = self._conn.execute(
                    "DELETE FROM memory WHERE id NOT IN ("
                    " SELECT id FROM memory ORDER BY created_at DESC, rowid DESC LIMIT ?"
                    ")",
                    (max(keep_latest, 0),),
                )
                deleted += r.rowcount
            self._sync_fts_after_delete()
        return deleted

    def forget(self, item_id: str) -> bool:
        """Delete one row by id. Returns whether anything was deleted."""
        deleted = False
        with self._conn:
            r = self._conn.execute("DELETE FROM memory WHERE id=?", (item_id,))
            deleted = r.rowcount > 0
            self._sync_fts_after_delete()
        return deleted

    def _sync_fts_after_delete(self) -> None:
        """Keep the FTS mirror from outliving the rows it indexes.

        Driven off `memory` (the source of truth) rather than threading id
        lists through every delete path -- one rule, checked after every
        deletion: an fts row with no matching base row does not belong.
        """
        if self.fts5_available:
            self._conn.execute("DELETE FROM memory_fts WHERE id NOT IN (SELECT id FROM memory)")

    # ---------------------------------------------------------------- health

    def health(self) -> dict:
        """Machine-checkable status -- the honesty check for FTS5 degradation.

        A caller that wants to know whether `recall()` is doing real full-text
        ranking or a recency-ordered LIKE scan reads this rather than assuming.
        """
        return {
            "fts5_available": self.fts5_available,
            "search_mode": "fts5_bm25" if self.fts5_available else "like_fallback_recency_order",
            "row_count": self.count(),
            "path": self.path,
        }

    # --------------------------------------------------------------- export

    def export_to_vault(self, item: MemoryItem, vault: Vault, *, folder: str = "sessions") -> Path:
        """Write one item as a human-readable note through the existing vault
        writer. Optional and separate from the store itself: `Vault` is an
        export target here, never a dependency -- `MemoryStore` never
        constructs one, so the store works identically when this is never
        called.
        """
        summary = f"{item.kind.value} ({item.provenance.value}): {item.subject}"
        body = (
            f"**Provenance:** {item.provenance.value}\n"
            f"**Source:** {item.source_ref}\n\n"
            f"{item.body}\n"
        )
        return vault.write_page(
            folder, f"{item.kind.value}-{item.subject}", summary, body, sources=[item.source_ref]
        )


__all__ = [
    "MEMORY_SCHEMA",
    "MemoryItem",
    "MemoryKind",
    "MemoryStore",
    "Provenance",
]
