"""Claim store — outside knowledge enters as a labelled claim, never as a rule.

The failure this prevents: a confident recommendation from a video, blog post, or
another model gets written into an agent instruction, and from then on every worker
inherits it. Unlike a bad commit it leaves no diff to revert and no test goes red —
the swarm just quietly does the wrong thing forever.

So promotion is mechanical, not editorial. `promote()` REFUSES unless the claim has
an evidence link, has been observed from more than one independent source, and does
not contradict an already-promoted claim. Refusing is the feature.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from ..contracts import new_id, now

# An unreplicated claim is an anecdote. Two independent sources is the floor.
MIN_SOURCES_TO_PROMOTE = 2


class ClaimType(str, Enum):
    DEMONSTRATED_FACT = "demonstrated_fact"    # shown working on screen / in output
    RECOMMENDATION = "recommendation"          # "you should do X"
    OPINION = "opinion"
    PREDICTION = "prediction"
    TOOL_INSTRUCTION = "tool_instruction"      # concrete command / config
    CODE = "code"
    WARNING = "warning"                        # "this breaks when..."


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"        # single source, no corroboration
    CORROBORATED = "corroborated"    # seen from independent sources
    VERIFIED = "verified"            # we reproduced it ourselves
    CONTRADICTED = "contradicted"    # conflicts with something we trust
    PROMOTED = "promoted"            # cleared the gate; agents may follow it


# Types that may never become an agent instruction regardless of corroboration.
# Ten people repeating a prediction does not make it a fact, and a stale
# tool instruction is worse than no instruction because it fails confidently.
NEVER_PROMOTABLE = frozenset({ClaimType.OPINION, ClaimType.PREDICTION})

CLAIMS_SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    claim_key TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    time_sensitive INTEGER NOT NULL DEFAULT 0,
    topics TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    promoted_at REAL
);
CREATE INDEX IF NOT EXISTS idx_claim_key ON claims(claim_key);
CREATE INDEX IF NOT EXISTS idx_claim_status ON claims(status);

-- One row per independent sighting. Corroboration is counted, never asserted.
CREATE TABLE IF NOT EXISTS claim_sources (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    source_ref TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    locator TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    UNIQUE(claim_id, source_ref)
);
CREATE INDEX IF NOT EXISTS idx_src_claim ON claim_sources(claim_id);

CREATE TABLE IF NOT EXISTS claim_conflicts (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    conflicts_with TEXT NOT NULL REFERENCES claims(id),
    detail TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
"""


def claim_key(text: str) -> str:
    """Normalised identity so the same claim from two sources collapses to one row."""
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: new_id("clm"))
    text: str
    type: ClaimType
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    time_sensitive: bool = False
    topics: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=now)
    promoted_at: float | None = None

    @property
    def key(self) -> str:
        return claim_key(self.text)

    @property
    def agents_may_follow(self) -> bool:
        return self.status is VerificationStatus.PROMOTED


class PromotionResult(BaseModel):
    """Explicit outcome — a refusal always carries its reason."""

    promoted: bool
    reason: str


class ClaimStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(CLAIMS_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ add

    def add(
        self,
        text: str,
        type: ClaimType,
        *,
        source_ref: str,
        source_hash: str = "",
        locator: str = "",
        confidence: float = 0.5,
        time_sensitive: bool = False,
        topics: list[str] | None = None,
    ) -> Claim:
        """Record a sighting. Same text from a new source corroborates rather than duplicates."""
        key = claim_key(text)
        existing = self._conn.execute(
            "SELECT * FROM claims WHERE claim_key=?", (key,)
        ).fetchone()

        if existing:
            claim = self._row_to_claim(existing)
        else:
            claim = Claim(
                text=text,
                type=type,
                confidence=confidence,
                time_sensitive=time_sensitive,
                topics=topics or [],
            )
            with self._conn:
                self._conn.execute(
                    "INSERT INTO claims (id, text, claim_key, type, status, confidence,"
                    " time_sensitive, topics, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        claim.id,
                        claim.text,
                        key,
                        claim.type.value,
                        claim.status.value,
                        claim.confidence,
                        int(claim.time_sensitive),
                        json.dumps(claim.topics),
                        claim.created_at,
                    ),
                )

        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO claim_sources (id, claim_id, source_ref, source_hash,"
                " locator, created_at) VALUES (?,?,?,?,?,?)",
                (new_id("src"), claim.id, source_ref, source_hash, locator, now()),
            )

        # Corroboration is derived from counted sources, never set by a caller.
        if (
            self.source_count(claim.id) >= MIN_SOURCES_TO_PROMOTE
            and claim.status is VerificationStatus.UNVERIFIED
        ):
            self._set_status(claim.id, VerificationStatus.CORROBORATED)

        return self.get(claim.id)  # type: ignore[return-value]

    def mark_verified(self, claim_id: str) -> None:
        """We reproduced it ourselves. The only status a caller may assert."""
        self._set_status(claim_id, VerificationStatus.VERIFIED)

    def add_conflict(self, claim_id: str, conflicts_with: str, detail: str = "") -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO claim_conflicts (id, claim_id, conflicts_with, detail, created_at)"
                " VALUES (?,?,?,?,?)",
                (new_id("cfl"), claim_id, conflicts_with, detail, now()),
            )
        self._set_status(claim_id, VerificationStatus.CONTRADICTED)

    # -------------------------------------------------------------- promote

    def promote(self, claim_id: str) -> PromotionResult:
        """The gate. Refuses unless every condition holds."""
        claim = self.get(claim_id)
        if claim is None:
            return PromotionResult(promoted=False, reason="no such claim")

        if claim.type in NEVER_PROMOTABLE:
            return PromotionResult(
                promoted=False,
                reason=f"type {claim.type.value} is never promotable — repetition does not make it a fact",
            )

        if claim.status is VerificationStatus.CONTRADICTED:
            return PromotionResult(promoted=False, reason="claim is contradicted")

        if self.conflicts(claim_id):
            return PromotionResult(promoted=False, reason="unresolved conflict recorded")

        sources = self.sources(claim_id)
        if not sources:
            return PromotionResult(promoted=False, reason="no evidence link")

        if not any(s["source_hash"] for s in sources):
            return PromotionResult(
                promoted=False,
                reason="no source hash — evidence must be content-addressed to be checkable",
            )

        verified = claim.status is VerificationStatus.VERIFIED
        if not verified and len(sources) < MIN_SOURCES_TO_PROMOTE:
            return PromotionResult(
                promoted=False,
                reason=f"only {len(sources)} source(s); need {MIN_SOURCES_TO_PROMOTE} or self-verification",
            )

        if claim.time_sensitive and not verified:
            return PromotionResult(
                promoted=False,
                reason="time-sensitive claims require self-verification, not corroboration",
            )

        with self._conn:
            self._conn.execute(
                "UPDATE claims SET status=?, promoted_at=? WHERE id=?",
                (VerificationStatus.PROMOTED.value, now(), claim_id),
            )
        return PromotionResult(promoted=True, reason="evidence linked, corroborated, no conflict")

    # ----------------------------------------------------------------- read

    def _row_to_claim(self, r: sqlite3.Row) -> Claim:
        return Claim(
            id=r["id"],
            text=r["text"],
            type=ClaimType(r["type"]),
            status=VerificationStatus(r["status"]),
            confidence=r["confidence"],
            time_sensitive=bool(r["time_sensitive"]),
            topics=json.loads(r["topics"]),
            created_at=r["created_at"],
            promoted_at=r["promoted_at"],
        )

    def _set_status(self, claim_id: str, status: VerificationStatus) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE claims SET status=? WHERE id=?", (status.value, claim_id)
            )

    def get(self, claim_id: str) -> Claim | None:
        r = self._conn.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
        return self._row_to_claim(r) if r else None

    def sources(self, claim_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM claim_sources WHERE claim_id=? ORDER BY created_at", (claim_id,)
        ).fetchall()

    def source_count(self, claim_id: str) -> int:
        r = self._conn.execute(
            "SELECT COUNT(*) AS n FROM claim_sources WHERE claim_id=?", (claim_id,)
        ).fetchone()
        return int(r["n"])

    def conflicts(self, claim_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM claim_conflicts WHERE claim_id=?", (claim_id,)
        ).fetchall()

    def by_status(self, status: VerificationStatus) -> list[Claim]:
        rows = self._conn.execute(
            "SELECT * FROM claims WHERE status=? ORDER BY created_at", (status.value,)
        ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def instructions(self) -> list[Claim]:
        """The only claims agents are allowed to act on."""
        return self.by_status(VerificationStatus.PROMOTED)

    def stats(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM claims GROUP BY status"
        ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}


__all__ = [
    "CLAIMS_SCHEMA",
    "Claim",
    "ClaimStore",
    "ClaimType",
    "MIN_SOURCES_TO_PROMOTE",
    "NEVER_PROMOTABLE",
    "PromotionResult",
    "VerificationStatus",
    "claim_key",
]
