"""Cost tracker — persists savings history across sessions."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from ._sqlite import connect as get_db

TRACKER_DB = ".forgeos/cost_tracker.db"

class CostTracker:
    def __init__(self, db_path: str = TRACKER_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        db = get_db(self.db_path)
        db.execute("""
            CREATE TABLE IF NOT EXISTS savings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                task_type TEXT,
                saved_usd REAL NOT NULL,
                saved_tokens INTEGER DEFAULT 0,
                strategy TEXT,
                metadata TEXT
            )
        """)
        db.commit()

    def record(
        self,
        task_type: str,
        saved_usd: float,
        saved_tokens: int = 0,
        strategy: str = "",
        metadata: dict | None = None,
    ) -> int:
        """Record a savings event. Returns row ID."""
        ts = datetime.now(timezone.utc).isoformat()
        meta = json.dumps(metadata or {})
        db = get_db(self.db_path)
        cur = db.execute(
            "INSERT INTO savings (timestamp, task_type, saved_usd, saved_tokens, strategy, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, task_type, saved_usd, saved_tokens, strategy, meta),
        )
        db.commit()
        return cur.lastrowid

    def total_saved(self) -> dict:
        """Get total savings across all recorded events."""
        db = get_db(self.db_path)
        row = db.execute(
            "SELECT COUNT(*), SUM(saved_usd), SUM(saved_tokens) FROM savings"
        ).fetchone()
        return {
            "total_events": row[0] or 0,
            "total_usd": round(row[1] or 0, 4),
            "total_tokens": row[2] or 0,
        }

    def by_task_type(self) -> list[dict]:
        """Break down savings by task type."""
        db = get_db(self.db_path)
        rows = db.execute(
            "SELECT task_type, COUNT(*), SUM(saved_usd), SUM(saved_tokens) FROM savings GROUP BY task_type ORDER BY SUM(saved_usd) DESC"
        ).fetchall()
        return [
            {
                "task_type": r[0],
                "count": r[1],
                "total_usd": round(r[2], 4),
                "total_tokens": r[3],
            }
            for r in rows
        ]

    def monthly_projection(self) -> dict:
        """Project monthly/yearly savings from historical data."""
        total = self.total_saved()
        if total["total_events"] == 0:
            return {"monthly_usd": 0, "yearly_usd": 0}
        # Assume events span ~30 days for projection
        monthly = total["total_usd"] * 22  # ~22 work days
        yearly = monthly * 12
        return {"monthly_usd": round(monthly, 2), "yearly_usd": round(yearly, 2)}

    def recent(self, limit: int = 10) -> list[dict]:
        """Get most recent savings events."""
        db = get_db(self.db_path)
        rows = db.execute(
            "SELECT timestamp, task_type, saved_usd, saved_tokens, strategy FROM savings ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "timestamp": r[0],
                "task_type": r[1],
                "saved_usd": r[2],
                "saved_tokens": r[3],
                "strategy": r[4],
            }
            for r in rows
        ]
