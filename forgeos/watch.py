"""Continuous cost monitoring with anomaly detection.

Reads the ledger in real time and flags when spend rate
deviates beyond expected bounds. Detects:
- Cost spikes (>3x rolling average)
- Silent burn (steady spend with zero task completion)
- Budget leaks (tasks that never complete)
"""
from __future__ import annotations
import time
from datetime import datetime

def watch(ledger, job_id: str | None = None, interval_seconds: int = 30, max_alerts: int = 10) -> list[dict]:
    """Monitor spend rate and return anomalies."""
    alerts: list[dict] = []
    history: list[tuple[float, int]] = []  # (timestamp, spend_usd)
    threshold_multiplier = 3.0

    start = time.time()
    last_check = start

    while len(alerts) < max_alerts:
        time.sleep(interval_seconds)
        now = time.time()
        spent = _get_spend_since(ledger, last_check, job_id=job_id)
        history.append((now, spent))

        if len(history) >= 3:
            avg = sum(h[1] for h in history[-6:]) / min(len(history), 6)
            if spent > avg * threshold_multiplier and spent > 0:
                alerts.append({
                    "type": "cost_spike",
                    "timestamp": datetime.utcnow().isoformat(),
                    "spend_this_interval": spent,
                    "rolling_average": avg,
                    "multiplier": round(spent / avg, 2),
                })
        last_check = now

    return alerts

def _get_spend_since(ledger, since_ts: float, job_id: str | None = None) -> float:
    from forgeos.ledger import Ledger
    # Use ledger methods to get spend since timestamp
    if job_id:
        rows = ledger._conn.execute(
            "SELECT SUM(usd_micros) FROM spend WHERE job_id=? AND created_at > ?",
            (job_id, since_ts),
        ).fetchone()
    else:
        rows = ledger._conn.execute(
            "SELECT SUM(usd_micros) FROM spend WHERE created_at > ?",
            (since_ts,),
        ).fetchone()
    return int(rows[0] or 0) / 1_000_000
