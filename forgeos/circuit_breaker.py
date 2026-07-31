"""Per-worker circuit breaker: trip/stay-tripped/auto-recover."""
from __future__ import annotations
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class BreakerRecord:
    worker_id: str
    total_calls: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state: BreakerState = BreakerState.CLOSED
    cooldown_until: float = 0.0
    circuit_opened_at: float = 0.0

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failures / self.total_calls

    @property
    def is_open(self) -> bool:
        return self.state is BreakerState.OPEN


class CircuitBreaker:
    DEFAULT_FAILURE_THRESHOLD = 3
    DEFAULT_COOLDOWN_SECONDS = 60.0
    DEFAULT_MAX_COOLDOWN_SECONDS = 600.0
    DEFAULT_HALF_OPEN_ALLOWANCE = 1

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        max_cooldown_seconds: float = DEFAULT_MAX_COOLDOWN_SECONDS,
        half_open_allowance: int = DEFAULT_HALF_OPEN_ALLOWANCE,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_cooldown_seconds = max_cooldown_seconds
        self.half_open_allowance = half_open_allowance
        self._records: dict[str, BreakerRecord] = {}
        self._lock = Lock()

    def record_success(self, worker_id: str) -> None:
        with self._lock:
            record = self._records.setdefault(worker_id, BreakerRecord(worker_id=worker_id))
            record.total_calls += 1
            record.consecutive_failures = 0
            record.last_success_time = time.time()
            if record.state is BreakerState.HALF_OPEN:
                record.state = BreakerState.CLOSED
                record.cooldown_until = 0.0
            elif record.state is BreakerState.OPEN:
                record.state = BreakerState.HALF_OPEN

    def record_failure(self, worker_id: str) -> None:
        with self._lock:
            record = self._records.setdefault(worker_id, BreakerRecord(worker_id=worker_id))
            record.total_calls += 1
            record.failures += 1
            record.consecutive_failures += 1
            record.last_failure_time = time.time()
            if record.consecutive_failures >= self.failure_threshold:
                if record.state is not BreakerState.OPEN:
                    record.state = BreakerState.OPEN
                    record.circuit_opened_at = time.time()
                    record.cooldown_until = time.time() + self.cooldown_seconds

    def is_available(self, worker_id: str) -> bool:
        with self._lock:
            record = self._records.get(worker_id)
            if record is None:
                return True
            if record.state is BreakerState.CLOSED:
                return True
            if record.state is BreakerState.OPEN:
                now = time.time()
                if now >= record.cooldown_until:
                    record.state = BreakerState.HALF_OPEN
                    return True
                return False
            return True  # HALF_OPEN — allow probe

    def get_state(self, worker_id: str) -> BreakerState:
        with self._lock:
            record = self._records.get(worker_id)
            if record is None:
                return BreakerState.CLOSED
            return record.state

    def get_all_states(self) -> dict[str, BreakerState]:
        with self._lock:
            return {wid: rec.state for wid, rec in self._records.items()}

    def reset(self, worker_id: str | None = None) -> None:
        with self._lock:
            if worker_id is None:
                self._records.clear()
            else:
                self._records.pop(worker_id, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                wid: {
                    "state": rec.state.value,
                    "total_calls": rec.total_calls,
                    "failures": rec.failures,
                    "failure_rate": round(rec.failure_rate, 3),
                    "consecutive_failures": rec.consecutive_failures,
                    "cooldown_until": rec.cooldown_until,
                }
                for wid, rec in self._records.items()
            }
