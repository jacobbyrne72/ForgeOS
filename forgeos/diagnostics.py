"""Degradation recorder — process-local record of "this subsystem is not working
correctly, but execution continued" facts.

MEASURED FINDING (audit, 2026-07-31): zero of forgeos's 116 modules import Python's
`logging`. Meanwhile ~49 `except Exception` handlers exist across the codebase, 9 of
them swallowing with a bare `pass`, and at least three of those change behaviour
invisibly: entry-point adapter discovery failing quietly means a smaller fleet than
configured with no reason surfaced (`adapter/auto_discover.py`); psutil sampling
failing quietly means pool sizes are computed from defaults instead of the real
machine (`core/resources.py`); a profiler's ledger history failing to load means
routing decisions get made blind (`cli.py`). ForgeOS records DOMAIN facts superbly
-- see `events.py` (the append-only mission/task ledger), `ledger.py` (spend), and
every worker report -- but until this module existed, it recorded nothing when a
SUBSYSTEM itself silently degraded. AGENTS.md's own rule -- `GateStatus.UNAVAILABLE`
in `core/verify.py`, "'we could not check' is not 'it is fine'" -- was being
violated inside forgeos's own code: a failure was being treated as a pass.

This module is a DEGRADATION RECORDER, not a logging framework. Deliberately, it
has:

- no severity levels -- everything recorded here already survived an
  `except Exception` clause, i.e. the caller already decided it was not fatal;
  re-grading it is the job a WARNING/ERROR split does, and that is not this job
- no handlers, no formatters, no config, and no destination other than one
  in-process list an operator can read straight back with `degradations()`

A logging framework is precisely the thing that gets configured to filter itself
into silence -- a logger level nobody set, a handler nobody wired up, a format
string nobody reads. This is the opposite: run `python -m forgeos doctor` and see
exactly what degraded, in plain language, with no configuration step that could
have hidden it.

Process-local by design, deliberately NOT persisted to disk:

- these are facts about what is wrong with THIS process right now, not a
  historical record another process needs to resume from -- `events.py` already
  owns "what happened to this job/task, durably, queryable by job_id/task_id"
- a second SQLite store for "the same shape of fact, but without the job/task
  keys that make `events.py` queryable" would duplicate `events.py` while being
  strictly worse at the one thing `events.py` is for
- a degradation that matters beyond this process's lifetime is its caller's to
  promote into `events.py` or a real report -- this module's job stops at making
  it visible to `doctor`, not at making it durable

Bounded (ring buffer, `_MAX_DEGRADATIONS` entries) for the same reason
`gateway/client.py`'s `_MAX_INFLIGHT` bounds its in-flight map: a long-lived
daemon (`watch.py`) must not let a pathological loop of failures grow this
module's memory without limit. Repeats of the SAME degradation (same subsystem +
same `what_failed`) do not consume a new slot -- they increment a count on the
existing entry, which also keeps a tight failure loop from evicting older, still
-relevant degradations out of the buffer.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

# Same discipline as gateway/client.py's _MAX_INFLIGHT: a hard cap so a
# pathological loop of distinct failures cannot grow this module's memory
# without bound. Oldest-first eviction, same as _MAX_INFLIGHT.
_MAX_DEGRADATIONS = 200

# A dashboard line for an operator, not a debugger -- full tracebacks belong in
# whatever the caller does next (a real report, a re-raise upstream), not here.
_REASON_MAX_LEN = 300


@dataclass
class Degradation:
    """One "a subsystem is not working correctly" fact."""

    subsystem: str
    what_failed: str
    reason: str
    consequence: str
    count: int = 1


_lock = threading.Lock()
_degradations: dict[tuple[str, str], Degradation] = {}


def _format_reason(exc: BaseException | None) -> str:
    if exc is None:
        return ""
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= _REASON_MAX_LEN else text[: _REASON_MAX_LEN - 3] + "..."


def record_degradation(
    subsystem: str,
    what_failed: str,
    exc: BaseException | None = None,
    *,
    consequence: str,
) -> None:
    """Record that `subsystem` degraded. Never raises.

    Called from inside an `except Exception:` clause that has already decided
    to swallow and continue -- so a failure in here (a broken clock, a corrupt
    module global, anything) must never become the caller's problem. Recording
    a diagnostics fact is strictly a bonus on top of already-decided control
    flow, never a gate on it.
    """
    try:
        reason = _format_reason(exc)
        key = (subsystem, what_failed)
        with _lock:
            existing = _degradations.get(key)
            if existing is not None:
                existing.count += 1
                existing.reason = reason
                existing.consequence = consequence
            else:
                if len(_degradations) >= _MAX_DEGRADATIONS:
                    oldest_key = next(iter(_degradations), None)
                    if oldest_key is not None:
                        del _degradations[oldest_key]
                _degradations[key] = Degradation(
                    subsystem=subsystem,
                    what_failed=what_failed,
                    reason=reason,
                    consequence=consequence,
                )
    except Exception:
        # The one deliberate empty `except` in this module: a diagnostics
        # failure must never become the failure, and there is nothing lower
        # to record it to -- this is the recorder.
        pass


def degradations() -> list[Degradation]:
    """Snapshot for readers (e.g. `python -m forgeos doctor`). Insertion order."""
    with _lock:
        return list(_degradations.values())


def clear() -> None:
    """Reset all recorded state. For tests."""
    with _lock:
        _degradations.clear()


__all__ = ["Degradation", "record_degradation", "degradations", "clear"]
