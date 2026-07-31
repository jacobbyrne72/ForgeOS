"""Deterministic tool-output reduction. Never let raw output reach a model.

A 12,000-line test log costs input tokens AND causes the model to reason about
noise it did not need. Deterministic extraction beats asking a model to
summarise: it is free, it is exact, and it cannot quietly editorialise a real
failure into "looks fine". This module turns raw stdout/stderr into a bounded
`ReducedOutput` — the full text is never discarded, only withheld: it is
content-addressed by `artifact_ref` (a sha256 of the original text) so it can
always be retrieved by whatever stores logs, even though this module does not
store them itself.

Two entry points:

- `reduce_pytest` parses pytest's own summary + FAILURES sections. Structure
  is introspected, not chunked: the short test summary (always present by
  default since pytest 6) already lists every FAILED/ERROR node id, so that
  section is kept close to verbatim — nothing is silently dropped. The heavy
  FAILURES section (full tracebacks, which is where a loop bug can print the
  same error hundreds of times) is instead parsed into one deduplicated entry
  per distinct root cause, with a `count` of how many node ids share it.

- `reduce_generic` has no structure to introspect, so it falls back to the
  classic head/tail/grep-for-signal approach for arbitrary command output.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .preflight import count_tokens

if TYPE_CHECKING:
    from .avoidance import AvoidanceLog

# --------------------------------------------------------------------- regex

_SHORT_SUMMARY_HEADER_RE = re.compile(r"^=+\s*short test summary info\s*=+$", re.IGNORECASE)
_FAILURES_HEADER_RE = re.compile(r"^=+\s*(FAILURES|ERRORS)\s*=+$", re.IGNORECASE)
_BAR_LINE_RE = re.compile(r"^=+.*=+$")
_BLOCK_HEADER_RE = re.compile(r"^_{3,}\s*(.+?)\s*_{3,}$")
_FAILED_SUMMARY_RE = re.compile(r"^(FAILED|ERROR)\s+(\S+)(?:\s*-\s*(.*))?$")
_FAILED_INLINE_RE = re.compile(r"^(\S+::\S+)\s+(FAILED|ERROR)\b")
_FRAME_LONG_RE = re.compile(r"^(?P<file>[^\s:]+\.py):(?P<line>\d+): in (?P<func>\S+)\s*$")
_FRAME_SHORT_RE = re.compile(r"^(?P<file>[^\s:]+\.py):(?P<line>\d+):\s*(?P<rest>.*)$")
_E_LINE_RE = re.compile(r"^E\s+(.*)$")
_BARE_SUMMARY_RE = re.compile(
    r"^(?:\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected)"
    r"(?:,\s*)?)+\s*in\s+[\d.]+s?\b.*$",
    re.IGNORECASE,
)

_LIB_MARKERS = ("site-packages", ".venv", "dist-packages")

_ERROR_PATTERN = re.compile(r"\b(error|exception|traceback|fatal|fail(?:ed|ure)?)\b", re.IGNORECASE)
_WARNING_PATTERN = re.compile(r"\bwarn(?:ing)?\b", re.IGNORECASE)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_library_frame(path: str) -> bool:
    norm = path.replace("\\", "/")
    if any(marker in norm for marker in _LIB_MARKERS):
        return True
    parts = norm.split("/")
    return "venv" in parts or "vendor" in parts


def _pick_frame(block_lines: list[str]) -> str:
    """The most relevant frame in a traceback block: last project frame wins.

    A traceback runs from the test's call site down to wherever the exception
    was actually raised. The deepest frame is usually the most relevant one,
    UNLESS it is inside a library the project merely calls — in that case the
    project's own last frame (the call that triggered the library failure) is
    more actionable than "somewhere inside site-packages".
    """
    frames: list[tuple[str, str, str | None]] = []
    for ln in block_lines:
        m = _FRAME_LONG_RE.match(ln.strip())
        if m:
            frames.append((m["file"], m["line"], m["func"]))
            continue
        m = _FRAME_SHORT_RE.match(ln.strip())
        if m:
            frames.append((m["file"], m["line"], None))
    if not frames:
        return ""
    for file, line, func in reversed(frames):
        if not _is_library_frame(file):
            return f"{file}:{line}" + (f" in {func}" if func else "")
    file, line, func = frames[-1]
    return f"{file}:{line}" + (f" in {func}" if func else "")


def _short_name(nodeid: str) -> str:
    """The name pytest prints in a FAILURES block header for this node id."""
    parts = nodeid.split("::")[1:]
    return ".".join(parts) if parts else nodeid


def _parse_failure_blocks(lines: list[str]) -> dict[str, list[str]]:
    """Group FAILURES/ERRORS section text into one block per test, keyed by
    the name pytest prints in that section's own header line.

    Inter-frame dividers (`_ _ _ _ ...`, spaced) never match the block-header
    pattern (`_{3,}`, contiguous), so a multi-frame traceback stays inside a
    single block — every frame remains available to `_pick_frame`.
    """
    blocks: dict[str, list[str]] = {}
    in_section = False
    current_name: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_name is not None:
            blocks[current_name] = current_lines

    for ln in lines:
        stripped = ln.strip()
        if _FAILURES_HEADER_RE.match(stripped):
            _flush()
            current_name, current_lines = None, []
            in_section = True
            continue
        if not in_section:
            continue
        if _BAR_LINE_RE.match(stripped):
            _flush()
            in_section, current_name, current_lines = False, None, []
            continue
        m = _BLOCK_HEADER_RE.match(stripped)
        if m:
            _flush()
            current_name, current_lines = m.group(1), []
            continue
        if current_name is not None:
            current_lines.append(ln.rstrip())
    if in_section:
        _flush()
    return blocks


def _fallback_reason(nodeid: str, blocks: dict[str, list[str]]) -> str:
    block = blocks.get(_short_name(nodeid), [])
    for ln in block:
        m = _E_LINE_RE.match(ln.strip())
        if m:
            return m.group(1).strip()
    return "unknown error (no summary reason, no E-line found)"


def _find_result_line(lines: list[str]) -> str:
    for ln in reversed(lines):
        s = ln.strip()
        if _BAR_LINE_RE.match(s) and (" in " in s or "no tests ran" in s.lower()):
            return s
    # Fallback: the bare summary line without pytest's ===== ruler. A headless
    # CLI worker's final message is often exactly "3 passed in 0.52s" — real
    # pytest output it quoted, minus the decoration — and requiring the ruler
    # made the merge gate refuse verified work as "nothing was actually
    # verified" (observed live). The grammar is strict — count clauses then a
    # timing — so prose that merely mentions "passed" still never parses.
    for ln in reversed(lines):
        s = ln.strip()
        if _BARE_SUMMARY_RE.match(s):
            return s
    return ""


def _parse_failed_entries(lines: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (verbatim short-summary lines, [(nodeid, reason), ...]) in order.

    Prefers the "short test summary info" section (always emitted by default
    since pytest 6, and already a compact FAILED/ERROR-per-line listing — the
    cheapest possible source of truth for "every failed node id"). Falls back
    to scanning inline `... FAILED` progress lines when that section is
    missing, so an unusual invocation still keeps every id.
    """
    short_lines: list[str] = []
    entries: list[tuple[str, str]] = []
    in_short_summary = False
    for ln in lines:
        stripped = ln.strip()
        if _SHORT_SUMMARY_HEADER_RE.match(stripped):
            in_short_summary = True
            continue
        if not in_short_summary:
            continue
        if _BAR_LINE_RE.match(stripped):
            break
        m = _FAILED_SUMMARY_RE.match(stripped)
        if m:
            entries.append((m.group(2), (m.group(3) or "").strip()))
            short_lines.append(stripped)

    if entries:
        return short_lines, entries

    seen: set[str] = set()
    for ln in lines:
        m = _FAILED_INLINE_RE.match(ln.strip())
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            entries.append((m.group(1), ""))
    return [], entries


class Failure(BaseModel):
    """The first unique occurrence of one root-cause error.

    `test`/`frame`/`error` are the three fields a caller asked for. `count`
    and `also` exist so a duplicated failure can be collapsed to one entry
    without silently losing which (and how many) other node ids share it —
    losing a real failure to save tokens is the one unacceptable outcome.
    """

    test: str
    frame: str = ""
    error: str = ""
    count: int = Field(default=1, ge=1)
    also: list[str] = Field(default_factory=list)


class ReducedOutput(BaseModel):
    """A bounded, honest reduction of one tool call's raw output.

    `artifact_ref` is a sha256 content address of the untouched original text.
    Nothing is lost by reduction — only withheld from this object — as long as
    whatever ran the tool keeps the full text keyed by that hash.
    """

    exit_code: int
    summary: str
    passed: int = 0
    failed: int = 0
    failures: list[Failure] = Field(default_factory=list)
    kept_lines: list[str] = Field(default_factory=list)
    original_lines: int = Field(ge=0)
    original_tokens_estimate: int = Field(ge=0)
    reduced_tokens_estimate: int = Field(ge=0)
    artifact_ref: str

    @property
    def saved_tokens(self) -> int:
        """Never negative: a reduction that grew the payload saved nothing."""
        return max(0, self.original_tokens_estimate - self.reduced_tokens_estimate)

    @property
    def saved_pct(self) -> float:
        if not self.original_tokens_estimate:
            return 0.0
        return round(100.0 * self.saved_tokens / self.original_tokens_estimate, 2)

    def record_to(self, avoidance_log: "AvoidanceLog", job_id: str, task_id: str | None = None) -> str:
        """Persist this reduction's counterfactual saving to the avoidance ledger.

        `baseline_source` names exactly how the withheld baseline was counted
        (a token estimate of the full original text, addressed by its hash) —
        a guessed baseline is not evidence, per `Avoidance._measured_not_guessed`.
        """
        from .avoidance import AvoidanceMethod  # local import: avoids a hard dep for callers who never record

        return avoidance_log.record(
            job_id,
            AvoidanceMethod.DETERMINISTIC,
            baseline_tokens=self.original_tokens_estimate,
            actual_tokens=self.reduced_tokens_estimate,
            baseline_source=(
                f"token estimate of the full {self.original_lines}-line tool output, "
                f"content-addressed as sha256:{self.artifact_ref[:16]}"
            ),
            task_id=task_id,
        )


def reduce_pytest(output: str) -> ReducedOutput:
    """Extract exit status, counts, every FAILED id, and one entry per root cause.

    Never crashes on malformed or empty input — a log that does not parse
    cleanly degrades to an honest "no summary found" rather than raising.
    """
    text = output or ""
    lines = text.splitlines()
    artifact_ref = _content_hash(text)
    original_tokens = count_tokens(text).tokens

    result_line = _find_result_line(lines)
    no_tests = bool(result_line) and "no tests ran" in result_line.lower()

    passed = 0
    result_failed: int | None = None
    if result_line:
        m_passed = re.search(r"(\d+)\s+passed", result_line)
        m_failed = re.search(r"(\d+)\s+failed", result_line)
        m_error = re.search(r"(\d+)\s+error", result_line)
        passed = int(m_passed.group(1)) if m_passed else 0
        result_failed = (int(m_failed.group(1)) if m_failed else 0) + (int(m_error.group(1)) if m_error else 0)

    _, failed_entries = _parse_failed_entries(lines)
    blocks = _parse_failure_blocks(lines)

    groups: dict[str, list[str]] = {}
    for nodeid, reason in failed_entries:
        key = reason if reason else _fallback_reason(nodeid, blocks)
        groups.setdefault(key, []).append(nodeid)

    # Every FAILED id is kept verbatim (never dropped), but the (possibly long)
    # error text is written ONCE per group instead of once per id — repeating
    # an identical reason hundreds of times is exactly the waste this module
    # exists to remove. The reason itself still lives in `failures` below.
    failures: list[Failure] = []
    kept_lines: list[str] = []
    for reason, nodeids in groups.items():
        rep = nodeids[0]
        frame = _pick_frame(blocks.get(_short_name(rep), []))
        failures.append(Failure(test=rep, frame=frame, error=reason, count=len(nodeids), also=nodeids[1:]))
        if len(nodeids) == 1:
            kept_lines.append(f"FAILED {rep} - {reason}")
        else:
            kept_lines.append(f"FAILED x{len(nodeids)} (identical error, see failures[]): " + ", ".join(nodeids))

    failed = result_failed if result_failed is not None else len(failed_entries)

    raw_failure_lines = sum(len(b) for b in blocks.values())
    if raw_failure_lines and failures:
        kept_lines.append(
            f"... {raw_failure_lines} traceback line(s) across {len(failed_entries)} failure(s) "
            f"condensed into {len(failures)} unique error group(s) ..."
        )
    if result_line:
        kept_lines.append(result_line)

    if no_tests:
        exit_code = 5
    elif failed > 0:
        exit_code = 1
    elif result_line:
        exit_code = 0
    else:
        exit_code = 5  # no discernible summary: unknown/no-tests-collected, never claim success on a guess

    if result_line:
        m = re.match(r"^=+\s*(.*?)\s*=+$", result_line)
        summary = m.group(1) if m else result_line
    elif text.strip():
        summary = "no pytest summary line found in output"
    else:
        summary = "empty output"

    reduced_repr = "\n".join(kept_lines) + "\n" + "\n".join(
        f"{f.test} {f.frame} {f.error} x{f.count}" for f in failures
    )
    reduced_tokens = count_tokens(reduced_repr).tokens

    return ReducedOutput(
        exit_code=exit_code,
        summary=summary,
        passed=passed,
        failed=failed,
        failures=failures,
        kept_lines=kept_lines,
        original_lines=len(lines),
        original_tokens_estimate=original_tokens,
        reduced_tokens_estimate=reduced_tokens,
        artifact_ref=artifact_ref,
    )


def _dedup_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ln in lines:
        key = ln.strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(ln)
    return out


def reduce_generic(output: str, *, max_lines: int = 40) -> ReducedOutput:
    """Head + tail + deduplicated error/warning lines, for output with no known structure.

    `exit_code` here is a heuristic derived only from whether an error/warning
    pattern was seen in the text — not a real process exit status. Callers
    that have the actual subprocess return code should prefer that.
    """
    text = output or ""
    lines = text.splitlines()
    artifact_ref = _content_hash(text)
    original_tokens = count_tokens(text).tokens

    if not lines:
        return ReducedOutput(
            exit_code=0,
            summary="empty output",
            passed=0,
            failed=0,
            failures=[],
            kept_lines=[],
            original_lines=0,
            original_tokens_estimate=0,
            reduced_tokens_estimate=0,
            artifact_ref=artifact_ref,
        )

    matched = [ln for ln in lines if _ERROR_PATTERN.search(ln) or _WARNING_PATTERN.search(ln)]
    truncated = len(lines) > max_lines

    if not truncated:
        kept = _dedup_lines(lines)
    else:
        # Budget is allocated up front so error/warning lines from the middle
        # always have room — a naive head+tail split at half each leaves zero
        # space for them, which would silently drop the one thing most worth
        # keeping. Trimming (when the matched set overflows its share) only
        # ever removes from the middle bucket; head and tail are never cut.
        matched_cap = max(1, max_lines // 4)
        half = max(1, (max_lines - matched_cap) // 2)
        head = lines[:half]
        tail = lines[-half:]
        head_set, tail_set = set(head), set(tail)
        omitted = len(lines) - len(head) - len(tail)

        matched_extra = _dedup_lines([m for m in matched if m not in head_set and m not in tail_set])
        kept_middle = matched_extra[:matched_cap]
        hidden_middle = len(matched_extra) - len(kept_middle)

        combined = list(head)
        if omitted > 0:
            combined.append(f"... {omitted} line(s) omitted ...")
        combined.extend(kept_middle)
        if hidden_middle > 0:
            combined.append(f"... {hidden_middle} more matched line(s) withheld ...")
        combined.extend(tail)
        kept = _dedup_lines(combined)

    exit_code = 1 if matched else 0
    summary = f"{len(lines)} line(s)"
    if matched:
        summary += f", {len(matched)} error/warning line(s) matched"
    if truncated:
        summary += f", truncated to {len(kept)} kept line(s)"

    reduced_tokens = count_tokens("\n".join(kept)).tokens

    return ReducedOutput(
        exit_code=exit_code,
        summary=summary,
        passed=0,
        failed=len(matched),
        failures=[],
        kept_lines=kept,
        original_lines=len(lines),
        original_tokens_estimate=original_tokens,
        reduced_tokens_estimate=reduced_tokens,
        artifact_ref=artifact_ref,
    )


# --------------------------------------------------------- clear_tool_results


class ToolResult(BaseModel):
    """One tool call's result within a `Turn` — the unit `clear_tool_results` may clear.

    `cleared` makes clearing idempotent: a transcript grows turn by turn and
    this function runs again on every pass, so an already-cleared result must
    never be re-"cleared" (that would count a placeholder's own tokens as
    freed a second time) and must never lose its placeholder to a rewrite.
    """

    tool_name: str
    content: str
    cleared: bool = False


class Turn(BaseModel):
    """One turn of a provider-agnostic transcript: content plus zero or more tool results.

    Deliberately minimal — no `role` enum or provider message shape — because
    `clear_tool_results` only needs to tell "tool result payload" (clearable)
    apart from "everything else" (never touched).
    """

    content: str = ""
    tool_results: list[ToolResult] = Field(default_factory=list)


def _turn_tokens(turn: Turn) -> int:
    total = count_tokens(turn.content).tokens
    for result in turn.tool_results:
        total += count_tokens(result.content).tokens
    return total


def clear_tool_results(
    turns: list[Turn],
    *,
    trigger_tokens: int,
    keep_last_n: int,
    clear_at_least_tokens: int,
    exclude_tools: tuple[str, ...] = (),
) -> tuple[list[Turn], int]:
    """Provider-agnostic reimplementation of Anthropic's `clear_tool_uses_20250919`.

    Once the transcript's total tokens cross `trigger_tokens`, the oldest
    eligible tool results are replaced with a short placeholder — oldest
    turn first, never touching the last `keep_last_n` turns or any tool named
    in `exclude_tools` — until at least `clear_at_least_tokens` have been
    freed. Below the trigger this is a no-op (both the input `turns` list and
    saved-token count of 0 are returned unchanged): clearing a transcript
    that already fits is pure waste, and a floor-guarded trigger exists
    specifically to avoid thrashing a prompt cache on tiny, frequent clears.

    Only tool RESULT content is ever replaced. Ordinary turn content,
    already-cleared results, protected turns, and excluded tools all pass
    through byte-for-byte — this function's one job is picking what to clear, not
    rewriting anything it wasn't asked to touch. `turns` itself is never
    mutated; clearing happens on a deep copy so a caller holding the original
    is unaffected.
    """
    total_tokens = sum(_turn_tokens(t) for t in turns)
    if total_tokens < trigger_tokens:
        return turns, 0

    protected_from = max(0, len(turns) - keep_last_n) if keep_last_n > 0 else len(turns)
    new_turns = [t.model_copy(deep=True) for t in turns]
    freed = 0

    for i, turn in enumerate(new_turns):
        if freed >= clear_at_least_tokens:
            break
        if i >= protected_from:
            continue
        for result in turn.tool_results:
            if freed >= clear_at_least_tokens:
                break
            if result.cleared or result.tool_name in exclude_tools:
                continue
            original = count_tokens(result.content).tokens
            placeholder = f"[tool result cleared to free context: {result.tool_name}, {original} tokens]"
            result.content = placeholder
            result.cleared = True
            freed += max(0, original - count_tokens(placeholder).tokens)

    return new_turns, freed


__all__ = [
    "Failure",
    "ReducedOutput",
    "ToolResult",
    "Turn",
    "clear_tool_results",
    "reduce_generic",
    "reduce_pytest",
]
