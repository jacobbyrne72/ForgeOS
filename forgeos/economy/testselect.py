"""The verification ladder: run the cheapest check that could still catch the bug.

Running the full suite after every edit is the biggest wall-clock waste in an
agent loop. The ladder climbs one rung at a time — SYNTAX, LINT, DIRECT,
PACKAGE, INTEGRATION, FULL — only escalating when a lower rung actually
passes, and never silently choosing to run nothing.

"No tests selected" reported as green is a false green (AGENTS.md rule 8): a
change with no known related tests must escalate to a broader rung rather
than pass by default, because passing by omission is indistinguishable from
passing on the merits until the bug ships.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum, IntEnum

from pydantic import BaseModel, Field


class Level(IntEnum):
    """One rung of the ladder. Ordered cheapest to most expensive."""

    SYNTAX = 0
    LINT = 1
    DIRECT = 2
    PACKAGE = 3
    INTEGRATION = 4
    FULL = 5


_LOCKFILE_NAMES = {
    "poetry.lock",
    "uv.lock",
    "Pipfile.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
}
_CONFIG_NAMES = {"pyproject.toml", "setup.py", "setup.cfg", "conftest.py", "Dockerfile", "Makefile"}
_CONFIG_SUFFIXES = (".cfg", ".ini", ".toml", ".yaml", ".yml")


def _norm(path: str) -> str:
    return path.replace("\\", "/")


def _basename(path: str) -> str:
    norm = _norm(path)
    return norm.rsplit("/", 1)[-1]


def _dirname(path: str) -> str:
    norm = _norm(path)
    return norm.rsplit("/", 1)[0] if "/" in norm else ""


def _is_lockfile(path: str) -> bool:
    return _basename(path) in _LOCKFILE_NAMES


def _is_config(path: str) -> bool:
    name = _basename(path)
    return name in _CONFIG_NAMES or name.endswith(_CONFIG_SUFFIXES) or name.startswith("requirements")


def _as_iterable(tests: str | Iterable[str]) -> Iterable[str]:
    if isinstance(tests, str):
        return (tests,)
    return tests


class Selection(BaseModel):
    """What actually runs for one call to `select()`.

    `run_all=True` means the caller should run everything known at this rung
    rather than trust `tests` (used for the coarse INTEGRATION/FULL rungs,
    and as the last-resort fallback when even PACKAGE-level lookup finds
    nothing). `escalated=True` flags that the returned `level` is higher than
    what was requested, because the graph had no answer for the narrower one.
    """

    level: Level
    tests: list[str] = Field(default_factory=list)
    run_all: bool = False
    escalated: bool = False
    reason: str = ""


class TestGraph:
    """Deterministic map from changed code to the tests that exercise it.

    Two mappings, `symbol -> tests` and `file -> tests`, registered by
    whatever instruments the codebase (coverage data, import graph, etc). No
    guessing: a symbol or file with no registered mapping is treated as
    genuinely unknown, which is exactly the condition that must escalate
    rather than silently select nothing.
    """

    # Not a pytest test class, despite the leading "Test" (see contracts.TestResults
    # for the same fix). Without this, pytest tries to collect it and warns every run.
    __test__ = False

    def __init__(self) -> None:
        self._by_symbol: dict[str, set[str]] = {}
        self._by_file: dict[str, set[str]] = {}

    def register_symbol(self, symbol: str, tests: str | Iterable[str]) -> None:
        self._by_symbol.setdefault(symbol, set()).update(_as_iterable(tests))

    def register_file(self, file: str, tests: str | Iterable[str]) -> None:
        self._by_file.setdefault(_norm(file), set()).update(_as_iterable(tests))

    def tests_for(self, changed_files: Iterable[str], changed_symbols: Iterable[str] = ()) -> list[str]:
        found: set[str] = set()
        for f in changed_files:
            found.update(self._by_file.get(_norm(f), ()))
        for s in changed_symbols:
            found.update(self._by_symbol.get(s, ()))
        return sorted(found)

    def _package_tests(self, changed_files: Iterable[str]) -> list[str]:
        packages = {_dirname(f) for f in changed_files}
        found: set[str] = set()
        for file, tests in self._by_file.items():
            if _dirname(file) in packages:
                found.update(tests)
        return sorted(found)

    def level_for(self, changed_files: Iterable[str]) -> Level:
        """The rung a change demands based on what kind of file it touches.

        A lockfile can break anything that depends on it, so it demands FULL.
        A config file (build config, CI, fixtures in conftest.py) demands
        INTEGRATION. An ordinary source edit only demands DIRECT — the
        targeted tests mapped to it.
        """
        files = list(changed_files)
        if not files:
            return Level.SYNTAX
        if any(_is_lockfile(f) for f in files):
            return Level.FULL
        if any(_is_config(f) for f in files):
            return Level.INTEGRATION
        return Level.DIRECT

    def select(
        self,
        changed_files: Iterable[str],
        changed_symbols: Iterable[str] = (),
        level: Level | None = None,
    ) -> Selection:
        """The concrete test selection for one rung of the ladder.

        SYNTAX/LINT are not pytest selections at all (a different kind of
        check runs there). DIRECT and PACKAGE return concrete node ids when
        the graph has an answer, and escalate — never to an empty, silently
        green selection — when it does not. INTEGRATION/FULL are coarse by
        definition: they mean "run everything known", not an enumerable list.
        """
        changed_files = list(changed_files)
        changed_symbols = list(changed_symbols)
        requested = level if level is not None else self.level_for(changed_files)
        changed_anything = bool(changed_files or changed_symbols)

        if requested <= Level.LINT:
            return Selection(
                level=requested,
                reason="syntax/lint rungs are not a pytest selection",
            )

        if requested == Level.DIRECT:
            tests = self.tests_for(changed_files, changed_symbols)
            if tests or not changed_anything:
                return Selection(level=Level.DIRECT, tests=tests, reason="tests mapped directly to the change")
            pkg_tests = self._package_tests(changed_files)
            if pkg_tests:
                return Selection(
                    level=Level.PACKAGE,
                    tests=pkg_tests,
                    escalated=True,
                    reason="no test maps directly to the change; escalated to the enclosing package",
                )
            return Selection(
                level=Level.PACKAGE,
                run_all=True,
                escalated=True,
                reason=(
                    "no known tests at file, symbol, or package level; escalated to PACKAGE and "
                    "running everything known rather than reporting a false green"
                ),
            )

        if requested == Level.PACKAGE:
            tests = sorted(set(self.tests_for(changed_files, changed_symbols)) | set(self._package_tests(changed_files)))
            if tests or not changed_anything:
                return Selection(level=Level.PACKAGE, tests=tests, reason="package-level test set for the change")
            return Selection(
                level=Level.PACKAGE,
                run_all=True,
                escalated=True,
                reason="no known tests at package level; running everything known rather than nothing",
            )

        # INTEGRATION and FULL are coarse rungs: run broadly by definition.
        return Selection(level=requested, run_all=True, reason=f"{requested.name} runs the full known suite")


def next_level(current: Level, passed: bool) -> Level | None:
    """The next rung to run, or None once FULL has passed (fully verified).

    Climbing happens only on a pass — "refuses to skip a rung" means this
    always moves exactly one step up, never further. A failure keeps you on
    the same rung: there is nothing to gain from testing broader on top of
    code that already failed the narrow check.
    """
    if not passed:
        return current
    if current >= Level.FULL:
        return None
    return Level(current + 1)


# --------------------------------------------------------------- flake check


class FlakeVerdict(str, Enum):
    """How confidently a failing test can be attributed to the current diff.

    Only LIKELY_FLAKY is a safe reason to skip an escalation, and it requires
    positive evidence: a coverage record for this specific test showing zero
    overlap with the changed lines. RELEVANT and UNKNOWN both default to
    "treat it as real" — missing evidence must never read as evidence of
    flakiness, or a genuine regression could ride a coverage gap straight past
    the governor (the same "unavailable is not a pass" rule that makes an
    unmapped change escalate in `TestGraph.select` rather than pass by
    default).
    """

    RELEVANT = "relevant"
    LIKELY_FLAKY = "likely_flaky"
    UNKNOWN = "unknown"


def classify_failure_relevance(
    failed_test: str,
    covered_lines_by_file: Mapping[str, set[int]] | None,
    changed_lines_by_file: Mapping[str, set[int]],
) -> FlakeVerdict:
    """DeFlaker's core idea in one function: did this failing test execute any changed line?

    `covered_lines_by_file` must be per-test coverage — the lines THIS test
    ran, not the whole suite's aggregate coverage. Aggregate coverage would
    make nearly every failure look RELEVANT regardless of what the failing
    test itself touched, which defeats the point of the check. Wired to a
    real run, this would come from coverage.py's dynamic contexts (run with
    `--context=test-function`, then `CoverageData.lines(file, contexts=...)`
    per failing test's context); `changed_lines_by_file` would come from
    parsing a unified diff into added/modified line numbers per file (this
    repo already does that shape of parsing for security scanning — see
    `forgeos/security_diff.py`'s `DiffResult.lines_added`).

    `covered_lines_by_file=None` (no coverage recorded for this test)
    verdicts UNKNOWN, never LIKELY_FLAKY — an absent measurement is not the
    positive "ran none of the changed code" evidence a skip-escalation
    decision requires. `failed_test` itself is not read by the verdict logic
    (the check is a pure line-intersection test); it is part of the signature
    so a governor calling this once per currently-failing test id has a
    matching identity to log or key its own per-test verdict map by, without
    a second lookup back to the test that produced it.

    Both mappings are keyed by file path; only slash direction is normalized
    (matching `TestGraph`'s own `_norm`), so callers must supply both under
    the same base (repo-relative paths recommended) or genuinely overlapping
    lines will silently look disjoint.
    """
    if covered_lines_by_file is None:
        return FlakeVerdict.UNKNOWN

    changed = {_norm(f): lines for f, lines in changed_lines_by_file.items()}
    for f, lines in covered_lines_by_file.items():
        if set(lines) & changed.get(_norm(f), set()):
            return FlakeVerdict.RELEVANT
    return FlakeVerdict.LIKELY_FLAKY


__all__ = [
    "FlakeVerdict",
    "Level",
    "Selection",
    "TestGraph",
    "classify_failure_relevance",
    "next_level",
]
