"""The verification ladder and merge gate.

Three ideas, all about not trusting a green test run:

1. **Climb the ladder.** Running the full suite after every edit is the largest
   wall-clock waste in an agent loop. Syntax, then lint on changed files, then the
   directly related tests, then the package, then security, then the full suite —
   and only ever the next rung after the current one passes.

2. **Passing tests are not sufficient evidence for merge.** Published work
   (SecureAgentBench) shows coding agents produce patches that are functionally
   correct and still insecure. So the merge gate requires a clean security scan of
   the diff *in addition to* green tests, and it requires evidence that a real
   command produced those results.

3. **A green result can be gamed at the source.** SpecBench (arXiv 2605.21384)
   and the RLVR reward-hacking survey (arXiv 2604.15149) both document agents
   under a pass/fail loop learning to edit the test instead of the code —
   deleting assertions, loosening bounds, rewriting the check to match whatever
   the implementation now does. Neither the ladder nor the security scan above
   catches this; both trust whatever test the diff shipped with. So the merge
   gate also flags a diff that edits a test in the same change as the source it
   verifies, and requires a human to say `tamper_reviewed=True` before it merges.

Independent review is also not automatically independent. LLM evaluators recognize
and favor their own generations, and the bias scales with how well a model
recognizes its own output (Panickssery et al., NeurIPS 2024, arXiv 2404.13076);
position/order effects compound this further, worst exactly when the two
candidates are closest in quality (Judging the Judges, arXiv 2406.07791).
`reviewer_worker == implementer_worker` is a hard block below; `reviewer_family`
vs. `implementer_family` extends the same instinct one level up — same provider
family is not the same worker, but it is plausibly the same blind spot, so it is
a WARN, not a block.

Every gate returns the command it ran and that command's real output. A gate that
reports a verdict without the evidence behind it is indistinguishable from a guess.
"""

from __future__ import annotations

import json
import subprocess
from enum import Enum, IntEnum
from pathlib import Path

from pydantic import BaseModel, Field

from ..toolpath import resolve_tool

# Scanners are optional. A machine without them must still be able to run forgeos —
# but a MISSING scanner is reported as unavailable, never as a pass. Treating
# "could not check" as "clean" is how insecure code reaches a merge.
SEMGREP = "semgrep"
GITLEAKS = "gitleaks"
RUFF = "ruff"

DEFAULT_TIMEOUT = 300
SECURITY_TIMEOUT = 30


class Gate(IntEnum):
    SYNTAX = 0
    LINT = 1
    DIRECT_TESTS = 2
    PACKAGE_TESTS = 3
    SECURITY = 4
    FULL_SUITE = 5


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"  # tool not installed — NOT a pass
    SKIPPED = "skipped"


class Finding(BaseModel):
    rule: str = ""
    path: str = ""
    line: int = 0
    severity: str = ""
    message: str = ""


class GateResult(BaseModel):
    gate: Gate
    status: GateStatus
    command: str = ""
    evidence: str = ""
    findings: list[Finding] = Field(default_factory=list)
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status is GateStatus.PASS

    @property
    def blocks_merge(self) -> bool:
        """UNAVAILABLE blocks too. 'We could not check' is not 'it is fine'."""
        return self.status in (GateStatus.FAIL, GateStatus.UNAVAILABLE)


def _run(cmd: list[str], *, cwd: str | None = None, timeout: int = DEFAULT_TIMEOUT):
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def _tool_available(name: str) -> bool:
    """Trusted-path check. A binary in the repo under analysis is not available;
    it is an attempt to get forgeos to execute attacker-supplied code."""
    return resolve_tool(name) is not None


# --------------------------------------------------------------------- gates


def run_lint(paths: list[str], *, cwd: str | None = None) -> GateResult:
    """Ruff on changed files only. Cheap, and catches the trivial breakage early."""
    if not paths:
        return GateResult(gate=Gate.LINT, status=GateStatus.SKIPPED, evidence="no paths")
    if not _tool_available(RUFF):
        return GateResult(gate=Gate.LINT, status=GateStatus.UNAVAILABLE,
                          evidence="ruff not on PATH")
    cmd = [resolve_tool(RUFF) or RUFF, "check", "--", *paths]
    r = _run(cmd, cwd=cwd)
    return GateResult(
        gate=Gate.LINT,
        status=GateStatus.PASS if r.returncode == 0 else GateStatus.FAIL,
        command=" ".join(cmd),
        evidence=(r.stdout or r.stderr or "")[-4000:],
    )


def run_semgrep(paths: list[str], *, cwd: str | None = None) -> GateResult:
    """Static analysis over the changed paths, parsed from JSON not scraped text."""
    if not paths:
        return GateResult(gate=Gate.SECURITY, status=GateStatus.SKIPPED, evidence="no paths")
    if not _tool_available(SEMGREP):
        return GateResult(gate=Gate.SECURITY, status=GateStatus.UNAVAILABLE,
                          evidence="semgrep not on PATH")

    # Pinned ruleset + metrics off: `--config auto` fetches rules from the network
    # at scan time and reports a project hash home, which is unexpected outbound
    # disclosure when scanning someone else's private repository.
    cmd = [resolve_tool(SEMGREP) or SEMGREP, "--config", "p/security-audit",
           "--metrics=off", "--json", "--quiet", "--error", "--", *paths]
    try:
        r = _run(cmd, cwd=cwd, timeout=SECURITY_TIMEOUT)
    except subprocess.TimeoutExpired:
        return GateResult(gate=Gate.SECURITY, status=GateStatus.UNAVAILABLE,
                          command=" ".join(cmd), evidence="semgrep timed out")

    findings: list[Finding] = []
    try:
        data = json.loads(r.stdout or "{}")
        for res in data.get("results", []):
            extra = res.get("extra") or {}
            findings.append(
                Finding(
                    rule=str(res.get("check_id", "")),
                    path=str(res.get("path", "")),
                    line=int((res.get("start") or {}).get("line") or 0),
                    severity=str(extra.get("severity", "")),
                    message=str(extra.get("message", ""))[:300],
                )
            )
    except json.JSONDecodeError:
        # Unparseable output is not a pass. Report it as unavailable so the merge
        # gate blocks rather than assuming silence means clean.
        return GateResult(gate=Gate.SECURITY, status=GateStatus.UNAVAILABLE,
                          command=" ".join(cmd),
                          evidence=(r.stdout or r.stderr or "")[-2000:])

    return GateResult(
        gate=Gate.SECURITY,
        status=GateStatus.FAIL if findings else GateStatus.PASS,
        command=" ".join(cmd),
        evidence=f"{len(findings)} finding(s)",
        findings=findings,
    )


def run_gitleaks(paths: list[str] | None = None, *, cwd: str | None = None) -> GateResult:
    """Secret detection over the working tree, scoped to the changed paths.

    Two deliberate choices, both learned the hard way:

    **`--no-git` — scan the tree, not the history.** `gitleaks detect` defaults to
    walking every commit. For a *merge gate* that is the wrong question: the
    verdict stops depending on the change under review, and a single finding
    anywhere in the past blocks every future merge forever, with no action the
    author of the current change can take to clear it. This repo demonstrated it
    on itself — a deliberately realistic fake token in a redaction test became a
    permanent history finding the moment the tree was first committed, and every
    merge-gate test began failing on a change that touched none of it. Auditing
    history for leaked credentials is worth doing; it is a different job, on a
    different schedule, with a different response (rotate the key), and wiring it
    into the merge gate makes the gate useless without making the audit better.

    **`paths` is honoured.** It used to be accepted and ignored, which meant the
    gate scanned everything and reported findings the change was not responsible
    for. Findings outside the changed paths are excluded from the verdict — they
    are real, but they are not this change's to answer for, and a gate that
    cannot be satisfied is a gate people learn to bypass.
    """
    if not _tool_available(GITLEAKS):
        return GateResult(gate=Gate.SECURITY, status=GateStatus.UNAVAILABLE,
                          evidence="gitleaks not on PATH")

    sources = list(dict.fromkeys(paths or ["."]))
    commands: list[list[str]] = []
    all_findings: list[Finding] = []
    for source in sources:
        cmd = [resolve_tool(GITLEAKS) or GITLEAKS, "detect", "--no-git", "--no-banner", "--redact",
               "--report-format", "json", "--report-path", "-"]
        cmd += ["--source", source]
        commands.append(cmd)
        try:
            r = _run(cmd, cwd=cwd, timeout=SECURITY_TIMEOUT)
        except subprocess.TimeoutExpired:
            return GateResult(gate=Gate.SECURITY, status=GateStatus.UNAVAILABLE,
                              command=" && ".join(" ".join(c) for c in commands),
                              evidence="gitleaks timed out")

        raw = (r.stdout or "").strip()
        if raw and raw not in ("null", "[]"):
            try:
                all_findings.extend(
                    Finding(
                        rule=str(item.get("RuleID", "")),
                        path=str(item.get("File", "")),
                        line=int(item.get("StartLine") or 0),
                        severity="high",
                        # --redact means the secret itself is never captured here.
                        message=str(item.get("Description", ""))[:200],
                    )
                    for item in json.loads(raw) or []
                )
            except json.JSONDecodeError:
                return GateResult(gate=Gate.SECURITY, status=GateStatus.UNAVAILABLE,
                                  command=" && ".join(" ".join(c) for c in commands),
                                  evidence=raw[-2000:])

    scanned = len(all_findings)
    findings = all_findings
    findings = _only_in(findings, paths)
    elsewhere = scanned - len(findings)

    evidence = f"{len(findings)} secret(s) detected"
    if elsewhere:
        # Named, never silently dropped. A filtered finding is still a real one,
        # and hiding the count would make the gate look cleaner than the repo is.
        evidence += f" ({elsewhere} outside the changed paths, not this change's to answer for)"

    return GateResult(
        gate=Gate.SECURITY,
        status=GateStatus.FAIL if findings else GateStatus.PASS,
        command=" && ".join(" ".join(c) for c in commands),
        evidence=evidence,
        findings=findings,
    )


def _only_in(findings: list[Finding], paths: list[str] | None) -> list[Finding]:
    """Keep findings that land on one of the changed paths.

    No paths given means no scoping — everything is kept, because "the caller did
    not say what changed" must not quietly become "nothing is in scope". That
    default is the difference between a filter and a mute button.

    Path shapes are compared with separators normalised, since gitleaks reports
    repo-relative paths that differ from the caller's by separator on Windows.
    """
    if not paths:
        return findings

    def norm(p: str) -> str:
        return p.replace("\\", "/").lstrip("./").lower()

    wanted = {norm(p) for p in paths}
    kept = []
    for f in findings:
        fp = norm(f.path)
        # A finding matches if its path is one of the changed paths, or lives
        # under one of them when the caller passed a directory.
        if fp in wanted or any(fp.startswith(w.rstrip("/") + "/") for w in wanted):
            kept.append(f)
    return kept


def run_security(paths: list[str], *, cwd: str | None = None) -> GateResult:
    """Both scanners combined. Either one failing fails the gate."""
    scan_paths = paths
    if cwd and paths:
        root = Path(cwd)
        scan_paths = [
            path
            for path in paths
            if (Path(path) if Path(path).is_absolute() else root / path).exists()
        ]
        if not scan_paths:
            return GateResult(
                gate=Gate.SECURITY,
                status=GateStatus.SKIPPED,
                evidence="no existing paths to scan",
            )

    sem = run_semgrep(scan_paths, cwd=cwd)
    leaks = run_gitleaks(scan_paths, cwd=cwd)

    findings = sem.findings + leaks.findings
    if sem.status is GateStatus.FAIL or leaks.status is GateStatus.FAIL:
        status = GateStatus.FAIL
    elif GateStatus.UNAVAILABLE in (sem.status, leaks.status):
        status = GateStatus.UNAVAILABLE
    elif sem.status is GateStatus.SKIPPED and leaks.status is GateStatus.SKIPPED:
        status = GateStatus.SKIPPED
    else:
        status = GateStatus.PASS

    return GateResult(
        gate=Gate.SECURITY,
        status=status,
        command=f"{sem.command} && {leaks.command}".strip(" &"),
        evidence=f"semgrep: {sem.evidence} | gitleaks: {leaks.evidence}",
        findings=findings,
    )


# ---------------------------------------------------------------- merge gate


def _norm_path(path: str) -> str:
    """Same separator/relative-prefix normalisation as `_only_in` — gitleaks and
    callers can disagree on separators, and a mismatch here would silently miss
    a real pairing."""
    return path.replace("\\", "/").lstrip("./")


def _test_name_stem(path: str) -> str | None:
    """The source stem a test file is named for, or None if `path` isn't a test.

    Matches this repo's own convention (`tests/test_<name>.py`) and the wider
    ecosystem's alternative (`<name>_test.py`). Compared case-insensitively —
    Windows paths and callers disagree on case the same way they disagree on
    separators.
    """
    name = _norm_path(path).rsplit("/", 1)[-1].lower()
    if name.startswith("test_") and name.endswith(".py"):
        return name[len("test_"):-len(".py")]
    if name.endswith("_test.py"):
        return name[:-len("_test.py")]
    return None


def detect_test_tampering(files_touched: list[str]) -> list[str]:
    """Flag a diff that edits both a test and the source it is meant to verify.

    Per SpecBench (arXiv 2605.21384) and the RLVR reward-hacking survey (arXiv
    2604.15149), agents optimizing against a visible pass/fail loop learn to
    edit the test instead of the code under test — deleting assertions,
    loosening bounds, rewriting the check to match whatever the implementation
    now does. A test that passed because it was rewritten proves nothing about
    the code it is supposed to verify, and it is a blind spot neither the ladder
    nor the security scan above covers: both trust whatever test the diff
    shipped with.

    Two tiers, both surfaced, never collapsed into one message:

    - **Named pairing** (`tests/test_foo.py` and `foo.py` both touched) is the
      strong signal — the exact test for the exact code moved together.
    - **General case** (any test file touched alongside any non-test `.py`
      source, no name match) is weaker evidence but still a real coincidence,
      so it is reported too, at minimum as a WARN-level reason, rather than
      silently passing because the heuristic could not prove a specific pair.

    Deterministic and offline — no model call, so this check cannot itself be
    gamed the way an LLM-judged one could.
    """
    tests = [p for p in files_touched if _test_name_stem(p) is not None]
    if not tests:
        return []
    sources = [
        p for p in files_touched
        if _norm_path(p).lower().endswith(".py") and _test_name_stem(p) is None
    ]
    if not sources:
        return []

    source_by_stem = {
        _norm_path(p).rsplit("/", 1)[-1][: -len(".py")].lower(): p for p in sources
    }
    reasons: list[str] = []
    paired: set[str] = set()
    for t in tests:
        src = source_by_stem.get(_test_name_stem(t))
        if src:
            reasons.append(
                f"test-tampering: {t} and its source {src} were edited in the same "
                f"change — a test and the code it verifies changing together needs "
                f"human sign-off (tamper_reviewed=True) before merge"
            )
            paired.add(t)

    unpaired = [t for t in tests if t not in paired]
    if unpaired:
        reasons.append(
            f"test-tampering (WARN): {len(unpaired)} test file(s) edited alongside "
            f"{len(sources)} non-test source file(s) in the same diff, with no "
            f"confirmed name pairing — still worth a human's eyes "
            f"(tamper_reviewed=True to acknowledge)"
        )
    return reasons


class MergeVerdict(BaseModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    gates: list[GateResult] = Field(default_factory=list)

    @property
    def blocking(self) -> list[GateResult]:
        return [g for g in self.gates if g.blocks_merge]


class MergeGate:
    """The last deterministic check before anything reaches the base branch."""

    def evaluate(
        self,
        *,
        gates: list[GateResult],
        tests_passed: int,
        tests_failed: int,
        evidence: str,
        commands_run: list[str],
        reviewer_verdict: str | None = None,
        reviewer_worker: str = "",
        implementer_worker: str = "",
        reviewer_family: str = "",
        implementer_family: str = "",
        files_touched: list[str] | None = None,
        tamper_reviewed: bool = False,
    ) -> MergeVerdict:
        reasons: list[str] = []
        warnings: list[str] = []

        if tests_failed > 0:
            reasons.append(f"{tests_failed} test(s) failing")
        if tests_passed == 0:
            # Zero-passed is not green. "No tests ran" reported as success is the
            # single most common false green in an agent pipeline.
            reasons.append("no tests passed — nothing was actually verified")

        if not evidence.strip():
            reasons.append("no evidence recorded")
        if not commands_run:
            reasons.append("no command was run to produce the evidence")

        # A security gate must be PRESENT, not merely non-failing. Iterating whatever
        # was handed in meant `gates=[]` merged clean — the gate enforced its headline
        # contract only when a caller volunteered the evidence against itself.
        if not any(g.gate is Gate.SECURITY for g in gates):
            reasons.append("no security gate was run — absence is not a pass")

        for g in gates:
            if g.status is GateStatus.FAIL:
                reasons.append(f"{g.gate.name.lower()} failed ({len(g.findings)} finding(s))")
            elif g.status is GateStatus.UNAVAILABLE:
                reasons.append(f"{g.gate.name.lower()} could not be checked — not treated as clean")

        # `files_touched` is new and optional. Every existing caller — forge.py's
        # own call site included — predates this check and has no way to supply it
        # without a coordinated second change; treating "not supplied" as "tampering"
        # would block every one of them on a parameter they were never asked for. So
        # None (the default) skips the check outright. An explicitly empty list is
        # different: the caller looked and found nothing changed, and the check
        # honours that by finding nothing too. Only a real path list can produce a
        # real finding — this is the one place absence is allowed to read as "not
        # applicable" rather than "not a pass", and it is allowed only because the
        # parameter itself, not just the value, is new.
        if files_touched is not None:
            tamper_reasons = detect_test_tampering(files_touched)
            if tamper_reasons:
                if tamper_reviewed:
                    warnings.extend(
                        f"{r} (acknowledged: tamper_reviewed=True)" for r in tamper_reasons
                    )
                else:
                    reasons.extend(tamper_reasons)

        # Approval is one exact token. Comparing against the literal "fail" meant
        # "FAIL", "failed" and "reject" all read as approval — the failure mode where
        # a rejecting reviewer is silently counted as consenting.
        if reviewer_verdict is None:
            reasons.append("no independent review")
        elif str(reviewer_verdict).strip().lower() != "pass":
            reasons.append(f"reviewer did not approve (verdict: {reviewer_verdict!r})")

        # No truthiness guard: an absent reviewer id is a missing identity, not a
        # reason to skip the independence check. Self-review is not review — the
        # model that wrote the bug shares the blind spot that produced it.
        if reviewer_verdict is not None:
            if not reviewer_worker or not implementer_worker:
                reasons.append("reviewer identity missing — independence unverifiable")
            elif reviewer_worker == implementer_worker:
                reasons.append("reviewer must not be the implementer")
            elif implementer_worker in reviewer_worker or reviewer_worker in implementer_worker:
                # A decorated id ("reviewer::a" for implementer "a") is the same
                # worker wearing a label. Exact comparison alone let that through,
                # which is precisely how a self-review passes as independent.
                reasons.append(
                    f"reviewer id {reviewer_worker!r} is derived from implementer "
                    f"{implementer_worker!r} — that is the same worker relabelled"
                )

        # Different worker is not the same as different blind spot. LLM evaluators
        # recognize and favor their own generations (Panickssery et al., NeurIPS
        # 2024, arXiv 2404.13076), and position/order effects compound the risk
        # further when the two outputs are close in quality (arXiv 2406.07791) —
        # exactly the case a merge gate cares most about getting right. This is a
        # WARN, not a block: same provider family is weaker evidence of a shared
        # blind spot than the same worker id above, and blocking on it outright
        # would refuse every single-vendor fleet regardless of actual review quality.
        if reviewer_family and implementer_family and reviewer_family == implementer_family:
            warnings.append(
                f"reviewer and implementer are both {reviewer_family!r} — same-family "
                f"review carries self-preference bias risk; consider a cross-family reviewer"
            )

        return MergeVerdict(allowed=not reasons, reasons=reasons, warnings=warnings, gates=gates)


def next_gate(current: Gate, passed: bool) -> Gate | None:
    """Climb one rung, and only after the current rung passes."""
    if not passed:
        return None
    if current >= Gate.FULL_SUITE:
        return None
    return Gate(int(current) + 1)


__all__ = [
    "DEFAULT_TIMEOUT",
    "detect_test_tampering",
    "Finding",
    "Gate",
    "GateResult",
    "GateStatus",
    "MergeGate",
    "MergeVerdict",
    "next_gate",
    "run_gitleaks",
    "run_lint",
    "run_security",
    "run_semgrep",
]
