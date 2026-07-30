"""The verification ladder and merge gate.

Two ideas, both about not trusting a green test run:

1. **Climb the ladder.** Running the full suite after every edit is the largest
   wall-clock waste in an agent loop. Syntax, then lint on changed files, then the
   directly related tests, then the package, then security, then the full suite —
   and only ever the next rung after the current one passes.

2. **Passing tests are not sufficient evidence for merge.** Published work
   (SecureAgentBench) shows coding agents produce patches that are functionally
   correct and still insecure. So the merge gate requires a clean security scan of
   the diff *in addition to* green tests, and it requires evidence that a real
   command produced those results.

Every gate returns the command it ran and that command's real output. A gate that
reports a verdict without the evidence behind it is indistinguishable from a guess.
"""

from __future__ import annotations

import json
import subprocess
from enum import Enum, IntEnum

from pydantic import BaseModel, Field

from ..toolpath import resolve_tool

# Scanners are optional. A machine without them must still be able to run hive —
# but a MISSING scanner is reported as unavailable, never as a pass. Treating
# "could not check" as "clean" is how insecure code reaches a merge.
SEMGREP = "semgrep"
GITLEAKS = "gitleaks"
RUFF = "ruff"

DEFAULT_TIMEOUT = 300


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
    it is an attempt to get hive to execute attacker-supplied code."""
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
        r = _run(cmd, cwd=cwd)
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
    """Secret detection. A leaked credential in a diff is a hard merge blocker."""
    if not _tool_available(GITLEAKS):
        return GateResult(gate=Gate.SECURITY, status=GateStatus.UNAVAILABLE,
                          evidence="gitleaks not on PATH")

    cmd = [resolve_tool(GITLEAKS) or GITLEAKS, "detect", "--no-banner", "--redact",
           "--report-format", "json", "--report-path", "-"]
    if cwd:
        cmd += ["--source", "."]
    try:
        r = _run(cmd, cwd=cwd)
    except subprocess.TimeoutExpired:
        return GateResult(gate=Gate.SECURITY, status=GateStatus.UNAVAILABLE,
                          command=" ".join(cmd), evidence="gitleaks timed out")

    findings: list[Finding] = []
    raw = (r.stdout or "").strip()
    if raw and raw not in ("null", "[]"):
        try:
            for item in json.loads(raw) or []:
                findings.append(
                    Finding(
                        rule=str(item.get("RuleID", "")),
                        path=str(item.get("File", "")),
                        line=int(item.get("StartLine") or 0),
                        severity="high",
                        # --redact means the secret itself is never captured here.
                        message=str(item.get("Description", ""))[:200],
                    )
                )
        except json.JSONDecodeError:
            return GateResult(gate=Gate.SECURITY, status=GateStatus.UNAVAILABLE,
                              command=" ".join(cmd), evidence=raw[-2000:])

    return GateResult(
        gate=Gate.SECURITY,
        status=GateStatus.FAIL if findings else GateStatus.PASS,
        command=" ".join(cmd),
        evidence=f"{len(findings)} secret(s) detected",
        findings=findings,
    )


def run_security(paths: list[str], *, cwd: str | None = None) -> GateResult:
    """Both scanners combined. Either one failing fails the gate."""
    sem = run_semgrep(paths, cwd=cwd)
    leaks = run_gitleaks(paths, cwd=cwd)

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


class MergeVerdict(BaseModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)
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
    ) -> MergeVerdict:
        reasons: list[str] = []

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

        return MergeVerdict(allowed=not reasons, reasons=reasons, gates=gates)


def next_gate(current: Gate, passed: bool) -> Gate | None:
    """Climb one rung, and only after the current rung passes."""
    if not passed:
        return None
    if current >= Gate.FULL_SUITE:
        return None
    return Gate(int(current) + 1)


__all__ = [
    "DEFAULT_TIMEOUT",
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
