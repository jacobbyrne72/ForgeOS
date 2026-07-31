"""Diff-aware security scanner — scans git diff, not whole files."""

from __future__ import annotations
import subprocess
import re
from dataclasses import dataclass, field


@dataclass
class DiffLine:
    path: str
    old_line: int
    new_line: int
    diff_type: str
    content: str
    hunk_header: str = ""


@dataclass
class DiffResult:
    files_changed: list[str] = field(default_factory=list)
    lines_added: list[DiffLine] = field(default_factory=list)
    lines_removed: list[DiffLine] = field(default_factory=list)
    hunks: list[str] = field(default_factory=list)


NEW_FILE_RE2 = re.compile(r"^\+\+\+\s+(?:b/)?(.+)$")
HUNK_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@")


def get_diff(
    staged: bool = False,
    commit_range: str | None = None,
    file_paths: list[str] | None = None,
    cwd: str | None = None,
) -> DiffResult:
    cmd = ["git"]
    if cwd:
        cmd += ["-C", cwd]
    if staged:
        cmd += ["diff", "--cached"]
    elif commit_range:
        cmd += ["diff", commit_range]
    else:
        cmd += ["diff", "--no-ext-diff"]
    if file_paths:
        cmd += ["--"]
        cmd.extend(file_paths)
    cmd += ["--no-color", "--unified=3"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return DiffResult()
    return _parse_diff(result.stdout or "")


def _parse_diff(raw: str) -> DiffResult:
    files_changed: list[str] = []
    lines_added: list[DiffLine] = []
    lines_removed: list[DiffLine] = []
    current_file = ""
    current_old_line = 0
    current_new_line = 0
    in_hunk = False
    for raw_line in raw.splitlines():
        new_match = NEW_FILE_RE2.match(raw_line)
        if new_match and new_match.group(1) != "/dev/null":
            if current_file:
                files_changed.append(current_file)
            current_file = new_match.group(1)
            in_hunk = False
            continue
        if raw_line.startswith("---") or raw_line.startswith("+++"):
            continue
        hunk_match = HUNK_RE.match(raw_line)
        if hunk_match:
            current_old_line = int(hunk_match.group(1)) if hunk_match.lastindex >= 1 else 0
            current_new_line = int(hunk_match.group(2)) if hunk_match.lastindex >= 2 else 0
            if hunk_match.lastindex < 2:
                current_new_line = int(hunk_match.group(1))
            in_hunk = True
            continue
        if in_hunk and current_file and raw_line:
            line_type = raw_line[0] if raw_line else " "
            content = raw_line[1:] if len(raw_line) > 1 else ""
            if line_type == "+":
                lines_added.append(
                    DiffLine(
                        path=current_file,
                        old_line=0,
                        new_line=current_new_line,
                        diff_type="+",
                        content=content,
                    )
                )
                current_new_line += 1
            elif line_type == "-":
                lines_removed.append(
                    DiffLine(
                        path=current_file,
                        old_line=current_old_line,
                        new_line=0,
                        diff_type="-",
                        content=content,
                    )
                )
                current_old_line += 1
            elif line_type == " ":
                current_old_line += 1
                current_new_line += 1
    if current_file and current_file not in files_changed:
        files_changed.append(current_file)
    return DiffResult(
        files_changed=files_changed,
        lines_added=lines_added,
        lines_removed=lines_removed,
    )


def _tool_available(name: str) -> bool:
    from forgeos.toolpath import resolve_tool

    return resolve_tool(name) is not None


def _resolve_tool(name: str) -> str:
    from forgeos.toolpath import resolve_tool

    t = resolve_tool(name)
    return t or name


def run_semgrep_on_diff(diff: DiffResult, *, cwd: str | None = None):
    if not diff.lines_added:
        return {"status": "skipped", "evidence": "no added lines in diff"}
    if not _tool_available("semgrep"):
        return {"status": "unavailable", "evidence": "semgrep not on PATH"}
    cmd = [
        _resolve_tool("semgrep"),
        "--config",
        "p/security-audit",
        "--metrics=off",
        "--json",
        "--quiet",
        "--error",
        "--diff",
        "-",
    ]
    try:
        r = subprocess.run(
            cmd, input=_reconstruct_diff(diff), capture_output=True, text=True, timeout=120, cwd=cwd
        )
        import json as _json

        try:
            data = _json.loads(r.stdout or "{}")
            findings = data.get("results", [])
            return {
                "status": "fail" if findings else "pass",
                "findings_count": len(findings),
                "tool": "semgrep",
                "evidence": f"{len(findings)} finding(s) on changed lines",
            }
        except (ValueError, _json.JSONDecodeError):
            return {"status": "unavailable", "evidence": r.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"status": "unavailable", "evidence": "semgrep timed out"}


def run_gitleaks_on_diff(diff: DiffResult, *, cwd: str | None = None):
    if not diff.lines_added:
        return {"status": "skipped", "evidence": "no added lines in diff"}
    if not _tool_available("gitleaks"):
        return {"status": "unavailable", "evidence": "gitleaks not on PATH"}
    cmd = [
        _resolve_tool("gitleaks"),
        "detect",
        "--no-git",
        "--no-banner",
        "--redact",
        "--report-format",
        "json",
        "--report-path",
        "-",
    ]
    try:
        r = subprocess.run(
            cmd, input=_reconstruct_diff(diff), capture_output=True, text=True, timeout=120, cwd=cwd
        )
        import json as _json

        findings = []
        try:
            for item in _json.loads(r.stdout) or []:
                findings.append(
                    {
                        "rule": item.get("RuleID", ""),
                        "path": item.get("File", ""),
                        "line": item.get("StartLine", 0),
                        "message": item.get("Description", "")[:200],
                    }
                )
        except (ValueError, _json.JSONDecodeError):
            pass
        return {
            "status": "fail" if findings else "pass",
            "findings": findings,
            "tool": "gitleaks",
            "evidence": f"{len(findings)} secret(s) in changed lines",
        }
    except subprocess.TimeoutExpired:
        return {"status": "unavailable", "evidence": "gitleaks timed out"}


def run_diff_security(diff: DiffResult, *, cwd: str | None = None):
    sem = run_semgrep_on_diff(diff, cwd=cwd)
    leaks = run_gitleaks_on_diff(diff, cwd=cwd)
    sem_count = sem.get("findings_count", 0) or 0
    leak_count = len(leaks.get("findings", []) or [])
    total = sem_count + leak_count
    if sem.get("status") == "fail" or leaks.get("status") == "fail":
        status = "fail"
    elif "unavailable" in (sem.get("status"), leaks.get("status")):
        status = "unavailable"
    else:
        status = "pass"
    return {
        "status": status,
        "semgrep": sem,
        "gitleaks": leaks,
        "total_findings": total,
        "files_scanned": len(diff.files_changed),
        "lines_added": len(diff.lines_added),
        "command": "semgrep + gitleaks on diff",
    }


def _reconstruct_diff(diff: DiffResult) -> str:
    lines = []
    for path in diff.files_changed:
        lines.append(f"--- a/{path}")
        lines.append(f"+++ b/{path}")
        lines.append("@@ -0,0 +0,0 @@")
        for l in diff.lines_removed:
            if l.path == path:
                lines.append(f"-{l.content}")
        for l in diff.lines_added:
            if l.path == path:
                lines.append(f"+{l.content}")
    return "\n".join(lines)
