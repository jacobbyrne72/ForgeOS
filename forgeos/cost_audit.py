"""Cost audit - scan any project for AI cost waste."""
from __future__ import annotations

import re
from pathlib import Path

WASTE_PATTERNS = [
    {
        "name": "missing_budget",
        "regex": r"GPT-4|gpt-4|claude-3-opus",
        "severity": "medium",
        "savings_usd": 0.03,
        "fix": "Consider cheaper model for this task",
    },
    {
        "name": "redundant_model_call",
        "regex": r"client.chat.completions.create",
        "severity": "high",
        "savings_usd": 0.10,
        "fix": "Batch calls or add caching layer",
    },
    {
        "name": "missing_cache",
        "regex": r"for .* in .*:.*(run|execute|call)",
        "severity": "high",
        "savings_usd": 0.05,
        "fix": "Cache repetitive calls with same inputs",
    },
]

_SKIP_DIRS = frozenset(
    {
        ".git", ".forgeos", ".forgeos-live", "__pycache__", ".venv", "venv",
        "site-packages", "node_modules", "vendor", "dist", "build",
        ".pytest_cache", ".ruff_cache", "coverage", ".tox",
    }
)

class CostAuditor:
    def __init__(self):
        self.issues = []
        self.total_savings = 0.0

    def audit_file(self, filepath):
        path = Path(filepath)
        if not path.exists():
            return []
        # ForgeOS runs on Windows as well as POSIX.  `Path.read_text()` uses
        # the process locale there (often cp1252), so a perfectly valid UTF-8
        # source file could crash the audit before it reported anything.  An
        # audit should be best-effort over source text: decode explicitly and
        # preserve the scan with replacement characters if a file is malformed.
        text = path.read_text(encoding="utf-8", errors="replace")
        file_issues = []
        for wp in WASTE_PATTERNS:
            for m in re.finditer(wp["regex"], text):
                line_no = text[:m.start()].count(chr(10)) + 1
                snippet = text.split(chr(10))[line_no - 1].strip()[:80]
                issue = {
                    "file": filepath,
                    "line": line_no,
                    "type": wp["name"],
                    "severity": wp["severity"],
                    "savings_usd": wp["savings_usd"],
                    "fix": wp["fix"],
                    "snippet": snippet,
                }
                file_issues.append(issue)
                self.issues.append(issue)
                self.total_savings += wp["savings_usd"]
        return file_issues

    def audit_directory(self, directory="."):
        self.issues = []
        self.total_savings = 0.0
        files = 0
        for path in Path(directory).rglob("*.py"):
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            files += 1
            self.audit_file(str(path))
        by_severity = {}
        for i in self.issues:
            s = i["severity"]
            by_severity[s] = by_severity.get(s, 0) + 1
        return {
            "files_audited": files,
            "total_issues": len(self.issues),
            "total_potential_savings_usd": round(self.total_savings, 2),
            "by_severity": by_severity,
        }
