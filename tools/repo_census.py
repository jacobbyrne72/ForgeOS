"""Census of every cloned repo: what it is, what licence, how big.

713 repos were cloned and 34 vendored. Nothing recorded what the other 679
were, so the backlog was invisible and "have we looked at this one" had no
answer. This produces that answer mechanically -- no model call, so it can run
over the whole corpus for free and be re-run whenever the set changes.

Deliberately shallow. It answers "what IS this" (language, licence, size,
entry points, stated purpose) so a human or a later pass can decide what to
read properly. Judging value needs reading; this is the index that makes
reading targetable.

Licence is captured because it decides whether code can be copied at all. One
repo in this corpus forbids commercial reuse, which would not have been noticed
without checking -- and noticing after porting is much worse than before.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOTS = [
    Path(r"C:\Users\byrne\Downloads\ForgeOS\vendor"),
    Path(r"C:\Users\byrne\Downloads\ForgeOS-catalog-p0-2026-07-31"),
    Path(r"C:\Users\byrne\Downloads\ForgeOS-catalog-p1-partial-2026-07-31"),
    Path(r"C:\Users\byrne\Downloads\ForgeOS-catalog-p2-partial-2026-07-31"),
    Path(r"C:\Users\byrne\Downloads\ForgeOS-catalog-other-partial-2026-07-31"),
    Path(r"C:\Users\byrne\Downloads\ForgeOS-catalog-replacements-2026-07-31"),
]

# Extension -> language. Only what matters for "can we port this into Python".
_LANG = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "JavaScript",
    ".jsx": "JavaScript", ".rs": "Rust", ".go": "Go", ".java": "Java", ".kt": "Kotlin",
    ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".cpp": "C++", ".c": "C", ".h": "C",
    ".swift": "Swift", ".sh": "Shell", ".md": "Markdown", ".zig": "Zig", ".ex": "Elixir",
}

_SKIP_DIRS = frozenset({
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    "target", ".mypy_cache", ".pytest_cache", "vendor", "site-packages",
    ".next", "coverage", ".tox", "third_party",
})

# Ordered: the first pattern that matches wins, so "Apache" is not reported for
# a file that merely mentions it in a compatibility note.
_LICENCES = (
    ("AGPL-3.0", r"GNU AFFERO GENERAL PUBLIC LICENSE"),
    ("GPL-3.0", r"GNU GENERAL PUBLIC LICENSE\s*\n?\s*Version 3"),
    ("GPL-2.0", r"GNU GENERAL PUBLIC LICENSE\s*\n?\s*Version 2"),
    ("LGPL", r"GNU LESSER GENERAL PUBLIC LICENSE"),
    ("MPL-2.0", r"Mozilla Public License Version 2\.0"),
    ("Apache-2.0", r"Apache License\s*\n?\s*Version 2\.0"),
    ("BSD-3", r"Redistributions in binary form must reproduce"),
    ("MIT", r"Permission is hereby granted, free of charge"),
    ("Unlicense", r"This is free and unencumbered software"),
)

# Licences that forbid or complicate copying code into a permissively-licensed
# project. Surfaced separately because discovering one AFTER porting is much
# worse than before.
COPYLEFT = frozenset({"AGPL-3.0", "GPL-3.0", "GPL-2.0", "LGPL"})


def detect_licence(repo: Path) -> str:
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING", "LICENSE-MIT"):
        path = repo / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:6000]
        except OSError:
            continue
        for label, pattern in _LICENCES:
            if re.search(pattern, text, re.I):
                return label
        return "other"
    return "none-found"


def first_meaningful_line(repo: Path) -> str:
    """The README's first real sentence -- what the project says it is."""
    for name in ("README.md", "readme.md", "README.rst", "README"):
        path = repo / name
        if not path.exists():
            continue
        try:
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
                line = raw.strip().lstrip("#").strip()
                # Skip badges, images and empty decoration.
                if not line or line.startswith(("![", "[!", "<", "---", "=", "|")):
                    continue
                if len(line) < 12:
                    continue
                return re.sub(r"\s+", " ", line)[:180]
        except OSError:
            pass
    return ""


def survey(repo: Path, *, max_files: int = 8000) -> dict:
    langs: Counter[str] = Counter()
    files = 0
    entry_points: list[str] = []
    truncated = False

    for path in repo.rglob("*"):
        if files >= max_files:
            truncated = True
            break
        # Relative to the repo, never absolute. Checking absolute parts meant
        # that when a ROOT was itself named `vendor/`, every file beneath it
        # matched the skip set and the whole tree surveyed as "unknown".
        try:
            rel_parts = set(path.relative_to(repo).parts[:-1])
        except ValueError:
            continue
        if rel_parts & _SKIP_DIRS:
            continue
        if not path.is_file():
            continue
        files += 1
        lang = _LANG.get(path.suffix)
        if lang:
            langs[lang] += 1
        name = path.name
        if name in ("pyproject.toml", "setup.py", "package.json", "Cargo.toml",
                    "go.mod", "pom.xml", "docker-compose.yml", "Dockerfile"):
            rel = path.relative_to(repo).as_posix()
            if rel.count("/") <= 1:
                entry_points.append(rel)

    primary = langs.most_common(1)[0][0] if langs else "unknown"
    return {
        "name": repo.name,
        "path": str(repo),
        "primary_language": primary,
        "languages": dict(langs.most_common(5)),
        "files": files,
        "truncated": truncated,
        "licence": detect_licence(repo),
        "entry_points": sorted(set(entry_points))[:6],
        "purpose": first_meaningful_line(repo),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/repo_census.json")
    ap.add_argument("--markdown", default="artifacts/repo_census.md")
    ap.add_argument("--limit", type=int, default=0, help="0 = every repo")
    args = ap.parse_args()

    repos: list[dict] = []
    for root in ROOTS:
        if not root.exists():
            print(f"  (missing root: {root})")
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            repos.append(survey(child))
            if args.limit and len(repos) >= args.limit:
                break
        if args.limit and len(repos) >= args.limit:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(repos, indent=2, sort_keys=True), encoding="utf-8")

    by_lang = Counter(r["primary_language"] for r in repos)
    by_lic = Counter(r["licence"] for r in repos)
    restricted = [r for r in repos if r["licence"] in COPYLEFT]
    python_repos = [r for r in repos if r["primary_language"] == "Python"]

    lines = [
        "# Repo census", "",
        f"{len(repos)} cloned repositories surveyed. Mechanical: no model call, "
        "so this is cheap to re-run whenever the set changes.", "",
        "## Primary language", "",
        "| language | repos |", "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in by_lang.most_common()]
    lines += ["", "## Licence", "", "| licence | repos |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in by_lic.most_common()]
    lines += [
        "", "## Copyleft — code cannot simply be copied", "",
        "Listed because discovering this AFTER porting is far worse than before. "
        "These may still be read for design, and used as separate processes.", "",
    ]
    lines += [f"- `{r['name']}` ({r['licence']})" for r in restricted] or ["- none"]
    lines += [
        "", f"## Python repos ({len(python_repos)}) — portable without a rewrite", "",
        "| repo | files | licence | purpose |", "|---|---|---|---|",
    ]
    for r in sorted(python_repos, key=lambda r: -r["files"])[:80]:
        purpose = (r["purpose"] or "")[:90].replace("|", "/")
        lines.append(f"| `{r['name']}` | {r['files']} | {r['licence']} | {purpose} |")

    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"surveyed {len(repos)} repos -> {out} and {args.markdown}")
    print(f"  languages: {dict(by_lang.most_common(6))}")
    print(f"  licences:  {dict(by_lic.most_common(6))}")
    print(f"  copyleft:  {len(restricted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
