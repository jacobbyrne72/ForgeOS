"""Mission compiler — natural language → decomposed TaskSpecs.

Deterministic decomposition using tree-sitter analysis. Never
calls a model to decompose. Conservative: produces few well-scoped
tasks with verifiable acceptance criteria.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from .contracts import Budget, Scope, TaskSpec, new_id

LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".toml": "toml",
    ".cfg": "ini",
    ".ini": "ini",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".rs": "rust",
    ".go": "go",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".java": "java",
    ".sh": "bash",
    ".bash": "bash",
}
TREE_EXTS = set(LANG_MAP.keys())
MAX_SOURCE = 200
MAX_SYMBOLS = 50


@dataclass
class Symbol:
    name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    language: str
    docstring: str = ""
    signature: str = ""

    @property
    def range(self) -> str:
        return f"{self.path}:L{self.start_line}-L{self.end_line}"


@dataclass
class FileAnalysis:
    path: str
    language: str
    symbols: list = field(default_factory=list)
    imports: list = field(default_factory=list)
    test_functions: list = field(default_factory=list)


@dataclass
class Mission:
    objective: str
    tasks: list[TaskSpec] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    task_map: dict[str, str] = field(default_factory=dict)

    def verify(self) -> list[str]:
        w = []
        for t in self.tasks:
            if not t.acceptance:
                w.append(f"task '{t.subject}' has no acceptance criteria")
            if not t.scope.paths:
                w.append(f"task '{t.subject}' has empty scope")
        return w


class CompilerError(RuntimeError):
    pass


OBJECTIVE_KINDS = [
    (re.compile(r"\b(add|implement|create|build|write|new)\b", re.I), "implementation"),
    (re.compile(r"\b(fix|repair|correct|resolve|patch)\b", re.I), "bugfix"),
    (re.compile(r"\b(test|spec|verify|validate)\b", re.I), "testing"),
    (re.compile(r"\b(review|audit|inspect|check|security)\b", re.I), "review"),
    (re.compile(r"\b(refactor|restructure|clean|simplify)\b", re.I), "refactor"),
    (re.compile(r"\b(update|upgrade|migrate|port)\b", re.I), "update"),
    (re.compile(r"\b(document|docs|readme)\b", re.I), "documentation"),
]


def classify_objective(objective: str) -> str:
    best, best_score = "implementation", 0
    for pat, kind in OBJECTIVE_KINDS:
        score = len(pat.findall(objective))
        if score > best_score:
            best_score = score
            best = kind
    return best


def _scope_from_objective(objective: str) -> list[str]:
    tokens = objective.split()
    explicit: list[str] = []
    for t in tokens:
        t = t.strip(".,:;()[]{}\"'\x60")
        if "/" in t or any(t.endswith(ext) for ext in TREE_EXTS):
            explicit.append(t)
    return explicit


def _budget_for(kind: str) -> Budget:
    budgets = {
        "implementation": Budget(max_usd=5.0, max_seconds=1800, max_iterations=8),
        "bugfix": Budget(max_usd=3.0, max_seconds=1200, max_iterations=6),
        "refactor": Budget(max_usd=4.0, max_seconds=1800, max_iterations=6),
        "testing": Budget(max_usd=2.0, max_seconds=600, max_iterations=4),
        "review": Budget(max_usd=2.0, max_seconds=600, max_iterations=3),
        "update": Budget(max_usd=5.0, max_seconds=1800, max_iterations=8),
        "documentation": Budget(max_usd=1.0, max_seconds=300, max_iterations=3),
    }
    return budgets.get(kind, Budget())


def _acceptance_for(kind: str, scope_paths: list[str]) -> list[str]:
    has_py = any(p.endswith(".py") for p in scope_paths)
    has_js = any(p.endswith((".js", ".ts", ".tsx", ".jsx")) for p in scope_paths)
    a: list[str] = []
    if kind == "testing":
        a.append("pytest passes")
    elif kind == "bugfix":
        a.append("pytest -x -q")
    elif kind == "implementation":
        if has_py:
            a.append("pytest -x -q")
        elif has_js:
            a.append("npx jest --passWithNoTests")
        a.append("ruff check --quiet" if has_py else "eslint --quiet")
    elif kind == "refactor":
        if has_py:
            a.append("pytest -x -q")
        a.append("ruff check --quiet" if has_py else "")
    elif kind == "review":
        a.append("No security findings")
        a.append("All existing tests pass")
    elif kind == "documentation":
        a.append("No lint errors introduced")
    else:
        a.append("The change works correctly")
    return [x for x in a if x]


def _caps_for(kind: str) -> list[str]:
    mapping = {
        "implementation": ["edit", "implement", "python"],
        "bugfix": ["edit", "implement", "debug"],
        "refactor": ["edit", "refactor", "python"],
        "testing": ["test", "verify", "python"],
        "review": ["review", "verify"],
        "update": ["edit", "implement"],
        "documentation": ["edit"],
    }
    return mapping.get(kind, ["edit", "implement"])


def _infer_scope(analyses: list, objective: str) -> list[str]:
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "through",
        "during",
        "before",
        "after",
        "all",
        "each",
        "every",
        "few",
        "more",
        "most",
        "only",
        "own",
        "same",
        "some",
        "such",
        "not",
        "no",
    }
    words = set(objective.lower().split()) - stop
    if not words:
        return []
    scored = []
    for fa in analyses:
        score = 0
        for sym in fa.symbols:
            for w in words:
                if w in sym.name.lower():
                    score += 3
                if w in (sym.signature or "").lower():
                    score += 1
        for imp in fa.imports:
            for w in words:
                if w in imp.lower():
                    score += 2
        if fa.test_functions and "test" not in words:
            score -= len(fa.test_functions) * 2
        if score > 0:
            scored.append((score, fa.path))
    scored.sort(reverse=True)
    return [p for _, p in scored[:20]]


def _walk_ts(node, source, path, lang, symbols, depth=0) -> None:
    if depth > 100:
        return
    kinds = (
        "function_definition",
        "function_declaration",
        "method_definition",
        "class_definition",
        "struct_definition",
        "enum_definition",
        "type_definition",
        "variable_declaration",
    )
    if node.type in kinds:
        name = ""
        for child in node.children:
            if child.type == "identifier":
                name = child.text.decode("utf-8", errors="replace")
                break
        lines = source.splitlines()
        sig = lines[node.start_point[0]][:200] if node.start_point[0] < len(lines) else ""
        symbols.append(
            Symbol(
                name=name or node.type,
                kind=node.type,
                path=path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                language=lang,
                signature=sig,
            )
        )
    for child in node.children:
        _walk_ts(child, source, path, lang, symbols, depth + 1)


def analyze_codebase(cwd: str) -> list[FileAnalysis]:
    analyses: list[FileAnalysis] = []
    try:
        from tree_sitter import Parser
    except ImportError:
        return analyses
    cwd_path = Path(cwd).resolve()
    source_files: list[Path] = []
    for root, dirs, files in os.walk(cwd_path):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d
            not in {"__pycache__", "vendor", "node_modules", "venv", ".venv", ".forgeos", ".forgeos-live"}
        ]
        for f in sorted(files):
            ext = Path(f).suffix.lower()
            if ext in TREE_EXTS:
                source_files.append(Path(root) / f)
            if len(source_files) >= MAX_SOURCE:
                break
        if len(source_files) >= MAX_SOURCE:
            break
    for sp in source_files[:100]:
        try:
            lang = LANG_MAP.get(sp.suffix.lower(), "")
            if not lang:
                continue
            mod = __import__(f"tree_sitter_{lang}", fromlist=["language"])
            L = mod.language() if hasattr(mod, "language") else mod.Language(mod.language_grammar())  # type: ignore[attr-defined]
            parser = Parser()
            parser.set_language(L)
            raw = sp.read_bytes()
            if not raw:
                continue
            tree = parser.parse(raw)
            syms: list[Symbol] = []
            _walk_ts(tree.root_node, raw.decode("utf-8", errors="replace"), str(sp), lang, syms)
            syms = syms[:MAX_SYMBOLS]
            if syms:
                fa = FileAnalysis(path=str(sp), language=lang, symbols=syms)
                fa.imports = [
                    s.signature
                    for s in syms
                    if s.kind in ("import_statement", "import_declaration", "require_statement")
                ]
                fa.test_functions = [
                    s
                    for s in syms
                    if any(m in s.name.lower() for m in ("test_", "spec_", "_test", "_spec", "should_"))
                ]
                analyses.append(fa)
        except Exception:
            continue
    return analyses


def compile_mission(objective: str, *, cwd: str = ".", max_tasks: int = 6) -> Mission:
    """Compile a natural-language objective into decomposed tasks."""
    objective = objective.strip()
    if not objective:
        raise CompilerError("objective must not be blank")
    kind = classify_objective(objective)
    analyses = analyze_codebase(cwd)
    explicit = _scope_from_objective(objective)
    scope = explicit if explicit else (_infer_scope(analyses, objective) or ["."])
    tasks: list[TaskSpec] = []
    deps: dict[str, list[str]] = {}
    if kind == "bugfix" and max_tasks >= 2:
        t1 = TaskSpec(
            id=new_id("task"),
            job_id="",
            subject=f"Diagnose: {objective[:100]}",
            description=f"Investigate root cause. Scope: {' '.join(scope[:5])}.",
            acceptance=["pytest -x -q"],
            scope=Scope(paths=scope[:5]),
            budget=Budget(max_usd=2.0, max_seconds=600, max_iterations=4),
            capabilities=["search", "debug", "implement"],
        )
        tasks.append(t1)
        t2 = TaskSpec(
            id=new_id("task"),
            job_id="",
            subject=f"Fix: {objective[:100]}",
            description=f"Implement the fix. Scope: {' '.join(scope[:5])}.",
            acceptance=_acceptance_for("bugfix", scope),
            scope=Scope(paths=scope[:5]),
            budget=_budget_for("bugfix"),
            capabilities=["edit", "implement", "debug"],
            depends_on=[t1.id],
        )
        tasks.append(t2)
        deps[t2.id] = [t1.id]
    elif kind == "refactor" and max_tasks >= 2:
        t1 = TaskSpec(
            id=new_id("task"),
            job_id="",
            subject=f"Identify refactoring targets: {objective[:80]}",
            description=f"Analyze codebase for refactoring. Scope: {' '.join(scope[:5])}.",
            acceptance=["pytest -x -q"],
            scope=Scope(paths=scope[:5]),
            budget=Budget(max_usd=2.0, max_seconds=600, max_iterations=4),
            capabilities=["search", "review", "python"],
        )
        tasks.append(t1)
        t2 = TaskSpec(
            id=new_id("task"),
            job_id="",
            subject=f"Apply refactoring: {objective[:150]}",
            description=f"Apply the refactoring. Scope: {' '.join(scope[:5])}.",
            acceptance=_acceptance_for("refactor", scope),
            scope=Scope(paths=scope[:5]),
            budget=_budget_for("refactor"),
            capabilities=["edit", "refactor", "python"],
            depends_on=[t1.id],
        )
        tasks.append(t2)
        deps[t2.id] = [t1.id]
    else:
        tasks.append(
            TaskSpec(
                id=new_id("task"),
                job_id="",
                subject=objective[:150],
                description=f"Work on: {objective}. Scope: {' '.join(scope[:10])}",
                acceptance=_acceptance_for(kind, scope),
                scope=Scope(paths=scope[:10]),
                budget=_budget_for(kind),
                capabilities=_caps_for(kind),
            )
        )
    mission = Mission(objective=objective, tasks=tasks, dependencies=deps)
    mission.task_map = {t.id: t.subject for t in tasks}
    return mission
