"""Context compression that returns whole symbols, not orphaned lines.

This module used to open with "AST-aware context compression" and then say, in
its own next sentence, "Uses string matching". Both could not be true and the
second one was. It also claimed "cutting prompt tokens by 60-90% on typical
tasks", which nothing measured. Recorded here rather than quietly deleted: a
docstring overstating what the code does is the exact failure this project
argues against, and finding one in our own source is worth more than the lines
it cost.

WHY AN AST AND NOT A LINE WINDOW.

The old approach matched objective words against lines and kept a +/-2 line
window around each hit. That produces fragments: a `def` line without its body,
a body without its signature, an `if` without what it guards. A model handed
those either fills the gap by guessing or asks for the file again — and the
second outcome costs more than sending the function whole would have.

A syntax tree knows where a symbol starts and stops. Ranking whole symbols and
sending complete ones means every byte spent is a byte that parses, and
anything that clears the bar arrives with its decorators, signature, docstring
and body intact.

FALLBACK IS LABELLED. Python parses with the stdlib `ast`; any other language,
or a file with a syntax error (common mid-edit), falls back to the line-window
approach — and `CompressionResult.method` says which one ran. Silently
degrading to fragments while claiming symbol extraction is precisely how the
original docstring came to be wrong.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum

# Around each matched line in fallback mode. Deliberately small: a fragment is
# a fragment, and widening it does not make it a function.
_WINDOW = 2

# Words in nearly every objective, discriminating nothing.
_STOPWORDS = frozenset("""
a an the and or of to in on at by as is are was be been it its this that these
those for from with without into under over what which who when where why how
do does did can could should would add fix update change make use using write
implement create build test run get set new
""".split())


class Method(str, Enum):
    SYMBOLS = "ast-symbols"      # whole functions/classes, syntactically complete
    LINE_WINDOW = "line-window"  # fragments; the fallback
    UNCHANGED = "unchanged"      # nothing matched, so nothing was dropped


@dataclass
class CompressionResult:
    files: list[tuple[str, str]]
    method: Method
    chars_before: int = 0
    chars_after: int = 0
    symbols_kept: int = 0
    symbols_total: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def reduction_pct(self) -> float:
        """Measured on THIS input, and only this input.

        Never a headline figure: compression varies enormously with how much of
        a file the objective actually touches, and quoting one number as typical
        is the overstatement the old docstring made.
        """
        if self.chars_before <= 0:
            return 0.0
        return 100.0 * (1 - self.chars_after / self.chars_before)

    def render(self) -> str:
        return (f"{self.method.value}: {self.chars_before:,} -> {self.chars_after:,} chars "
                f"({self.reduction_pct:.0f}% smaller), "
                f"{self.symbols_kept}/{self.symbols_total} symbols")


def objective_terms(objective: str) -> set[str]:
    """Distinguishing words from an objective, with snake_case parts split out."""
    terms = {t.strip(".,()[]{}:;\"'`").lower() for t in (objective or "").split()}
    out: set[str] = set()
    for t in terms:
        if len(t) < 3 or t in _STOPWORDS:
            continue
        out.add(t)
        out.update(p for p in t.replace(".", "_").split("_") if len(p) > 2)
    return out


@dataclass
class _Symbol:
    name: str
    start: int          # 1-based, inclusive of decorators
    end: int            # 1-based, inclusive
    source: str


def _symbols(source: str) -> list[_Symbol] | None:
    """Top-level and class-level defs/classes, whole. None if it will not parse.

    Decorators are included by starting at the earliest decorator line: a method
    delivered without its `@property` reads as a plain method, and the model
    then reasons about something that does not exist.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    lines = source.splitlines()
    found: list[_Symbol] = []

    def add(node) -> None:
        start = node.lineno
        for dec in getattr(node, "decorator_list", []) or []:
            start = min(start, getattr(dec, "lineno", start))
        end = getattr(node, "end_lineno", None) or start
        found.append(_Symbol(
            name=getattr(node, "name", "?"), start=start, end=end,
            source="\n".join(lines[start - 1:end]),
        ))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node)
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        add(child)
    return found


def _score(symbol: _Symbol, terms: set[str]) -> int:
    """A name match counts for more than a body match.

    The symbol's name is usually what the objective is about; a term appearing
    once in a body may be a local variable that happens to share a word. Same
    field-weighting principle as the retrieval ranker, for the same reason.
    """
    name = symbol.name.lower()
    body = symbol.source.lower()
    score = 0
    for term in terms:
        if term in name:
            score += 5
        if term in body:
            score += 1
    return score


def compress_context(
    objective: str,
    files: list[tuple[str, str]],
    language: str = "python",
) -> list[tuple[str, str]]:
    """Compressed `(path, source)` pairs. Signature unchanged for existing callers.

    Returns the input untouched when nothing matches: sending nothing is not
    compression, it is losing the task's context, and the return value alone
    cannot tell a caller which happened.
    """
    return compress(objective, files, language=language).files


def compress(
    objective: str,
    files: list[tuple[str, str]],
    *,
    language: str = "python",
    min_score: int = 1,
) -> CompressionResult:
    """The same work with the evidence attached.

    `method` reports which strategy actually ran, so a caller can tell whole
    symbols from fallback fragments rather than assuming.
    """
    terms = objective_terms(objective)
    before = sum(len(src) for _, src in files)
    if not terms:
        return CompressionResult(files, Method.UNCHANGED, before, before,
                                 notes=["objective carried no distinguishing terms"])

    out: list[tuple[str, str]] = []
    kept = total = 0
    used_fallback = False
    notes: list[str] = []

    for path, source in files:
        symbols = _symbols(source) if language == "python" else None
        if symbols is None:
            used_fallback = True
            if language == "python":
                notes.append(f"{path}: would not parse; fell back to line windows")
            trimmed = _line_window(source, terms)
            if trimmed:
                out.append((path, trimmed))
            continue

        total += len(symbols)
        chosen = [s for s in symbols if _score(s, terms) >= min_score]
        # A matched method arrives inside its matched class; keeping both sends
        # the same body twice.
        chosen = _drop_contained(chosen)
        if chosen:
            kept += len(chosen)
            out.append((path, "\n\n".join(s.source for s in chosen)))

    if not out:
        return CompressionResult(files, Method.UNCHANGED, before, before,
                                 symbols_total=total,
                                 notes=notes + ["nothing scored above the threshold"])

    after = sum(len(src) for _, src in out)
    method = Method.LINE_WINDOW if used_fallback else Method.SYMBOLS
    return CompressionResult(out, method, before, after, kept, total, notes)


def _drop_contained(symbols: list[_Symbol]) -> list[_Symbol]:
    """Drop symbols wholly inside another chosen symbol's line range."""
    ordered = sorted(symbols, key=lambda s: (s.start, -s.end))
    out: list[_Symbol] = []
    for sym in ordered:
        if any(o.start <= sym.start and sym.end <= o.end for o in out):
            continue
        out.append(sym)
    return out


def _line_window(source: str, terms: set[str]) -> str:
    """The original strategy, kept only as a fallback and labelled as one."""
    lines = source.splitlines()
    keep: set[int] = set()
    for i, line in enumerate(lines):
        low = line.lower()
        if any(t in low for t in terms):
            keep.update(range(max(0, i - _WINDOW), min(len(lines), i + _WINDOW + 1)))
    return "\n".join(lines[i] for i in sorted(keep))
