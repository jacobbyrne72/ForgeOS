"""Packing techniques ported from the codex-brain ripper (`atlas.py`).

Naive packing takes items until the budget runs out. That fails in four specific
ways, and each function here answers one of them:

1. **First-come-first-served spends the budget on whatever was ranked first.**
   `allocate_sections` splits the budget by ROLE instead, so evidence cannot crowd
   out the summary cards or the overflow notice.

2. **Truncating at a byte offset cuts a function in half.** `expand_to_boundary`
   grows an excerpt out to the nearest structural line, so the model never receives
   a fragment that starts mid-body.

3. **One directory dominates the pack.** `diversify` caps sources per directory, so
   fourteen slots do not all go to the same module.

4. **Contradictions get silently blended.** `find_tensions` surfaces where sources
   DISAGREE. Handing a model a confident blend of conflicting evidence is worse than
   handing it less evidence, because the conflict is invisible and unresolvable.

And `next_reads` makes the overflow explicit — a worker that knows what was withheld
can ask for it, which is the difference between a bounded pack and a lossy one.
"""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, Field

# Budget split by section role. Evidence dominates, but never to the point of
# starving the map (cards) or the honesty (tensions, next_reads).
SECTION_SPLIT: dict[str, float] = {
    "cards": 0.15,
    "evidence": 0.62,
    "tensions": 0.08,
    "next_reads": 0.08,
}

# Structural lines an excerpt may start or end on. Cutting anywhere else hands the
# model a fragment whose beginning it cannot see.
BOUNDARY_RE = re.compile(r"^(#{1,6} |def |class |function |async def |## |---|=== |\s*$)")

MAX_SOURCES = 14
MAX_SOURCES_PER_DIR = 3
MAX_EXCERPTS_PER_SOURCE = 3
BOUNDARY_SEARCH_LINES = 40

# Words that signal a claim's polarity. Crude on purpose — the goal is to FLAG a
# possible disagreement for a human or model to resolve, not to adjudicate it.
POSITIVE_HINTS = (
    "works", "passes", "passed", "confirmed", "viable", "correct", "fixed",
    "succeeds", "stable", "verified", "supported",
)
NEGATIVE_HINTS = (
    "fails", "failed", "broken", "rejected", "dead", "deprecated", "unsupported",
    "regression", "crashes", "does not work", "doesn't work", "removed",
)


class Excerpt(BaseModel):
    path: str
    start_line: int
    end_line: int
    text: str
    tokens: int = 0
    reason: str = ""


class Tension(BaseModel):
    """Two sources that appear to disagree about the same term."""

    term: str
    positive_source: str
    negative_source: str
    positive_line: str
    negative_line: str

    def render(self) -> str:
        return (
            f"CONFLICT on '{self.term}':\n"
            f"  {self.positive_source}: {self.positive_line.strip()[:160]}\n"
            f"  {self.negative_source}: {self.negative_line.strip()[:160]}"
        )


class Overflow(BaseModel):
    """What did not fit. Named so it can be requested, not silently dropped."""

    path: str
    reason: str
    est_tokens: int = 0


class SectionBudget(BaseModel):
    cards: int
    evidence: int
    tensions: int
    next_reads: int

    @property
    def total(self) -> int:
        return self.cards + self.evidence + self.tensions + self.next_reads


def allocate_sections(budget: int, split: dict[str, float] | None = None) -> SectionBudget:
    """Split a token budget by section role.

    Any rounding remainder goes to evidence, which is the section that most benefits
    from one extra excerpt.
    """
    split = split or SECTION_SPLIT
    cards = int(budget * split.get("cards", 0.0))
    tensions = int(budget * split.get("tensions", 0.0))
    next_reads = int(budget * split.get("next_reads", 0.0))
    evidence = max(0, budget - cards - tensions - next_reads)
    return SectionBudget(cards=cards, evidence=evidence, tensions=tensions,
                         next_reads=next_reads)


def expand_to_boundary(
    lines: list[str],
    start: int,
    end: int,
    *,
    max_search: int = BOUNDARY_SEARCH_LINES,
) -> tuple[int, int]:
    """Grow [start, end) outward to the nearest structural boundaries.

    Bounded by `max_search` so a file with no boundaries at all cannot cause the
    excerpt to swallow the whole file — the guard against a fix becoming the bug.
    """
    if not lines:
        return (0, 0)

    start = max(0, min(start, len(lines)))
    end = max(start, min(end, len(lines)))

    new_start = start
    for i in range(start, max(-1, start - max_search), -1):
        if i < len(lines) and BOUNDARY_RE.match(lines[i]):
            new_start = i
            break

    new_end = end
    for i in range(end, min(len(lines), end + max_search)):
        if BOUNDARY_RE.match(lines[i]):
            new_end = i
            break
    else:
        new_end = min(len(lines), end)

    return (new_start, max(new_start + 1, new_end))


def diversify(
    sources: list[dict],
    *,
    max_sources: int = MAX_SOURCES,
    max_per_dir: int = MAX_SOURCES_PER_DIR,
    path_key: str = "path",
) -> tuple[list[dict], list[Overflow]]:
    """Cap sources overall and per directory. Returns (kept, overflow).

    Without the per-directory cap, a tightly-clustered match set spends every slot
    inside one module and the pack answers a narrower question than it appears to.
    """
    kept: list[dict] = []
    overflow: list[Overflow] = []
    per_dir: dict[str, int] = defaultdict(int)

    for src in sources:
        path = str(src.get(path_key, ""))
        directory = path.replace("\\", "/").rsplit("/", 1)[0] if "/" in path else "."

        if len(kept) >= max_sources:
            overflow.append(Overflow(path=path, reason=f"over the {max_sources}-source cap",
                                     est_tokens=int(src.get("tokens", 0) or 0)))
            continue
        if per_dir[directory] >= max_per_dir:
            overflow.append(Overflow(path=path,
                                     reason=f"{directory} already has {max_per_dir} sources",
                                     est_tokens=int(src.get("tokens", 0) or 0)))
            continue

        per_dir[directory] += 1
        kept.append(src)

    return kept, overflow


def find_tensions(sources: list[dict], terms: list[str], *, max_tensions: int = 5) -> list[Tension]:
    """Surface where sources disagree about the same term.

    This is the technique naive packing lacks entirely. A model given two
    contradictory sources with no signal will confidently synthesise them into a
    single wrong answer; given the conflict explicitly, it can say which it trusts
    or ask. Flagging beats adjudicating — this function never decides who is right.
    """
    tensions: list[Tension] = []

    for term in terms:
        low_term = term.lower()
        positives: list[tuple[str, str]] = []
        negatives: list[tuple[str, str]] = []

        for src in sources:
            path = str(src.get("path", ""))
            text = str(src.get("text", ""))
            for line in text.splitlines():
                low = line.lower()
                if low_term not in low:
                    continue
                if any(h in low for h in NEGATIVE_HINTS):
                    negatives.append((path, line))
                elif any(h in low for h in POSITIVE_HINTS):
                    positives.append((path, line))

        for (ppath, pline) in positives:
            for (npath, nline) in negatives:
                # A single source contradicting itself is usually narrative
                # ("X failed, then X passed"), not a genuine conflict between sources.
                if ppath == npath:
                    continue
                tensions.append(
                    Tension(term=term, positive_source=ppath, negative_source=npath,
                            positive_line=pline, negative_line=nline)
                )
                break
            if len(tensions) >= max_tensions:
                return tensions

    return tensions[:max_tensions]


def next_reads(overflow: list[Overflow], *, limit: int = 6) -> list[str]:
    """Name what was withheld, so it can be asked for.

    Silent truncation is the failure this prevents: a worker that does not know
    something was excluded reports a confident conclusion drawn from a partial view.
    """
    lines: list[str] = []
    for item in overflow[:limit]:
        est = f" (~{item.est_tokens} tok)" if item.est_tokens else ""
        lines.append(f"{item.path}{est} — {item.reason}")
    if len(overflow) > limit:
        lines.append(f"…and {len(overflow) - limit} more excluded")
    return lines


def spend(section: list, items: list, budget_tokens: int, *, token_key: str = "tokens") -> int:
    """Fill a section up to its budget. Returns tokens spent.

    Stops at the first item that would exceed the budget rather than skipping ahead
    to smaller ones — preserving rank order matters more than packing tightly, since
    the ranking is what made the pack relevant.
    """
    spent = 0
    for item in items:
        cost = int(getattr(item, token_key, None) or (item.get(token_key, 0) if isinstance(item, dict) else 0))
        if spent + cost > budget_tokens:
            break
        section.append(item)
        spent += cost
    return spent


class PackReport(BaseModel):
    """The honest accounting for one pack."""

    kept_sources: int
    excluded_sources: int
    tensions_found: int
    baseline_tokens: int
    actual_tokens: int
    sections: SectionBudget

    @property
    def saved_tokens(self) -> int:
        return max(0, self.baseline_tokens - self.actual_tokens)

    @property
    def saved_pct(self) -> float:
        if self.baseline_tokens <= 0:
            return 0.0
        return round(100.0 * self.saved_tokens / self.baseline_tokens, 2)

    def render(self) -> str:
        parts = [
            f"{self.kept_sources} sources kept, {self.excluded_sources} excluded",
            f"{self.saved_tokens:,} tokens saved ({self.saved_pct}%)",
        ]
        if self.tensions_found:
            parts.append(f"{self.tensions_found} conflict(s) flagged")
        return " | ".join(parts)


__all__ = [
    "BOUNDARY_RE",
    "Excerpt",
    "MAX_EXCERPTS_PER_SOURCE",
    "MAX_SOURCES",
    "MAX_SOURCES_PER_DIR",
    "NEGATIVE_HINTS",
    "POSITIVE_HINTS",
    "Overflow",
    "PackReport",
    "SECTION_SPLIT",
    "SectionBudget",
    "Tension",
    "allocate_sections",
    "diversify",
    "expand_to_boundary",
    "find_tensions",
    "next_reads",
    "spend",
]
