"""Rank 713 blobless clones by what their FILE TREES say they contain.

The catalogue clones were made with `--filter=blob:none`. Their `.git`
directories are ~80KB and hold commit and tree objects only: every filename and
every path is known, and not one byte of file CONTENT is present. `git show
HEAD:README.md` returns "bad object" on all of them. So "read every repo" is
not something the disk can answer -- it would take 713 network fetches, and
this machine has already been crashed once by parallel cloning.

What IS available locally, instantly and for free, is the full path listing of
every repo. That turns out to be a lot. A directory called `sandbox/` or
`landlock/` tells you a project does process isolation; `retry.py` and
`backoff.py` tell you it has a retry policy worth reading; `embeddings/` and
`vector_store/` say it does semantic memory. Filenames are what maintainers
chose to call things, and they name capabilities.

So this scores every repo on how many ForgeOS-relevant capability signals its
tree shows, and produces a hydration order. Hydrating the top N sequentially is
a bounded, resumable job; hydrating all 713 is not.

No model call, no network. `git ls-tree` reads local objects only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOTS = [
    Path(r"C:\Users\byrne\Downloads\ForgeOS-catalog-p0-2026-07-31"),
    Path(r"C:\Users\byrne\Downloads\ForgeOS-catalog-p1-partial-2026-07-31"),
    Path(r"C:\Users\byrne\Downloads\ForgeOS-catalog-p2-partial-2026-07-31"),
    Path(r"C:\Users\byrne\Downloads\ForgeOS-catalog-other-partial-2026-07-31"),
    Path(r"C:\Users\byrne\Downloads\ForgeOS-catalog-replacements-2026-07-31"),
]

# Capability -> path substrings that betray it, and what it is worth to ForgeOS.
#
# Weights encode where ForgeOS is actually WEAK, not what sounds impressive.
# Sandboxing scores highest because ForgeOS has none at all and ships a
# shell-execution tool; retrieval scores low because ForgeOS's capsule and
# two-stage reranker are already ahead of most of what is out there.
SIGNALS: dict[str, tuple[int, tuple[str, ...]]] = {
    "sandboxing":      (10, ("sandbox", "landlock", "seccomp", "seatbelt", "bwrap",
                             "firejail", "isolate", "jail", "execpolicy")),
    "durable_workflow": (8, ("checkpoint", "resume", "replay", "workflow_state",
                             "saga", "idempot", "dead_letter", "deadletter")),
    "retry_policy":     (7, ("retry", "backoff", "circuit_break", "resilien")),
    "scheduling":       (7, ("scheduler", "dag", "task_graph", "topolog", "dispatch")),
    "cost_tracking":    (7, ("token_count", "pricing", "cost_track", "usage_track",
                             "budget", "spend")),
    "memory":           (6, ("embedding", "vector_store", "vectorstore", "memory",
                             "knowledge_graph", "recall")),
    "prompt_cache":     (6, ("prompt_cache", "cache_control", "kv_cache", "prefix_cache")),
    "eval_harness":     (6, ("benchmark", "eval", "harness", "scorer", "rubric")),
    "routing":          (5, ("router", "routing", "model_select", "provider_pool",
                             "failover", "load_balanc")),
    "observability":    (5, ("telemetry", "tracing", "otel", "span", "metrics")),
    "compression":      (5, ("compress", "summariz", "prune", "distill", "truncat")),
    "sub_agents":       (4, ("subagent", "sub_agent", "swarm", "crew", "team",
                             "orchestrat", "multi_agent")),
    "crawling":         (4, ("crawl", "scrape", "fetcher", "spider", "playwright")),
    "security":         (4, ("gitleaks", "semgrep", "secret_scan", "redact", "sanitiz")),
    "hooks":            (3, ("hook", "middleware", "interceptor", "plugin")),
    "mcp":              (3, ("mcp", "model_context_protocol")),
}

# Trees that are big enough to hold real engineering. A 3-file repo is a gist.
MIN_INTERESTING_FILES = 12


def tree(repo: Path) -> list[str]:
    """Every path in HEAD. Local objects only -- never triggers a fetch."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-tree", "-r", "HEAD", "--name-only"],
            capture_output=True, text=True, timeout=60,
            env={"GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0",
                 "PATH": __import__("os").environ.get("PATH", "")},
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def score(paths: list[str]) -> tuple[int, dict[str, int]]:
    """Total capability score and the per-signal hit counts.

    Counts DISTINCT files per signal rather than raw substring hits: one file
    named `retry_retry_retry.py` is not three times the evidence of a retry
    policy that `retry.py` plus `backoff.py` is.
    """
    low = [p.lower() for p in paths]
    hits: dict[str, int] = {}
    total = 0
    for name, (weight, needles) in SIGNALS.items():
        matched = {p for p in low if any(n in p for n in needles)}
        if matched:
            hits[name] = len(matched)
            # Diminishing: evidence of a capability saturates quickly. Twenty
            # files mentioning "retry" does not make it twenty times better.
            total += weight * min(len(matched), 4)
    return total, hits


def languages(paths: list[str]) -> dict[str, int]:
    ext = Counter(Path(p).suffix.lower() for p in paths)
    lang = {".py": "Python", ".ts": "TypeScript", ".rs": "Rust", ".go": "Go",
            ".java": "Java", ".js": "JavaScript", ".rb": "Ruby", ".c": "C",
            ".cpp": "C++", ".cs": "C#"}
    out: Counter[str] = Counter()
    for suffix, count in ext.items():
        if suffix in lang:
            out[lang[suffix]] += count
    return dict(out.most_common(4))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/repo_triage.json")
    ap.add_argument("--markdown", default="artifacts/repo_triage.md")
    ap.add_argument("--top", type=int, default=60, help="how many to list for hydration")
    args = ap.parse_args()

    rows: list[dict] = []
    empty = 0
    for root in ROOTS:
        if not root.exists():
            continue
        for repo in sorted(p for p in root.iterdir() if p.is_dir()):
            paths = tree(repo)
            if not paths:
                empty += 1
                continue
            total, hits = score(paths)
            langs = languages(paths)
            rows.append({
                "name": repo.name,
                "path": str(repo),
                "files": len(paths),
                "score": total,
                "signals": hits,
                "languages": langs,
                "python": langs.get("Python", 0),
            })

    rows.sort(key=lambda r: (-r["score"], -r["python"], r["name"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    worthwhile = [r for r in rows if r["files"] >= MIN_INTERESTING_FILES and r["score"] > 0]
    signal_totals: Counter[str] = Counter()
    for r in rows:
        for name in r["signals"]:
            signal_totals[name] += 1

    lines = [
        "# Repo triage — ranked from file trees alone", "",
        "The catalogue clones are `--filter=blob:none`: every FILENAME is local, no",
        "file CONTENT is. `git show HEAD:README.md` returns `bad object` on all of",
        "them. Reading them means 713 network fetches, so this ranks them first and",
        "hydrates only what earns it.", "",
        f"- repos with a readable tree: **{len(rows)}**",
        f"- repos whose tree could not be read: {empty}",
        f"- scoring above zero and larger than {MIN_INTERESTING_FILES} files: **{len(worthwhile)}**", "",
        "Scored on capability signals in path names, weighted by where ForgeOS is",
        "actually weak — sandboxing highest (ForgeOS has none and ships a shell tool),",
        "retrieval lowest (the capsule and two-stage reranker are already ahead).", "",
        "## How common is each capability across the corpus", "",
        "| signal | repos showing it |", "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in signal_totals.most_common()]
    lines += [
        "", f"## Hydration order — top {args.top}", "",
        "Fetch these sequentially (never in parallel; this machine has been crashed",
        "by parallel cloning). Each is `git -C <path> fetch --refetch` or a checkout",
        "of the paths that matter.", "",
        "| # | repo | score | files | py | signals |", "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(worthwhile[:args.top], 1):
        sig = ", ".join(f"{k}×{v}" for k, v in sorted(r["signals"].items(),
                                                      key=lambda kv: -kv[1])[:5])
        lines.append(f"| {i} | `{r['name']}` | {r['score']} | {r['files']} | "
                     f"{r['python']} | {sig} |")

    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"triaged {len(rows)} readable trees ({empty} unreadable)")
    print(f"  worth hydrating: {len(worthwhile)}")
    print(f"  most common signals: {dict(signal_totals.most_common(6))}")
    print(f"  -> {args.out}, {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
