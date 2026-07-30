"""Fetch arXiv papers relevant to agent orchestration and token economics.

Uses the official arXiv API (authorised interface, 1 request / 3 seconds as their
docs require) rather than scraping pages — structured Atom in, abstracts included,
a fraction of the bytes of rendered HTML.

Saves metadata + abstracts only. Abstracts are the right granularity for deciding
what deserves a full read; pulling every PDF up front is exactly the waste this
harness is built to avoid.

Attribution, as arXiv asks: thank you to arXiv for use of its open access
interoperability.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "http://export.arxiv.org/api/query"
OUT = Path(r"C:\Users\byrne\Downloads\forgeos\research\papers")
RATE_LIMIT_SECONDS = 3.1  # arXiv asks for 1 request per 3 seconds

QUERIES: dict[str, str] = {
    "prompt_compression": 'all:"prompt compression" OR all:"context compression"',
    "kv_cache": 'all:"KV cache" AND all:"inference"',
    "prefix_caching": 'all:"prefix caching" OR all:"prompt caching"',
    "llm_routing": 'all:"LLM routing" OR all:"model routing" OR all:"model selection"',
    "multi_agent_llm": 'all:"multi-agent" AND all:"large language model"',
    "agent_orchestration": 'all:"agent orchestration" OR all:"agentic workflow"',
    "token_efficiency": 'all:"token efficiency" OR all:"token reduction"',
    "coding_agents": 'all:"coding agent" OR all:"software engineering agent"',
    "tool_use": 'all:"tool use" AND all:"language model" AND all:"agent"',
    "retrieval_context": 'all:"context engineering" OR all:"retrieval augmented" AND all:"code"',
    "cost_aware_inference": 'all:"cost-aware" AND all:"inference"',
    "self_verification": 'all:"self-verification" OR all:"self-correction" AND all:"language model"',
    "agent_memory": 'all:"agent memory" OR all:"long-term memory" AND all:"llm"',
    "speculative_decoding": 'all:"speculative decoding"',
    "cascade_inference": 'all:"model cascade" OR all:"LLM cascade"',
}

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def fetch(query: str, max_results: int = 40) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    req = urllib.request.Request(f"{API}?{params}", headers={"User-Agent": "forgeos-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed arXiv host
        root = ET.fromstring(r.read())

    out = []
    for e in root.findall("a:entry", NS):
        def txt(tag: str) -> str:
            el = e.find(tag, NS)
            return (el.text or "").strip() if el is not None else ""

        out.append(
            {
                "id": txt("a:id").rsplit("/", 1)[-1],
                "title": " ".join(txt("a:title").split()),
                "abstract": " ".join(txt("a:summary").split()),
                "published": txt("a:published"),
                "updated": txt("a:updated"),
                "authors": [
                    (a.find("a:name", NS).text or "").strip()
                    for a in e.findall("a:author", NS)
                    if a.find("a:name", NS) is not None
                ],
                "primary_category": (
                    e.find("arxiv:primary_category", NS).get("term")
                    if e.find("arxiv:primary_category", NS) is not None
                    else ""
                ),
                "comment": txt("arxiv:comment"),
                "url": txt("a:id"),
            }
        )
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index: dict[str, list[dict]] = {}
    seen: set[str] = set()
    total_new = 0

    for topic, q in QUERIES.items():
        try:
            papers = fetch(q)
        except Exception as exc:  # network/parse — report, do not abort the sweep
            print(f"{topic:24} FAILED {type(exc).__name__}: {exc}")
            time.sleep(RATE_LIMIT_SECONDS)
            continue

        fresh = [p for p in papers if p["id"] not in seen]
        seen.update(p["id"] for p in fresh)
        index[topic] = papers
        total_new += len(fresh)
        print(f"{topic:24} {len(papers):>3} results, {len(fresh):>3} new")
        time.sleep(RATE_LIMIT_SECONDS)

    (OUT / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")

    # Flat, deduped, newest first — the list a human or agent actually triages from.
    flat = {p["id"]: p for ps in index.values() for p in ps}
    ordered = sorted(flat.values(), key=lambda p: p["published"], reverse=True)
    (OUT / "papers.json").write_text(json.dumps(ordered, indent=1), encoding="utf-8")

    lines = ["# arXiv sweep — agent orchestration & token economics", ""]
    lines.append("Source: arXiv API. Thank you to arXiv for use of its open access interoperability.")
    lines.append("")
    for p in ordered:
        lines.append(f"- **{p['title']}** ({p['published'][:10]}, {p['primary_category']}) — {p['url']}")
    (OUT / "papers.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\ntopics: {len(index)}  unique papers: {len(flat)}")
    print(f"wrote {OUT / 'papers.json'} and papers.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
